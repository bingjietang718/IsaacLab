import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .assembly_env_cfg import AssemblyEnvCfg, OBS_DIM_CFG, STATE_DIM_CFG
# from .torch_jit_utils import quat_to_angle_axis
from isaaclab.utils.math import axis_angle_from_quat

import warp as wp
from . import industreal_algo_utils as industreal_algo

class AssemblyEnv(DirectRLEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        # cfg.observation_space += cfg.action_space
        # cfg.state_space += cfg.action_space
        self.cfg_task = cfg.tasks[cfg.task_name]

        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_mesh_in_warp(self.cfg_task.held_asset_cfg.obj_path, 
                                                                                                            self.cfg_task.fixed_asset_cfg.obj_path, 
                                                                                                            self.cfg_task.num_mesh_sample_points, 
                                                                                                            self.wp_device)

    def _set_body_inertias(self):
        """ Note: this is to account for the asset_options.armature parameter in IGE. """
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """ Set parameters defining dynamic interactions. """
        self.default_gains = torch.tensor(
            self.cfg.ctrl.default_task_prop_gains,
            device=self.device
        ).repeat((self.num_envs, 1))

        self.pos_threshold = torch.tensor(
            self.cfg.ctrl.pos_action_threshold,
            device=self.device
        ).repeat((self.num_envs, 1))
        self.rot_threshold = torch.tensor(
            self.cfg.ctrl.rot_action_threshold,
            device=self.device
        ).repeat((self.num_envs, 1))

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """ Update material properties for a given asset. """
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """ Initialize tensors once. """
        self.identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros(
            (self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros(
            (self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        held_base_z_offset = 0.0

        self.held_base_pos_local = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device
        ).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        self.gripper_goal_pos_local = torch.tensor(
            # [0.0, 0.0, 0.03], device=self.device
            [0.0, 0.0, 0.01], device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index('panda_leftfinger')
        self.right_finger_body_idx = self._robot.body_names.index('panda_rightfinger')
        self.fingertip_body_idx = self._robot.body_names.index('panda_fingertip_centered')

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros(
            (self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros(
            (self.num_envs, self.cfg_task.num_keypoints, 3),device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = 0.0

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # SBC
        if self.cfg_task.if_sbc:
            self.curr_max_disp = self.cfg_task.initial_max_disp
        else:
            self.curr_max_disp = self.cfg_task.curriculum_height_bound[0]

    def _get_keypoint_offsets(self, num_keypoints):
        """ Get uniformly-spaced keypoints along a line of unit length, centered at 0. """
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """ Initialize simulation scene. """
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -0.4)
        )

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func("/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711))

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """ Get values computed from raw tensors. This includes adding noise. """
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            self.identity_quat,
            self.gripper_goal_pos_local,
        )

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat,
            torch_utils.quat_conjugate(self.prev_fingertip_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)

        self.ee_angvel_fd = (rot_diff_aa[0].unsqueeze(-1) * rot_diff_aa[1]) / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos,
            self.held_base_quat_local, self.held_base_pos_local)
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos,
            self.identity_quat, self.fixed_success_pos_local)

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos,
                self.identity_quat, keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat, self.target_held_base_pos,
                self.identity_quat, keypoint_offset.repeat(self.num_envs, 1))[1]

        self.keypoint_dist = torch.norm(
            self.keypoints_held - self.keypoints_fixed,
            p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """ Get actor/critic inputs using assymetric critic. """
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            'joint_pos': self.joint_pos[:, 0:7],
            'fingertip_pos': self.fingertip_midpoint_pos,
            'fingertip_quat': self.fingertip_midpoint_quat,
            'fingertip_goal_pos': self.gripper_goal_pos,
            'fingertip_goal_quat': self.identity_quat,
            'delta_pos': self.gripper_goal_pos - self.fingertip_midpoint_pos, 
        }

        state_dict = {
            'joint_pos': self.joint_pos[:, 0:7],
            'joint_vel': self.joint_vel[:, 0:7],
            'fingertip_pos': self.fingertip_midpoint_pos,
            'fingertip_quat': self.fingertip_midpoint_quat,
            'ee_linvel': self.fingertip_midpoint_linvel,
            'ee_angvel': self.fingertip_midpoint_angvel,
            'fingertip_goal_pos': self.gripper_goal_pos,
            'fingertip_goal_quat': self.identity_quat,
            'held_pos': self.held_pos,
            'held_quat': self.held_quat,
            'delta_pos': self.gripper_goal_pos - self.fingertip_midpoint_pos, 
        }
        # obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ['prev_actions']]
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order]
        obs_tensors = torch.cat(obs_tensors, dim=-1)

        # state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ['prev_actions']]
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order]
        state_tensors = torch.cat(state_tensors, dim=-1)

        return {'policy': obs_tensors, 'critic': state_tensors}

    def _reset_buffers(self, env_ids):
        """ Reset buffers. """
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """ Apply policy actions with smoothing. """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.cfg.ctrl.ema_factor*action.clone().to(self.device) + (1-self.cfg.ctrl.ema_factor)*self.actions

    def close_gripper_in_place(self):
        """ Keep gripper in current position as gripper closes. """
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat,
            self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(
            torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat),
            dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """ Apply actions for policy as delta targets from current position. """
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1])
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                self.num_envs, 1
            )
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)
        
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.):
        """ Set robot gains using critical damping. """
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """ Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm). """
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,#_fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device)

        # set target for gripper joints to use GYM's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """ Update intermediate values used for rewards and observations. """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_rewards(self):
        """ Update rewards and compute success statistics. """
        # Get successful and failed envs at current timestep

        curr_successes = industreal_algo.check_plug_inserted_in_socket(
            self.held_pos, 
            self.fixed_pos,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.success_height_thresh,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf
        )

        rew_buf = self._update_rew_buf(curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

            if self.cfg_task.if_sbc:
            
                rew_buf *= industreal_algo.get_curriculum_reward_scale(
                    cfg_task=self.cfg_task,
                    curr_max_disp=self.curr_max_disp
                )

                self.curr_max_disp = industreal_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(curr_successes) / self.num_envs,
                    cfg_task=self.cfg_task,
                    curr_max_disp=self.curr_max_disp
                )

            self.extras["curr_max_disp"] = self.curr_max_disp

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """ Compute reward at current timestep. """
        rew_dict = {}

        # SDF-based reward.
        rew_dict['sdf'] = industreal_algo.get_sdf_reward(
                            self.plug_mesh,
                            self.plug_sample_points,
                            self.held_pos,
                            self.held_quat,
                            self.fixed_pos,
                            self.fixed_quat,
                            self.wp_device,
                            self.device,
                        )
        
        rew_dict['curr_successes'] = curr_successes.clone().float()

        curr_engaged = industreal_algo.check_plug_inserted_in_socket(
            self.held_pos, 
            self.fixed_pos,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.engage_height_thresh,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf
        )
        rew_dict['curr_engaged'] = curr_engaged.clone().float()

        rew_buf = rew_dict['sdf'] + rew_dict['curr_engaged'] + rew_dict['curr_successes']

        for rew_name, rew in rew_dict.items():
            self.extras[f'logs_rew_{rew_name}'] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(
            joints=[0.0, 0.0, 0.0, -1.870, 0.0, 1.8675, 0.785398],
            env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        """ Move assets to default pose before randomization. """
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_state_to_sim(held_state, env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """Define grasp pose for plug and move gripper to pose."""

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            self.identity_quat,
            self.gripper_goal_pos_local,
        )

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        ctrl_target_fingertip_centered_euler = (
            torch.tensor(
                self.cfg_task.hand_init_orn,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2],
        )

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(self, env_ids):
        """ Set robot joint position using DLS IK. """
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type='geometric',
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method='dls',
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device)
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """ Return Franka to its default joint position. """
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        print(f'Resetting {len(env_ids)} envs...')
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """ Step the simulation without an action. Used for resets. """
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise,
            dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        fixed_state[:, 3] += 0.1435

        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0],
            fixed_orn_euler[:, 1],
            fixed_orn_euler[:, 2])
        fixed_state[:, 3:7]  = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(
            self.cfg.obs_rand.fixed_asset_pos,
            dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):

        curr_curriculum_disp_range = self.curr_max_disp - self.cfg_task.curriculum_height_bound[0]
        if pre_grasp:
            self.curriculum_disp = self.cfg_task.curriculum_height_bound[0] + curr_curriculum_disp_range * (torch.rand((self.num_envs,), dtype=torch.float32, device=self.device))

            rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
            held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            held_asset_init_pos_rand = torch.tensor(
                self.cfg_task.held_asset_init_pos_noise,
                dtype=torch.float32, device=self.device)
            self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)
        # held_state[:, 0:3] += held_pos_init_rand + self.scene.env_origins[env_ids]

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0
        
        held_state[env_ids, 2] += self.cfg_task.fixed_asset_cfg.height
        held_state[env_ids, 2] += self.cfg_task.fixed_asset_cfg.base_height 
        held_state[env_ids, 2] -= self.curriculum_disp

        plug_partial_insert_idx = torch.argwhere(
            self.curriculum_disp < 0.0
        )
        held_state[plug_partial_insert_idx, :2] += self.held_pos_init_rand[plug_partial_insert_idx, :2]

        self._held_asset.write_root_state_to_sim(held_state)
        self._held_asset.reset()

        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids):
        """ Randomize initial state and perform any episode-level randomization. """
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos,
            self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self.randomize_held_initial_state(env_ids, pre_grasp=True)

        self._move_gripper_to_grasp_pose(env_ids)

        # self.randomize_held_initial_state(env_ids, pre_grasp=False)

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(
            self.cfg.ctrl.reset_task_prop_gains,
            device=self.device
        ).repeat((self.num_envs, 1))
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

            diff = self.target_held_base_pos - self.held_base_pos

        bad_idxs = env_ids[diff[:, 1] > 0.05]
        # print('Bad:', bad_idxs)

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        print('Done Reset')