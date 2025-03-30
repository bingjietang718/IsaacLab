import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .assembly_env_cfg import AssemblyEnvCfg, OBS_DIM_CFG, STATE_DIM_CFG
from .assembly_env import AssemblyEnv

import json
import wandb
from datetime import datetime
import warp as wp
from . import industreal_algo_utils as industreal_algo
from . import automate_algo_utils as automate_algo
from . import automate_log_utils as automate_log
from soft_dtw_cuda import SoftDTW

class AssemblySparseEnv(AssemblyEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        
        super().__init__(cfg, render_mode, **kwargs)

    def _get_rewards(self):
        """ Update rewards and compute success statistics. """
        # Get successful and failed envs at current timestep

        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos, 
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf
        )

        rew_buf = self._update_rew_buf(curr_successes)
        #self.ep_succeeded = torch.logical_or(self.ep_succeeded, curr_successes)

        wandb.log(self.extras)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(self.ep_succeeded) / self.num_envs

            sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
                    curr_max_disp=self.curr_max_disp,
                    curriculum_height_bound=self.curriculum_height_bound,
                )

            rew_buf *= sbc_rwd_scale

            if self.cfg_task.if_sbc:

                self.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(self.ep_succeeded) / self.num_envs,
                    cfg_task=self.cfg_task,
                    curriculum_height_bound=self.curriculum_height_bound, 
                    curriculum_height_step=self.curriculum_height_step,
                    curr_max_disp=self.curr_max_disp
                )

            self.extras["curr_max_disp"] = self.curr_max_disp

            wandb.log({
                'success': torch.mean(self.ep_succeeded.float()),
                'reward': torch.mean(rew_buf),
                'sbc_rwd_scale': sbc_rwd_scale
                }
            )

            if self.cfg_task.if_logging_eval:
                self.success_log = torch.cat(
                        [
                            self.success_log, 
                            self.ep_succeeded.reshape((self.num_envs, 1))
                        ], 
                    dim=0)

                if self.success_log.shape[0] >= self.cfg_task.num_eval_trials:
                    automate_log.write_log_to_hdf5(
                        self.held_asset_pose_log,
                        self.fixed_asset_pose_log,
                        self.success_log,
                        self.eval_logging_filename
                    )
                    exit(0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """ Compute reward at current timestep. """
        rew_dict = {}

        rew_dict['curr_successes'] = curr_successes.clone().float()

        # Sparse Reward
        self.ep_succeeded = torch.logical_or(self.ep_succeeded, curr_successes)
        rew_dict['sparse'] = self.ep_succeeded 

        rew_buf = rew_dict['sparse']

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
            joints=self.cfg.ctrl.reset_joints,
            env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self.cfg_task.if_logging_eval:
            self.held_asset_pose_log = torch.cat(
                    [
                        self.held_asset_pose_log, 
                        torch.cat([self.held_pos, self.held_quat], dim=1)
                    ], 
                dim=0)
            self.fixed_asset_pose_log = torch.cat(
                    [
                        self.fixed_asset_pose_log, 
                        torch.cat([self.fixed_pos, self.fixed_quat], dim=1)
                    ], 
                dim=0)

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos-self.gripper_goal_pos).unsqueeze(1) # (num_envs, 1, 3)
        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(prev_fingertip_midpoint_pos, 
                                                                    self.cfg_task.num_point_robot_traj, 
                                                                    dim=1) # (num_envs, num_point_robot_traj, 3)

    def _set_assets_to_default_pose(self, env_ids):
        """ Move assets to default pose before randomization. """
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """Define grasp pose for plug and move gripper to pose."""

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            gripper_goal_quat, 
            gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        # ctrl_target_fingertip_centered_euler = (
        #     torch.tensor(
        #         self.cfg_task.hand_init_orn,
        #         device=self.device,
        #     )
        #     .unsqueeze(0)
        #     .repeat(self.num_envs, 1)
        # )

        # self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
        #     ctrl_target_fingertip_centered_euler[:, 0],
        #     ctrl_target_fingertip_centered_euler[:, 1],
        #     ctrl_target_fingertip_centered_euler[:, 2],
        # )
        self.ctrl_target_fingertip_midpoint_quat = gripper_goal_quat.clone()

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(self, env_ids):
        """ Set robot joint position using DLS IK. """
        ik_time = 0.0
        while ik_time < 0.50:
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
        # gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        # gripper_width = self.cfg_task.hand_width_max / 3.0
        gripper_width = self.gripper_open_width
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

        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp
        if pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (torch.rand((self.num_envs,), dtype=torch.float32, device=self.device))

            if self.cfg_task.sample_from == 'rand':
                
                rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
                held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                held_asset_init_pos_rand = torch.tensor(
                    self.cfg_task.held_asset_init_pos_noise,
                    dtype=torch.float32, device=self.device)
                self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)

            if self.cfg_task.sample_from == 'gp':
                rand_sample = torch.rand((self.cfg_task.num_gp_candidates, 3), dtype=torch.float32, device=self.device)
                held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                held_asset_init_pos_rand = torch.tensor(
                    self.cfg_task.held_asset_init_pos_noise,
                    dtype=torch.float32, device=self.device)
                held_asset_init_candidates = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)
                self.held_pos_init_rand, _  = automate_algo.propose_failure_samples_batch_from_gp(
                                                self.gp, 
                                                held_asset_init_candidates.cpu().detach().numpy(), 
                                                len(env_ids),
                                                self.device)
                
                # self.curriculum_disp = self.cfg_task.curriculum_height_bound[0] * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

            if self.cfg_task.sample_from == 'gmm':
                self.held_pos_init_rand = automate_algo.sample_rel_pos_from_gmm(self.gmm, len(env_ids), self.device)
                # self.curriculum_disp = self.cfg_task.curriculum_height_bound[0] * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0
        
        # held_state[env_ids, 2] += self.cfg_task.fixed_asset_cfg.height
        # held_state[env_ids, 2] += self.cfg_task.fixed_asset_cfg.base_height 
        # held_state[env_ids, 2] += self.disassembly_dists
        held_state[env_ids, 2] += self.curriculum_disp

        plug_in_freespace_idx = torch.argwhere(
            self.curriculum_disp > self.disassembly_dists
        )
        held_state[plug_in_freespace_idx, :2] += self.held_pos_init_rand[plug_in_freespace_idx, :2]

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

        self.randomize_held_initial_state(env_ids, pre_grasp=False)

        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

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
