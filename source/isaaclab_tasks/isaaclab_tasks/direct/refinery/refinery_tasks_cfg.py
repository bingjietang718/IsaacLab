import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = '/home/bingjie/Downloads/all_assembly_asset'

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

@configclass
class FixedAssetCfg:
    usd_path: str = ''
    friction: float = 0.75
    mass: float = 0.05

@configclass
class HeldAssetCfg:
    usd_path: str = ''
    friction: float = 0.75
    mass: float = 0.05

@configclass
class RobotCfg:
    robot_usd: str = ''
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75

@configclass
class AssemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ''
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    palm_to_finger_dist: float = 0.1134

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0., 0., 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_threading task only.
    action_penalty_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    # SDF reward
    sdf_rwd_scale: float = 1.0
    num_mesh_sample_points: int = 1000

    # Imitation reward
    imitation_rwd_scale: float = 1.0
    soft_dtw_gamma: float = 0.01 # set to 0 if want to use the original DTW without any smoothing
    num_point_robot_traj: int = 10 # number of waypoints included in the end-effector trajectory
    
    # SBC
    initial_max_disp: float = 0.01  # max initial downward displacement of plug at beginning of curriculum
    curriculum_success_thresh: float = 0.8  # success rate threshold for increasing curriculum difficulty
    curriculum_failure_thresh: float = 0.5  # success rate threshold for decreasing curriculum difficulty
    curriculum_freespace_range: float = 0.01
    num_curriculum_step: int = 10
    curriculum_height_step: list = [-0.005, 0.003]  # how much to increase max initial downward displacement after hitting success or failure thresh
    
    if_sbc: bool = False

    # Logging evaluation results
    if_logging_eval: bool = False
    num_eval_trials: int = 1000
    eval_filename: str = 'evaluation_10000_4.h5'

    # Fine-tuning
    sample_from: str = 'gmm'
    num_gp_candidates: int = 1000
    acquisition_function: str = 'ucb'

@configclass
class HeldAsset(HeldAssetCfg):
    usd_path = '10000_4.usd'
    obj_path = '10000_4.obj'
    diameter = 0.007986
    height = 0.050
    mass = 0.019

@configclass
class FixedAsset(FixedAssetCfg):
    usd_path = '10000_0.usd'
    obj_path = '10000_0.obj'
    diameter = 0.0081
    height = 0.050896
    base_height = 0.0

## start: add assembled asset config classes
@configclass
class AssembledAsset1(FixedAssetCfg):
    usd_path = '10000_1.usd'
    obj_path = '10000_1.obj'
    diameter = 0.007986
    height = 0.050
    mass = 0.019

@configclass
class AssembledAsset2(FixedAssetCfg):
    usd_path = '10000_2.usd'
    obj_path = '10000_2.obj'
    diameter = 0.007986
    height = 0.050
    mass = 0.019

@configclass
class AssembledAsset3(FixedAssetCfg):
    usd_path = '10000_3.usd'
    obj_path = '10000_3.obj'
    diameter = 0.007986
    height = 0.050
    mass = 0.019

## end: add assembled asset config classes


@configclass
class Insertion(AssemblyTask):
    name = 'insertion'

    assembly_id = '10000'
    step_id = '4'
    assembly_dir = f'{ASSET_DIR}/{assembly_id}/'

    fixed_asset_cfg = FixedAsset()
    held_asset_cfg = HeldAsset()
    ## start: add assembled asset in task classes
    assembled_asset_1_cfg = AssembledAsset1()
    assembled_asset_2_cfg = AssembledAsset2()
    assembled_asset_3_cfg = AssembledAsset3()
    ## end: add assembled asset in task classes
    

    asset_size = 8.0
    duration_s = 10.0

    plug_grasp_path = f'{assembly_dir}/{assembly_id}_{step_id}.out'
    disassembly_dist_json = f'{assembly_dir}/disassembly_dist.json'
    disassembly_direction = f'{assembly_dir}/disassembly_direction.json'
    disassembly_path_json = f'{assembly_dir}/{assembly_id}_{step_id}_disassembly_traj.json'

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    hand_width_max: float = 0.080  # maximum opening width of gripper

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # Held Asset (applies to all tasks)
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    engage_height_thresh: float = 0.01
    success_height_thresh: float = 0.003
    close_error_thresh: float = 0.015

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f'{assembly_dir}{fixed_asset_cfg.usd_path}',
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True, # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={}
        ),
        actuators={}
    )

    ## start: add assembled asset articulationcfg
    assembled_asset1: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/AssembledAsset1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f'{assembly_dir}{assembled_asset_1_cfg.usd_path}',
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True, # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={}
    )

    assembled_asset2: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/AssembledAsset2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f'{assembly_dir}{assembled_asset_2_cfg.usd_path}',
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True, # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={}
    )

    assembled_asset3: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/AssembledAsset3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f'{assembly_dir}{assembled_asset_3_cfg.usd_path}',
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True, # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={}
    )

    ## end: add assembled asset articulationcfg
    

    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f'{assembly_dir}{held_asset_cfg.usd_path}',
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
                enabled_self_collisions=True,
                fix_root_link=False, # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
