from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg

import omni.isaac.lab.sim as sim_utils

# ASSET_DIR = '../IsaacLab_benchmark/source/extensions/omni.isaac.lab_assets/data/Factory'
ASSET_DIR = '/home/bingjie/Downloads/assembly_asset'
DATA_DIR = ' '
# ASSEMBLY_ID= "factory_8mm"
ASSEMBLY_ID = '15654'

@configclass
class FixedAssetCfg:
    usd_path: str = ''
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0 # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05

@configclass
class HeldAssetCfg:
    usd_path: str = ''
    diameter: float = 0.0 # Used for gripper width.
    height: float = 0.0
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
    # fixed_asset_init_orn_range_deg: float = 360.0
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
    keypoint_coef_baseline: list = [5, 4]  # General movement towards fixed object.
    keypoint_coef_coarse: list = [50, 2]  # Movement to align the assets.
    keypoint_coef_fine: list = [100, 0]  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f'{ASSET_DIR}/{ASSEMBLY_ID}_1.usd'
    obj_path = f'{ASSET_DIR}/{ASSEMBLY_ID}_1.obj'
    # usd_path = f'{ASSET_DIR}/00731_plug.usd'
    diameter = 0.007986
    height = 0.050
    mass = 0.019

@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f'{ASSET_DIR}/{ASSEMBLY_ID}_0.usd'
    obj_path = f'{ASSET_DIR}/{ASSEMBLY_ID}_0.obj'
    # usd_path = f'{ASSET_DIR}/00731_socket.usd'
    diameter = 0.0081
    # height = 0.025
    height = 0.050896
    base_height = 0.0

@configclass
class Insertion(AssemblyTask):
    name = 'insertion'
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    # SDF reward
    num_mesh_sample_points = 1000

    # SBC
    initial_max_disp: float = 0.01  # max initial downward displacement of plug at beginning of curriculum
    curriculum_success_thresh: float = 0.75  # success rate threshold for increasing curriculum difficulty
    curriculum_failure_thresh: float = 0.5  # success rate threshold for decreasing curriculum difficulty
    curriculum_height_step: list = [-0.005, 0.003]  # how much to increase max initial downward displacement after hitting success or failure thresh
    curriculum_height_bound: list = [-0.01, 0.01]  # max initial downward displacement of plug at hardest and easiest stages of curriculum

    if_sbc: bool = True 

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    # fixed_asset_init_orn_range_deg: float = 360.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
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
            usd_path=fixed_asset_cfg.usd_path,
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
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={}
        ),
        actuators={}
    )
