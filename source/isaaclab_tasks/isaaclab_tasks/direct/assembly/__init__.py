import gymnasium as gym
from . import agents
# from .assembly_env import AssemblyEnv, AssemblyEnvCfg
##
# Register Gym environments.
##
# gym.register(
#     id="Assembly-Direct-v0",
#     entry_point="omni.isaac.lab_tasks.direct.assembly:AssemblyEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": AssemblyEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
#     },
# )