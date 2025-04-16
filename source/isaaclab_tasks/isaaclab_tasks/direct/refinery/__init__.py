import gymnasium as gym
from . import agents
from .refinery_env import RefineryEnv, RefineryEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Refinery-Direct-v0",
    entry_point="isaaclab_tasks.direct.refinery:RefineryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RefineryEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)