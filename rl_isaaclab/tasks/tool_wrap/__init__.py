import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Tool-Wrap-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.tool_wrap.tool_wrap_env:SharpaWaveToolWrapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tool_wrap_env_cfg:SharpaWaveToolWrapEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Tool-Wrap-Cache-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.tool_wrap.tool_wrap_cache_env:SharpaWaveToolWrapCacheEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tool_wrap_cache_env_cfg:SharpaWaveToolWrapCacheEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Tool-Wrap-Deploy-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.tool_wrap.tool_wrap_deploy_env:SharpaWaveToolWrapDeployEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tool_wrap_deploy_env_cfg:SharpaWaveToolWrapDeployEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)
