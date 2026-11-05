from isaaclab.utils import configclass

from .tool_wrap_env_cfg import SharpaWaveToolWrapEnvCfg


@configclass
class SharpaWaveToolWrapCacheEnvCfg(SharpaWaveToolWrapEnvCfg):
    cache_mode = "first_contact"    # "first_contact" | "pinch_lift"; set by scripts/gen_tool_wrap_cache.py
    cache_size = 20000              # states to collect before saving
    episode_length_s = 6.0
    grip_phase_s = 100.0            # never start the stability test here
    reset_mix = (0.0, 0.0, 1.0)     # always start open, tool on the floor
    max_tool_push = 0.03            # discard a first contact that shoved the tool this far
    # pinch_lift: squeeze a little past the trigger, lower the floor, hold, keep states where the pinch held
    squeeze_s = 0.25
    lift_range = (0.01, 0.03)
    lift_speed = 0.05
    hold_s = 1.0
    max_tool_speed = 0.05
    # no perturbations while collecting
    randomize_pd_gains = False
