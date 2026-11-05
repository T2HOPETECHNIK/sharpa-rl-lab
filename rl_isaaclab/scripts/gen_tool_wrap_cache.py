# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Collect tool-wrap reset states: python rl_isaaclab/scripts/gen_tool_wrap_cache.py --mode first_contact --headless'''

import argparse
import shutil
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate tool-wrap reset caches.")
parser.add_argument("--task", type=str, default="Isaac-Tool-Wrap-Cache-Sharpa-Wave-v0", help="Name of the task.")
parser.add_argument("--mode", type=str, default="first_contact", choices=["first_contact", "pinch_lift"])
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--num_states", type=int, default=None, help="States to collect (default: cfg.cache_size).")
parser.add_argument("--cache", type=str, default=None, help="Cache path prefix (default: cfg.grasp_cache_path).")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

import rl_isaaclab.tasks.tool_wrap
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree('outputs/', ignore_errors=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.cache_mode = args_cli.mode
    if args_cli.num_states is not None:
        env_cfg.cache_size = args_cli.num_states
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    while simulation_app.is_running() and not env.unwrapped.cache_done:
        env.step(env.zero_actions())
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
