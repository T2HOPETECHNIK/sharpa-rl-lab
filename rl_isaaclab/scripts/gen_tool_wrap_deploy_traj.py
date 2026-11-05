'''
Write the closing trajectory the real hand replays before the tool-wrap policy takes over (no simulator needed):

    python rl_isaaclab/scripts/gen_tool_wrap_deploy_traj.py --depth 0.04

--depth is the expected depth of the handle centre below the palm face (gap above the tool + handle radius), the
same quantity the sim keys its scripted close on. Output is in SharpaWave SDK joint order, radians.
'''

import argparse
import os

import numpy as np
import torch

from rl_isaaclab.tasks.tool_wrap.poses import ISAACLAB_JOINT_ORDER, KEYFRAME_DEPTHS, keyframe_tables
from rl_isaaclab.utils.misc import dof_isaaclab2sharpa

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=float, default=0.04, help="handle centre depth below the palm face (m)")
parser.add_argument("--steps", type=int, default=60, help="trajectory points (deploy sleeps init_traj_step_s between them)")
parser.add_argument("--out", type=str, default="cache/tool_wrap_deploy_init_traj.npy")
args = parser.parse_args()

depths, open_pose, desc, swept = (np.array(x) for x in keyframe_tables(ISAACLAB_JOINT_ORDER))
d = float(np.clip(args.depth, KEYFRAME_DEPTHS[0], KEYFRAME_DEPTHS[-1]))
desc_d = np.array([np.interp(d, depths, desc[:, j]) for j in range(desc.shape[1])])
swept_d = np.array([np.interp(d, depths, swept[:, j]) for j in range(swept.shape[1])])

half = args.steps // 2
a = np.linspace(0.0, 1.0, half)[:, None]
b = np.linspace(0.0, 1.0, args.steps - half + 1)[1:, None]
traj = np.concatenate([open_pose + (desc_d - open_pose) * a, desc_d + (swept_d - desc_d) * b])
traj_sdk = dof_isaaclab2sharpa(torch.tensor(traj, dtype=torch.float32)).numpy()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.save(args.out, traj_sdk)
print(f"saved {traj_sdk.shape} closing trajectory for depth {d:.3f} m to {args.out}")
