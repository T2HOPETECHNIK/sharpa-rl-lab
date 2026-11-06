'''Run: python -m pytest rl_isaaclab/tasks/tool_wrap/tests -q   (no hand or simulator needed)'''

import json
import types

import numpy as np
import torch

from rl_isaaclab.utils.deploy_recorder import DeployRecorder


class FakeEnv:
    def __init__(self):
        self.cfg = types.SimpleNamespace(hand_side=1, control_freq=20, init_traj_path=None, tactile_calibration_path=None)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.hand_dof_pos = torch.zeros(1, 22)
        self.cur_targets = torch.zeros(1, 22)
        self.policy_start_wall = None

    @property
    def unwrapped(self):
        return self

    def raw_fingertip_forces(self):
        return np.ones((5, 3))


def test_parts_merge_into_one_file(tmp_path):
    env = FakeEnv()
    args = types.SimpleNamespace(load_path=None)
    rec = DeployRecorder(env, str(tmp_path), "task", args, flush_every=50)
    for i in range(130):
        env.episode_length_buf[0] = i
        rec.add(torch.full((1, 222), float(i)), torch.zeros(1, 22))
    assert len(rec.parts) == 2 and len(rec.rows["t"]) == 30  # flushed rows leave memory
    rec.close()
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [rec.path.split("/")[-1]]  # parts removed
    d = np.load(rec.path, allow_pickle=False)
    assert d["obs"].shape == (130, 222) and d["raw_f6"].shape == (130, 5, 3)
    assert np.array_equal(d["episode_step"], np.arange(130)) and np.array_equal(d["obs"][:, 0], np.arange(130))
    assert json.loads(str(d["meta_json"]))["task"] == "task"
