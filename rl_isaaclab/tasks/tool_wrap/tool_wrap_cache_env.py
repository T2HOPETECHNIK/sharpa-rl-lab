'''
Collects reset states for the tool-wrap task by running the scripted close (the same one deploy replays):

    first_contact: the state at the pinch trigger (thumb + 2 fingertips touching something).
    pinch_lift:    after the trigger, squeeze a little more, lower the floor 1-3 cm, hold, and keep the state if the
                   pinch is still holding the tool off the floor. At training reset the floor goes back to the
                   recorded height, so the tool starts pinched and slightly lifted by the fingers.

Saved to {grasp_cache_path}_{cache_mode}.npy, CACHE_FIELDS floats per state.
'''

from __future__ import annotations

import os
import time

import numpy as np
import torch

from isaaclab.utils.math import quat_conjugate, quat_mul

from . import grip_metrics as gm
from .tool_wrap_env import CACHE_FIELDS, SharpaWaveToolWrapEnv

STAGE_CLOSE, STAGE_SQUEEZE, STAGE_LIFT, STAGE_HOLD = 0, 1, 2, 3


class SharpaWaveToolWrapCacheEnv(SharpaWaveToolWrapEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.stage = torch.zeros(n, device=dev, dtype=torch.long)
        self.stage_time = torch.zeros(n, device=dev)
        self.lift_goal = torch.zeros(n, device=dev)
        self.tool_start = torch.zeros((n, 3), device=dev)
        self.record = torch.zeros(n, device=dev, dtype=torch.bool)
        self.discard = torch.zeros(n, device=dev, dtype=torch.bool)
        self.saved = []
        self.num_saved = 0
        self.cache_done = False
        self.out_path = f"{cfg.grasp_cache_path}_{cfg.cache_mode}.npy"

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        was_closing = self.scripted.clone()
        self._advance_scripted_close()
        targets = self._scripted_targets()
        self.cur_targets[:, self.actuated_dof_indices] = torch.maximum(torch.minimum(
            targets[:, self.actuated_dof_indices], self.hand_dof_upper_limits[:, self.actuated_dof_indices]),
            self.hand_dof_lower_limits[:, self.actuated_dof_indices])
        handed_over = was_closing & ~self.scripted
        triggered = handed_over & (self.close_progress < 2.0 + cfg.close_finger_delay)
        # the close ran out without the tactile trigger (often the thumb touches with the side of its distal segment,
        # not the pad). The policy takes over from there on the real hand too, so keep it if it is a pinch.
        ran_out = handed_over & ~triggered
        if ran_out.any():
            ran_out &= self._compute_metrics()["pinch"]
        pushed = torch.linalg.norm(self.object_pos - self.tool_start, dim=-1) > cfg.max_tool_push
        started = (triggered | ran_out) & ~pushed
        if cfg.cache_mode == "first_contact":
            self.record |= started
        else:
            self._pinch_lift_stages(started)
        self.discard |= handed_over & ~started

    def _pinch_lift_stages(self, started):
        cfg, dt = self.cfg, self.step_dt
        self.stage_time += dt
        start = started & (self.stage == STAGE_CLOSE)
        self.stage[start], self.stage_time[start] = STAGE_SQUEEZE, 0.0
        # squeeze: keep advancing the close a little past the trigger (targets always follow the close here)
        squeezing = self.stage == STAGE_SQUEEZE
        self.close_progress += squeezing.float() * self.close_speed * dt
        done_squeeze = squeezing & (self.stage_time >= cfg.squeeze_s)
        if done_squeeze.any():
            ids = done_squeeze.nonzero(as_tuple=False).squeeze(-1)
            self.stage[ids], self.stage_time[ids] = STAGE_LIFT, 0.0
            self.lift_goal[ids] = self.floor_top0[ids] - torch.empty(len(ids), device=self.device).uniform_(*cfg.lift_range)
        # lower the floor (relative to the hand this is the fingers lifting the tool)
        lifting = (self.stage == STAGE_LIFT).nonzero(as_tuple=False).squeeze(-1)
        if len(lifting) > 0:
            self.floor_top[lifting] = torch.maximum(self.floor_top[lifting] - cfg.lift_speed * dt, self.lift_goal[lifting])
            self._write_floor(lifting)
            reached = lifting[self.floor_top[lifting] <= self.lift_goal[lifting] + 1e-5]
            self.stage[reached], self.stage_time[reached] = STAGE_HOLD, 0.0
        held = (self.stage == STAGE_HOLD) & (self.stage_time >= cfg.hold_s)
        if held.any():
            m = self._compute_metrics()
            clearance = m["grip_point"][:, 2] - self.tool_radius - self.floor_top
            still = torch.linalg.norm(self.object_linvel, dim=-1) < cfg.max_tool_speed
            ok = held & m["pinch"] & (clearance > 0.005) & still
            self.record |= ok
            self.discard |= held & ~ok

    def _get_dones(self):
        self._refresh_lab()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        lost = torch.linalg.norm(self.object_pos - self.tool_start, dim=-1) > 0.15
        if self.record.any():
            self._save_states(self.record.nonzero(as_tuple=False).squeeze(-1))
        done = self.record | self.discard | lost
        return done, time_out

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _save_states(self, ids):
        if self.cache_done:
            return
        rows = torch.zeros((len(ids), CACHE_FIELDS), device=self.device)
        rows[:, 0] = self.tool_id[ids].float()
        rows[:, 1] = self.palm_point[ids, 2] - self.floor_top[ids]
        rows[:, 2:24] = self.hand_dof_pos[ids]
        rows[:, 24:46] = self.cur_targets[ids]
        rows[:, 46:49] = gm.quat_apply_inverse(self.hand_quat[ids], self.object_pos[ids] - self.hand_pos[ids])
        rows[:, 49:53] = quat_mul(quat_conjugate(self.hand_quat[ids]), self.object_rot[ids])
        self.saved.append(rows.cpu())
        self.num_saved += len(ids)
        print(f"[{time.strftime('%H:%M:%S')}] {self.cfg.cache_mode}: {self.num_saved}/{self.cfg.cache_size} states")
        if self.num_saved >= self.cfg.cache_size:
            data = torch.cat(self.saved)[: self.cfg.cache_size].numpy()
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            np.save(self.out_path, data)
            counts = np.bincount(data[:, 0].astype(int), minlength=self.cfg.num_tools)
            print(f"saved {len(data)} states to {self.out_path}; per tool: {counts.tolist()}")
            self.cache_done = True

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.stage[env_ids] = STAGE_CLOSE
        self.stage_time[env_ids] = 0.0
        self.record[env_ids] = False
        self.discard[env_ids] = False
        self.tool_start[env_ids] = self.object_pos[env_ids]
