'''
Real-hand loop for the tool-wrap task. Same flow as the rotation deploy ('e' start, 'w' freeze, 'q' home):
on start the hand replays the closing trajectory until the thumb and 2 fingertips read contact, then the policy
runs. After grip_phase_s the grip is done; on_grip_phase_done() is where your arm should start lifting.

Differences from the rotation deploy: the tactile observation is the fingertip force vector (from F6) instead of
its magnitude, and the frame layout is the tool-wrap one (74 values).
'''

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_deploy_env import SharpaWaveInhandRotateDeployEnv, unscale
from rl_isaaclab.utils.misc import dof_sharpa2isaaclab

from .f6_calibration import FINGERS


class SharpaWaveToolWrapDeployEnv(SharpaWaveInhandRotateDeployEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        rotation, sign = cfg.tactile_force_rotation, cfg.tactile_force_sign
        path = getattr(cfg, "tactile_calibration_path", None)
        if path and os.path.exists(path):
            with open(path) as f:
                calib = json.load(f)
            if calib.get("hand_side") not in (None, cfg.hand_side):
                print(f"[tool_wrap] WARNING: {path} was made for hand_side {calib['hand_side']}, deploying hand_side {cfg.hand_side}")
            rotation = [calib["digits"][f]["rotation"] for f in FINGERS]
            sign = [calib["digits"][f]["sign"] for f in FINGERS]
            print(f"[tool_wrap] loaded F6 calibration from {path} ({calib.get('created', 'unknown date')})")
        else:
            print(f"[tool_wrap] WARNING: no F6 calibration at {path}; fingertip forces use the uncalibrated fallback. "
                  f"Run rl_isaaclab/scripts/calibrate_tool_wrap_f6.py first.")
        self.force_rotation = torch.tensor(rotation, dtype=torch.float32, device=cfg.device)
        self.force_sign = torch.tensor(sign, dtype=torch.float32, device=cfg.device).reshape(5)
        self.policy_start_wall = None
        self.grip_phase_reported = False
        super().__init__(cfg, render_mode, **kwargs)

    def tactile_callback(self, frame):
        ch = frame['channel']
        img, f6, deform, p = None, None, None, None
        if frame['content'].get("RAW") is not None: img = frame['content']['RAW'].squeeze()
        if frame['content'].get('F6') is not None: f6 = frame['content']['F6']
        if frame['content'].get('CONTACT_POINT') is not None: p = frame['content']['CONTACT_POINT']

        f_vec = torch.zeros(3) if f6 is None else torch.tensor(f6, dtype=torch.float32)[:3]
        contact_pos = torch.zeros((3), dtype=torch.float32, device=self.device)
        if p is not None:
            p = torch.tensor(p).reshape(-1, 3)
            centroid = p[torch.argmax(p[:, 2])]
            contact_pos = self.tac_uv_map[ch-5*(1-self.cfg.hand_side)][int(centroid[1]), int(centroid[0])]
            contact_pos = torch.tensor(contact_pos[:3]) / 1000.0

        self.frames[ch] = img, f_vec, deform, contact_pos

    def get_tactile_info(self):
        '''Returns ((1, 15) fingertip force vectors in the sim elastomer frames, (1, 15) contact positions).'''
        force = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        contact_pos = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        if not self.cfg.enable_tactile:
            return force.reshape(1, -1), contact_pos.reshape(1, -1)
        for ch in range(5):
            _, f_vec, _, contact_pos_ch = self.frames[4-ch+5*(1-self.cfg.hand_side)]
            if f_vec is not None:
                force[ch] = self.force_sign[ch] * (self.force_rotation[ch] @ f_vec.to(self.device))
            if contact_pos_ch is not None:
                contact_pos[ch] = torch.as_tensor(contact_pos_ch, dtype=torch.float32, device=self.device)
        force[self.cfg.disable_tactile_ids] = 0.0
        contact_pos[self.cfg.disable_tactile_ids] = 0.0
        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        force *= self.cfg.force_scale
        norm = torch.linalg.norm(force, dim=-1, keepdim=True)
        below = (norm < self.cfg.contact_threshold).squeeze(-1)
        force[below] = 0.0
        contact_pos[below] = 0.0
        force = force * (self.cfg.tactile_force_clip / norm.clamp(min=self.cfg.tactile_force_clip))
        return force.reshape(1, -1), contact_pos.reshape(1, -1)

    def raw_fingertip_forces(self) -> np.ndarray:
        '''(5, 3) raw F6 force vectors (sensor frames, no calibration or scaling), thumb..pinky.'''
        out = np.zeros((5, 3))
        for ch in range(5):
            f_vec = self.frames[4-ch+5*(1-self.cfg.hand_side)][1]
            if f_vec is not None:
                out[ch] = f_vec.cpu().numpy()
        return out

    def compute_observations(self):
        forces, contact_pos = self.get_tactile_info()
        frame = torch.cat([
            unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
            self.cur_targets,
            forces,
            contact_pos,
        ], dim=-1).unsqueeze(1).float()  # the SDK may report angles as float64
        self.obs_buf_lag_history[:] = torch.cat([self.obs_buf_lag_history[:, 1:], frame], dim=1)
        at_reset = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset) > 0:
            self.obs_buf_lag_history[at_reset] = frame[at_reset].expand(-1, self.obs_buf_lag_history.shape[1], -1)
            self.at_reset_buf[at_reset] = 0
        obs_buf = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1).clone()
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        return obs_buf

    def _pinch_triggered(self) -> bool:
        forces, _ = self.get_tactile_info()
        norm = torch.linalg.norm(forces.reshape(5, 3), dim=-1)
        thumb = norm[0] > self.cfg.pinch_trigger_threshold
        fingers = int((norm[1:] > self.cfg.pinch_trigger_threshold).sum()) >= self.cfg.pinch_trigger_min_fingers
        return bool(thumb and fingers)

    def _reset_idx(self, env_ids):
        if self.cfg.warm_up:
            self.hand.set_speed_coeff(0.3)
            self.hand.set_current_coeff(self.cfg.current_coef)
            self.hand.set_joint_position([0] * 22)
            time.sleep(3)

        if self.cfg.keyboard_listen:
            if self.deploy_state_flag.get() == 0: return
            self.deploy_state_flag.set(3)

        self.episode_length_buf[env_ids] = 0
        self.hand.set_speed_coeff(self.cfg.speed_coef)
        self.hand.set_current_coeff(self.cfg.current_coef)

        # scripted close until the pinch trigger, mirroring the sim
        traj = np.load(self.cfg.init_traj_path)
        j = 0
        self.hand.set_joint_position(traj[j])
        time.sleep(0.5)
        while not self._pinch_triggered() and j + 1 < len(traj):
            j += 1
            self.hand.set_joint_position(traj[j])
            time.sleep(self.cfg.init_traj_step_s)
        print(f"pinch trigger at step {j} of {len(traj)}" + ("" if j + 1 < len(traj) else " (trajectory ended without a trigger)"))
        self.prev_targets[env_ids] = dof_sharpa2isaaclab(torch.tensor(traj[j], dtype=torch.float32, device=self.device))
        self.cur_targets[env_ids] = self.prev_targets[env_ids].clone()

        self._refresh_lab()
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        self.policy_start_wall = time.perf_counter()
        self.grip_phase_reported = False

    def step(self, action):
        out = super().step(action)
        if (self.policy_start_wall is not None and not self.grip_phase_reported
                and time.perf_counter() - self.policy_start_wall >= self.cfg.grip_phase_s):
            self.grip_phase_reported = True
            self.on_grip_phase_done()
        return out

    def on_grip_phase_done(self):
        '''Hook for the arm: the policy keeps running while the arm lifts the tool.'''
        print("[tool_wrap] grip phase done: lift the arm now")
