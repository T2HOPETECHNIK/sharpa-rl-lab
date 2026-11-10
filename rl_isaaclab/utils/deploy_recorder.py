'''
Per-step recorder for deploy.py (--record). Read-only: it copies state after each step and never changes what the
hand is sent. Every `flush_every` steps it writes the new rows to a part file next to the output
(<name>.partNNNN.npz, uncompressed, so the control loop isn't held up) and drops them from memory; close() merges the
parts into <name>.npz and removes them. A crash loses at most `flush_every` steps; the parts it leaves behind load
with np.load and concatenate in order.

Each row is one control step, recorded after the model's forward pass and before the action is sent, so obs,
dof_pos and cur_targets are the state the action was computed from. Per-step arrays (N = steps):
    t               (N,)      s since the recorder started
    deploy_state    (N,)      keyboard state: 0 home, 1 freeze, 2 reset, 3 run policy; -1 without keyboard
    episode_step    (N,)      env.episode_length_buf
    policy_time     (N,)      s since the last scripted close ended (tool-wrap only, nan otherwise)
    obs             (N, D)    policy observation before normalisation, exactly what deploy.py feeds the model
    action          (N, A)    model output (only applied while deploy_state == 3)
    dof_pos         (N, 22)   measured joint angles, Isaac Lab order
    cur_targets     (N, 22)   joint targets before this action, Isaac Lab order
    raw_f6          (N, 5, 3) raw fingertip F6 forces, sensor frames, thumb..pinky (tool-wrap only, nan otherwise)
    f6_baseline     (N, 5, 3) the rest reading deploy subtracts from raw_f6 before calibration (tool-wrap rest
                              re-zero; zeros when off, nan for envs without it). Added after v4; older files lack it.
Metadata: meta_json (args, cfg values, git commit, checkpoint md5, F6 calibration), init_traj (closing trajectory).
'''

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np

CFG_KEYS = ["hand_side", "enable_on_board", "control_freq", "speed_coef", "current_coef", "action_scale",
            "clip_actions", "init_traj_path", "init_traj_step_s", "grip_phase_s", "pinch_trigger_threshold",
            "pinch_trigger_min_fingers", "force_scale", "contact_threshold", "tactile_force_clip",
            "enable_contact_pos", "disable_tactile_ids", "tactile_calibration_path", "rest_tare", "rest_tare_s",
            "rest_warn_n", "rest_tare_max_spread_n"]


def _md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class DeployRecorder:
    def __init__(self, env, out_dir, task, args, flush_every=200):
        self.env = env.unwrapped
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".npz")
        self.flush_every = flush_every
        self.t0 = time.perf_counter()
        self.rows = {k: [] for k in ["t", "deploy_state", "episode_step", "policy_time", "obs", "action",
                                     "dof_pos", "cur_targets", "raw_f6", "f6_baseline"]}
        self.parts = []

        cfg = self.env.cfg
        meta = {
            "task": task,
            "args": vars(args),
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "checkpoint_md5": _md5(args.load_path) if args.load_path else None,
            "tactile_config": os.environ.get("SHARPA_TACTILE_CONFIG"),
            "cfg": {k: getattr(cfg, k) for k in CFG_KEYS if hasattr(cfg, k)},
        }
        calib_path = getattr(cfg, "tactile_calibration_path", None)
        if calib_path and os.path.exists(calib_path):
            with open(calib_path) as f:
                meta["f6_calibration"] = json.load(f)
        self.meta_json = json.dumps(meta, default=str)
        traj_path = getattr(cfg, "init_traj_path", None)
        self.init_traj = np.load(traj_path) if traj_path and os.path.exists(traj_path) else np.zeros((0,))
        print(f"[record] writing {self.path}")

    def add(self, obs, action):
        env = self.env
        r = self.rows
        r["t"].append(time.perf_counter() - self.t0)
        r["deploy_state"].append(env.deploy_state_flag.get() if hasattr(env, "deploy_state_flag") else -1)
        r["episode_step"].append(int(env.episode_length_buf[0]))
        start = getattr(env, "policy_start_wall", None)
        r["policy_time"].append(np.nan if start is None else time.perf_counter() - start)
        r["obs"].append(_np(obs)[0].astype(np.float32))
        r["action"].append(_np(action)[0].astype(np.float32))
        r["dof_pos"].append(_np(env.hand_dof_pos)[0].astype(np.float32))
        r["cur_targets"].append(_np(env.cur_targets)[0].astype(np.float32))
        raw = env.raw_fingertip_forces() if hasattr(env, "raw_fingertip_forces") else np.full((5, 3), np.nan)
        r["raw_f6"].append(np.asarray(raw, dtype=np.float32))
        base = getattr(env, "force_baseline", None)
        r["f6_baseline"].append(np.full((5, 3), np.nan, dtype=np.float32) if base is None else _np(base).astype(np.float32))
        if len(r["t"]) >= self.flush_every:
            self._write_part()

    def _write_part(self):
        if not self.rows["t"]:
            return
        arrays = {k: np.stack(v) if k in ("obs", "action", "dof_pos", "cur_targets", "raw_f6", "f6_baseline") else np.asarray(v)
                  for k, v in self.rows.items()}
        part = f"{self.path[:-len('.npz')]}.part{len(self.parts):04d}.npz"
        np.savez(part, **arrays)
        self.parts.append(part)
        for v in self.rows.values():
            v.clear()

    def close(self):
        self._write_part()
        if not self.parts:
            return
        chunks = [np.load(p) for p in self.parts]
        arrays = {k: np.concatenate([c[k] for c in chunks]) for k in self.rows}
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, meta_json=self.meta_json, init_traj=self.init_traj, **arrays)
        os.replace(tmp, self.path)
        for p in self.parts:
            os.remove(p)
        print(f"[record] saved {len(arrays['t'])} steps to {self.path}")
