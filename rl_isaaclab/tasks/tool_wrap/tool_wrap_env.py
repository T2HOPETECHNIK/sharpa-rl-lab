'''
Tool wrap-grasp task: the palm-down SharpaWave hovers over an elongated tool lying on a floor. A scripted close
brings the thumb and fingertips onto opposite sides of the handle; the policy then lifts the tool slightly with
the fingers and works it into a wrap (fingers under the handle, handle against the palm). The episode ends with a
stability test: the floor drops away (the arm lifting, in the real setup) and the tool is pulled down and twisted.

Episode timeline (per env, from reset):
    [scripted close, fresh resets only] -> policy grip phase until grip_phase_s -> stability test until the end

Policy inputs only use what the hand has: joint angles, joint targets, and per-fingertip force vectors and contact
positions. Rewards and privileged info use simulator ground truth.
'''

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, saturate

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env import transform_between_frames, unscale

from . import grip_metrics as gm
from .poses import keyframe_tables, finger_of_joint, FINGERS
from .tool_library import TOOL_ID_ATTR, tool_metadata

if TYPE_CHECKING:
    from .tool_wrap_env_cfg import SharpaWaveToolWrapEnvCfg

CACHE_FIELDS = 53  # tool_id, floor_depth, joint_pos(22), joint_targets(22), tool_pos_hand(3), tool_quat_hand(4)
RESET_FIRST_CONTACT, RESET_PINCH_LIFT, RESET_FRESH = 0, 1, 2

# links whose centre lines are sampled for the cage check (joint origins, base -> tip)
_CHAINS = {
    "thumb": ["right_thumb_MC", "right_thumb_PP", "right_thumb_DP", "right_thumb_fingertip"],
    **{f: [f"right_{f}_PP", f"right_{f}_MP", f"right_{f}_DP", f"right_{f}_fingertip"] for f in ["index", "middle", "ring", "pinky"]},
}
_SAMPLES_PER_SEGMENT = 4


class SharpaWaveToolWrapEnv(DirectRLEnv):
    cfg: SharpaWaveToolWrapEnvCfg

    def __init__(self, cfg: SharpaWaveToolWrapEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.num_hand_dofs = self.hand.num_joints

        # joints and bodies
        self.actuated_dof_indices = sorted(self.hand.joint_names.index(j) for j in cfg.actuated_joint_names)
        self.fingertip_ids = [self.hand.body_names.index(b) for b in cfg.fingertip_body_names]
        self.elastomer_ids = [self.hand.body_names.index(b) for b in cfg.elastomer_body_names]
        self.chain_ids = torch.tensor([[self.hand.body_names.index(b) for b in _CHAINS[f]] for f in FINGERS], device=dev)
        self.joint_finger = torch.tensor([FINGERS.index(finger_of_joint(j)) for j in self.hand.joint_names], device=dev)

        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(dev)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0] * cfg.dof_limits_scale
        self.hand_dof_upper_limits = joint_pos_limits[..., 1] * cfg.dof_limits_scale
        self.effort_limits = self.hand.data.joint_effort_limits.clone() * cfg.effort_limit_scale

        self.p_gain_default = self.hand.data.default_joint_stiffness[:, self.actuated_dof_indices].clone()
        self.d_gain_default = self.hand.data.default_joint_damping[:, self.actuated_dof_indices].clone()
        self.p_gain = self.p_gain_default.clone()
        self.d_gain = self.d_gain_default.clone()
        if cfg.torque_control:
            self.hand.data.default_joint_stiffness = torch.zeros_like(self.p_gain_default)
            self.hand.data.default_joint_damping = torch.zeros_like(self.d_gain_default)
            for act in self.hand.actuators.values():
                act.stiffness = torch.zeros_like(act.stiffness)
                act.damping = torch.zeros_like(act.damping)

        # scripted close keyframes in this articulation's joint order
        depths, open_pose, desc, swept = keyframe_tables(self.hand.joint_names)
        self.kf_depths = torch.tensor(depths, device=dev)
        self.pose_open = torch.tensor(open_pose, device=dev)
        self.kf_desc = torch.tensor(desc, device=dev)
        self.kf_swept = torch.tensor(swept, device=dev)

        # hand frame (fixed base): everything below is relative to the env origin
        self.hand_pos = torch.tensor(cfg.hand_pos, device=dev).repeat(n, 1)
        self.hand_quat = torch.tensor(cfg.hand_rot, device=dev).repeat(n, 1)
        ex, ey, ez = (torch.eye(3, device=dev)[i].repeat(n, 1) for i in range(3))
        self.palm_normal = gm.quat_apply(self.hand_quat, ex)  # points at the floor
        self.finger_dir = gm.quat_apply(self.hand_quat, ez)
        self.palm_point = self.hand_pos + cfg.palm_face_offset * self.palm_normal
        self.grip_ref = self.hand_pos + gm.quat_apply(
            self.hand_quat, torch.tensor([cfg.palm_face_offset + cfg.grip_ref_depth, 0.0, cfg.grip_center_offset], device=dev).repeat(n, 1))
        self.down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(n, 1)
        palm_grid = [(cfg.palm_face_offset - 0.008, y, z) for y in (-0.03, 0.0, 0.03) for z in (0.045, 0.07, 0.095)]
        self.palm_points_local = torch.tensor(palm_grid, device=dev)
        t = torch.arange(_SAMPLES_PER_SEGMENT, device=dev, dtype=torch.float) / _SAMPLES_PER_SEGMENT
        self.segment_t = t
        finger_ids = [f for f in range(5) for _ in range(3 * _SAMPLES_PER_SEGMENT + 1)] + [-1] * len(palm_grid)
        self.point_finger_ids = torch.tensor(finger_ids, device=dev)

        # which tool each env got
        stage = get_current_stage()
        tool_ids = [stage.GetPrimAtPath(f"{p}/object").GetAttribute(TOOL_ID_ATTR).Get() for p in self.scene.env_prim_paths]
        self.tool_id = torch.tensor(tool_ids, device=dev, dtype=torch.long)
        meta = [tool_metadata(spec) for spec in cfg.tool_specs]
        def per_env(key):
            return torch.tensor([m[key] for m in meta], device=dev)[self.tool_id]
        self.tool_radius = per_env("radius")
        self.tool_half_length = per_env("half_length")
        self.tool_rest_height = per_env("rest_height")
        self.tool_grip_min = per_env("grip_min")
        self.tool_grip_max = per_env("grip_max")

        # buffers
        self.prev_targets = torch.zeros((n, self.num_hand_dofs), device=dev)
        self.cur_targets = torch.zeros((n, self.num_hand_dofs), device=dev)
        self.obs_buf_lag_history = torch.zeros((n, 80, cfg.frame_dim), device=dev)
        self.proprio_hist_buf = torch.zeros((n, cfg.prop_hist_len, cfg.frame_dim), device=dev)
        self.priv_info_buf = torch.zeros((n, cfg.priv_info_dim), device=dev)
        self.at_reset_buf = torch.ones(n, device=dev, dtype=torch.long)
        self.last_forces = torch.zeros((n, 5, 3), device=dev)
        self.prev_pad_forces = torch.zeros((n, 5, 3), device=dev)
        self.floor_top0 = torch.zeros(n, device=dev)
        self.floor_top = torch.zeros(n, device=dev)
        self.scripted = torch.zeros(n, device=dev, dtype=torch.bool)
        self.close_progress = torch.zeros(n, device=dev)
        self.close_speed = torch.ones(n, device=dev)
        self.finger_delay = torch.zeros((n, 5), device=dev)
        self.depth_est = torch.full((n,), 0.04, device=dev)
        self.in_test = torch.zeros(n, device=dev, dtype=torch.bool)
        self.test_ref_pos = torch.zeros((n, 3), device=dev)
        self.test_torque = torch.zeros((n, 3), device=dev)
        self.ext_forces = torch.zeros((n, 1, 3), device=dev)
        self.ext_torques = torch.zeros((n, 1, 3), device=dev)
        self.reset_source = torch.full((n,), RESET_FRESH, device=dev, dtype=torch.long)
        self.metrics: dict[str, torch.Tensor] = {}
        self._metrics_stale = True
        self.grip_steps = int(round(cfg.grip_phase_s / self.step_dt))
        self._contact_body_ids_disable = torch.tensor(cfg.disable_tactile_ids, dtype=torch.long)

        self._randomize_physics()
        self.caches = {
            RESET_FIRST_CONTACT: self._load_cache(f"{cfg.grasp_cache_path}_first_contact.npy"),
            RESET_PINCH_LIFT: self._load_cache(f"{cfg.grasp_cache_path}_pinch_lift.npy"),
        }

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.floor = RigidObject(self.cfg.floor_cfg)
        # catch plane far below the per-env floors
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["floor"] = self.floor
        self._contact_sensor = []
        for i, sensor_cfg in enumerate(self.cfg.contact_sensor):
            self._contact_sensor.append(ContactSensor(sensor_cfg))
            self.scene.sensors[f"contact_sensor_{i}"] = self._contact_sensor[i]
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _randomize_physics(self):
        cfg, n = self.cfg, self.num_envs
        cpu_ids = torch.arange(n, device="cpu")
        if cfg.randomize_friction:
            scale = torch.empty(n, 1).uniform_(*cfg.friction_scale_range)
            self._set_friction(self.object, scale * cfg.object_base_friction)
            materials = self.hand.root_physx_view.get_material_properties()
            assert materials.shape[1] == 26, "hand shape count changed: update the elastomer material ids below"
            hand_friction = scale.repeat(1, 26) * cfg.metal_base_friction
            elastomer_material_ids = [19, 20, 22, 24, 25]  # same asset as the in-hand rotation task
            hand_friction[:, elastomer_material_ids] = scale * cfg.elastomer_base_friction
            self._set_friction(self.hand, hand_friction)
            floor_friction = torch.empty(n, 1).uniform_(*cfg.floor_friction_range)
            self._set_friction(self.floor, floor_friction)
            self.priv_info_buf[:, 12] = scale.squeeze(-1).to(self.device)
            self.priv_info_buf[:, 13] = floor_friction.squeeze(-1).to(self.device)
        if cfg.randomize_mass:
            masses = self.object.root_physx_view.get_masses().clone()
            ratio = torch.empty_like(masses).uniform_(*cfg.mass_scale_range)
            inertias = self.object.root_physx_view.get_inertias().clone()
            self.object.root_physx_view.set_masses(masses * ratio, cpu_ids)
            self.object.root_physx_view.set_inertias(inertias * ratio.reshape(n, *([1] * (inertias.dim() - 1))), cpu_ids)
        if cfg.randomize_com:
            coms = self.object.root_physx_view.get_coms().clone()
            coms[:, :3] += torch.empty(n, 3).uniform_(-cfg.com_range, cfg.com_range)
            self.object.root_physx_view.set_coms(coms, cpu_ids)
        self.tool_mass = self.object.root_physx_view.get_masses().reshape(n).to(self.device)
        self.priv_info_buf[:, 8] = self.tool_mass
        self.priv_info_buf[:, 9:12] = self.object.root_physx_view.get_coms().reshape(n, -1)[:, :3].to(self.device)

    def _set_friction(self, asset, value):
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value
        materials[..., 1] = value
        asset.root_physx_view.set_material_properties(materials, torch.arange(self.num_envs, device="cpu"))

    def _load_cache(self, path):
        if not os.path.exists(path):
            print(f"[tool_wrap] no cache at {path}; those resets fall back to fresh (scripted close)")
            return None
        data = torch.from_numpy(np.load(path)).float().to(self.device)
        assert data.shape[1] == CACHE_FIELDS, f"{path}: expected {CACHE_FIELDS} fields per state, got {data.shape[1]}"
        tool = data[:, 0].long()
        order = torch.argsort(tool)
        data, tool = data[order], tool[order]
        counts = torch.bincount(tool, minlength=self.cfg.num_tools)
        print(f"[tool_wrap] loaded {len(data)} states from {path}")
        return {"data": data, "counts": counts, "starts": torch.cumsum(counts, 0) - counts}

    # ------------------------------------------------------------------ stepping

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        policy_targets = self.prev_targets + self.cfg.action_scale * self.actions
        if self.scripted.any():
            self._advance_scripted_close()
            policy_targets = torch.where(self.scripted.unsqueeze(-1), self._scripted_targets(), policy_targets)
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            policy_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self._step_stability_test()

    def _apply_action(self) -> None:
        self._refresh_lab()
        if self.cfg.torque_control:
            torques = self.p_gain * (self.cur_targets - self.hand_dof_pos) - self.d_gain * self.hand_dof_vel
            torques = torch.maximum(torch.minimum(torques, self.effort_limits), -self.effort_limits)
            self.hand.set_joint_effort_target(torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        else:
            self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    # ---- scripted close: OPEN -> DESC(depth) -> SWEPT(depth), stops at the pinch trigger

    def _keyframes_at(self, depth: torch.Tensor):
        d = depth.clamp(self.kf_depths[0], self.kf_depths[-1])
        hi = torch.searchsorted(self.kf_depths, d).clamp(1, len(self.kf_depths) - 1)
        lo = hi - 1
        w = ((d - self.kf_depths[lo]) / (self.kf_depths[hi] - self.kf_depths[lo])).unsqueeze(-1)
        desc = self.kf_desc[lo] * (1 - w) + self.kf_desc[hi] * w
        swept = self.kf_swept[lo] * (1 - w) + self.kf_swept[hi] * w
        return desc, swept

    def _scripted_targets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        ids = slice(None) if env_ids is None else env_ids
        desc, swept = self._keyframes_at(self.depth_est[ids])
        s = (self.close_progress[ids].unsqueeze(-1) - self.finger_delay[ids][:, self.joint_finger]).clamp(0.0, 2.0)
        open_pose = self.pose_open.expand_as(desc)
        stage1 = open_pose + (desc - open_pose) * s.clamp(max=1.0)
        stage2 = desc + (swept - desc) * (s - 1.0).clamp(min=0.0)
        return torch.where(s <= 1.0, stage1, stage2)

    def _advance_scripted_close(self):
        self.close_progress += self.scripted.float() * self.close_speed * self.step_dt
        net = self._fingertip_force_norm()
        thumb = net[:, 0] > self.cfg.pinch_trigger_threshold
        fingers = (net[:, 1:] > self.cfg.pinch_trigger_threshold).sum(-1) >= self.cfg.pinch_trigger_min_fingers
        # contact sensors still hold the previous episode's forces for a step or two after a reset
        armed = self.episode_length_buf >= 3
        finished = self.close_progress >= 2.0 + self.cfg.close_finger_delay
        self.scripted &= ~((thumb & fingers & armed) | finished)

    # ---- stability test: floor drops away, tool is pulled down and twisted

    def _step_stability_test(self):
        starting = (self.episode_length_buf >= self.grip_steps) & ~self.in_test
        if starting.any():
            ids = starting.nonzero(as_tuple=False).squeeze(-1)
            self.in_test[ids] = True
            self.test_ref_pos[ids] = self._tool_pos_in_hand()[ids]
            self.test_torque[ids] = torch.nn.functional.normalize(torch.randn(len(ids), 3, device=self.device), dim=-1) * self.cfg.test_twist_torque
        if self.in_test.any():
            ids = self.in_test.nonzero(as_tuple=False).squeeze(-1)
            self.floor_top[ids] = torch.maximum(self.floor_top[ids] - self.cfg.test_floor_speed * self.step_dt,
                                                self.floor_top0[ids] - self.cfg.test_floor_drop)
            self._write_floor(ids)
        t_test = ((self.episode_length_buf - self.grip_steps).float() * self.step_dt).clamp(min=0.0)
        ramp = (t_test / self.cfg.test_pull_ramp_s).clamp(max=1.0) * self.in_test.float()
        self.ext_forces[:, 0] = self.down * (self.cfg.test_pull_factor * 9.81 * self.tool_mass * ramp).unsqueeze(-1)
        self.ext_torques[:, 0] = self.test_torque * self.in_test.float().unsqueeze(-1)
        self.object.set_external_force_and_torque(forces=self.ext_forces, torques=self.ext_torques, is_global=True)

    def _write_floor(self, env_ids: torch.Tensor):
        pose = torch.zeros((len(env_ids), 7), device=self.device)
        pose[:, 0:2] = self.grip_ref[env_ids, 0:2]
        pose[:, 2] = self.floor_top[env_ids] - 0.5 * self.cfg.floor_size[2]
        pose[:, 3] = 1.0
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.floor.write_root_pose_to_sim(pose, env_ids)

    # ------------------------------------------------------------------ MDP

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        return {"policy": obs, "priv_info": self.priv_info_buf, "proprio_hist": self.proprio_hist_buf}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_lab()
        m = self._compute_metrics()
        lost = torch.linalg.norm(m["grip_point"] - self.grip_ref, dim=-1) > self.cfg.lost_distance
        fell = self.object_pos[:, 2] < self.floor_top0 - self.cfg.test_floor_drop - 0.05
        slipped = self.in_test & (torch.linalg.norm(self._tool_pos_in_hand() - self.test_ref_pos, dim=-1) > self.cfg.test_slip_limit)
        failed = lost | fell | slipped
        m["failed"] = failed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["lost"] = lost.float().mean()
        self.extras["test_slipped"] = slipped.float().mean()
        self.extras["time_out"] = time_out.float().mean()
        return failed, time_out

    def _get_rewards(self) -> torch.Tensor:
        cfg, m = self.cfg, self.metrics
        caged = m["caged"]
        r_pinch = m["pinch"].float() * (~caged).float()
        r_lift = torch.where(caged, torch.ones_like(m["lift"]), (m["lift"] / cfg.lift_target).clamp(0.0, 1.0))
        r_palm = torch.exp(-m["palm_gap"].clamp(min=0.0) / cfg.palm_reward_length)
        r_below = m["below"][:, 1:].float().mean(-1)
        hold = (self.in_test & ~m["failed"]).float()
        squeeze = self._fingertip_force_norm().sum(-1) * self.in_test.float()
        torque_penalty = (self.hand_dof_torque[:, self.actuated_dof_indices] ** 2).sum(-1)
        work_penalty = ((self.hand_dof_torque[:, self.actuated_dof_indices] * self.hand_dof_vel[:, self.actuated_dof_indices]).sum(-1)) ** 2
        reward = (
            cfg.pinch_reward_scale * r_pinch
            + cfg.lift_reward_scale * r_lift
            + cfg.palm_reward_scale * r_palm
            + cfg.coverage_reward_scale * m["coverage"]
            + cfg.below_reward_scale * r_below
            + cfg.cage_reward_scale * caged.float()
            + cfg.hold_reward_scale * hold
            + cfg.drift_penalty_scale * m["drift"]
            + cfg.squeeze_penalty_scale * squeeze
            + cfg.torque_penalty_scale * torque_penalty
            + cfg.work_penalty_scale * work_penalty
        )
        # the policy is not in control during the scripted close
        reward = torch.where(self.scripted, torch.zeros_like(reward), reward)
        reward += cfg.fail_penalty * (m["failed"] & ~self.scripted).float()

        for k, v in [("pinch", r_pinch), ("lift", m["lift"]), ("palm_gap", m["palm_gap"]), ("coverage", m["coverage"]),
                     ("below", r_below), ("caged", caged.float()), ("hold", hold), ("drift", m["drift"]),
                     ("squeeze", squeeze), ("scripted", self.scripted.float()), ("total_reward", reward)]:
            self.extras[k] = v.mean()
        caged_in_test = caged & self.in_test
        self.extras["caged_in_test"] = caged_in_test.float().sum() / self.in_test.float().sum().clamp(min=1.0)
        return reward

    def _compute_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        grip_point, axis = gm.closest_point_on_handle(self.object_pos, self.object_rot, self.tool_half_length, self.grip_ref)
        tool_force = self._fingertip_tool_force_norm()
        pinch = gm.pinch_formed(tool_force, self.fingertip_pos, grip_point, self.finger_dir, cfg.tool_contact_threshold)
        lift = grip_point[:, 2] - self.tool_radius - self.floor_top0
        palm_gap = gm.palm_gap(grip_point, self.tool_radius, self.palm_point, self.palm_normal)
        cage = gm.cage_metrics(self._hand_sample_points(), self.point_finger_ids, grip_point, axis, self.tool_radius,
                               self.down, near_margin=cfg.cage_near_margin)
        caged = cage["bottom_covered"] & (cage["max_gap_deg"] <= cfg.cage_max_gap_deg) & (palm_gap <= cfg.cage_palm_tolerance)
        along = ((grip_point - self.grip_ref) * self.finger_dir).sum(-1)
        drift = (along.abs() - cfg.drift_tolerance).clamp(min=0.0)
        self.metrics = {"grip_point": grip_point, "axis": axis, "tool_force": tool_force, "pinch": pinch, "lift": lift,
                        "palm_gap": palm_gap, "caged": caged, "drift": drift, **cage}
        return self.metrics

    # ------------------------------------------------------------------ reset

    def _rand_pd_scales(self, lower, upper, num_envs, n_dofs):
        rand_scale_s = torch.distributions.Uniform(lower, 1).sample((num_envs, n_dofs)).to(self.device)
        rand_scale_l = torch.distributions.Uniform(1, upper).sample((num_envs, n_dofs)).to(self.device)
        mask_choice = torch.rand((num_envs, n_dofs), device=self.device) > 0.5
        return torch.where(mask_choice, rand_scale_s, rand_scale_l)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        cfg, m = self.cfg, len(env_ids)

        if cfg.randomize_pd_gains:
            self.p_gain[env_ids] = self.p_gain_default[env_ids] * self._rand_pd_scales(
                cfg.randomize_p_gain_scale_lower, cfg.randomize_p_gain_scale_upper, m, self.num_hand_dofs)
            self.d_gain[env_ids] = self.d_gain_default[env_ids] * self._rand_pd_scales(
                cfg.randomize_d_gain_scale_lower, cfg.randomize_d_gain_scale_upper, m, self.num_hand_dofs)

        # pick a reset source per env; cache sources fall back to fresh when the tool has no cached states
        source = torch.multinomial(torch.tensor(cfg.reset_mix, device=self.device), m, replacement=True)
        joint_pos = torch.zeros((m, self.num_hand_dofs), device=self.device)
        joint_tgt = torch.zeros_like(joint_pos)
        tool_pose = torch.zeros((m, 7), device=self.device)
        floor_top = torch.zeros(m, device=self.device)
        use_fresh = source == RESET_FRESH
        for src in (RESET_FIRST_CONTACT, RESET_PINCH_LIFT):
            sel = source == src
            cache = self.caches[src]
            if not sel.any():
                continue
            if cache is None:
                use_fresh |= sel
                continue
            rows, ok = self._sample_cache(cache, env_ids[sel])
            use_fresh[sel] |= ~ok
            idx = sel.nonzero(as_tuple=False).squeeze(-1)[ok]
            rows = rows[ok]
            hand_pos, hand_quat = self.hand_pos[env_ids[idx]], self.hand_quat[env_ids[idx]]
            floor_top[idx] = self.palm_point[env_ids[idx], 2] - rows[:, 1]
            joint_pos[idx] = rows[:, 2:24]
            joint_tgt[idx] = rows[:, 24:46]
            tool_pose[idx, :3] = hand_pos + gm.quat_apply(hand_quat, rows[:, 46:49])
            tool_pose[idx, 3:7] = quat_mul(hand_quat, rows[:, 49:53])
        source = torch.where(use_fresh, torch.full_like(source, RESET_FRESH), source)

        fresh = use_fresh.nonzero(as_tuple=False).squeeze(-1)
        if len(fresh) > 0:
            fresh_ids = env_ids[fresh]
            pose, ftop, depth = self._sample_tool_on_floor(fresh_ids)
            tool_pose[fresh], floor_top[fresh] = pose, ftop
            noise = 0.05 * (2.0 * torch.rand((len(fresh), self.num_hand_dofs), device=self.device) - 1.0)
            joint_pos[fresh] = self.pose_open + noise
            joint_tgt[fresh] = joint_pos[fresh]
            self.depth_est[fresh_ids] = depth + cfg.depth_estimate_noise * torch.randn(len(fresh), device=self.device)
            self.close_progress[fresh_ids] = 0.0
            self.close_speed[fresh_ids] = torch.empty(len(fresh), device=self.device).uniform_(*cfg.close_speed_range)
            self.finger_delay[fresh_ids] = cfg.close_finger_delay * torch.rand((len(fresh), 5), device=self.device)
        self.scripted[env_ids] = use_fresh
        self.reset_source[env_ids] = source

        lower, upper = self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids]
        joint_pos = saturate(joint_pos, lower, upper)
        joint_tgt = saturate(joint_tgt, lower, upper)

        # floor, tool, hand
        self.floor_top0[env_ids] = floor_top
        self.floor_top[env_ids] = floor_top
        self._write_floor(env_ids)
        tool_pose[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(tool_pose, env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros((m, 6), device=self.device), env_ids)
        hand_state = self.hand.data.default_root_state.clone()[env_ids]
        hand_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.hand.write_root_state_to_sim(hand_state, env_ids)
        self.prev_targets[env_ids] = joint_tgt
        self.cur_targets[env_ids] = joint_tgt
        self.hand.set_joint_position_target(joint_tgt, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

        self.in_test[env_ids] = False
        self.test_torque[env_ids] = 0.0
        self.last_forces[env_ids] = 0.0
        self.prev_pad_forces[env_ids] = 0.0
        self.proprio_hist_buf[env_ids] = 0.0
        self.at_reset_buf[env_ids] = 1
        self._metrics_stale = True
        self._refresh_lab()

    def _sample_cache(self, cache, env_ids):
        tid = self.tool_id[env_ids]
        counts = cache["counts"][tid]
        ok = counts > 0
        offset = (torch.rand(len(env_ids), device=self.device) * counts.clamp(min=1)).long()
        offset = torch.minimum(offset, (counts - 1).clamp(min=0))
        return cache["data"][cache["starts"][tid] + offset], ok

    def _sample_tool_on_floor(self, env_ids: torch.Tensor):
        '''Tool lying on the floor under the hand: returns (tool pose rel. env origin, floor top z, handle centre depth).'''
        cfg, m, dev = self.cfg, len(env_ids), self.device
        rest = self.tool_rest_height[env_ids]
        gap = torch.empty(m, device=dev).uniform_(*cfg.gap_range)
        floor_top = self.palm_point[env_ids, 2] - gap - 2.0 * rest
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)  # which way the head points
        yaw = side * 0.5 * math.pi + math.radians(cfg.yaw_range_deg) * (2.0 * torch.rand(m, device=dev) - 1.0)
        quat = quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(m, 1))
        s = self.tool_grip_min[env_ids] + torch.rand(m, device=dev) * (self.tool_grip_max[env_ids] - self.tool_grip_min[env_ids])
        target = self.grip_ref[env_ids].clone()
        target += self.finger_dir[env_ids] * torch.empty(m, 1, device=dev).uniform_(*cfg.grip_offset_range)
        target[:, 2] = floor_top + rest + 0.002
        axis = gm.quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(m, 1))
        pose = torch.cat([target - s.unsqueeze(-1) * axis, quat], dim=-1)
        depth = self.palm_point[env_ids, 2] - (floor_top + rest)
        return pose, floor_top, depth

    # ------------------------------------------------------------------ sensing helpers

    def _refresh_lab(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_ids] - self.scene.env_origins.unsqueeze(1)
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.hand_dof_torque = self.hand.data.applied_torque
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w

    def _tool_pos_in_hand(self) -> torch.Tensor:
        return gm.quat_apply_inverse(self.hand_quat, self.object_pos - self.hand_pos)

    def _fingertip_forces_w(self) -> torch.Tensor:
        '''(N, 5, 3) contact force on each fingertip pad, as the F6 sensor feels it: the normal forces from every contact
        (net_forces_w is normal-only) plus the friction from the tool and the floor. Finger-on-finger friction is left out.'''
        normal = torch.cat([self._contact_sensor[i].data.net_forces_w[:, :1, :] for i in range(5)], dim=1)
        friction = torch.cat([torch.nan_to_num(self._contact_sensor[i].data.friction_forces_w[:, :1]).sum(dim=2) for i in range(5)], dim=1)
        return normal + self.cfg.pad_friction_sign * friction

    def _fingertip_force_norm(self) -> torch.Tensor:
        return torch.linalg.norm(self._fingertip_forces_w(), dim=-1)

    def _fingertip_tool_force_norm(self) -> torch.Tensor:
        '''(N, 5) contact force between each finger's pad or distal phalanx and the tool (sim only).'''
        f = torch.stack([torch.linalg.norm(self._contact_sensor[i].data.force_matrix_w[:, 0, 0, :], dim=-1) for i in range(10)], dim=1)
        return torch.maximum(f[:, :5], f[:, 5:])

    def _hand_sample_points(self) -> torch.Tensor:
        n = self.num_envs
        joints = self.hand.data.body_pos_w[:, self.chain_ids.reshape(-1)].reshape(n, 5, 4, 3) - self.scene.env_origins.reshape(n, 1, 1, 3)
        a, b = joints[:, :, :-1], joints[:, :, 1:]
        t = self.segment_t.reshape(1, 1, 1, -1, 1)
        seg = (a.unsqueeze(3) + (b - a).unsqueeze(3) * t).reshape(n, 5, -1, 3)
        fingers = torch.cat([seg, joints[:, :, -1:]], dim=2).reshape(n, -1, 3)
        palm = self.hand_pos.unsqueeze(1) + gm.quat_apply(
            self.hand_quat.unsqueeze(1).expand(-1, len(self.palm_points_local), -1),
            self.palm_points_local.unsqueeze(0).expand(n, -1, -1))
        return torch.cat([fingers, palm], dim=1)

    def compute_observations(self):
        cfg = self.cfg
        # fingertip force vectors in each elastomer's frame, smoothed, delayed and noised like the rotate task
        forces_now = self._fingertip_forces_w()
        forces_w = forces_now * cfg.contact_smooth + self.prev_pad_forces * (1.0 - cfg.contact_smooth)
        self.prev_pad_forces = forces_now
        elastomer_quat = self.hand.data.body_quat_w[:, self.elastomer_ids]
        forces = gm.quat_apply_inverse(elastomer_quat, forces_w)
        forces = forces + cfg.tactile_force_noise * torch.randn_like(forces)
        norm = torch.linalg.norm(forces, dim=-1, keepdim=True)
        forces = torch.where(norm < cfg.tactile_force_threshold, torch.zeros_like(forces), forces)
        forces = forces * (cfg.tactile_force_clip / norm.clamp(min=cfg.tactile_force_clip))
        hold = (torch.rand((self.num_envs, 5, 1), device=self.device) < cfg.contact_latency).float()
        self.last_forces = self.last_forces * hold + forces * (1.0 - hold)
        sensed = self.last_forces.clone()
        sensed[:, self._contact_body_ids_disable] = 0.0

        # contact position on the pad (as in the rotate task; off by default)
        contact_mask = torch.linalg.norm(sensed, dim=-1) > 0.0
        contact_pos = torch.cat([self._contact_sensor[i].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for i in range(5)], dim=1)
        contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
        tactile_frame = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        world_quat = torch.zeros_like(tactile_frame[..., 3:7])
        world_quat[..., 0] = 1.0
        contact_pos[contact_mask] = transform_between_frames(contact_pos[contact_mask] - tactile_frame[..., :3][contact_mask],
                                                             world_quat[contact_mask], tactile_frame[..., 3:7][contact_mask])
        contact_pos[~contact_mask] = 0.0
        if not cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        if not cfg.enable_tactile:
            sensed[:] = 0.0
            contact_pos[:] = 0.0

        joint_noise = (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0) * cfg.joint_noise_scale
        frame = torch.cat([
            unscale(joint_noise + self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
            self.cur_targets,
            sensed.reshape(self.num_envs, -1),
            contact_pos.reshape(self.num_envs, -1),
        ], dim=-1).unsqueeze(1)
        self.obs_buf_lag_history[:] = torch.cat([self.obs_buf_lag_history[:, 1:], frame], dim=1)
        at_reset = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset) > 0:
            self.obs_buf_lag_history[at_reset] = frame[at_reset].expand(-1, self.obs_buf_lag_history.shape[1], -1)
            self.at_reset_buf[at_reset] = 0
        obs = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1).clone()
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -cfg.prop_hist_len:].clone()

        # privileged info (sim ground truth): slots 8-13 are set in _randomize_physics
        if self._metrics_stale or not self.metrics:
            self._compute_metrics()
            self._metrics_stale = False
        mt = self.metrics
        self.priv_info_buf[:, 0:3] = gm.quat_apply_inverse(self.hand_quat, mt["grip_point"] - self.grip_ref)
        self.priv_info_buf[:, 3:6] = gm.quat_apply_inverse(self.hand_quat, mt["axis"])
        self.priv_info_buf[:, 6] = self.tool_radius
        self.priv_info_buf[:, 7] = self.tool_half_length
        self.priv_info_buf[:, 14] = mt["lift"]
        self.priv_info_buf[:, 15] = mt["palm_gap"]
        self.priv_info_buf[:, 16:21] = (mt["tool_force"] > cfg.tool_contact_threshold).float()
        self.priv_info_buf[:, 21:24] = self.object_linvel
        return obs
