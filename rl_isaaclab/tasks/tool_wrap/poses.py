'''
Hand poses for the scripted pinch close, shared by the sim env, the cache generator and the
deploy trajectory script. Pure python on purpose: deploy.py imports it without Isaac Lab.

Frame used below (right hand base link, right_hand_C_MC):
    x: palmar normal (points down when the palm faces the floor), palm face at x ~= +0.025 m
    y: lateral, index side +y, pinky side -y
    z: wrist -> fingertips, knuckle (MCP) line at z ~= 0.097 m

The close has two stages, keyed by the expected depth of the handle centre below the palm face:
    OPEN -> DESC(depth): fingertips come down in front of the handle (z ~= 0.142), thumb behind it (z ~= 0.072)
    DESC -> SWEPT(depth): fingertips sweep back toward the wrist (z ~= 0.095), thumb forward (z ~= 0.128)
so the fingertips meet the far side of the handle and the thumb the near side, near the handle's mid height.

KEYFRAMES were solved by forward kinematics on assets/SharpaWave/right_sharpa_wave.usda joint frames with the
joint limits scaled by 0.9 (dof_limits_scale). Abduction joints and pinky CMC are held at 0.
In sim the thumb often meets the handle with the side of its distal segment rather than the pad, so the pad
may stay under the trigger force and the close runs to the end; both sim and deploy then hand over anyway.
A pad-first thumb approach is worth solving for once the thumb pad direction is confirmed on hardware.
Check them visually in sim before trusting them on hardware.
'''

import math

# Isaac Lab articulation joint order for right_sharpa_wave (matches the limits table in
# tasks/inhand_rotate/sharpa_wave_deploy_env.py and the index maps in utils/misc.py).
ISAACLAB_JOINT_ORDER = [
    "right_index_MCP_FE", "right_middle_MCP_FE", "right_pinky_CMC", "right_ring_MCP_FE", "right_thumb_CMC_FE",
    "right_index_MCP_AA", "right_middle_MCP_AA", "right_pinky_MCP_FE", "right_ring_MCP_AA", "right_thumb_CMC_AA",
    "right_index_PIP", "right_middle_PIP", "right_pinky_MCP_AA", "right_ring_PIP", "right_thumb_MCP_FE",
    "right_index_DIP", "right_middle_DIP", "right_pinky_PIP", "right_ring_DIP", "right_thumb_MCP_AA",
    "right_pinky_DIP", "right_thumb_IP",
]

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# Fingers raised in a loose claw above the handle, thumb opposed but held back toward the wrist (degrees).
OPEN_DEG = {
    "right_thumb_CMC_FE": 29.5, "right_thumb_CMC_AA": -17.8, "right_thumb_MCP_FE": 71.4,
    "right_thumb_MCP_AA": -1.7, "right_thumb_IP": 89.1,
    "right_index_MCP_FE": 0.0, "right_index_MCP_AA": 0.0, "right_index_PIP": 30.0, "right_index_DIP": 20.0,
    "right_middle_MCP_FE": 0.0, "right_middle_MCP_AA": 0.0, "right_middle_PIP": 30.0, "right_middle_DIP": 20.0,
    "right_ring_MCP_FE": 0.0, "right_ring_MCP_AA": 0.0, "right_ring_PIP": 30.0, "right_ring_DIP": 20.0,
    "right_pinky_CMC": 0.0, "right_pinky_MCP_FE": 0.0, "right_pinky_MCP_AA": 0.0, "right_pinky_PIP": 30.0,
    "right_pinky_DIP": 20.0,
}

_FINGER_JOINTS = ["MCP_FE", "PIP", "DIP"]


def _kf(thumb, index, middle, ring, pinky):
    '''thumb: (CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP); fingers: (MCP_FE, PIP, DIP), degrees.'''
    pose = dict(zip([f"right_thumb_{j}" for j in ["CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP"]], thumb))
    for name, flex in zip(["index", "middle", "ring", "pinky"], [index, middle, ring, pinky]):
        pose.update({f"right_{name}_{j}": v for j, v in zip(_FINGER_JOINTS, flex)})
        pose[f"right_{name}_MCP_AA"] = 0.0
    pose["right_pinky_CMC"] = 0.0
    return pose


# depth of the handle centre below the palm face (m) -> {"desc": pose, "swept": pose}
KEYFRAMES = {
    0.02: {
        "desc": _kf((22.7, -7.8, 71.3, -4.3, 89.9), (-9.0, 75.0, 54.0), (-9.0, 81.0, 48.0), (-6.0, 64.5, 64.5), (-9.0, 63.0, 58.5)),
        "swept": _kf((35.7, 11.0, 45.9, 3.8, 28.2), (21.0, 90.0, 28.5), (27.0, 90.0, 22.5), (16.5, 90.0, 33.0), (7.5, 90.0, 42.0)),
    },
    0.03: {
        "desc": _kf((33.0, -16.3, 71.0, -13.0, 86.5), (1.5, 67.5, 48.0), (1.5, 73.5, 42.0), (3.0, 60.0, 55.5), (1.5, 55.5, 54.0)),
        "swept": _kf((41.6, 10.7, 40.7, -4.3, 28.7), (24.0, 90.0, 25.5), (28.5, 90.0, 21.0), (21.0, 90.0, 28.5), (13.5, 90.0, 36.0)),
    },
    0.04: {
        "desc": _kf((35.6, -16.1, 66.1, -17.8, 82.3), (12.0, 60.0, 40.5), (12.0, 66.0, 33.0), (13.5, 51.0, 48.0), (12.0, 48.0, 45.0)),
        "swept": _kf((49.3, 10.3, 33.5, -9.6, 27.3), (33.0, 90.0, 16.5), (36.0, 90.0, 13.5), (30.0, 90.0, 19.5), (24.0, 90.0, 25.5)),
    },
    0.05: {
        "desc": _kf((46.0, -17.7, 53.6, -17.6, 85.7), (22.5, 52.5, 27.0), (24.0, 54.0, 27.0), (24.0, 43.5, 36.0), (22.5, 40.5, 33.0)),
        "swept": _kf((61.6, 5.7, 28.5, -7.4, 17.3), (40.5, 82.5, 16.5), (43.5, 84.0, 9.0), (37.5, 84.0, 15.0), (33.0, 79.5, 25.5)),
    },
    0.06: {
        "desc": _kf((47.0, -17.9, 44.2, -16.8, 88.1), (36.0, 39.0, 15.0), (36.0, 43.5, 12.0), (37.5, 28.5, 27.0), (36.0, 28.5, 18.0)),
        "swept": _kf((64.7, 8.5, 20.5, -13.2, 9.5), (49.5, 69.0, 10.5), (52.5, 70.5, 3.0), (46.5, 72.0, 7.5), (42.0, 70.5, 12.0)),
    },
}

KEYFRAME_DEPTHS = sorted(KEYFRAMES)


def pose_rad(pose_deg: dict, joint_order: list[str] = ISAACLAB_JOINT_ORDER) -> list[float]:
    return [math.radians(pose_deg[name]) for name in joint_order]


def keyframe_tables(joint_order: list[str] = ISAACLAB_JOINT_ORDER):
    '''Returns (depths, open, desc, swept) as nested lists in radians, rows ordered by depth.'''
    depths = KEYFRAME_DEPTHS
    desc = [pose_rad(KEYFRAMES[d]["desc"], joint_order) for d in depths]
    swept = [pose_rad(KEYFRAMES[d]["swept"], joint_order) for d in depths]
    return depths, pose_rad(OPEN_DEG, joint_order), desc, swept


def finger_of_joint(name: str) -> str:
    return next(f for f in FINGERS if f"_{f}_" in name)
