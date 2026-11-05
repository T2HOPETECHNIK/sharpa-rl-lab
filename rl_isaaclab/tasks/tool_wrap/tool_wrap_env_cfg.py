import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env_cfg import SharpaWaveEnvCfg as _InhandRotateCfg

from .poses import OPEN_DEG
from .tool_library import ToolSpawnerCfg, build_tool_specs

# hand asset, actuators and fingertip contact sensors are shared with the in-hand rotation task
_ROTATE = _InhandRotateCfg()

HAND_POS = (0.0, 0.0, 0.5)
HAND_ROT = (0.70710678, 0.0, 0.70710678, 0.0)  # palm down, fingers along world +x, tool axis along world y

NUM_TOOLS = 32
TOOL_SEED = 0


@configclass
class SharpaWaveToolWrapEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.0
    grip_phase_s = 5.0          # policy adjusts the grip; the stability test runs for the rest of the episode
    action_space = 22
    frame_dim = 74              # 22 joint pos + 22 joint targets + 5x3 fingertip force vectors + 5x3 contact positions
    observation_space = 3 * 74  # last 3 frames
    prop_hist_len = 30          # frames seen by ProprioAdapt
    priv_info_dim = 24          # must match agents/ppo_cfg.yaml
    state_space = 0
    asymmetric_obs = False
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    effort_limit_scale = 1.0    # TODO: set to the torque the real hand reaches at deploy current_coef
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )
    # hand: fixed base, palm facing the floor
    hand_pos = HAND_POS
    hand_rot = HAND_ROT
    palm_face_offset = 0.025    # palm face along hand base +x (palmar normal)
    grip_center_offset = 0.115  # where the handle should sit along hand base +z: 1-2 cm past the knuckle line
    grip_ref_depth = 0.03       # reference depth below the palm face for "where is the handle" checks
    robot_cfg: ArticulationCfg = _ROTATE.robot_cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_POS,
            rot=HAND_ROT,
            joint_pos={name: math.radians(deg) for name, deg in OPEN_DEG.items()},
        ),
    )
    # 5 elastomers, then 5 distal phalanges. The elastomers are filtered against the tool and the floor with friction
    # tracking, so the pad force can include shear like the real F6; the phalanges are filtered against the tool.
    contact_sensor = [
        s.replace(track_friction_forces=True, filter_prim_paths_expr=["/World/envs/env_.*/object", "/World/envs/env_.*/floor"])
        if i < 5 else s
        for i, s in enumerate(_ROTATE.contact_sensor)
    ]
    # friction_forces_w is already the force on the sensor body, like net_forces_w (checked: a cube held still on a
    # floor by friction under sideways gravity reads +3 N friction against a -3 m/s^2 pull)
    pad_friction_sign = 1.0
    actuated_joint_names = _ROTATE.actuated_joint_names
    fingertip_body_names = _ROTATE.fingertip_body_names  # thumb, index, middle, ring, pinky
    elastomer_body_names = [f"right_{f}_elastomer" for f in ["thumb", "index", "middle", "ring", "pinky"]]
    # tools
    num_tools = NUM_TOOLS
    tool_specs = build_tool_specs(NUM_TOOLS, TOOL_SEED)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                ToolSpawnerCfg(**spec, physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5))
                for spec in build_tool_specs(NUM_TOOLS, TOOL_SEED)
            ],
            random_choice=False,  # env i gets tool i % num_tools; the env reads the id back from the prim
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.115, 0.0, 0.42)),
    )
    # floor: one kinematic box per env so its height can change per reset and drop during the stability test
    floor_size = (0.5, 0.5, 0.05)
    floor_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/floor",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.33, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.115, 0.0, 0.38)),
    )
    # GUI camera: env 0 seen from the pinky side, looking along the handle
    viewer: ViewerCfg = ViewerCfg(eye=(0.30, -0.45, 0.52), lookat=(0.11, 0.0, 0.43), origin_type="env", env_index=0)
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)
    # placement at reset (fresh resets and the cache generator)
    gap_range = (0.005, 0.045)          # palm face to tool top; FK says opposition holds to ~5.5-6.5 cm
    grip_offset_range = (-0.015, 0.025) # handle centre along the fingers, relative to grip_center_offset
    yaw_range_deg = 15.0                # tool axis relative to perpendicular-to-fingers
    depth_estimate_noise = 0.01         # error in the handle depth the closing keyframes are keyed on
    # scripted close (fresh resets, cache generator, and mirrored by the deploy trajectory)
    close_speed_range = (0.9, 1.6)      # keyframe stages per second; the close has 2 stages
    close_finger_delay = 0.25           # max per-finger start delay (stages)
    pinch_trigger_threshold = 0.5       # N, net fingertip force (floor counts, like the real sensor)
    pinch_trigger_min_fingers = 2       # plus the thumb
    # reset sources: first-contact cache, pinched-and-lifted cache, fresh (open hand + scripted close)
    grasp_cache_path = "cache/tool_wrap"  # prefix; train.py/play.py --cache overrides it
    reset_mix = (0.5, 0.3, 0.2)
    # rewards
    tool_contact_threshold = 0.2        # N, fingertip-to-tool contact (filtered sensor, sim only)
    pinch_reward_scale = 0.5
    lift_reward_scale = 1.0
    lift_target = 0.02                  # "lift slightly": reward saturates at 2 cm of clearance
    palm_reward_scale = 1.0
    palm_reward_length = 0.02
    coverage_reward_scale = 1.0
    below_reward_scale = 0.5
    cage_reward_scale = 3.0
    hold_reward_scale = 5.0
    drift_penalty_scale = -20.0
    drift_tolerance = 0.015             # handle may wander this far along the fingers before it costs
    squeeze_penalty_scale = -0.05
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    fail_penalty = -20.0
    # cage check
    cage_near_margin = 0.014           # link centre lines sit ~1.1 cm inside the link surface
    cage_max_gap_deg = 120.0
    cage_palm_tolerance = 0.012
    # terminations
    lost_distance = 0.12                # handle point under the hand this far from the grip reference
    test_slip_limit = 0.03              # tool moves this far in the hand frame during the test
    # stability test (stands in for the arm lifting the gripped tool)
    test_floor_drop = 0.10
    test_floor_speed = 0.10
    test_pull_factor = 1.0              # extra downward pull, in tool weights, ramped in
    test_pull_ramp_s = 1.0
    test_twist_torque = 0.03            # N m, random direction per episode
    # tactile observation (fingertips only, as on the hand)
    enable_tactile = True
    enable_contact_pos = False          # implemented in sim and deploy, untested in the original repo
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_latency = 0.005
    tactile_force_threshold = 0.2       # N, zeroed below this (deploy contact_threshold)
    tactile_force_noise = 0.05          # N per axis
    tactile_force_clip = 15.0           # N
    joint_noise_scale = 0.02
    dof_limits_scale = 0.9
    # randomization
    randomize_pd_gains = True
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2.0
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2.0
    randomize_friction = True
    friction_scale_range = (0.8, 1.6)   # not slippery
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    floor_friction_range = (0.4, 1.0)
    randomize_com = True
    com_range = 0.005
    randomize_mass = True
    mass_scale_range = (0.8, 1.25)
    # kept so play.py's overrides apply cleanly
    reset_random_quat = False
    randomize_joint_pos_offset = False
    gravity_curriculum = False
