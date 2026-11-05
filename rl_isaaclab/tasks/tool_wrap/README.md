# Tool wrap-grasp (SharpaWave)

The hand hovers palm-down over an elongated tool lying on a flat surface, with the fingers perpendicular to the
tool. A scripted close puts the thumb and fingertips on opposite sides of the handle. The policy then lifts the
tool slightly **with the fingers**, works the fingers under the handle and pulls it into a wrap: fingers and thumb
around the handle, carrying its weight, instead of a frictional fingertip squeeze. The arm stays still until the
grip is done.

Only sensors the Wave has are used as policy inputs: joint angles, joint targets, and per-fingertip force vectors
and contact positions. Rewards and the privileged (teacher) inputs use simulator ground truth.

## Setup

Tested with Isaac Lab v2.3.2 / Isaac Sim 5.1 (the repo's README says release/2.2.0 and 2.3.0 work too).

```bash
conda activate isaaclab          # your Isaac Lab env
cd /path/to/sharpa-rl-lab
pip install -e .                 # or: export PYTHONPATH=$PWD
python -m pytest rl_isaaclab/tasks/tool_wrap/tests -q   # grip-metric unit tests, no simulator
```

Run everything from the repo root: caches and checkpoints are written relative to it (`cache/`, `logs/`).
`train.py` and `play.py` delete `outputs/` in the current directory on start (that's Hydra's scratch folder).

## Workflow

1. **Look at the scripted close first.** The close keyframes come from forward kinematics, not from a sim run.
   Watch a few envs before trusting them:
   ```bash
   python rl_isaaclab/scripts/gen_tool_wrap_cache.py --mode first_contact --num_envs 16 --num_states 200
   ```
2. **Generate reset caches** (each takes `cache_size` = 20000 states by default):
   ```bash
   python rl_isaaclab/scripts/gen_tool_wrap_cache.py --mode first_contact --headless
   python rl_isaaclab/scripts/gen_tool_wrap_cache.py --mode pinch_lift --headless
   ```
   These write `cache/tool_wrap_first_contact.npy` and `cache/tool_wrap_pinch_lift.npy`. Without caches, training
   still runs: every reset falls back to a fresh start with the scripted close, which is slower.
3. **Train the teacher** (sees privileged info):
   ```bash
   python rl_isaaclab/scripts/train.py --task Isaac-Tool-Wrap-Sharpa-Wave-v0 --num_envs 8192 --headless
   ```
   Watch `caged`, `caged_in_test`, `hold`, `lift` and `test_slipped` in TensorBoard (`logs/tool_wrap/.../stage1_tb`).
4. **Distil to the deployable student** (proprioception + touch history only):
   ```bash
   python rl_isaaclab/scripts/train.py --task Isaac-Tool-Wrap-Sharpa-Wave-v0 --num_envs 8192 --headless --algorithm ProprioAdapt --load_path logs/tool_wrap/<run>/stage1_nn/last.pth
   ```
   Like the rotation task, `ProprioAdapt` ignores `--max_agent_steps` and runs until you stop it. It saves
   `stage2_nn/best.pth` as it improves.
5. **Play:**
   ```bash
   python rl_isaaclab/scripts/play.py --task Isaac-Tool-Wrap-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path <stage2 .pth>
   ```
6. **Calibrate the fingertip force axes** on the hand, where `deploy.py` runs (once per hand; about 5 minutes):
   ```bash
   python3 rl_isaaclab/scripts/calibrate_tool_wrap_f6.py --hand_side 1
   python3 rl_isaaclab/scripts/calibrate_tool_wrap_f6.py --hand_side 1 --verify
   ```
   The hand opens flat. For each fingertip, thumb to pinky, you take your hand off for a baseline, then make three
   pushes with your own fingertip on the hand's pad. Each is held for 2 s:
   - **press:** straight into the pad;
   - **toward the tip:** press lightly and drag the pad's skin toward the tip;
   - **sideways:** press lightly and drag it sideways (toward the thumb side for the fingers, toward the palm side
     for the thumb).

   It fits a rotation and a sign per fingertip and saves them to `cache/tool_wrap_f6_calibration.json`.
   The deploy env loads that file automatically, and warns if it is missing. `--verify` prints live readings as
   "into pad / toward tip / sideways": each push should show up positive on its own axis.
7. **Deploy.** To check the deploy path after edits without the hand, run
   `PYTHONPATH=$PWD python rl_isaaclab/tasks/tool_wrap/tests/dry_run_deploy.py <stage2 .pth>`. It runs `deploy.py`
   against a fake SDK. On the hand:
   ```bash
   python rl_isaaclab/scripts/gen_tool_wrap_deploy_traj.py --depth 0.04   # gap above the tool + handle radius, metres
   python rl_isaaclab/scripts/deploy.py --task Isaac-Tool-Wrap-Deploy-Sharpa-Wave-v0 --hand_side 1 --load_path <stage2 .pth>
   ```
   Same keys as the rotation demo: `e` start, `w` freeze, `q` home. After `grip_phase_s` it prints
   "grip phase done": override `SharpaWaveToolWrapDeployEnv.on_grip_phase_done()` to trigger your arm's lift.

## How the task is set up

**Hand frame** (base link `right_hand_C_MC`): +x palmar normal (palm face at +0.025 m), +y index side, +z wrist to
fingertips (knuckle line at ~0.097 m). The hand is fixed at `(0, 0, 0.5)` with rotation `(0.7071, 0, 0.7071, 0)`:
palm down, fingers along world +x, tool axis along world y (±15°).

**Reach (from the asset's link lengths and 0.9-scaled joint limits):** fingertips reach about 8 cm below the palm
face. Thumb–fingertip opposition holds from about 7.6 cm up to 1.6 cm below the palm face, for a handle centred
1–2 cm past the knuckles. So the fingers can lift a pinched handle about 6 cm, nearly into the palm. The gap
between the palm and the tool top is randomised over 0.5–4.5 cm.

**Episode** (8 s at 20 Hz):
1. Scripted close (fresh resets only): OPEN → DESC(depth) → SWEPT(depth) from `poses.py`. Fingertips come down in
   front of the handle, then sweep back toward the wrist while the thumb comes forward. It stops when the thumb
   and 2 fingertips feel more than 0.5 N; the real hand uses the same trigger.
2. Grip phase (policy) until 5 s.
3. Stability test: the floor box drops 10 cm (standing in for the arm lifting), a downward pull ramps to 1× the
   tool's weight, and a small random twist is applied. The episode fails if the tool moves more than 3 cm in the
   hand frame.

**Reset sources** (`reset_mix`): 50% first-contact cache, 30% pinched-and-lifted cache, 20% fresh with the
scripted close. The pinched-and-lifted states are made by squeezing past the trigger and lowering the floor
1–3 cm. Only the resulting state is stored, so at training reset the tool starts held 1–3 cm above the floor by the
fingers alone.

**Rewards** (`tool_wrap_env_cfg.py`): pinch formed → grip region lifted (saturates at 2 cm) → handle close to the
palm, with angular coverage around the handle and fingers below it → cage check passes → held through the test
with low squeeze. Penalties: the handle drifting out of the reachable zone along the fingers, torque, work, and
failure. The cage check (`grip_metrics.py`) samples points along the link centre lines and bins them by angle
around the handle axis. It counts as caged when the bin straight below the handle is covered, no empty arc exceeds
120°, and the handle is within 1.2 cm of the palm.

**Policy frame (74 values, 3 frames stacked → 222):** 22 unscaled joint positions, 22 joint targets, 5×3 fingertip
force vectors in each elastomer frame, 5×3 contact positions (zeroed unless `enable_contact_pos`).

**Privileged info (24):** 0–2 handle point under the hand (hand frame, relative to the grip reference), 3–5 handle
axis (hand frame), 6 radius, 7 half length, 8 mass, 9–11 COM, 12 tool friction scale, 13 floor friction, 14 lift,
15 palm gap, 16–20 per-finger tool contact, 21–23 tool velocity.

**Cache row (53 floats):** tool id, floor depth below the palm face, joint positions (22), joint targets (22), tool
position in the hand frame (3), tool quaternion in the hand frame (4).

**Tools** (`tool_library.py`): 32 procedural tools. Cylinder, capsule and flat handles from 2 to 4.5 cm across and
15 to 30 cm long, plus hammer-like (block head, 2500–5000 kg/m³) and spatula-like (blade) tools, giving roughly
0.05–0.5 kg. Each env gets tool `i % 32`; the env reads the id back from the prim.

## What has been checked

On Isaac Lab v2.3.2 / Isaac Sim 5.1, RTX PRO 5000 (48 GB), 8192 envs:

- **First-contact cache:** 95% of recorded states are a real pinch (thumb on the near side, at least 2 fingertips
  on the far side, all touching the tool); usually all 5 fingertips touch. About 1 close in 4 is discarded.
- **Pinched-and-lifted cache:** every recorded state is a pinch holding the tool off the floor: 0.65–2.8 cm for 90% of
  states, median 1.6 cm. Only about 12% of attempts survive the 1 s hold, which is cheap to make up.
- **Teacher (PPO):** 60M steps at about 23,000 steps/s (43 min). Best reward 944. At the end, 96% of envs in the
  stability test pass the cage check, the tool slips in 0.1% of test steps, and about 1.6 of the 4 fingers sit
  under the handle.
- **Student (ProprioAdapt):** levels off at about 890 reward after 30M steps (stopped by hand), close to the
  teacher's running mean of about 875.
- **Deploy:** not run on a hand yet. `tests/dry_run_deploy.py` runs the student through `deploy.py` against a fake
  SDK: it loads, the pinch trigger fires, and observations stay finite. The trajectory script runs.

Findings from those runs:

- Contact sensors keep the previous episode's forces for a step or two after reset, so the pinch trigger is only
  armed from the 3rd step.
- In most closes the thumb meets the handle with the side of its distal segment, so the thumb pad stays under the
  0.5 N trigger and the close runs to the end. The policy then takes over from the end of the close, on the real
  hand as in sim, and the cache keeps those end states when they are real pinches. A thumb pose that brings the pad
  first was tried and made pinches much rarer (27%). Revisit it once the thumb pad direction is confirmed on
  hardware.

## Calibration and open items

- **Shear in the sim force vectors:** Isaac Lab's `net_forces_w` is the sum of *normal* contact forces only, while
  the Wave's F6 measures the total force on the pad. So the five elastomer sensors also track friction forces
  against the tool and the floor, and the observation is normal plus friction. The sign of `friction_forces_w`
  was checked with a cube held by friction under sideways gravity (`pad_friction_sign`). Contact with the hand's
  own links is not counted.
- **F6 force frame (required before deploy):** run `calibrate_tool_wrap_f6.py` (workflow step 6). All five sim
  fingertip frames share one convention: the pad faces −z, −y points toward the tip, and +x is sideways. The repo's
  tactile normal maps point roughly −z/−y, so the F6 axes may already match the sim frame; the calibration checks.
  The identity values in `tool_wrap_deploy_env_cfg.py` are only a fallback.
- **Torque cap:** sim torque limits come from the USD drive limits (MCP 1.86, PIP 0.64, DIP 0.19, thumb CMC 3.3 Nm).
  Deploy runs at `current_coef = 0.3`. Measure what that means in torque and set `effort_limit_scale` to match.
- **Deploy depth:** the real close is keyed on one `--depth`. If your arm's estimate of the tool height varies a
  lot, regenerate per grasp, or switch to a table of trajectories.
- **Soft pads:** sim fingertips are rigid. Try compliant contact on the elastomer material if rolling or slip
  differs on hardware.
- **Critic:** the repo's actor and critic share one network. A separate critic on full state would likely help this
  harder task.
- **Scripted steps in PPO:** while an env is in the scripted close, its actions are ignored and its reward is zero,
  but the steps still enter the PPO batch. Caches avoid most of this.
- **Friction-free drop check** (holds with near-zero friction ⇒ a true wrap): not implemented. It needs per-env
  material changes mid-episode; the cage check stands in for it.
