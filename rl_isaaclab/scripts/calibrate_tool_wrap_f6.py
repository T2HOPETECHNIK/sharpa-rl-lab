'''
Calibrate the fingertip F6 force axes for the tool-wrap deploy. Run it where deploy.py runs, with the hand connected:

    python3 rl_isaaclab/scripts/calibrate_tool_wrap_f6.py --hand_side 1
    python3 rl_isaaclab/scripts/calibrate_tool_wrap_f6.py --hand_side 1 --verify

The hand opens flat and holds still. For each fingertip (thumb to pinky) you record a hands-off baseline, then three
pushes with your own fingertip on the hand's pad: straight into it, dragging its skin toward the tip, and dragging it
sideways. The script fits, per fingertip, the rotation and sign that turn F6 readings into the sim's fingertip frame,
and writes them to cfg.tactile_calibration_path, which the tool-wrap deploy env loads at start-up.

--verify prints live calibrated forces as "into pad / toward tip / sideways" so you can check each push reads the
right way round.
'''

import argparse
import json
import os
import time

parser = argparse.ArgumentParser(description="Calibrate SharpaWave fingertip F6 axes for the tool-wrap policy.")
parser.add_argument("--hand_side", type=int, default=1, help="0 for left hand, 1 for right hand.")
parser.add_argument("--enable_on_board", action="store_true", help="Enable on-board Tactile.")
parser.add_argument("--seconds", type=float, default=2.0, help="Recording time per push.")
parser.add_argument("--min_force", type=float, default=0.5, help="N; samples below this (after baseline) are ignored.")
parser.add_argument("--out", type=str, default=None, help="Output path (default: cfg.tactile_calibration_path).")
parser.add_argument("--verify", action="store_true", help="Only print live calibrated forces.")
args = parser.parse_args()

import numpy as np

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_deploy_env import SharpaWaveManager
from rl_isaaclab.tasks.tool_wrap.f6_calibration import COMPONENTS, FINGERS, PUSHES, fit_pad_rotation, instruction
from rl_isaaclab.tasks.tool_wrap.tool_wrap_deploy_env import SharpaWaveToolWrapDeployEnv
from rl_isaaclab.tasks.tool_wrap.tool_wrap_deploy_env_cfg import SharpaWaveToolWrapDeployEnvCfg

RESIDUAL_WARN_DEG = 20.0


def record(env, ch, seconds, rate=100.0):
    samples = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        samples.append(env.raw_fingertip_forces()[ch].copy())
        time.sleep(1.0 / rate)
    return np.array(samples)


def record_push(env, ch, finger, push, baseline):
    while True:
        print(f"\n  {instruction(push, finger)}")
        input("  Press Enter, then start pushing; recording begins in 1 s ...")
        time.sleep(1.0)
        print(f"  recording {args.seconds:.1f} s ...")
        f = record(env, ch, args.seconds) - baseline
        mag = np.linalg.norm(f, axis=1)
        keep = mag > args.min_force
        if keep.mean() < 0.3:
            print(f"  too little force (peak {mag.max():.2f} N, needs > {args.min_force} N for most of the recording); try again")
            continue
        mean = f[keep].mean(axis=0)
        print(f"  ok: {np.linalg.norm(mean):.2f} N on average over {keep.sum()} samples")
        return mean


def calibrate(env, out_path):
    digits = {}
    for ch, finger in enumerate(FINGERS):
        print(f"\n=== {finger} ({ch + 1}/5) ===")
        input(f"  Take everything off the {finger} pad, then press Enter to record the baseline ...")
        baseline = record(env, ch, 1.0).mean(axis=0)
        measured = {push: record_push(env, ch, finger, push, baseline) for push in PUSHES}
        R, sign, residual = fit_pad_rotation(measured)
        worst = max(residual.values())
        print(f"  fit: sign {sign:+.0f}, residuals " + ", ".join(f"{k} {v:.1f} deg" for k, v in residual.items()))
        if worst > RESIDUAL_WARN_DEG:
            print(f"  WARNING: {worst:.0f} deg residual. The pushes didn't line up well; consider re-running this finger.")
        digits[finger] = {
            "rotation": R.round(6).tolist(),
            "sign": sign,
            "residual_deg": {k: round(v, 2) for k, v in residual.items()},
            "measured": {k: v.round(4).tolist() for k, v in measured.items()},
            "baseline": baseline.round(4).tolist(),
        }
    calib = {"hand_side": args.hand_side, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "digits": digits}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=1)
    print(f"\nSaved {out_path}. The tool-wrap deploy env loads it automatically; check it with --verify.")
    for finger, d in digits.items():
        angle = np.degrees(np.arccos(np.clip((np.trace(np.array(d["rotation"])) - 1) / 2, -1, 1)))
        print(f"  {finger:6s} sign {d['sign']:+.0f}, {angle:5.1f} deg away from the uncalibrated default, worst residual {max(d['residual_deg'].values()):.1f} deg")


def verify(env):
    print("\nLive calibrated forces (N, after force_scale), Ctrl-C to stop. Push each pad and check the signs.")
    names = list(COMPONENTS)
    while True:
        forces, _ = env.get_tactile_info()
        f = forces.reshape(5, 3).cpu().numpy()
        cells = []
        for ch, finger in enumerate(FINGERS):
            c = [f[ch] @ COMPONENTS[n] for n in names]
            cells.append(f"{finger[:5]:5s} " + " ".join(f"{v:+5.1f}" for v in c))
        print("\r" + " | ".join(cells) + f"   ({' / '.join(names)})", end="", flush=True)
        time.sleep(0.2)


def main():
    cfg = SharpaWaveToolWrapDeployEnvCfg()
    cfg.device = "cpu"
    cfg.keyboard_listen = False
    cfg.hand_side = args.hand_side
    cfg.enable_on_board = args.enable_on_board
    out_path = args.out or cfg.tactile_calibration_path
    if args.out:
        cfg.tactile_calibration_path = args.out
    env = SharpaWaveToolWrapDeployEnv(cfg)
    try:
        print("Opening the hand flat ...")
        env.go_home()
        input("Keep everything off the fingertips, then press Enter to zero the tactile sensors ...")
        env.calib_tactile()
        if args.verify:
            verify(env)
        else:
            calibrate(env, out_path)
    except KeyboardInterrupt:
        print()
    finally:
        env.hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()


if __name__ == "__main__":
    main()
