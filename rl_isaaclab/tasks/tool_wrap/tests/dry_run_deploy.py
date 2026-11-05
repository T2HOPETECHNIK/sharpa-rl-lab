'''Dry-run rl_isaaclab/scripts/deploy.py for the tool-wrap task against a fake SharpaWave SDK (no hand needed):

    PYTHONPATH=$PWD python rl_isaaclab/tasks/tool_wrap/tests/dry_run_deploy.py logs/tool_wrap/<run>/stage2_nn/best.pth

The fake hand reports contact on the thumb, index and middle pads after the 25th closing-trajectory command, so the
pinch trigger fires; the policy then runs for 115 steps with keyboard control off. Prints observation shapes,
whether they are finite, action sizes and the fingertip forces the policy sees. Needs cache/tool_wrap_deploy_init_traj.npy.
'''
import os, sys, threading, time, types, runpy
import numpy as np
state = {"cmds": 0, "last": [0.0] * 22, "policy_steps": 0, "contact_after": 25}
class Err:  code = 0; message = ""
class FakeHand:
    def get_device_info(self): return types.SimpleNamespace(ip="0.0.0.0")
    def set_control_mode(self, m): return Err()
    def set_speed_coeff(self, c): return Err()
    def set_current_coeff(self, c): return Err()
    def set_control_source(self, s): return Err()
    def set_tactile_callback(self, cb): self.cb = cb
    def calib_tactile(self): return True
    def stop(self): pass
    def start(self):
        def run():
            while True:
                touching = state["cmds"] > state["contact_after"]
                for ch in range(5):   # right hand: channels 0..4; deploy maps channel 4-k to finger k (thumb=4)
                    finger = 4 - ch
                    f = [0.3, -0.4, 1.2, 0, 0, 0] if touching and finger in (0, 1, 2) else [0.0] * 6
                    self.cb({"channel": ch, "content": {"F6": f, "CONTACT_POINT": None}})
                time.sleep(0.006)
        threading.Thread(target=run, daemon=True).start()
    def set_joint_position(self, q):
        state["cmds"] += 1; state["last"] = list(np.asarray(q, dtype=float))
    def get_states(self): return types.SimpleNamespace(angles=state["last"])
class Manager:
    @staticmethod
    def get_instance(): return Manager()
    def get_all_device_sn(self): return ["FAKE-SN"]
    def connect(self, sn, cfg): return FakeHand()
    def disconnect_all(self): pass
sharpa = types.ModuleType("sharpa")
sharpa.SharpaWaveManager = Manager
sharpa.ControlMode = types.SimpleNamespace(POSITION="POSITION"); sharpa.ControlSource = types.SimpleNamespace(SDK="SDK")
sharpa.SharpaWaveConfig = type("SharpaWaveConfig", (), {})
sys.modules["sharpa"] = sharpa
pynput = types.ModuleType("pynput"); pynput.keyboard = types.SimpleNamespace(Listener=object, Key=None)
sys.modules["pynput"] = pynput; sys.modules["pynput.keyboard"] = pynput.keyboard

from rl_isaaclab.tasks.tool_wrap.tool_wrap_deploy_env_cfg import SharpaWaveToolWrapDeployEnvCfg as C
from rl_isaaclab.tasks.tool_wrap import tool_wrap_deploy_env as de
C.keyboard_listen = False                     # policy starts on its own instead of waiting for 'e'
de.SharpaWaveToolWrapDeployEnv.change_tactile_config = lambda self, on_board=False: True
orig_step = de.SharpaWaveToolWrapDeployEnv.step
def step(self, action):
    out = orig_step(self, action)
    state["policy_steps"] += 1
    obs = out[0]["policy"]; k = state["policy_steps"]
    if k in (1, 20, 60, 110) or k == 115:
        a = action.detach().cpu().numpy()
        print(f"[dry-run] step {k}: obs {tuple(obs.shape)} finite={bool(np.isfinite(obs.cpu().numpy()).all())} "
              f"proprio_hist {tuple(out[0]['proprio_hist'].shape)} |action| max {np.abs(a).max():.2f} "
              f"fingertip forces (obs, first frame) {obs[0, 44:59].cpu().numpy().round(2).reshape(5, 3).tolist()}", flush=True)
    if k >= 115:
        print("[dry-run] done", flush=True); os._exit(0)
    return out
de.SharpaWaveToolWrapDeployEnv.step = step
sys.argv = ["deploy.py", "--task", "Isaac-Tool-Wrap-Deploy-Sharpa-Wave-v0", "--hand_side", "1", "--device", "cpu",
            "--load_path", sys.argv[1]]
runpy.run_path("rl_isaaclab/scripts/deploy.py", run_name="__main__")
