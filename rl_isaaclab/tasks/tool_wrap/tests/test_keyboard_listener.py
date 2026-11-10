'''Run: python -m pytest rl_isaaclab/tasks/tool_wrap/tests -q   (no hand or simulator needed)'''

import os
import pty
import termios
import time

from rl_isaaclab.utils.keyboard_listener import KeyboardListener
from rl_isaaclab.utils.misc import ThreadSafeValue


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_keys_come_from_its_own_terminal_and_settings_are_restored():
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "rb", buffering=0)
    before = termios.tcgetattr(slave)
    state, calib = ThreadSafeValue(0), ThreadSafeValue(0)
    kl = KeyboardListener(state, calib, hand_ip="0.0.0.0", stream=stream)
    kl.start()
    try:
        assert wait_for(lambda: kl._saved_attrs is not None)
        lflag = termios.tcgetattr(slave)[3]
        assert not lflag & termios.ICANON and not lflag & termios.ECHO and lflag & termios.ISIG  # cbreak, Ctrl-C kept
        os.write(master, b"e")                      # single key, no Enter
        assert wait_for(lambda: state.get() == 2)
        state.set(3)
        os.write(master, b"xz")                     # other keys ignored
        time.sleep(0.1)
        assert state.get() == 3
        os.write(master, b"w")
        assert wait_for(lambda: state.get() == 1)
        os.write(master, b"w")
        assert wait_for(lambda: state.get() == 3)   # continue restores the state before the freeze
        os.write(master, b"t")
        assert wait_for(lambda: calib.get() == 1)
        os.write(master, b"q")
        assert wait_for(lambda: state.get() == 0)
    finally:
        kl.stop()
        kl.thread.join(timeout=2)
    assert termios.tcgetattr(slave) == before
    stream.close()
    os.close(master)


def test_not_a_terminal_leaves_state_alone(tmp_path):
    f = open(tmp_path / "keys.txt", "wb+")
    f.write(b"e")
    f.flush()
    state = ThreadSafeValue(0)
    kl = KeyboardListener(state, ThreadSafeValue(0), hand_ip="0.0.0.0", stream=f)
    kl.start()
    kl.thread.join(timeout=2)
    assert state.get() == 0
    f.close()
