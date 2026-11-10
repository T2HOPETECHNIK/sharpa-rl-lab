'''
Deploy keys, read from this process's own terminal.

The old listener used pynput, which receives key presses from every window on the desktop: typing in another window
while deploy ran sent 'e' to the hand and started closes. This one reads stdin with the terminal in cbreak mode
(single keys, no Enter, no echo), so only keys typed into the deploy terminal while it has focus count. Ctrl-C still
raises SIGINT. The terminal's settings are restored on stop() and at exit.

    q  move home            w  freeze / continue
    e  start (scripted close, then the policy)
    t  tactile re-zero (SDK calib_tactile; hand open, nothing touching)
'''
import atexit
import os
import select
import sys
import threading

try:
    import termios
    import tty
except ImportError:  # not POSIX
    termios = tty = None

from rl_isaaclab.utils.misc import ThreadSafeValue


class KeyboardListener:
    def __init__(self,
                 deploy_state_flag: ThreadSafeValue,
                 calib_tactile_flag: ThreadSafeValue,
                 hand_ip: str,
                 stream=None):
        self.deploy_state_flag = deploy_state_flag
        self.hand_ip = hand_ip
        self.calib_tactile_flag = calib_tactile_flag
        self.last_deploy_state_flag = self.deploy_state_flag.get()
        self.stream = stream if stream is not None else sys.stdin
        self._saved_attrs = None
        self._fd = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def handle_key(self, ch: str):
        '''
        deploy_state_flag:
            0: move home
            1: freeze actions
            2: reset env
            3: run policy
        calib_tactile_flag:
            0: silence
            1: calibrate tactile
        '''
        if ch == 'q':
            print('[Keyboard] Moving home.')
            self.deploy_state_flag.set(0)
        elif ch == 'w':
            if self.deploy_state_flag.get() != 1:
                self.last_deploy_state_flag = self.deploy_state_flag.get()
                print('[Keyboard] Freeze actions.')
                self.deploy_state_flag.set(1)
            else:
                print('[Keyboard] Continue actions.')
                self.deploy_state_flag.set(self.last_deploy_state_flag)
        elif ch == 'e':
            print('[Keyboard] Start policy.')
            self.deploy_state_flag.set(2)
        elif ch == 't':
            self.calib_tactile_flag.set(1)
            print("[Keyboard] Tactile calibration.")

    def _restore(self):
        if self._saved_attrs is not None and termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            except (termios.error, OSError, ValueError):
                pass
            self._saved_attrs = None

    def _run(self):
        try:
            fd = self.stream.fileno()
        except (AttributeError, OSError, ValueError):
            fd = None
        if fd is None or termios is None or not os.isatty(fd):
            print("[Keyboard] WARNING: stdin is not a terminal, so keys can't be read. Run deploy in a terminal "
                  "window (not piped or in the background) to use q/w/e/t.")
            return
        self._fd = fd
        self._saved_attrs = termios.tcgetattr(fd)
        atexit.register(self._restore)
        tty.setcbreak(fd)
        print("[Keyboard] Keys are read from this terminal only (q home, w freeze, e start, t tactile re-zero).")
        try:
            while not self.stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                data = os.read(fd, 64)
                if not data:  # EOF
                    break
                for ch in data.decode(errors="ignore"):
                    self.handle_key(ch)
        finally:
            self._restore()
        print("[Keyboard] Keyboard listener stopped.")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self._restore()
