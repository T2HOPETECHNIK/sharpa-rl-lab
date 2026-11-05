'''Run: python -m pytest rl_isaaclab/tasks/tool_wrap/tests -q   (numpy only, no hand needed)'''

import numpy as np
import pytest

from rl_isaaclab.tasks.tool_wrap.f6_calibration import PUSHES, TARGETS, fit_pad_rotation


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def simulate_pushes(R_true, sign_true, rng, noise=0.0, normal_leak=0.0):
    '''Sensor readings for the three pushes, given sim = sign * R @ sensor.'''
    out = {}
    for k in PUSHES:
        force_sim = 3.0 * TARGETS[k]
        if k != "press":
            force_sim = force_sim + normal_leak * 3.0 * TARGETS["press"]  # dragging also presses
        sensor = sign_true * R_true.T @ force_sim
        out[k] = sensor + noise * rng.normal(size=3)
    return out


@pytest.mark.parametrize("sign_true", [1.0, -1.0])
def test_recovers_rotation_and_sign(sign_true):
    rng = np.random.default_rng(0)
    R_true = random_rotation(rng)
    R, sign, residual = fit_pad_rotation(simulate_pushes(R_true, sign_true, rng, normal_leak=0.6))
    assert sign == sign_true
    assert np.allclose(R, R_true, atol=1e-6)
    assert max(residual.values()) < 1e-4
    assert np.isclose(np.linalg.det(R), 1.0)


def test_identity_when_sensor_matches_sim_frame():
    rng = np.random.default_rng(1)
    R, sign, _ = fit_pad_rotation(simulate_pushes(np.eye(3), 1.0, rng))
    assert sign == 1.0 and np.allclose(R, np.eye(3), atol=1e-6)


def test_noisy_pushes_stay_close():
    rng = np.random.default_rng(2)
    R_true = random_rotation(rng)
    R, sign, residual = fit_pad_rotation(simulate_pushes(R_true, 1.0, rng, noise=0.2, normal_leak=0.8))
    assert sign == 1.0
    err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert err < 10.0
    assert max(residual.values()) < 15.0
