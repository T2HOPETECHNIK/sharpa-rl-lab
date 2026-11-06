'''
Fit the rotation (and sign) that maps each fingertip's F6 force axes into the sim elastomer frame, which is what the
tool-wrap policy was trained on. Pure numpy so it can be tested without the hand.

Sim elastomer frame, the same for all five digits (from the joint frames in assets/SharpaWave/right_sharpa_wave.usda):
    -z  out of the pad (the pad faces -z)
    -y  toward the fingertip
    +x  sideways: toward the thumb side for the fingers; for the thumb, away from the fingers, toward the side the
        palm faces (hand open flat; checked on the joint frames: the thumb pad -> index pad direction is -0.6 along it)
Forces are the force acting ON the pad, the convention of Isaac Lab contact sensors.
The F6 frames turned out to match these to within ~15 deg on all five digits (2026-11-05 calibration), so every digit
should fit with the same sign; a digit whose sign differs had a push done the wrong way round.

Calibration pushes (done with your own fingertip on the hand's pad) and the force they put on the pad:
    press       straight into the pad            -> +z
    toward_tip  drag the pad skin toward the tip -> -y
    sideways    drag it sideways (see above)     -> +x
'''

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PUSHES = ["press", "toward_tip", "sideways"]
TARGETS = {
    "press": np.array([0.0, 0.0, 1.0]),
    "toward_tip": np.array([0.0, -1.0, 0.0]),
    "sideways": np.array([1.0, 0.0, 0.0]),
}
# component names for printing calibrated forces: force on the pad . direction
COMPONENTS = {"into pad": TARGETS["press"], "toward tip": TARGETS["toward_tip"], "sideways": TARGETS["sideways"]}


def instruction(push: str, finger: str) -> str:
    side = ("away from the fingers, toward the side the palm faces (down when the palm faces down)" if finger == "thumb"
            else "toward the thumb side of the hand")
    return {
        "press": f"Press straight into the {finger} pad (the soft tip, palm side), about 2-5 N, and hold.",
        "toward_tip": f"Rest your fingertip on the {finger} pad, press lightly, and drag the pad skin toward the tip "
                      f"of the {finger} without sliding off. Hold the drag.",
        "sideways": f"Rest your fingertip on the {finger} pad, press lightly, and drag the pad skin sideways {side} "
                    f"without sliding off. Hold the drag.",
    }[push]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero force vector")
    return v / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def fit_pad_rotation(measured: dict[str, np.ndarray]) -> tuple[np.ndarray, float, dict[str, float]]:
    '''Fit target ~= sign * R @ measured for one fingertip.

    Args:
        measured: push name -> mean F6 force vector for that push, baseline removed, sensor frame.

    Returns:
        (R, sign, residual_deg): R is a proper rotation (3x3), sign is +1 or -1 (-1 when the sensor reports the
        reaction force), residual_deg is the angle left between each fitted push and its target.

    Dragging the skin always presses on it too, so the shear pushes have their component along the press direction
    removed before fitting.
    '''
    n = _unit(measured["press"])
    m = {"press": n}
    for k in ("toward_tip", "sideways"):
        m[k] = _unit(measured[k] - (measured[k] @ n) * n)
    H = sum(np.outer(m[k], TARGETS[k]) for k in PUSHES)
    U, _, Vt = np.linalg.svd(H)
    M = Vt.T @ U.T
    sign = 1.0 if np.linalg.det(M) > 0 else -1.0
    R = sign * M
    residual = {k: _angle_deg(sign * R @ m[k], TARGETS[k]) for k in PUSHES}
    return R, sign, residual
