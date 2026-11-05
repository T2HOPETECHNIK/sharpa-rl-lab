'''Run: python -m pytest rl_isaaclab/tasks/tool_wrap/tests -q   (plain torch, no simulator needed)'''

import math

import torch

from rl_isaaclab.tasks.tool_wrap.grip_metrics import (
    cage_metrics, closest_point_on_handle, palm_gap, pinch_formed, quat_apply, quat_apply_inverse,
)

DOWN = torch.tensor([[0.0, 0.0, -1.0]])
AXIS = torch.tensor([[0.0, 1.0, 0.0]])  # handle along world y
CENTER = torch.zeros(1, 3)
RADIUS = torch.tensor([0.015])


def ring_points(angles_deg, rho=0.025, finger=1):
    # angle 0 = straight below the handle, positive toward +x
    pts = [[rho * math.sin(math.radians(a)), 0.0, -rho * math.cos(math.radians(a))] for a in angles_deg]
    return torch.tensor([pts]), torch.full((len(angles_deg),), finger, dtype=torch.long)


def test_quat_roundtrip():
    q = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    v = torch.randn(8, 3)
    assert torch.allclose(quat_apply_inverse(q, quat_apply(q, v)), v, atol=1e-5)


def test_full_wrap_is_caged():
    pts, ids = ring_points(range(0, 360, 30))
    m = cage_metrics(pts, ids, CENTER, AXIS, RADIUS, DOWN)
    assert m["bottom_covered"].item()
    assert m["max_gap_deg"].item() == 0.0
    assert m["coverage"].item() == 1.0


def test_side_pinch_is_not_caged():
    pts, ids = ring_points([90, -90])
    m = cage_metrics(pts, ids, CENTER, AXIS, RADIUS, DOWN)
    assert not m["bottom_covered"].item()
    assert m["max_gap_deg"].item() >= 150.0
    assert not m["below"].any()


def test_far_points_and_axial_window_ignored():
    pts, ids = ring_points([0], rho=0.2)
    assert cage_metrics(pts, ids, CENTER, AXIS, RADIUS, DOWN)["coverage"].item() == 0.0
    pts, ids = ring_points([0])
    pts[..., 1] = 0.5  # far along the handle
    assert cage_metrics(pts, ids, CENTER, AXIS, RADIUS, DOWN)["max_gap_deg"].item() == 360.0


def test_below_is_per_finger():
    pts, _ = ring_points([10, -20, 170])
    ids = torch.tensor([1, 2, 0])
    below = cage_metrics(pts, ids, CENTER, AXIS, RADIUS, DOWN)["below"][0]
    assert below.tolist() == [False, True, True, False, False]


def test_palm_gap_and_closest_point():
    gap = palm_gap(torch.tensor([[0.0, 0.0, 0.40]]), RADIUS, torch.tensor([[0.0, 0.0, 0.43]]), DOWN)
    assert abs(gap.item() - 0.015) < 1e-6
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])  # tool +x -> world +y
    p, axis = closest_point_on_handle(torch.zeros(1, 3), q, torch.tensor([0.1]), torch.tensor([[0.05, 0.3, 0.0]]))
    assert torch.allclose(axis, AXIS, atol=1e-6)
    assert torch.allclose(p, torch.tensor([[0.0, 0.1, 0.0]]), atol=1e-6)


def test_pinch_needs_opposition():
    finger_dir = torch.tensor([[1.0, 0.0, 0.0]])
    tips = torch.tensor([[[-0.02, 0, 0], [0.02, 0, 0], [0.02, 0, 0], [0.02, 0, 0], [0.02, 0, 0]]])
    force = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    assert pinch_formed(force, tips, CENTER, finger_dir, 0.2).item()
    tips_same_side = tips.clone()
    tips_same_side[0, 0, 0] = 0.02
    assert not pinch_formed(force, tips_same_side, CENTER, finger_dir, 0.2).item()
    assert not pinch_formed(torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]), tips, CENTER, finger_dir, 0.2).item()
