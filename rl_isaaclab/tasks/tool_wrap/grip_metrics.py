'''
Grip measurements used by the tool-wrap rewards. Pure torch (no Isaac Lab) so they can be unit tested.

Everything here reads simulator ground truth (link poses, tool pose, per-fingertip contact force with the tool).
None of it is a policy input, so it adds no sensor to the hand.

Quaternions are (w, x, y, z), matching Isaac Lab.
'''

import math

import torch


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return quat_apply(q_conj, v)


def closest_point_on_handle(tool_pos: torch.Tensor, tool_quat: torch.Tensor, handle_half_length: torch.Tensor,
                            ref_point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Point on the handle axis (tool local +x through the tool origin) nearest to ref_point.

    Returns (point, axis), both (N, 3) in the frame of the inputs.
    '''
    ex = torch.zeros_like(tool_pos)
    ex[:, 0] = 1.0
    axis = quat_apply(tool_quat, ex)
    t = ((ref_point - tool_pos) * axis).sum(-1)
    t = torch.maximum(torch.minimum(t, handle_half_length), -handle_half_length)
    return tool_pos + t.unsqueeze(-1) * axis, axis


def palm_gap(center: torch.Tensor, radius: torch.Tensor, palm_point: torch.Tensor,
             palm_normal: torch.Tensor) -> torch.Tensor:
    '''Distance from the palm face to the handle surface along the palm normal (negative = penetrating).'''
    return ((center - palm_point) * palm_normal).sum(-1) - radius


def cage_metrics(points: torch.Tensor, point_finger_ids: torch.Tensor, center: torch.Tensor, axis: torch.Tensor,
                 radius: torch.Tensor, down: torch.Tensor, near_margin: float = 0.018, axial_window: float = 0.06,
                 num_bins: int = 12, below_half_angle_deg: float = 60.0) -> dict[str, torch.Tensor]:
    '''How far the hand surrounds the handle cross-section under the palm.

    Args:
        points: (N, P, 3) sample points on the hand's links (segment centre lines), world frame.
        point_finger_ids: (P,) long, 0..4 = thumb..pinky, -1 = palm.
        center: (N, 3) point on the handle axis under the hand.
        axis: (N, 3) unit handle axis.
        radius: (N,) handle radius.
        down: (N, 3) world down (gravity direction).
        near_margin: a point counts as touching the handle within radius + near_margin of the axis
            (link centre lines sit ~1 cm inside the link surface).
        axial_window: only points within this distance along the axis from `center` count.
        num_bins: angular bins around the axis; bin num_bins // 2 is centred straight below the handle.

    Returns dict with:
        coverage: (N,) fraction of bins with a near point.
        max_gap_deg: (N,) largest run of empty bins, in degrees (360 if nothing is near).
        bottom_covered: (N,) a near point lies in the bin straight below the handle.
        below: (N, 5) per finger, a near point lies within below_half_angle_deg of straight below.
    '''
    v = points - center.unsqueeze(1)
    a = (v * axis.unsqueeze(1)).sum(-1)
    v_perp = v - a.unsqueeze(-1) * axis.unsqueeze(1)
    rho = torch.linalg.norm(v_perp, dim=-1)
    near = (rho < radius.unsqueeze(-1) + near_margin) & (a.abs() < axial_window)

    e1 = down - (down * axis).sum(-1, keepdim=True) * axis
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(1e-6)
    e2 = torch.cross(axis, e1, dim=-1)
    theta = torch.atan2((v_perp * e2.unsqueeze(1)).sum(-1), (v_perp * e1.unsqueeze(1)).sum(-1))  # 0 = straight below

    bin_width = 2.0 * math.pi / num_bins
    bins = torch.floor((theta + math.pi + 0.5 * bin_width) / bin_width).long() % num_bins  # bin 0 centred at the top
    one_hot = torch.nn.functional.one_hot(bins, num_bins).bool() & near.unsqueeze(-1)
    covered = one_hot.any(dim=1)

    # longest circular run of uncovered bins
    doubled = torch.cat([~covered, ~covered], dim=1).float()
    run = torch.zeros(covered.shape[0], device=covered.device)
    longest = torch.zeros_like(run)
    for k in range(doubled.shape[1]):
        run = (run + 1.0) * doubled[:, k]
        longest = torch.maximum(longest, run)
    longest = torch.clamp(longest, max=num_bins)

    below_pts = near & (theta.abs() < math.radians(below_half_angle_deg))
    below = torch.stack([(below_pts & (point_finger_ids == f)).any(dim=1) for f in range(5)], dim=1)

    return {
        "coverage": covered.float().mean(dim=1),
        "max_gap_deg": longest * (360.0 / num_bins),
        "bottom_covered": covered[:, num_bins // 2],
        "below": below,
    }


def pinch_formed(tool_force_norm: torch.Tensor, fingertip_pos: torch.Tensor, center: torch.Tensor,
                 finger_dir: torch.Tensor, threshold: float, min_fingers: int = 2) -> torch.Tensor:
    '''Thumb on the wrist side of the handle and at least min_fingers fingertips on the far side, all touching it.

    Args:
        tool_force_norm: (N, 5) contact force magnitude between each fingertip (thumb..pinky) and the tool.
        fingertip_pos: (N, 5, 3).
        center: (N, 3) handle axis point under the hand.
        finger_dir: (N, 3) unit wrist->fingertip direction of the hand.
    '''
    s = ((fingertip_pos - center.unsqueeze(1)) * finger_dir.unsqueeze(1)).sum(-1)
    touching = tool_force_norm > threshold
    thumb_ok = touching[:, 0] & (s[:, 0] < 0.0)
    fingers_ok = (touching[:, 1:] & (s[:, 1:] > 0.0)).sum(-1) >= min_fingers
    return thumb_ok & fingers_ok
