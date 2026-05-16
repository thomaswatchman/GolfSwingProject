"""
swing_metrics.py
Computes biomechanical angles and key swing metrics from pose landmarks.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------
# Low-level geometry helpers
# -----------------------------------------------------------------------

def _vec(a: tuple, b: tuple) -> np.ndarray:
    """Vector from point a to point b (2-D or 3-D)."""
    return np.array(b) - np.array(a)


def angle_between(a: tuple, vertex: tuple, b: tuple) -> float:
    """
    Angle at *vertex* formed by the rays vertex→a and vertex→b.

    Returns degrees in [0, 180].
    """
    v1 = _vec(vertex, a)
    v2 = _vec(vertex, b)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


# -----------------------------------------------------------------------
# Metric functions  (each takes a landmarks dict, returns float)
# -----------------------------------------------------------------------

def lead_arm_angle(lm: dict) -> float:
    """Angle at the lead (left) elbow: shoulder → elbow → wrist."""
    return angle_between(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])


def trail_arm_angle(lm: dict) -> float:
    """Angle at the trail (right) elbow: shoulder → elbow → wrist."""
    return angle_between(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])


def shoulder_rotation(lm: dict) -> float:
    """
    Approximate shoulder turn angle (degrees) relative to the camera plane.
    Uses the horizontal distance between shoulders normalised by their 3-D distance.
    """
    ls = np.array(lm["left_shoulder"])
    rs = np.array(lm["right_shoulder"])
    dist_3d = np.linalg.norm(ls - rs) + 1e-9
    dx = abs(ls[0] - rs[0])
    return float(np.degrees(np.arccos(np.clip(dx / dist_3d, 0.0, 1.0))))


def hip_rotation(lm: dict) -> float:
    """Same idea as shoulder_rotation but for the hips."""
    lh = np.array(lm["left_hip"])
    rh = np.array(lm["right_hip"])
    dist_3d = np.linalg.norm(lh - rh) + 1e-9
    dx = abs(lh[0] - rh[0])
    return float(np.degrees(np.arccos(np.clip(dx / dist_3d, 0.0, 1.0))))


def spine_tilt(lm: dict) -> float:
    """
    Lateral tilt of the spine (degrees from vertical).
    Measured from mid-hip to mid-shoulder in the image plane.
    """
    mid_hip = (
        (lm["left_hip"][0] + lm["right_hip"][0]) / 2,
        (lm["left_hip"][1] + lm["right_hip"][1]) / 2,
    )
    mid_shoulder = (
        (lm["left_shoulder"][0] + lm["right_shoulder"][0]) / 2,
        (lm["left_shoulder"][1] + lm["right_shoulder"][1]) / 2,
    )
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]          # y increases downward
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def knee_flex(lm: dict, side: str = "lead") -> float:
    """
    Knee flexion angle for 'lead' (left) or 'trail' (right) leg.
    hip → knee → ankle.
    """
    if side == "lead":
        return angle_between(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
    return angle_between(lm["right_hip"], lm["right_knee"], lm["right_ankle"])


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

ALL_METRICS = {
    "lead_arm_angle":    lead_arm_angle,
    "trail_arm_angle":   trail_arm_angle,
    "shoulder_rotation": shoulder_rotation,
    "hip_rotation":      hip_rotation,
    "spine_tilt":        spine_tilt,
    "lead_knee_flex":    lambda lm: knee_flex(lm, "lead"),
    "trail_knee_flex":   lambda lm: knee_flex(lm, "trail"),
}


def compute_metrics(landmarks: dict) -> dict[str, float]:
    """
    Compute all swing metrics from a landmarks dict.

    Args:
        landmarks: Output of PoseEstimator.process()

    Returns:
        Dict of metric_name → value (degrees).
    """
    return {name: fn(landmarks) for name, fn in ALL_METRICS.items()}
