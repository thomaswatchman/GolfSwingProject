"""
test_swing_metrics.py
Unit tests for the swing_metrics module.
"""

import math
import pytest
from src.analysis.swing_metrics import angle_between, compute_metrics, ALL_METRICS


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def straight_arm_landmarks():
    """Landmarks where the lead arm is perfectly straight (180°)."""
    return {
        "left_shoulder":  (0.3, 0.3, 0.0),
        "left_elbow":     (0.4, 0.3, 0.0),
        "left_wrist":     (0.5, 0.3, 0.0),
        "right_shoulder": (0.7, 0.3, 0.0),
        "right_elbow":    (0.6, 0.3, 0.0),
        "right_wrist":    (0.5, 0.3, 0.0),
        "left_hip":       (0.35, 0.6, 0.0),
        "right_hip":      (0.65, 0.6, 0.0),
        "left_knee":      (0.35, 0.75, 0.0),
        "right_knee":     (0.65, 0.75, 0.0),
        "left_ankle":     (0.35, 0.9, 0.0),
        "right_ankle":    (0.65, 0.9, 0.0),
        "nose":           (0.5, 0.1, 0.0),
    }


# -----------------------------------------------------------------------
# angle_between
# -----------------------------------------------------------------------

class TestAngleBetween:
    def test_right_angle(self):
        angle = angle_between((0, 1), (0, 0), (1, 0))
        assert abs(angle - 90.0) < 1e-6

    def test_straight_line(self):
        angle = angle_between((-1, 0), (0, 0), (1, 0))
        assert abs(angle - 180.0) < 1e-6

    def test_zero_angle(self):
        angle = angle_between((1, 0), (0, 0), (1, 0))
        assert abs(angle - 0.0) < 1e-6

    def test_45_degrees(self):
        angle = angle_between((1, 0), (0, 0), (1, 1))
        assert abs(angle - 45.0) < 1e-4


# -----------------------------------------------------------------------
# compute_metrics
# -----------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_keys(self, straight_arm_landmarks):
        result = compute_metrics(straight_arm_landmarks)
        assert set(result.keys()) == set(ALL_METRICS.keys())

    def test_values_are_floats(self, straight_arm_landmarks):
        result = compute_metrics(straight_arm_landmarks)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not a float"

    def test_values_in_valid_range(self, straight_arm_landmarks):
        result = compute_metrics(straight_arm_landmarks)
        for k, v in result.items():
            assert 0.0 <= v <= 180.0, f"{k}={v} out of range"

    def test_lead_arm_straight(self, straight_arm_landmarks):
        result = compute_metrics(straight_arm_landmarks)
        assert abs(result["lead_arm_angle"] - 180.0) < 1.0
