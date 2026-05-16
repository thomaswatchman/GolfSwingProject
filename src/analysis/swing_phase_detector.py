"""
swing_phase_detector.py
Classifies each frame of a golf swing into one of five phases:

    SETUP       → Golfer is addressing the ball (still)
    BACKSWING   → Club moving away from ball
    TRANSITION  → Top of backswing (brief pause / direction change)
    DOWNSWING   → Club moving toward impact
    FOLLOW_THROUGH → Post-impact, club continuing upward

Detection is based on the right wrist trajectory (trail hand) since it
has the largest and most reliable motion arc throughout the swing.
No ML model needed — pure signal processing on the pose landmarks.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np


# -----------------------------------------------------------------------
# Phase enum
# -----------------------------------------------------------------------

class SwingPhase(str, Enum):
    SETUP          = "Setup"
    BACKSWING      = "Backswing"
    TRANSITION     = "Transition"
    DOWNSWING      = "Downswing"
    FOLLOW_THROUGH = "Follow Through"
    UNKNOWN        = "Unknown"


# -----------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------

class SwingPhaseDetector:
    """
    Stateful, frame-by-frame swing phase classifier.

    Feed one landmarks dict per frame via .update() and read .phase.

    Algorithm
    ---------
    1. Track the trail wrist (right_wrist) y-coordinate over a rolling window.
    2. Compute a smoothed vertical velocity (dy/dt).
    3. Use velocity sign + cumulative displacement to drive a simple state machine:

        SETUP          : wrist barely moving  (|vel| < threshold)
        BACKSWING      : wrist rising (y decreasing in image coords)
        TRANSITION     : wrist near top and velocity crossing zero
        DOWNSWING      : wrist falling rapidly (y increasing)
        FOLLOW_THROUGH : wrist rising again after impact
    """

    # Tunable constants
    WINDOW        = 9    # frames for velocity smoothing
    STILL_THRESH  = 0.004  # normalised units/frame — below = "not moving"
    TOP_THRESH    = 0.006  # velocity near zero = transition zone
    MIN_SETUP_FRAMES = 5   # must be still this long to enter SETUP

    def __init__(self):
        self._y_history: deque[float] = deque(maxlen=self.WINDOW)
        self._phase = SwingPhase.UNKNOWN
        self._still_count = 0
        self._swing_started = False
        self._post_impact = False
        # y at top of backswing (minimum y = highest wrist position)
        self._min_y_seen = 1.0
        self._peak_locked = False

    # ------------------------------------------------------------------
    @property
    def phase(self) -> SwingPhase:
        return self._phase

    # ------------------------------------------------------------------
    def update(self, landmarks: dict | None) -> SwingPhase:
        """
        Feed one frame's landmarks and return the current phase.

        Args:
            landmarks: Output of PoseEstimator.process(), or None if no
                       pose was detected this frame.
        """
        if landmarks is None:
            return self._phase

        # Trail wrist y in normalised image coords (0=top, 1=bottom)
        wy = landmarks["right_wrist"][1]
        self._y_history.append(wy)

        if len(self._y_history) < 3:
            return self._phase

        # Smoothed velocity: positive = wrist moving DOWN (toward ground)
        vel = self._smoothed_velocity()

        # ── State machine ────────────────────────────────────────────
        if not self._swing_started:
            # Waiting for the golfer to settle at address
            if abs(vel) < self.STILL_THRESH:
                self._still_count += 1
            else:
                self._still_count = 0

            if self._still_count >= self.MIN_SETUP_FRAMES:
                self._phase = SwingPhase.SETUP

            # Swing starts when wrist begins moving upward (vel negative)
            if self._phase == SwingPhase.SETUP and vel < -self.STILL_THRESH:
                self._swing_started = True
                self._phase = SwingPhase.BACKSWING
                self._min_y_seen = wy

        elif self._phase == SwingPhase.BACKSWING:
            if wy < self._min_y_seen:
                self._min_y_seen = wy          # still going up

            # Transition: velocity crosses zero (wrist slows at top)
            if vel > -self.TOP_THRESH:
                self._phase = SwingPhase.TRANSITION

        elif self._phase == SwingPhase.TRANSITION:
            # Downswing begins when wrist accelerates downward
            if vel > self.TOP_THRESH:
                self._phase = SwingPhase.DOWNSWING

        elif self._phase == SwingPhase.DOWNSWING:
            # Impact zone: wrist passes back through its address height
            # Follow-through: wrist starts rising again after impact
            if vel < -self.STILL_THRESH:
                self._phase = SwingPhase.FOLLOW_THROUGH

        # FOLLOW_THROUGH is terminal — stays until next reset()
        return self._phase

    # ------------------------------------------------------------------
    def reset(self):
        """Call between swings to restart detection."""
        self._y_history.clear()
        self._phase = SwingPhase.UNKNOWN
        self._still_count = 0
        self._swing_started = False
        self._min_y_seen = 1.0
        self._peak_locked = False

    # ------------------------------------------------------------------
    def _smoothed_velocity(self) -> float:
        """
        Central-difference velocity at the most recent frame,
        averaged over available history for smoothness.
        """
        arr = np.array(self._y_history)
        if len(arr) < 3:
            return 0.0
        # Use last 3 points for a simple central difference
        return float((arr[-1] - arr[-3]) / 2.0)


# -----------------------------------------------------------------------
# Phase colour map (BGR for OpenCV overlay)
# -----------------------------------------------------------------------

PHASE_COLORS: dict[SwingPhase, tuple[int, int, int]] = {
    SwingPhase.SETUP:           (200, 200, 200),   # grey
    SwingPhase.BACKSWING:       (50,  200,  50),   # green
    SwingPhase.TRANSITION:      (0,   200, 255),   # yellow
    SwingPhase.DOWNSWING:       (50,   50, 255),   # red
    SwingPhase.FOLLOW_THROUGH:  (255, 150,  50),   # orange
    SwingPhase.UNKNOWN:         (128, 128, 128),   # grey
}
