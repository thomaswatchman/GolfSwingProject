"""
swing_phase_detector.py
Four phases:

    BACKSWING      -> Wrists rising away from ball
    DOWNSWING      -> Wrists accelerating downward toward impact
    BALL_CONTACT   -> Lowest wrist point = moment of impact
    FOLLOW_THROUGH -> Post-impact, wrists rising again

Swing starts as soon as the wrists begin moving upward.
Ball Contact fires when wrists rise CONTACT_RISE_THRESH above
their lowest point during the downswing.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np


class SwingPhase(str, Enum):
    BACKSWING      = "Backswing"
    DOWNSWING      = "Downswing"
    BALL_CONTACT   = "Ball Contact"
    FOLLOW_THROUGH = "Follow Through"
    UNKNOWN        = "Unknown"


class SwingPhaseDetector:
    WINDOW              = 9
    STILL_THRESH        = 0.004
    CONTACT_HOLD_FRAMES = 3
    CONTACT_RISE_THRESH = 0.0001

    def __init__(self):
        self._y_history: deque[float] = deque(maxlen=self.WINDOW)
        self._phase        = SwingPhase.UNKNOWN
        self._ds_lowest_y  = 0.0
        self._contact_hold = 0

    @property
    def phase(self) -> SwingPhase:
        return self._phase

    def update(self, landmarks: dict | None) -> SwingPhase:
        if landmarks is None:
            return self._phase

        wy = max(landmarks["right_wrist"][1], landmarks["left_wrist"][1])
        self._y_history.append(wy)
        if len(self._y_history) < 3:
            return self._phase

        vel = self._smoothed_velocity()

        # Ball Contact hold countdown
        if self._phase == SwingPhase.BALL_CONTACT:
            self._contact_hold -= 1
            if self._contact_hold <= 0:
                self._phase = SwingPhase.FOLLOW_THROUGH
            return self._phase

        # Follow Through is terminal
        if self._phase == SwingPhase.FOLLOW_THROUGH:
            return self._phase

        # UNKNOWN -> BACKSWING: wrist starts rising
        if self._phase == SwingPhase.UNKNOWN:
            if vel < -self.STILL_THRESH:
                self._phase = SwingPhase.BACKSWING

        # BACKSWING -> DOWNSWING: wrist reverses direction
        elif self._phase == SwingPhase.BACKSWING:
            if vel > self.STILL_THRESH:
                self._phase       = SwingPhase.DOWNSWING
                self._ds_lowest_y = wy

        # DOWNSWING -> BALL_CONTACT: wrist rises above its lowest point
        elif self._phase == SwingPhase.DOWNSWING:
            if wy > self._ds_lowest_y:
                self._ds_lowest_y = wy
            if wy < self._ds_lowest_y - self.CONTACT_RISE_THRESH:
                self._phase        = SwingPhase.BALL_CONTACT
                self._contact_hold = self.CONTACT_HOLD_FRAMES

        return self._phase

    def reset(self):
        self._y_history.clear()
        self._phase        = SwingPhase.UNKNOWN
        self._ds_lowest_y  = 0.0
        self._contact_hold = 0

    def _smoothed_velocity(self) -> float:
        arr = np.array(self._y_history)
        if len(arr) < 3:
            return 0.0
        return float((arr[-1] - arr[-3]) / 2.0)


PHASE_COLORS: dict[SwingPhase, tuple[int, int, int]] = {
    SwingPhase.BACKSWING:       (50,  205,  50),
    SwingPhase.DOWNSWING:       (50,   50, 255),
    SwingPhase.BALL_CONTACT:    (0,   255, 255),
    SwingPhase.FOLLOW_THROUGH:  (255, 150,  50),
    SwingPhase.UNKNOWN:         (100, 100, 100),
}