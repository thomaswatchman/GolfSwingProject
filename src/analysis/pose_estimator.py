"""
pose_estimator.py
Wraps MediaPipe Pose Landmarker (Tasks API — mediapipe >= 0.10)
to detect and track body landmarks per frame.

The old mp.solutions.pose API was removed in mediapipe 0.10.x on macOS/Apple Silicon.
This module uses the new mediapipe.tasks API instead.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode

# -----------------------------------------------------------------------
# Model download (cached locally)
# -----------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).parent / "pose_landmarker_full.task"


def _ensure_model() -> str:
    if not MODEL_PATH.exists():
        print(f"[INFO] Downloading pose model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")
    return str(MODEL_PATH)


# -----------------------------------------------------------------------
# Landmark name → index mapping (MediaPipe BlazePose 33-point model)
# -----------------------------------------------------------------------

GOLF_LANDMARK_IDX = {
    "nose":            0,
    "left_shoulder":   11,
    "right_shoulder":  12,
    "left_elbow":      13,
    "right_elbow":     14,
    "left_wrist":      15,
    "right_wrist":     16,
    "left_hip":        23,
    "right_hip":       24,
    "left_knee":       25,
    "right_knee":      26,
    "left_ankle":      27,
    "right_ankle":     28,
}

# Skeleton connections to draw (pairs of landmark names)
_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]

_GREEN  = (50, 205,  50)
_YELLOW = (0,  220, 220)


# -----------------------------------------------------------------------
# PoseEstimator
# -----------------------------------------------------------------------

class PoseEstimator:
    """
    Detect body landmarks in a single BGR frame using the Tasks API.

    Usage (context manager):
        with PoseEstimator() as pe:
            for frame in frames:
                landmarks = pe.process(frame)   # dict or None
                annotated = pe.draw(frame)
    """

    def __init__(self, model_complexity: str = "full"):
        model_path = _ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,   # works for both files and webcam
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        self._frame_ts_ms = 0  # monotonically increasing timestamp

    # ------------------------------------------------------------------
    def _next_ts(self) -> int:
        """Return a strictly increasing timestamp in milliseconds."""
        self._frame_ts_ms += 33          # ~30 fps default; exact value doesn't matter
        return self._frame_ts_ms

    # ------------------------------------------------------------------
    def process(self, bgr_frame: np.ndarray) -> dict[str, tuple[float, float, float]] | None:
        """
        Run pose estimation on one BGR frame.

        Returns:
            Dict mapping landmark name → (x, y, z) in normalised [0,1] coords,
            or None if no pose was detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_ts())

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]   # first (only) detected person
        return {
            name: (lm[idx].x, lm[idx].y, lm[idx].z)
            for name, idx in GOLF_LANDMARK_IDX.items()
        }

    # ------------------------------------------------------------------
    def draw(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a copy of the frame with the skeleton overlaid."""
        landmarks = self.process(bgr_frame)
        out = bgr_frame.copy()
        if landmarks is None:
            return out

        h, w = out.shape[:2]

        def px(name):
            x, y, _ = landmarks[name]
            return int(x * w), int(y * h)

        # Draw connections
        for a, b in _CONNECTIONS:
            if a in landmarks and b in landmarks:
                cv2.line(out, px(a), px(b), _GREEN, 2, cv2.LINE_AA)

        # Draw joint dots
        for name in landmarks:
            cv2.circle(out, px(name), 5, _YELLOW, -1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()