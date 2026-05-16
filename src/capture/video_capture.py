"""
video_capture.py
Handles reading from webcam or video file and yielding frames.
"""

import cv2
from pathlib import Path


class VideoCapture:
    """Context manager for reading video from a device or file."""

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720):
        """
        Args:
            source: Camera index (0 = default webcam) or path to a video file.
            width:  Desired capture width in pixels.
            height: Desired capture height in pixels.
        """
        self.source = source
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def frames(self):
        """Generator: yield BGR frames one at a time."""
        if self._cap is None:
            raise RuntimeError("Use VideoCapture as a context manager.")
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def frame_count(self) -> int:
        """Total frames (-1 for live webcam)."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else -1
