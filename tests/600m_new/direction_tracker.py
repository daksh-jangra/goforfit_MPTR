"""
direction_tracker.py
---------------------
Maintains a short history of centroid positions for a single lane and
estimates the runner's direction of travel along the Y axis.

Movement direction is inferred by comparing the earliest and most recent
sampled Y position within a rolling window, rather than a single
frame-to-frame delta, which makes the estimate robust to per-frame jitter
in the color-mask centroid.
"""

from collections import deque
from typing import Optional, Tuple


class DirectionTracker:
    TOWARD_CAMERA = "toward_camera"
    AWAY_FROM_CAMERA = "away_from_camera"
    STATIC = "static"

    def __init__(self, history_len: int = 8, noise_threshold_px: float = 4.0,
                 toward_camera_is_increasing_y: bool = True) -> None:
        self.history: deque = deque(maxlen=history_len)
        self.noise_threshold_px = noise_threshold_px
        self.toward_camera_is_increasing_y = toward_camera_is_increasing_y

    def update(self, centroid: Tuple[int, int], timestamp: float) -> None:
        self.history.append((centroid[0], centroid[1], timestamp))

    def reset(self) -> None:
        self.history.clear()

    def has_data(self) -> bool:
        return len(self.history) >= 2

    def get_direction(self) -> Optional[str]:
        """
        Returns TOWARD_CAMERA, AWAY_FROM_CAMERA, STATIC, or None if there
        isn't enough history yet.
        """
        if len(self.history) < 2:
            return None

        y_first = self.history[0][1]
        y_last = self.history[-1][1]
        delta_y = y_last - y_first

        if abs(delta_y) < self.noise_threshold_px:
            return self.STATIC

        moving_down_frame = delta_y > 0  # increasing Y = moving toward bottom of frame

        if moving_down_frame == self.toward_camera_is_increasing_y:
            return self.TOWARD_CAMERA
        return self.AWAY_FROM_CAMERA

    def latest_centroid(self) -> Optional[Tuple[int, int]]:
        if not self.history:
            return None
        x, y, _ = self.history[-1]
        return (x, y)
