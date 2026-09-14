"""
lap_counter.py
---------------
Implements the finish-line crossing state machine for a single lane.

A lap is only counted when ALL of the following hold:
  1. A runner of the lane's assigned color is currently detected (handled
     upstream by the detector; this module just receives a centroid).
  2. The centroid transitions from the FAR side of the finish line to the
     NEAR side (a genuine physical crossing, not just "currently on the
     near side" every frame).
  3. The estimated motion direction at the moment of crossing is
     TOWARD_CAMERA.
  4. At least `debounce_seconds` have passed since the last counted lap for
     this lane (protects against a runner oscillating/hovering on the
     line generating multiple counts).

Tracking the FAR->NEAR *transition* (rather than simple line-touching) is
what guarantees "one physical crossing = exactly one lap" even if the
runner's centroid jitters slightly around the finish line for several
frames -- once they are on the NEAR side, no further lap will be counted
for that lane until they are observed back on the FAR side again.
"""

from typing import Optional

from direction_tracker import DirectionTracker

FAR = "far"
NEAR = "near"


class LaneCrossingDetector:
    def __init__(self, debounce_seconds: float = 1.0) -> None:
        self.debounce_seconds = debounce_seconds
        self._last_side: Optional[str] = None
        self._last_lap_timestamp: float = float("-inf")

    def reset(self) -> None:
        self._last_side = None
        self._last_lap_timestamp = float("-inf")

    def check_crossing(
        self,
        centroid_y: int,
        direction: Optional[str],
        timestamp: float,
        finish_line_y: int,
    ) -> bool:
        """
        Update internal side-tracking state and return True if this frame
        represents a valid, newly-counted lap crossing.
        """
        # Side is defined relative to the finish line. FAR = still on the
        # track side, NEAR = has reached/passed the camera side of the line.
        current_side = FAR if centroid_y < finish_line_y else NEAR

        crossed = False
        if (
            self._last_side == FAR
            and current_side == NEAR
            and direction == DirectionTracker.TOWARD_CAMERA
            and (timestamp - self._last_lap_timestamp) >= self.debounce_seconds
        ):
            crossed = True
            self._last_lap_timestamp = timestamp

        self._last_side = current_side
        return crossed
