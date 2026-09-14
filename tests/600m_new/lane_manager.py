"""
lane_manager.py
----------------
Owns all per-lane state and orchestrates the per-frame update cycle:
  detections -> direction tracking -> crossing detection -> timers.

This module has no OpenCV GUI dependency (no imshow/waitKey), which is what
allows the whole detection+timing+lap-counting engine to be reused directly
inside a Flask/FastAPI backend later -- only `ui.py` and `main.py` deal with
on-screen windows.
"""

from typing import Dict, List, Optional, Tuple

from config import AppConfig
from detector import Detection
from direction_tracker import DirectionTracker
from lap_counter import LaneCrossingDetector
from timer import LaneTimer


class Lane:
    """All state belonging to a single running lane."""

    def __init__(self, lane_number: int, color: str, config: AppConfig) -> None:
        self.lane_number = lane_number
        self.color = color

        self.timer = LaneTimer()
        self.direction_tracker = DirectionTracker(
            history_len=config.direction_history_len,
            noise_threshold_px=config.direction_noise_threshold_px,
            toward_camera_is_increasing_y=config.toward_camera_is_increasing_y,
        )
        self.crossing_detector = LaneCrossingDetector(debounce_seconds=config.crossing_debounce_seconds)

        self.currently_detected: bool = False
        self.last_centroid: Optional[Tuple[int, int]] = None
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self) -> None:
        self.timer.reset()
        self.direction_tracker.reset()
        self.crossing_detector.reset()
        self.currently_detected = False
        self.last_centroid = None
        self.last_bbox = None


class LaneManager:
    """Coordinates the full set of lanes for a test session."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.lanes: Dict[int, Lane] = {
            lane_num: Lane(lane_num, color, config)
            for lane_num, color in sorted(config.lane_colors.items())
        }

    # -- lifecycle -----------------------------------------------------
    def start_all(self) -> None:
        for lane in self.lanes.values():
            lane.timer.start()

    def stop_all(self) -> None:
        for lane in self.lanes.values():
            lane.timer.stop()

    def reset_all(self) -> None:
        for lane in self.lanes.values():
            lane.reset()

    # -- per-frame update -----------------------------------------------
    def update(
        self,
        detections: Dict[str, Detection],
        dt: float,
        timestamp: float,
        counting_enabled: bool,
    ) -> List[int]:
        """
        Advance all lanes by one frame.

        detections       : color -> Detection for this frame (from a BaseDetector)
        dt                : seconds elapsed since the previous frame
        timestamp         : monotonic clock reading for this frame
        counting_enabled  : True only while the app is in the RUNNING state

        Returns the list of lane numbers that recorded a NEW lap this frame
        (used by the UI to trigger the finish-line flash animation).
        """
        finish_line_y = self.config.finish_line_y_px()
        newly_lapped: List[int] = []

        for lane in self.lanes.values():
            lane.timer.update(dt)

            detection = detections.get(lane.color)
            if detection is None:
                lane.currently_detected = False
                continue

            lane.currently_detected = True
            lane.last_centroid = detection.centroid
            lane.last_bbox = detection.bbox
            lane.direction_tracker.update(detection.centroid, timestamp)

            if not counting_enabled:
                continue

            direction = lane.direction_tracker.get_direction()
            crossed = lane.crossing_detector.check_crossing(
                centroid_y=detection.centroid[1],
                direction=direction,
                timestamp=timestamp,
                finish_line_y=finish_line_y,
            )
            if crossed:
                lane.timer.record_lap()
                newly_lapped.append(lane.lane_number)

        return newly_lapped

    def get_colors(self) -> List[str]:
        return [lane.color for lane in self.lanes.values()]
