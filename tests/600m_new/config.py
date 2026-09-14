"""
config.py
---------
Centralised configuration for the Computer Vision Fitness Test application.

Keeping every tunable value in one module makes the application easy to
re-configure (and, later, easy to expose through a Flask/FastAPI settings
endpoint) without touching the business logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# Colors available for lane assignment.
# ----------------------------------------------------------------------------
AVAILABLE_COLORS: List[str] = ["Red", "Blue", "Green", "Yellow", "Orange", "White"]

# ----------------------------------------------------------------------------
# HSV calibration ranges for each color.
#
# Each color maps to a LIST of (lower, upper) HSV tuples because some colors
# (notably Red) wrap around the Hue axis (0-10 and 170-180) and therefore
# need two ranges to be captured correctly with a single inRange() call each.
#
# These values are intentionally kept in their own structure so they can be
# re-calibrated for different lighting conditions without touching detector
# logic.
# ----------------------------------------------------------------------------
HSV_COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "Red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (180, 255, 255)),
    ],
    "Blue": [
        ((94, 80, 2), (126, 255, 255)),
    ],
    "Green": [
        ((36, 60, 60), (86, 255, 255)),
    ],
    "Yellow": [
        ((20, 100, 100), (34, 255, 255)),
    ],
    "Orange": [
        ((10, 120, 100), (19, 255, 255)),
    ],
    "White": [
        ((0, 0, 200), (180, 40, 255)),
    ],
}

# ----------------------------------------------------------------------------
# BGR colors used purely for UI rendering (swatches, lane labels, etc.)
# ----------------------------------------------------------------------------
COLOR_BGR: Dict[str, Tuple[int, int, int]] = {
    "Red": (0, 0, 255),
    "Blue": (255, 80, 0),
    "Green": (0, 200, 0),
    "Yellow": (0, 220, 220),
    "Orange": (0, 140, 255),
    "White": (235, 235, 235),
}


@dataclass
class AppConfig:
    """Holds every runtime-tunable parameter for a single test session."""

    # Physical setup
    camera_distance_m: float = 0.0

    # lane_number (1-based) -> color name. Populated by the configuration screen.
    lane_colors: Dict[int, str] = field(default_factory=dict)
    num_lanes: int = 5

    # Camera / rendering
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 720
    target_fps: int = 30

    # Finish line: expressed as a fraction of the frame height so it scales
    # automatically with resolution.
    finish_line_y_ratio: float = 0.6

    # Detection tuning
    min_contour_area: int = 800

    # Debounce: minimum time (seconds) that must pass between two counted
    # laps for the SAME lane. Prevents a runner oscillating on the line from
    # generating multiple laps.
    crossing_debounce_seconds: float = 1.0

    # How long (seconds) the finish line flashes BLACK after a valid lap.
    lap_flash_duration: float = 0.5

    # Number of centroid samples kept to estimate movement direction.
    direction_history_len: int = 8

    # Minimum pixel movement (in y) across the history window required to
    # classify movement as "toward" or "away" rather than "static / noise".
    direction_noise_threshold_px: float = 4.0

    # Assumption used to translate raw pixel-Y motion into a real-world
    # direction. See README "Assumptions" section for full explanation.
    # True  -> increasing Y (moving down the frame) means the runner is
    #          approaching the camera positioned at/behind the finish line.
    # False -> the opposite (flip this if your camera is mounted upside
    #          down or looking the other way down the track).
    toward_camera_is_increasing_y: bool = True

    def get_lane_for_color(self, color: str):
        """Return the lane number assigned to a given color, or None."""
        for lane, c in self.lane_colors.items():
            if c == color:
                return lane
        return None

    def finish_line_y_px(self) -> int:
        """Absolute pixel Y coordinate of the finish line for this frame size."""
        return int(self.frame_height * self.finish_line_y_ratio)
