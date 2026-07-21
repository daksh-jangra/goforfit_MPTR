"""
detector.py
-----------
Defines the detector abstraction used throughout the application.

Any concrete detector (HSV color segmentation today, a YOLO/deep-learning
model tomorrow) must implement `BaseDetector.detect(frame)` and return a
mapping of {lane_color: Detection}. Because lap_counter.py, lane_manager.py,
and ui.py only ever talk to this interface, swapping the detection backend
later requires touching nothing but a single line in main.py where the
concrete detector is instantiated.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Detection:
    """A single detected runner within one video frame."""
    color: str
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    confidence: float = 1.0  # HSV segmentation has no natural confidence; reserved for future detectors


class BaseDetector(ABC):
    """Abstract detector interface. All detectors are stateless per call."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Detection]:
        """
        Run detection on a single BGR frame.

        Returns a dict mapping color name -> Detection for every color that
        was found in the frame (at most one Detection per color, i.e. one
        runner per lane, per the application's constraints).
        """
        raise NotImplementedError
