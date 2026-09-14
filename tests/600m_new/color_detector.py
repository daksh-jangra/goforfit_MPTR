"""
color_detector.py
------------------
Concrete implementation of BaseDetector using HSV color segmentation.

For each color assigned to a lane, this detector:
  1. Builds a binary mask isolating pixels within that color's calibrated
     HSV range(s) (see config.HSV_COLOR_RANGES).
  2. Cleans the mask with morphological opening/closing to remove noise and
     fill small holes.
  3. Finds contours and keeps the single largest one above a minimum area
     threshold, since exactly one runner per lane is expected.
  4. Reports that contour's centroid and bounding box as the Detection for
     that lane's color.
"""

from typing import Dict, List

import cv2
import numpy as np

from config import HSV_COLOR_RANGES
from detector import BaseDetector, Detection


class ColorDetector(BaseDetector):
    def __init__(self, colors_to_detect: List[str], min_contour_area: int = 800) -> None:
        self.colors_to_detect = colors_to_detect
        self.min_contour_area = min_contour_area
        self._morph_kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame: np.ndarray) -> Dict[str, Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: Dict[str, Detection] = {}

        for color in self.colors_to_detect:
            mask = self._build_mask(hsv, color)
            if mask is None:
                continue

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2

            results[color] = Detection(color=color, centroid=(cx, cy), bbox=(x, y, w, h), area=area)

        return results

    def _build_mask(self, hsv: np.ndarray, color: str):
        ranges = HSV_COLOR_RANGES.get(color)
        if not ranges:
            return None

        mask = None
        for lower, upper in ranges:
            part = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        return mask