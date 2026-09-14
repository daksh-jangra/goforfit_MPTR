"""
utils.py
--------
Small, dependency-free helper functions shared across the UI and other
modules. Kept separate so drawing/formatting helpers don't pollute business
logic modules.
"""

import cv2
import numpy as np
from typing import Tuple


def format_time(seconds: float) -> str:
    """Format a duration in seconds as MM:SS.xx (matches spec example 00:45.28)."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:05.2f}"


def draw_rounded_rect(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = -1,
    radius: int = 12,
) -> None:
    """Draw a filled or outlined rounded rectangle onto img."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        # Filled: draw two rectangles + four filled circles/pies for corners
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Outline: draw lines between corners + arcs
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Thin wrapper around cv2.putText with sane defaults and anti-aliasing."""
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def text_size(text: str, font_scale: float = 0.6, thickness: int = 1, font=cv2.FONT_HERSHEY_SIMPLEX):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h, baseline


def clamp(value, low, high):
    return max(low, min(high, value))
