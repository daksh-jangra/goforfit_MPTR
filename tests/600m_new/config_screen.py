"""
config_screen.py
------------------
Pre-webcam configuration window.

Collects, before any camera frame is opened:
  1. Camera distance from the finish line (typed with the keyboard).
  2. Exactly 5 of the 6 available T-shirt colors (chosen by mouse-click on
     swatches). The click ORDER determines the lane mapping
     (1st click -> Lane 1, 2nd click -> Lane 2, ...).

Returns a populated AppConfig once the user confirms with ENTER.
"""

from typing import List

import cv2
import numpy as np

from config import AVAILABLE_COLORS, COLOR_BGR, AppConfig
from utils import draw_rounded_rect, put_text, text_size

WINDOW_NAME = "Fitness Test - Configuration"
WIN_W, WIN_H = 760, 560


class _ConfigState:
    def __init__(self) -> None:
        self.distance_text: str = ""
        self.selected_colors: List[str] = []
        self.swatch_rects = {}  # color -> (x1, y1, x2, y2)
        self.error_message: str = ""


def _build_swatch_rects() -> dict:
    rects = {}
    cols = 3
    swatch_w, swatch_h = 190, 90
    gap_x, gap_y = 30, 24
    start_x, start_y = 40, 220
    for i, color in enumerate(AVAILABLE_COLORS):
        row, col = divmod(i, cols)
        x1 = start_x + col * (swatch_w + gap_x)
        y1 = start_y + row * (swatch_h + gap_y)
        rects[color] = (x1, y1, x1 + swatch_w, y1 + swatch_h)
    return rects


def _mouse_callback(event, x, y, flags, state: _ConfigState) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for color, (x1, y1, x2, y2) in state.swatch_rects.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            if color in state.selected_colors:
                state.selected_colors.remove(color)
            elif len(state.selected_colors) < 5:
                state.selected_colors.append(color)
            state.error_message = ""
            return


def _render(state: _ConfigState) -> np.ndarray:
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:, :] = (30, 30, 30)

    put_text(canvas, "FITNESS TEST - CONFIGURATION", (40, 45), 1.0, (0, 210, 255), 2)
    put_text(canvas, "1) Enter camera distance from finish line (metres):", (40, 95), 0.6, (230, 230, 230), 1)

    draw_rounded_rect(canvas, (40, 110), (400, 155), (55, 55, 55), -1, 8)
    display_text = state.distance_text if state.distance_text else "e.g. 3.5"
    text_color = (255, 255, 255) if state.distance_text else (130, 130, 130)
    put_text(canvas, display_text, (55, 140), 0.65, text_color, 2)

    put_text(
        canvas,
        "2) Click 5 of the 6 colors below. Click order = Lane 1..5",
        (40, 190),
        0.6,
        (230, 230, 230),
        1,
    )

    for color, (x1, y1, x2, y2) in state.swatch_rects.items():
        bgr = COLOR_BGR[color]
        draw_rounded_rect(canvas, (x1, y1), (x2, y2), bgr, -1, 10)
        border = (0, 210, 255) if color in state.selected_colors else (90, 90, 90)
        draw_rounded_rect(canvas, (x1, y1), (x2, y2), border, 3, 10)

        label_color = (0, 0, 0) if color == "White" or color == "Yellow" else (255, 255, 255)
        put_text(canvas, color, (x1 + 12, y1 + 30), 0.6, label_color, 2)

        if color in state.selected_colors:
            lane_no = state.selected_colors.index(color) + 1
            put_text(canvas, f"Lane {lane_no}", (x1 + 12, y1 + 65), 0.6, label_color, 2)

    footer_y = WIN_H - 60
    put_text(
        canvas,
        f"Selected: {len(state.selected_colors)}/5      Press ENTER to start webcam, ESC to quit",
        (40, footer_y),
        0.55,
        (200, 200, 200),
        1,
    )
    if state.error_message:
        put_text(canvas, state.error_message, (40, footer_y + 28), 0.55, (0, 0, 255), 2)

    return canvas


def run_config_screen() -> AppConfig:
    """
    Blocks until the user finishes configuration. Returns a populated
    AppConfig, or raises SystemExit if the user presses ESC.
    """
    state = _ConfigState()
    state.swatch_rects = _build_swatch_rects()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, state)

    while True:
        frame = _render(state)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(WINDOW_NAME)
            raise SystemExit("Configuration cancelled by user.")

        elif key in (13, 10):  # ENTER
            error = _validate(state)
            if error:
                state.error_message = error
                continue
            cv2.destroyWindow(WINDOW_NAME)
            return _build_config(state)

        elif key == 8:  # Backspace
            state.distance_text = state.distance_text[:-1]

        elif key != 255:
            ch = chr(key) if 0 <= key < 256 else ""
            if ch.isdigit() or (ch == "." and "." not in state.distance_text):
                if len(state.distance_text) < 8:
                    state.distance_text += ch


def _validate(state: _ConfigState) -> str:
    if not state.distance_text:
        return "Please enter the camera distance (metres) before starting."
    try:
        value = float(state.distance_text)
        if value <= 0:
            return "Camera distance must be a positive number."
    except ValueError:
        return "Camera distance must be a valid number."

    if len(state.selected_colors) != 5:
        return "Please select exactly 5 colors."

    return ""


def _build_config(state: _ConfigState) -> AppConfig:
    config = AppConfig()
    config.camera_distance_m = float(state.distance_text)
    config.lane_colors = {i + 1: color for i, color in enumerate(state.selected_colors)}
    return config
