"""
ui.py
-----
All on-screen rendering for the main test window. This module is the only
place (besides main.py's window plumbing and config_screen.py) that touches
cv2.imshow-style presentation concerns -- the detection/timing/business
logic modules know nothing about how (or whether) their data gets drawn.

Layout:
  +---------------------------+--------------------+
  |                           |   STATUS / TIMER   |
  |      LIVE CAMERA FEED     +--------------------+
  |      + finish line        |  Lane 1 card       |
  |                           |  Lane 2 card       |
  |                           |  ...               |
  +---------------------------+--------------------+
"""

import time
from typing import Dict, Optional

import cv2
import numpy as np

from config import AppConfig, COLOR_BGR
from lane_manager import LaneManager
from state_machine import AppState
from utils import draw_rounded_rect, put_text, text_size, format_time

PANEL_BG = (32, 32, 32)
CARD_BG = (48, 48, 48)
TEXT_PRIMARY = (235, 235, 235)
TEXT_SECONDARY = (170, 170, 170)
ACCENT = (0, 210, 255)


class Renderer:
    def __init__(self, config: AppConfig, stats_panel_width: int = 380) -> None:
        self.config = config
        self.stats_panel_width = stats_panel_width
        self.canvas_width = config.frame_width + stats_panel_width
        self.canvas_height = config.frame_height

        # Finish-line flash state
        self._flash_until: float = 0.0

    def trigger_flash(self) -> None:
        self._flash_until = time.perf_counter() + self.config.lap_flash_duration

    def _finish_line_color(self) -> tuple:
        if time.perf_counter() < self._flash_until:
            return (0, 0, 0)  # BLACK during flash
        return (255, 255, 255)  # WHITE otherwise

    def render(
        self,
        frame: np.ndarray,
        lane_manager: LaneManager,
        state: AppState,
        global_elapsed: float,
    ) -> np.ndarray:
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas[:, :] = PANEL_BG

        camera_panel = self._build_camera_panel(frame, lane_manager)
        canvas[0 : self.config.frame_height, 0 : self.config.frame_width] = camera_panel

        stats_panel = self._build_stats_panel(lane_manager, state, global_elapsed)
        canvas[:, self.config.frame_width :] = stats_panel

        return canvas

    # ------------------------------------------------------------------
    def _build_camera_panel(self, frame: np.ndarray, lane_manager: LaneManager) -> np.ndarray:
        panel = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))

        finish_y = self.config.finish_line_y_px()
        line_color = self._finish_line_color()
        cv2.line(panel, (0, finish_y), (self.config.frame_width, finish_y), line_color, 3, cv2.LINE_AA)
        put_text(panel, "FINISH LINE", (10, finish_y - 10), 0.55, line_color, 2)

        # Draw a bounding box + label for every currently detected runner.
        for lane in lane_manager.lanes.values():
            if not lane.currently_detected or lane.last_bbox is None:
                continue
            x, y, w, h = lane.last_bbox
            color = COLOR_BGR.get(lane.color, (255, 255, 255))
            cv2.rectangle(panel, (x, y), (x + w, y + h), color, 2)
            label = f"L{lane.lane_number}:{lane.color}"
            put_text(panel, label, (x, max(15, y - 8)), 0.5, color, 2)
            if lane.last_centroid:
                cv2.circle(panel, lane.last_centroid, 4, color, -1)

        return panel

    # ------------------------------------------------------------------
    def _build_stats_panel(
        self, lane_manager: LaneManager, state: AppState, global_elapsed: float
    ) -> np.ndarray:
        panel = np.zeros((self.canvas_height, self.stats_panel_width, 3), dtype=np.uint8)
        panel[:, :] = PANEL_BG

        margin = 14
        y = margin

        # ---- Status header ----
        status_color = {
            AppState.READY: (0, 200, 255),
            AppState.RUNNING: (0, 220, 0),
            AppState.STOPPED: (0, 0, 220),
        }.get(state, TEXT_PRIMARY)

        draw_rounded_rect(panel, (margin, y), (self.stats_panel_width - margin, y + 70), CARD_BG, -1, 10)
        put_text(panel, f"STATUS: {state.value}", (margin + 14, y + 30), 0.75, status_color, 2)

        instruction = {
            AppState.READY: "Press SPACE to Start Test",
            AppState.RUNNING: "Press SPACE to Stop Test",
            AppState.STOPPED: "Press SPACE to Start Again",
        }[state]
        put_text(panel, instruction, (margin + 14, y + 55), 0.5, TEXT_SECONDARY, 1)
        y += 80

        # ---- Global timer ----
        draw_rounded_rect(panel, (margin, y), (self.stats_panel_width - margin, y + 40), CARD_BG, -1, 10)
        put_text(panel, f"Global Timer: {format_time(global_elapsed)}", (margin + 14, y + 27), 0.55, ACCENT, 2)
        y += 52

        # ---- Per-lane cards ----
        card_height = (self.canvas_height - y - margin) // max(1, len(lane_manager.lanes))
        card_height = min(card_height, 128)

        for lane in lane_manager.lanes.values():
            self._draw_lane_card(panel, lane, margin, y, self.stats_panel_width - margin, y + card_height - 8)
            y += card_height

        return panel

    def _draw_lane_card(self, panel: np.ndarray, lane, x1: int, y1: int, x2: int, y2: int) -> None:
        draw_rounded_rect(panel, (x1, y1), (x2, y2), CARD_BG, -1, 10)

        color = COLOR_BGR.get(lane.color, (255, 255, 255))
        # Color swatch
        cv2.rectangle(panel, (x1 + 10, y1 + 10), (x1 + 34, y1 + 34), color, -1)
        cv2.rectangle(panel, (x1 + 10, y1 + 10), (x1 + 34, y1 + 34), (10, 10, 10), 1)

        put_text(panel, f"Lane {lane.lane_number}", (x1 + 44, y1 + 24), 0.6, TEXT_PRIMARY, 2)
        put_text(panel, f"Color: {lane.color}", (x1 + 44, y1 + 44), 0.45, TEXT_SECONDARY, 1)

        detected_marker = "DETECTED" if lane.currently_detected else "--"
        marker_color = (0, 220, 0) if lane.currently_detected else (110, 110, 110)
        put_text(panel, detected_marker, (x2 - 105, y1 + 24), 0.45, marker_color, 1)

        line_y = y1 + 62
        put_text(panel, f"Timer: {format_time(lane.timer.current_lap_elapsed)}", (x1 + 12, line_y), 0.48, TEXT_PRIMARY, 1)
        put_text(panel, f"Laps: {lane.timer.lap_count}", (x1 + 12, line_y + 20), 0.48, TEXT_PRIMARY, 1)
        put_text(panel, f"Last Lap: {lane.timer.last_lap_time:.2f}s", (x1 + 12, line_y + 40), 0.45, TEXT_SECONDARY, 1)
        put_text(panel, f"Total: {format_time(lane.timer.total_elapsed)}", (x1 + 12, line_y + 60), 0.45, TEXT_SECONDARY, 1)
