"""
main.py
-------
Application entry point.

Flow:
  1. Show the configuration screen (camera distance + 5 lane colors).
  2. Open the webcam.
  3. Run the main loop:
       - grab a frame
       - run the (pluggable) detector
       - advance the lane manager (timers, direction, lap counting)
       - render the combined camera + stats UI
       - handle SPACE (start/stop/restart) and ESC (quit)

The detection backend is chosen in exactly one place below (`_build_detector`)
so it can later be swapped for a YOLO-based detector without touching any
other module.

--------------------------------------------------------------------------
GoForFit platform integration
--------------------------------------------------------------------------
This script is launched by the main GoForFit FastAPI app (app.py) as a
plain subprocess, exactly like the other AI test scripts (bmi__3_.py,
wingspan2.py, shuttleruntest.py, etc.) -- it still opens its own desktop
OpenCV window and is fully keyboard-driven; nothing about the core test
logic elsewhere in this package was changed.

Two small additions were made purely for platform integration:
  * --session-id is accepted on the command line (the web app passes this
    automatically when a test is launched from a student's session).
  * Whenever a lap is recorded, and again when the test is stopped, the
    current per-lane times are POSTed to the main app's
    POST /api/stage/run_600m endpoint. This stages the result so the
    coach can review/edit it on the "screen3" web page and click
    "Save to Database" -- the same staged-review flow every other AI
    test already uses. Staging on every lap (not just at the very end)
    means the latest results are already saved server-side even if the
    web UI has to force-kill this process via /api/stop/run_600m.
"""

import argparse
import os
import sys
import time

import cv2

from color_detector import ColorDetector
from config import AppConfig
from config_screen import run_config_screen
from detector import BaseDetector
from lane_manager import LaneManager
from state_machine import AppState, StateMachine
from timer import GlobalTimer
from ui import Renderer

try:
    import requests
except ImportError:  # pragma: no cover - staging is best-effort only
    requests = None

API_BASE_URL = os.environ.get("GOFORFIT_API_URL", "http://localhost:8080")

SPACE_KEY = 32
ESC_KEY = 27


def _parse_args():
    parser = argparse.ArgumentParser(description="600m multi-lane run (colour tracking)")
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Student session id (passed automatically by the GoForFit web app).",
    )
    return parser.parse_args()


def _stage_results(lane_manager: "LaneManager", session_id) -> None:
    """
    Best-effort POST of the current per-lane times to the main app so the
    coach can review/save them on screen3. Never raises -- a failed stage
    call must not interrupt the test.
    """
    if requests is None:
        return
    try:
        laps = [
            {
                "lane": lane.lane_number,
                "time_sec": round(lane.timer.total_elapsed, 3),
                "lap_count": lane.timer.lap_count,
            }
            for lane in lane_manager.lanes.values()
        ]
        payload = {"laps": laps}
        if session_id is not None:
            payload["session_id"] = session_id
        requests.post(f"{API_BASE_URL}/api/stage/run_600m", json=payload, timeout=1.5).raise_for_status()
    except Exception as exc:
        print(f"WARNING: could not stage 600m results ({exc})", file=sys.stderr)


def _build_detector(config: AppConfig) -> BaseDetector:
    """Single place to swap the detection backend (e.g. for a YOLO detector)."""
    colors = list(config.lane_colors.values())
    return ColorDetector(colors_to_detect=colors, min_contour_area=config.min_contour_area)


def _open_camera(config: AppConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {config.camera_index}.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    cap.set(cv2.CAP_PROP_FPS, config.target_fps)
    return cap


def main() -> None:
    args = _parse_args()
    session_id = args.session_id

    config = run_config_screen()

    lane_manager = LaneManager(config)
    detector = _build_detector(config)
    global_timer = GlobalTimer()
    renderer = Renderer(config)

    def on_start():
        global_timer.start()
        lane_manager.start_all()

    def on_stop():
        global_timer.stop()
        lane_manager.stop_all()
        # A STOPPED test is a completed result the coach can review, so
        # stage it right away (also covers the case where the web UI
        # force-kills this process straight after the STOP/SAVE button).
        _stage_results(lane_manager, session_id)

    def on_reset():
        global_timer.reset()
        lane_manager.reset_all()

    state_machine = StateMachine(on_start=on_start, on_stop=on_stop, on_reset=on_reset)

    cap = _open_camera(config)
    window_name = "Computer Vision Fitness Test"
    cv2.namedWindow(window_name)

    min_frame_time = 1.0 / config.target_fps
    last_time = time.perf_counter()

    print("Application started.")
    print(f"Camera distance from finish line: {config.camera_distance_m} m")
    print("Lane assignments:")
    for lane_no, color in sorted(config.lane_colors.items()):
        print(f"  Lane {lane_no}: {color}")
    print("Controls: SPACE = start/stop/restart, ESC = quit")

    try:
        while True:
            frame_start = time.perf_counter()
            dt = frame_start - last_time
            last_time = frame_start

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame from camera.", file=sys.stderr)
                break

            frame = cv2.resize(frame, (config.frame_width, config.frame_height))
            frame = cv2.flip(frame, 1)  # mirror for a more intuitive "toward camera" feel

            detections = detector.detect(frame)

            counting_enabled = state_machine.is_running()
            newly_lapped = lane_manager.update(
                detections=detections,
                dt=dt,
                timestamp=frame_start,
                counting_enabled=counting_enabled,
            )
            if newly_lapped:
                renderer.trigger_flash()
                # Keep the server-side staged result fresh after every lap,
                # so the coach always has the latest times to review even
                # if the browser has to force-stop this process later.
                _stage_results(lane_manager, session_id)

            global_timer.update(dt)

            canvas = renderer.render(
                frame=frame,
                lane_manager=lane_manager,
                state=state_machine.state,
                global_elapsed=global_timer.total_elapsed,
            )
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == SPACE_KEY:
                state_machine.handle_space()
            elif key == ESC_KEY:
                break

            # Basic frame pacing toward the target FPS.
            elapsed = time.perf_counter() - frame_start
            remaining = min_frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

    finally:
        if state_machine.state == AppState.STOPPED:
            _stage_results(lane_manager, session_id)
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()