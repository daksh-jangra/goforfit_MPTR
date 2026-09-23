"""
GoForFit NEP FMS — FastAPI Launcher  (with DB storage)
======================================================
Each test is an independent OpenCV/MediaPipe script that opens its own window.
This server acts as a launchpad: it starts them as subprocesses so they run
exactly as written (no code changes to the originals needed).

Converted from Flask to FastAPI. Behavior, routes, and JSON response shapes
are preserved 1:1 with the original app. The Jinja templates (index.html,
screen2.html, screen3.html, screen4.html, results.html) are used unmodified —
a small `url_for` compatibility shim below lets them keep using Flask-style
calls like `url_for('screen2', category='All Tests')`.

Usage:
    pip install -r requirements.txt
    python app.py
    Then open  http://localhost:5000  in your browser.

Interactive API docs (new, courtesy of FastAPI): http://localhost:5000/docs
"""

import os
# ── Force OpenCV into headless mode BEFORE cv2 is imported anywhere ────────
# Without this, OpenCV's Qt/GTK backend opens a blank window the moment
# VideoCapture is created, even when imshow() is never called.
os.environ.setdefault("DISPLAY", "")          # no X11 display → offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # Qt backend → offscreen
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # skip Windows Media Foundation GUI

import subprocess
import sys
import csv
import io
import time
import concurrent.futures
from pathlib import Path
import sqlite3
from typing import Any, Optional
from urllib.parse import urlencode

import cv2
import jinja2
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.templating import Jinja2Templates
from pydantic import BaseModel

from db import db  # unchanged database module

app = FastAPI(title="GoForFit NEP FMS")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
TESTS_DIR = BASE_DIR / "tests"
TEMPLATES_DIR = BASE_DIR / "templates"

# ── Wingspan: browser-controlled live feed (no desktop OpenCV window) ──────
# Every other test still opens its own OpenCV window via subprocess, as
# before. Wingspan is the one test that needed several distinct in-test
# actions (calibrate / measure / snapshot / save) which don't work from a
# phone with no keyboard, so it now runs in-process and streams MJPEG
# frames to <img> in screen4 instead, controlled by on-screen buttons.
sys.path.insert(0, str(TESTS_DIR))

# These hold the *actual* exception text when an engine module fails to
# import, so we can surface it to the browser instead of only ever
# printing it to a server console nobody is watching.
import traceback as _traceback

WEB_ENGINES_IMPORT_ERROR: Optional[str] = None
DUAL_FEED_IMPORT_ERROR: Optional[str] = None

try:
    from web_engines import (
        BmiEngine, CurlUpEngine, PullUpEngine, FullPullUpEngine, WallTossEngine,
        SitReachEngine, TrunkLiftEngine, ReverseCurlEngine,
        ShoulderStretchEngine,
        WingspanEngine, VerticalJumpEngine,
    )
except Exception as _e:  # pragma: no cover — surfaced to the browser, see below
    BmiEngine = CurlUpEngine = PullUpEngine = FullPullUpEngine = None
    WallTossEngine = SitReachEngine = TrunkLiftEngine = None
    ReverseCurlEngine = None
    ShoulderStretchEngine = None
    WingspanEngine = VerticalJumpEngine = None
    WEB_ENGINES_IMPORT_ERROR = f"{_e.__class__.__name__}: {_e}"
    print("=" * 70)
    print("[FATAL] Live-feed camera engines failed to import — EVERY AI test")
    print("        (including wingspan) will be UNAVAILABLE in the browser")
    print("        until this is fixed. Root cause:")
    print("-" * 70)
    _traceback.print_exc()
    print("=" * 70)

try:
    from dual_feed_engines import ShuttleRunEngine, Dash50mEngine, Run600mEngine
except Exception as _e:  # pragma: no cover — surfaced to the browser, see below
    ShuttleRunEngine = Dash50mEngine = Run600mEngine = None
    DUAL_FEED_IMPORT_ERROR = f"{_e.__class__.__name__}: {_e}"
    print("=" * 70)
    print("[FATAL] Dual-camera engines failed to import — shuttle_run/dash_50m/")
    print("        run_600m will be UNAVAILABLE in the browser until fixed.")
    print("        Root cause:")
    print("-" * 70)
    _traceback.print_exc()
    print("=" * 70)

_wingspan_engine = None      
_vertical_jump_engine = None
_shuttle_engine = None
_dash50m_engine = None
_run600m_engine = None
_bmi_engine = None           
_curl_up_engine = None
_pull_up_engine = None
_full_pull_up_engine = None
_wall_toss_engine = None
_sit_reach_engine = None     
_trunk_lift_engine = None
_reverse_curl_engine = None
_shoulder_stretch_engine = None


# ── Jinja templates, with a Flask-style url_for shim ─────────────────────────
# Starlette's built-in `url_for` only understands *path* params (e.g. /x/{id}),
# not Flask-style "extra kwargs become a query string" calls such as
# `url_for('screen2', category='All Tests')`. The templates use that pattern
# extensively, so we replace the default `url_for` global with one that
# replicates Flask's behavior — this means the .html files don't need to change.

@jinja2.pass_context
def _flask_style_url_for(context, name: str, **params):
    request: Request = context["request"]
    route = None
    for r in request.app.routes:
        if getattr(r, "name", None) == name:
            route = r
            break
    if route is None:
        raise ValueError(f"No route named '{name}'")

    path_param_names = set(getattr(route, "param_convertors", {}).keys())
    path_params  = {k: v for k, v in params.items() if k in path_param_names}
    query_params = {k: v for k, v in params.items() if k not in path_param_names}

    path = request.url_for(name, **path_params).path
    if query_params:
        path = f"{path}?{urlencode(query_params)}"
    return path


_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=jinja2.select_autoescape(),
)
_jinja_env.globals["url_for"] = _flask_style_url_for  # setdefault() inside Jinja2Templates keeps this
templates = Jinja2Templates(env=_jinja_env)


async def _json_body(request: Request) -> dict:
    """Mirror Flask's `request.get_json(silent=True, force=True) or {}` — never raises."""
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


# ── Test registry ──────────────────────────────────────────────────────────
TESTS = {
    "bmi": {
        "id":          "bmi",
        "title":       "BMI / Height Estimator",
        "subtitle":    "MediaPipe pose-based height measurement + BMI calculation",
        "icon":        "📏",
        "category":    "Anthropometry",
        "script":      "bmi__3_.py",
        "description": (
            "Estimates height from webcam using MediaPipe PoseLandmarker. "
            "Calibrate once with a known height, then measure anyone instantly. "
            "Outputs height in cm and calculates BMI when weight is provided."
        ),
        "controls": [
            ("SPACE", "Start 5-second scan"),
            ("C",     "Calibrate with known-height person"),
            ("R",     "Reset calibration"),
            ("S",     "Save readings to CSV"),
            ("Q",     "Quit"),
        ],
        "tip":    "Mount the camera level (pointing straight ahead) and stand with your whole body in the frame. Calibrate once with a known height — no distance measurement needed.",
        "params": [],
        "unit":   "cm",
        "live_feed":   True,
    },
    "wingspan": {
        "id":          "wingspan",
        "title":       "Wingspan Measurement",
        "subtitle":    "Middle-fingertip to middle-fingertip in real cm",
        "icon":        "🦅",
        "category":    "Anthropometry",
        "script":      "wingspan2.py",
        "description": (
            "Uses MediaPipe Hand + Pose landmarkers to measure arm span accurately. "
            "Auto-downloads model files on first run. Calibrate for your distance, "
            "then capture a 3-second median snapshot."
        ),
        "controls": [
            ("C",     "Enter calibration mode"),
            ("M",     "Switch to measure mode"),
            ("SPACE", "Take 3-second median snapshot"),
            ("S",     "Save result to CSV"),
            ("Q",     "Quit"),
        ],
        "tip":    "Extend both arms fully, palms forward, fingers spread. Calibrate once with a known wingspan — no distance measurement needed; just stand at the calibration spot.",
        "params": [],
        "unit":   "cm",
        "live_feed":   True,
    },
    "curl_up": {
        "id":          "curl_up",
        "title":       "Partial Curl-Up Counter",
        "subtitle":    "Timed abdominal strength test — 30 seconds",
        "icon":        "🏋️",
        "category":    "Muscular Endurance",
        "script":      "parcurlup.py",
        "description": (
            "Counts partial curl-ups in 30 seconds using hip-angle analysis and "
            "shoulder-rise detection. Full-screen display with live rep counter, "
            "countdown timer, and session CSV export."
        ),
        "controls": [
            ("SPACE", "Start 30-second timer"),
            ("R",     "Reset counter"),
            ("F",     "Toggle fullscreen"),
            ("Q",     "Quit"),
        ],
        "tip":    "Lie flat, knees bent ~90°. Camera should see you from the side.",
        "params": [],
        "unit":   "reps",
        "live_feed":   True,
    },
    "pull_up": {
        "id":          "pull_up",
        "title":       "Modified Pull-Up",
        "subtitle":    "Shoulder strength and endurance — max reps, one trial",
        "icon":        "💪",
        "category":    "Muscular Endurance",
        "script":      "pullup.py",
        "description": (
            "Counts modified pull-ups on a pull-up stand. The participant lies on "
            "their back under the bar, grips it with an overgrasp, and on 'GO' raises "
            "the body by flexing the arms until the chin reaches the level of the bar, "
            "then lowers back without the shoulders touching the ground. Because the "
            "hands are on the bar, the wrists mark its height, so the chin-to-bar "
            "criterion is checked directly rather than estimated. The test ends by "
            "itself once the participant pauses for two or more seconds."
        ),
        "controls": [
            ("SPACE", "Start counting (the 'GO' signal)"),
            ("R",     "Reset counter"),
            ("F",     "Toggle fullscreen"),
            ("Q",     "Quit"),
        ],
        "tip": ("Set the bar just within reach of the participant lying on their back "
                "on the mat. Point the camera so the face and one whole arm, hand included, "
                "are in frame. Keep the knees straight and the back straight; a "
                "steady, constant pace scores best."),
        "params": [],
        "unit":   "reps",
        "live_feed":   True,
    },
    "full_pull_up": {
        "id":          "full_pull_up",
        "title":       "Pull-Up",
        "subtitle":    "Upper-body strength and endurance — max reps, one trial",
        "icon":        "🏋️",
        "category":    "Muscular Endurance",
        "script":      "pullup_full.py",
        "description": (
            "Counts pull-ups from a straight-arm hang. The participant hangs from the "
            "bar with an overhand grip, pulls up until the chin is above the bar, then "
            "lowers back to a full hang. A rep only counts if the chin clears the bar "
            "and the arms straighten again before the next pull. Because the hands are "
            "on the bar, the wrists mark its height, so the chin-over-bar criterion is "
            "checked directly. The test ends by itself when the participant drops off "
            "the bar."
        ),
        "controls": [
            ("SPACE", "Start counting (the 'GO' signal)"),
            ("R",     "Reset counter"),
            ("F",     "Toggle fullscreen"),
            ("Q",     "Quit"),
        ],
        "tip": ("Place the camera a few metres away, roughly level with the bar, so the "
                "head, arms and bar are all in frame. The participant starts from a "
                "still, straight-arm hang; no swinging, kicking or bent knees."),
        "params": [],
        "unit":   "reps",
        "live_feed":   True,
    },
    "wall_toss": {
        "id":          "wall_toss",
        "title":       "Wall Toss Coordination Test",
        "subtitle":    "Alternate-hand ball toss — 30 seconds",
        "icon":        "🎯",
        "category":    "Coordination",
        "script":      "wtt.py",
        "description": (
            "Counts alternating wall-toss catches using HSV ball tracking and "
            "MediaPipe hand detection. Five-state machine ensures only true "
            "alternating catches are counted."
        ),
        "controls": [
            ("S",   "Start / restart test"),
            ("R",   "Reset to zero"),
            ("C",   "Click-to-calibrate ball colour"),
            ("1–5", "Switch ball colour preset"),
            ("Q",   "Quit"),
        ],
        "tip":    "Use an orange ball and stand ~1 m from the wall. Camera to your side.",
        "params": [],
        "unit":   "catches",
        "live_feed":   True,
    },
    "shuttle_run": {
        "id":          "shuttle_run",
        "title":       "10 m Shuttle Run Timer",
        "subtitle":    "Dual camera, whistle-triggered start — up to 4 lanes",
        "icon":        "🏃",
        "category":    "Agility",
        "script":      "shuttleruntest.py",
        "description": (
            "Times a 10 m shuttle run using two cameras facing each other across the "
            "track, one at each end. Up to 4 students run in fixed lanes; a whistle "
            "(2.4–4.1 kHz) detected via microphone starts every lane's timer at once. "
            "MediaPipe Pose tracks ankle position per lane to detect alternating "
            "line touches at each end — each student needs the configured number of "
            "touches to finish. Results are staged for review and also saved to a "
            "timestamped CSV plus a raw JSON event log."
        ),
        "controls": [
            ("Click", "Calibrate: mark lane boundaries + touch line on each camera"),
            ("C",     "Confirm calibration step"),
            ("R",     "Reset calibration points"),
            ("SPACE", "Manual start (if no whistle detected)"),
            ("Q",     "Quit"),
        ],
        "tip": ("Two cameras face each other 10 m apart, one at each end of the track. "
                "Mark 4 lanes on the ground. A laptop/USB mic must be able to hear the whistle."),
        "params": [
            {
                "name":        "cam_a_url",
                "label":       "Camera A URL/Index (End A — start line)",
                "type":        "text",
                "placeholder": "e.g. http://192.168.1.74:4747/video or 0",
                "arg":         "--cam-a",
                "camera":      True,
            },
            {
                "name":        "cam_b_url",
                "label":       "Camera B URL/Index (End B — 10 m end)",
                "type":        "text",
                "placeholder": "e.g. http://192.168.1.75:4747/video or 1",
                "arg":         "--cam-b",
                "camera":      True,
            },
            {
                "name":        "touches",
                "label":       "Line touches required to finish",
                "type":        "number",
                "min":         2,
                "max":         12,
                "placeholder": "e.g. 4",
                "arg":         "--touches",
            },
        ],
        "timed":  True,
        "live_feed": True,
    },
    "dash_50m": {
        "id":          "dash_50m",
        "title":       "50 m Dash Timer",
        "subtitle":    "Two-camera sprint timing with automatic finish detection",
        "icon":        "⚡",
        "category":    "Speed",
        "script":      "sprint_timer.py",
        "description": (
            "Automatically times a 50 m sprint using two cameras: a perpendicular "
            "FINISH camera for precise crossing detection, and an OVERHEAD camera "
            "for lane identity. Supports whistle-triggered start, simultaneous "
            "multi-lane timing, and lap logging to CSV."
        ),
        "controls": [
            ("SPACE", "Arrows move/resize strip during calibration"),
            ("s+N",   "Arm lane N individually"),
            ("a",     "Toggle arm-all-on-whistle"),
            ("r",     "Reset all lanes"),
            ("Q",     "Quit"),
        ],
        "tip":    "FINISH cam: perpendicular, level with the line. OVERHEAD cam: elevated, looking straight down.",
        "params": [
            {
                "name":        "finish_cam_url",
                "label":       "Finish camera URL (perpendicular)",
                "type":        "text",
                "placeholder": "e.g. http://192.168.1.5:4747/video",
                "arg":         "--finish-cam",
                "camera":      True,
            },
            {
                "name":        "overhead_cam_url",
                "label":       "Overhead camera URL (lane identity)",
                "type":        "text",
                "placeholder": "e.g. http://192.168.1.6:4747/video",
                "arg":         "--overhead-cam",
                "camera":      True,
            },
            {
                "name":        "lanes",
                "label":       "Number of lanes",
                "type":        "number",
                "min":         1,
                "max":         9,
                "placeholder": "e.g. 4",
                "arg":         "--lanes",
            },
            {
                "name":        "lane_width",
                "label":       "Lane width (metres)",
                "type":        "number",
                "min":         0.5,
                "max":         3,
                "placeholder": "e.g. 1.22",
                "arg":         "--lane-width",
            },
        ],
        "timed":  True,
        "live_feed": True,
    },
    "sit_reach": {
        "id":          "sit_reach",
        "title":       "Sit and Reach Test",
        "subtitle":    "Flexibility measurement via finger tracking",
        "icon":        "🤸",
        "category":    "Flexibility",
        "script":      "snr.py",
        "description": (
            "Measures forward reach distance in cm using MediaPipe pose landmarks. "
            "Calibrated with your hip-to-ankle length for accurate cm output. "
            "Tracks maximum forward displacement during the reach."
        ),
        "controls": [
            ("SPACE", "Set start position / lock final result"),
            ("R",     "Reset for new trial"),
            ("Q",     "Quit and print summary"),
        ],
        "tip": "Sit on the floor, legs straight. Enter your hip-to-ankle length below before launching.",
        "params": [
            {
                "name":        "leg_cm",
                "label":       "Hip-to-ankle length (cm)",
                "type":        "number",
                "min":         20,
                "max":         130,
                "placeholder": "e.g. 87",
                "arg":         "--leg-cm",
            }
        ],
        "unit":   "cm",
        "live_feed":   True,
    },
    "trunk_lift": {
        "id":          "trunk_lift",
        "title":       "Trunk Lift",
        "subtitle":    "Chin-to-floor height, best of two trials",
        "icon":        "🧎",
        "category":    "Musculoskeletal",
        "script":      "trunk_lift.py",
        "description": (
            "Measures trunk extension: the participant lies prone and lifts the upper "
            "body, and the chin height above the mat is measured at the held peak. "
            "Tap both ends of the measuring stick on the video once to set the scale "
            "and the floor line. Two trials, better one kept, capped at 12 in (30 cm)."
        ),
        "controls": [
            ("TAP x2", "Tap both ends of the measuring stick"),
            ("MARK",   "Capture the current height as a trial"),
            ("UNDO",   "Remove the last trial"),
            ("RESET",  "Start the trials over"),
        ],
        "tip": (
            "Camera on the floor, side-on, 1.5-2 m away, whole mat in frame. Lay the "
            "measuring stick on the mat in front of the chin - where the protocol "
            "already puts it - so it sits the same distance from the lens as the chin."
        ),
        "params": [
            {
                "name":        "stick_cm",
                "label":       "Measuring stick length (cm)",
                "type":        "number",
                "min":         5,
                "max":         300,
                "placeholder": "e.g. 30",
                "arg":         "--stick-cm",
            }
        ],
        "unit":   "cm",
        "live_feed":   True,
    },
    "reverse_curl": {
        "id":          "reverse_curl",
        "title":       "Reverse Curl",
        "subtitle":    "Lift a 1 lb weight to 45\u00b0, hold 2 s \u2014 pass/fail",
        "icon":        "\U0001f4aa",
        "category":    "Musculoskeletal",
        "script":      "reverse_curl.py",
        "description": (
            "Hand, wrist and arm strength. Seated, the 1 lb (0.5 kg) dumbbell starts "
            "on the midpoint of the thigh; the participant flexes the elbow to 45\u00b0 "
            "or less, holds it there for 2 seconds, then lowers it back to the thigh "
            "under control. One trial, decided by the camera: a correct curl is a "
            "PASS, a real attempt that falls short is a FAIL. Practise before "
            "pressing Start. The coach still watches that the descent is slower "
            "than gravity."
        ),
        "controls": [
            ("RESET", "Start the trial over"),
        ],
        "tip": (
            "Camera side-on to the working arm at seated-shoulder height, 1.5-2 m "
            "away, with the shoulder, elbow, wrist and thigh all in frame. A "
            "straight-on front view foreshortens the elbow angle and good curls "
            "will read as short."
        ),
        # No params and no arm buttons: the camera works out which arm is
        # curling by itself. (/api/launch/reverse_curl still accepts an "arm"
        # field for scripted launches.)
        "params": [],
        "unit":   "pass/fail",
        "live_feed":   True,
    },
    "shoulder_stretch": {
        "id":          "shoulder_stretch",
        "title":       "Shoulder Stretch",
        "subtitle":    "Touch the fingertips behind the back \u2014 both sides in 15 s, pass/fail",
        "icon":        "\U0001f932",
        "category":    "Musculoskeletal",
        "script":      "shoulder_stretch.py",
        "description": (
            "Upper-body flexibility of the shoulder girdle. One arm reaches over the "
            "shoulder and down the back, the other reaches across and up the back, and "
            "the participant tries to touch the fingertips together between the "
            "shoulder blades. No equipment. BOTH sides must be done \u2014 right arm "
            "over the right shoulder AND left arm over the left \u2014 inside ONE trial "
            "of 15 seconds, which starts at the first real attempt (not when the "
            "camera opens). The item is PASSED only if the fingers touch on BOTH "
            "sides. The app measures the fingertip gap against the hand's own size, "
            "names the arm from the hands and decides the verdict; the coach judges "
            "that the reach is a genuine attempt."
        ),
        "controls": [
            ("RESET", "Start the trial over (clears both sides and the 15 s clock)"),
        ],
        "tip": (
            "Camera BEHIND the participant at upper-back height, 1.5-2 m away, with "
            "the upper back, both shoulders and BOTH HANDS in frame. This is the only "
            "test in the app shot from behind: the hands meet between the shoulder "
            "blades, so a front view cannot see them at all and no trial will score. "
            "Only the student should be in frame when the test starts \u2014 the camera "
            "locks onto the one person it sees and will not start with two."
        ),
        # No pre-launch params and no side control: both sides are scored in
        # one trial and the arm is named per frame from the hands.
        "params": [],
        "unit":   "pass/fail",
        "live_feed":   True,
    },
    "vertical_jump": {
        "id":          "vertical_jump",
        "title":       "Vertical Jump Test",
        "subtitle":    "Standing reach + max jump height — 5 attempts",
        "icon":        "🐰",
        "category":    "Power",
        "script":      "vertical_jump.py",
        "description": (
            "Uses MediaPipe Pose to track reach height and measure vertical jump. "
            "Capture your standing reach, calibrate it in cm, then take 5 jump "
            "attempts. Reports best jump, average jump, and per-attempt results."
        ),
        "controls": [
            ("S",     "Capture standing reach baseline"),
            ("C",     "Enter standing reach in cm (calibrate)"),
            ("SPACE", "Start a jump attempt"),
            ("R",     "Reset calibration"),
            ("ESC",   "Exit"),
        ],
        "tip":    "Stand side-on to the camera with your reaching arm fully visible.",
        "params": [],
        "unit":   "cm",
        "live_feed":   True,
    },
    "run_600m": {
        "id":          "run_600m",
        "title":       "600 m Run",
        "subtitle":    "Multi-lane endurance run — coloured T-shirt tracking, up to 5 lanes",
        "icon":        "🏃‍♂️",
        "category":    "Endurance",
        "script":      "600m_new/main.py",
        "description": (
            "Tracks up to 5 runners at once using HSV colour segmentation on "
            "coloured T-shirts. A virtual finish line inside the frame counts "
            "each valid, correctly-directed crossing as a completed lap, with "
            "a debounce so a runner lingering on the line isn't double-counted. "
            "Per-lane timers run for the whole test; the coach stops the test "
            "once every runner has covered 600 m."
        ),
        "controls": [
            ("Click", "On the setup screen: pick 5 lane colours + enter camera distance"),
            ("SPACE", "Start / Stop / Restart the test"),
            ("ESC",   "Quit"),
        ],
        "tip": (
            "Assign each runner a distinct T-shirt colour on the setup screen before "
            "the run starts, and position the camera so the finish line is clearly in frame."
        ),
        "params": [
            {
                "name":        "camera_url",
                "label":       "Camera URL/Index",
                "type":        "text",
                "placeholder": "e.g. http://192.168.1.74:4747/video or 0",
                "arg":         "--camera-url",
                "camera":      True,
            },
            {
                "name":        "lane_colors",
                "label":       "Lane colours (comma-separated, e.g. Red,Blue,Green,Yellow,Orange)",
                "type":        "text",
                "placeholder": "Red,Blue,Green,Yellow,Orange",
                "arg":         "--lane-colors",
            },
            {
                "name":        "camera_distance_m",
                "label":       "Camera distance from track (m)",
                "type":        "number",
                "min":         0,
                "max":         50,
                "placeholder": "e.g. 5",
                "arg":         "--camera-distance-m",
            },
        ],
        "timed":  True,
        "unit":   "s",
        "live_feed":   True,
    },
}

# ── Track running processes ────────────────────────────────────────────────
_running: dict[str, subprocess.Popen] = {}


def _is_running(test_id: str) -> bool:
    if test_id == "wingspan" and _wingspan_engine is not None:
        return _wingspan_engine.is_running()
    if test_id == "vertical_jump" and _vertical_jump_engine is not None:
        return _vertical_jump_engine.is_running()
    if test_id == "bmi" and _bmi_engine is not None:
        return _bmi_engine.is_running()
    if test_id == "curl_up" and _curl_up_engine is not None:
        return _curl_up_engine.is_running()
    if test_id == "pull_up" and _pull_up_engine is not None:
        return _pull_up_engine.is_running()
    if test_id == "full_pull_up" and _full_pull_up_engine is not None:
        return _full_pull_up_engine.is_running()
    if test_id == "wall_toss" and _wall_toss_engine is not None:
        return _wall_toss_engine.is_running()
    if test_id == "sit_reach" and _sit_reach_engine is not None:
        return _sit_reach_engine.is_running()
    if test_id == "trunk_lift" and _trunk_lift_engine is not None:
        return _trunk_lift_engine.is_running()
    if test_id == "reverse_curl" and _reverse_curl_engine is not None:
        return _reverse_curl_engine.is_running()
    if test_id == "shoulder_stretch" and _shoulder_stretch_engine is not None:
        return _shoulder_stretch_engine.is_running()
    if test_id == "shuttle_run" and _shuttle_engine is not None:
        return _shuttle_engine.is_running()
    if test_id == "dash_50m" and _dash50m_engine is not None:
        return _dash50m_engine.is_running()
    if test_id == "run_600m" and _run600m_engine is not None:
        return _run600m_engine.is_running()
    proc = _running.get(test_id)
    if proc is None:
        return False
    if proc.poll() is not None:
        del _running[test_id]
        return False
    return True

# ── Pending (staged-but-not-saved) AI results ──────────────────────────────
# AI test scripts no longer write straight to the DB. They POST their raw
# result here instead, where it sits until a human reviews it on screen3 and
# explicitly clicks "Save". Keyed by test_id, since each test's screen4/screen3
# is single-session-at-a-time in this app's UI.
_pending: dict[str, dict] = {}


# ── Pydantic models for the /api/save/* and /api/session bodies ─────────────
# Fields are optional (mirroring the original's lenient request.get_json()),
# and we manually enforce "session_id required" to keep the exact same
# {"ok": False, "error": ...} + 400 response shape the originals returned.

class SessionCreate(BaseModel):
    student_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    school: Optional[str] = None
    grade: Optional[str] = None
    notes: Optional[str] = None


class BMISave(BaseModel):
    session_id: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    calibration_used: Optional[bool] = False


class WingspanSave(BaseModel):
    session_id: Optional[int] = None
    wingspan_cm: Optional[float] = None
    height_cm: Optional[float] = None
    calibration_distance_cm: Optional[float] = None


class CurlUpSave(BaseModel):
    session_id: Optional[int] = None
    reps: Optional[int] = None
    duration_sec: Optional[int] = 30
    rating: Optional[str] = None


class PullUpSave(BaseModel):
    session_id: Optional[int] = None
    reps: Optional[int] = None
    duration_sec: Optional[float] = None
    rating: Optional[str] = None


class FullPullUpSave(BaseModel):
    session_id: Optional[int] = None
    reps: Optional[int] = None
    duration_sec: Optional[float] = None
    rating: Optional[str] = None


class WallTossSave(BaseModel):
    session_id: Optional[int] = None
    catches: Optional[int] = None
    duration_sec: Optional[int] = 30
    ball_colour: Optional[str] = None
    rating: Optional[str] = None


class ShuttleRunSave(BaseModel):
    session_id: Optional[int] = None
    time_sec: Optional[float] = None
    lane: Optional[int] = 1
    rating: Optional[str] = None


class Dash50mSave(BaseModel):
    session_id: Optional[int] = None
    time_sec: Optional[float] = None
    lane: Optional[int] = 1
    start_mode: Optional[str] = "whistle"
    rating: Optional[str] = None


class SitReachSave(BaseModel):
    session_id: Optional[int] = None
    reach_cm: Optional[float] = None
    leg_cm: Optional[float] = None
    rating: Optional[str] = None

class TrunkLiftSave(BaseModel):
    session_id: Optional[int] = None
    lift_cm: Optional[float] = None
    trial1_cm: Optional[float] = None
    trial2_cm: Optional[float] = None
    stick_cm: Optional[float] = None
    rating: Optional[str] = None

class ReverseCurlSave(BaseModel):
    session_id: Optional[int] = None
    passed: Optional[bool] = None
    arm: Optional[str] = None
    peak_angle_deg: Optional[float] = None
    hold_s: Optional[float] = None
    rating: Optional[str] = None

class ShoulderStretchSave(BaseModel):
    session_id: Optional[int] = None
    passed: Optional[bool] = None          # both sides touched
    left_passed: Optional[bool] = None
    right_passed: Optional[bool] = None
    best_gap_ratio: Optional[float] = None
    best_gap_left: Optional[float] = None
    best_gap_right: Optional[float] = None
    rating: Optional[str] = None

class VerticalJumpSave(BaseModel):
    session_id: Optional[int] = None
    jump_cm: Optional[float] = None
    average_jump_cm: Optional[float] = None
    standing_reach_cm: Optional[float] = None
    attempts: Optional[list[float]] = None
class Run600mSave(BaseModel):
    session_id: Optional[int] = None
    time_sec: Optional[float] = None
    lane: Optional[int] = 1
    lap_count: Optional[int] = None
    rating: Optional[str] = None

# ── Page routes ───────────────────────────────────────────────────────────

@app.get("/", name="index")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"tests": list(TESTS.values())})


@app.get("/screen2", name="screen2")
def screen2(request: Request, category: Optional[str] = None):
    categories: dict[str, list] = {}
    for test in TESTS.values():
        categories.setdefault(test["category"], []).append(test)

    selected = category or "All Tests"
    visible_tests = list(TESTS.values()) if selected == "All Tests" else categories.get(selected, [])
    return templates.TemplateResponse(
        request,
        "screen2.html",
        {
            "categories": categories,
            "selected": selected,
            "tests": visible_tests,
        },
    )


# ── Laravel popup-sync config (injected into screen3.html) ───────────────────
# screen3.html posts a confirmed result back to the Laravel parent window that
# opened this app as a popup (nep.goforfit.in), then closes. Both values below
# are read by screen3.html — without them the template's {{ ...|tojson }} calls
# get Jinja Undefined and the page 500s for every test.
#   • Override the origin for staging/prod via the GOFORFIT_PARENT_ORIGIN env
#     var. "*" is safe for local testing: syncResultToLaravel() only fires when
#     window.opener exists, and direct navigation has none.
#   • LARAVEL_SYNC maps testId -> {exerciseType, scoreKey} Laravel expects. An
#     empty map is safe — screen3 falls back to {exerciseType: testId,
#     scoreKey: 'score'} per test. Add only entries confirmed against the
#     Laravel side.
GOFORFIT_PARENT_ORIGIN = os.environ.get("GOFORFIT_PARENT_ORIGIN", "*")
LARAVEL_SYNC: dict[str, dict] = {}


@app.get("/screen3", name="screen3")
def screen3(request: Request, test: Optional[str] = None):
    test_id = test or "wall_toss"
    test_cfg = TESTS.get(test_id, TESTS["wall_toss"])
    return templates.TemplateResponse(request, "screen3.html", {
        "test": test_cfg,
        "parent_origin": GOFORFIT_PARENT_ORIGIN,
        "sync_map": LARAVEL_SYNC,
    })


@app.get("/screen4", name="screen4")
def screen4(request: Request, test: Optional[str] = None):
    test_id = test or "wall_toss"
    test_cfg = TESTS.get(test_id, TESTS["wall_toss"])
    return templates.TemplateResponse(request, "screen4.html", {"test": test_cfg})


# ── Students — the roster screen3 picks from ───────────────────────────────
# A class belongs to the school, so the roster is not scoped per coach: two
# coaches testing the same class must see the same children.

class StudentCreate(BaseModel):
    name: str
    grade: str
    roll_no: Optional[str] = None
    reg_no: Optional[str] = None
    school: Optional[str] = None


@app.get("/api/classes")
def list_classes():
    return {"ok": True, "classes": db.list_classes()}


@app.get("/api/students")
def list_students(grade: Optional[str] = Query(default=None)):
    """?grade=Class%204-A for one class; omit it for the whole school."""
    return {"ok": True, "students": db.list_students(grade=grade)}


@app.post("/api/students")
def create_student(body: StudentCreate):
    name = (body.name or "").strip()
    grade = (body.grade or "").strip()
    if not name or not grade:
        return JSONResponse(
            {"ok": False, "error": "A student needs a name and a class."},
            status_code=400,
        )
    try:
        student_id = db.add_student(
            name=name, grade=grade,
            roll_no=(body.roll_no or "").strip() or None,
            reg_no=(body.reg_no or "").strip() or None,
            school=(body.school or "").strip() or None,
        )
    except sqlite3.IntegrityError:
        return JSONResponse(
            {"ok": False,
             "error": f"Roll number {body.roll_no!r} is already used in {grade}."},
            status_code=409,
        )
    return {"ok": True, "student": db.get_student(student_id)}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int):
    if not db.delete_student(student_id):
        return JSONResponse({"ok": False, "error": "No such student"}, status_code=404)
    return {"ok": True}


@app.get("/results", name="results")
def results_dashboard(request: Request):
    return templates.TemplateResponse(request, "results.html", {})


# ── Process management ───────────────────────────────────────────────────

# Small bounded pool just for connection tests — cv2.VideoCapture can block
# on an unreachable URL for a long time, and a plain call has no built-in
# timeout, so we run it in a worker thread and give up after a few seconds.
_camera_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _probe_camera(url: str) -> dict:
    # A plain digit (e.g. "0", "1") means a local USB/built-in webcam index —
    # these must be opened WITHOUT forcing the FFMPEG backend, which is a
    # network-stream decoder and doesn't understand device indices. Forcing
    # it here caused the Test button to falsely report local cameras as
    # unreachable even when they work fine for the actual test scripts.
    #
    # Windows webcams are also inconsistent about which capture backend
    # actually works (Media Foundation vs DirectShow vs the OS default), so
    # for a local index we try a short list of backends in turn instead of
    # giving up after the first failure.
    if url.strip().isdigit():
        idx = int(url.strip())
        backend_attempts = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        for backend in backend_attempts:
            candidate = cv2.VideoCapture(idx, backend)
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
        if cap is None:
            return {"ok": False, "error": (
                "Could not open this camera index with any backend. It may be in use by "
                "another app (Camera app, Zoom, Teams, a previous test run that crashed), "
                "blocked by Windows camera privacy settings, or simply not the right index — "
                "try 1 instead of 0, or vice versa."
            )}
    else:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return {"ok": False, "error": "Could not open stream — check the URL and that the phone/camera app is running."}
        ret, frame = cap.read()
        if not ret or frame is None:
            return {"ok": False, "error": "Stream opened but no frame was received."}
        h, w = frame.shape[:2]
        return {"ok": True, "width": w, "height": h}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        cap.release()


class CameraTestBody(BaseModel):
    url: str


@app.post("/api/test-camera")
def test_camera(b: CameraTestBody):
    url = (b.url or "").strip()
    if not url:
        return JSONResponse({"ok": False, "error": "No URL provided"}, status_code=400)

    future = _camera_test_executor.submit(_probe_camera, url)
    try:
        result = future.result(timeout=5.0)
    except concurrent.futures.TimeoutError:
        result = {"ok": False, "error": "Timed out after 5s — check the URL, and that the laptop and camera are on the same network."}
    return result


@app.post("/api/launch/{test_id}")
async def launch(test_id: str, request: Request):
    if test_id == "wingspan" and WingspanEngine is not None:
        return await _launch_wingspan(request)

    if test_id == "vertical_jump" and VerticalJumpEngine is not None:
        return await _launch_vertical_jump(request)

    if test_id == "bmi" and BmiEngine is not None:
        return await _launch_bmi(request)

    if test_id == "curl_up" and CurlUpEngine is not None:
        return await _launch_curl_up(request)

    if test_id == "pull_up" and PullUpEngine is not None:
        return await _launch_pull_up(request)

    if test_id == "full_pull_up" and FullPullUpEngine is not None:
        return await _launch_full_pull_up(request)

    if test_id == "wall_toss" and WallTossEngine is not None:
        return await _launch_wall_toss(request)

    if test_id == "sit_reach" and SitReachEngine is not None:
        return await _launch_sit_reach(request)

    if test_id == "trunk_lift" and TrunkLiftEngine is not None:
        return await _launch_trunk_lift(request)

    if test_id == "reverse_curl" and ReverseCurlEngine is not None:
        return await _launch_reverse_curl(request)

    if test_id == "shoulder_stretch" and ShoulderStretchEngine is not None:
        return await _launch_shoulder_stretch(request)

    if test_id == "shuttle_run" and ShuttleRunEngine is not None:
        return await _launch_shuttle_run(request)

    if test_id == "dash_50m" and Dash50mEngine is not None:
        return await _launch_dash_50m(request)

    if test_id == "run_600m" and Run600mEngine is not None:
        return await _launch_run_600m(request)

    cfg = TESTS.get(test_id)
    if cfg is None:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)

    # This test is supposed to run as an in-browser live feed (like wingspan) —
    # if we get this far it's because its engine class is None, i.e. its engine
    # module failed to import. Previously this silently fell through to
    # spawning the *original* desktop-window script instead, which is exactly
    # the "camera opens in another window, and Laravel never gets synced"
    # bug reports we've gotten — so refuse instead of silently degrading.
    if cfg.get("live_feed"):
        is_dual = test_id in ("shuttle_run", "dash_50m", "run_600m")
        detail = DUAL_FEED_IMPORT_ERROR if is_dual else WEB_ENGINES_IMPORT_ERROR
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    f"Live camera engine for '{test_id}' is unavailable "
                    f"(module failed to import: {detail or 'unknown error — check server console'}). "
                    "Refusing to fall back to a desktop OpenCV window — fix the "
                    "dependency shown above and restart the server."
                ),
            },
            status_code=503,
        )

    script = TESTS_DIR / cfg["script"]

    if not script.exists():
        return JSONResponse({"ok": False, "error": f"Script not found: {script.name}"}, status_code=500)

    cmd  = [sys.executable, str(script)]
    body = await _json_body(request)

    for param in cfg.get("params", []):
        value = body.get(param["name"])
        if value is None or str(value).strip() == "":
            return JSONResponse(
                {"ok": False, "error": f"Missing required parameter: {param['label']}"},
                status_code=400,
            )
        cmd += [param["arg"], str(value)]

    # Pass session_id to the test script so it can auto-save
    session_id = body.get("session_id")
    if session_id is not None:
        cmd += ["--session-id", str(session_id)]

    try:
        # These subprocesses open real desktop OpenCV windows the coach types
        # into — don't let them inherit the server's forced-offscreen env
        # (see top of file), or the window can't render/receive input and
        # ends up silently eating keyboard focus.
        child_env = os.environ.copy()
        child_env.pop("QT_QPA_PLATFORM", None)
        proc = subprocess.Popen(cmd, cwd=str(TESTS_DIR), stdin=subprocess.DEVNULL, env=child_env)
        _running[test_id] = proc
        _pending.pop(test_id, None)  # clear any stale unsaved result from a previous run
        return {"ok": True, "pid": proc.pid}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _launch_wingspan(request: Request):
    """Starts the wingspan camera/processing loop in-process and streams it
    to the browser instead of opening a desktop OpenCV window."""
    global _wingspan_engine

    body = await _json_body(request)
    session_id = body.get("session_id")

    # Real distance from the camera, entered on screen3. Optional — blank means
    # "measuring at the calibration distance" (no correction). Entering the
    # venue's actual distance rescales the reading so one calibration stays
    # correct where the exact calibration distance won't fit.
    distance_cm = body.get("distance_cm")
    if distance_cm is not None and str(distance_cm).strip() != "":
        try:
            distance_cm = float(distance_cm)
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "Distance from camera must be a number"}, status_code=400)
        if not (30 < distance_cm < 1000):
            return JSONResponse({"ok": False, "error": "Distance from camera must be between 30 and 1000 cm"}, status_code=400)
    else:
        distance_cm = None

    def _stage(cm: float):
        _pending["wingspan"] = {"session_id": session_id, "wingspan_cm": cm}

    try:
        engine = WingspanEngine(session_id=session_id, on_stage=_stage, distance_cm=distance_cm)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _wingspan_engine = engine
    _pending.pop("wingspan", None)  # clear any stale unsaved result from a previous run
    return {"ok": True, "pid": -1, "mode": "live_feed"}

async def _launch_bmi(request: Request):
    global _bmi_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    # Real distance from the camera, entered on screen3. Optional — blank falls
    # back to the standard 215 cm. Entering the true distance decouples the
    # reading from a fixed standing spot: one calibration then works at any
    # distance (the calibration factor only has to absorb the per-camera lens
    # constant, which does not change with distance).
    distance_cm = body.get("distance_cm")
    if distance_cm is not None and str(distance_cm).strip() != "":
        try:
            distance_cm = float(distance_cm)
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "Distance from camera must be a number"}, status_code=400)
        if not (50 < distance_cm < 1000):
            return JSONResponse({"ok": False, "error": "Distance from camera must be between 50 and 1000 cm"}, status_code=400)
    else:
        distance_cm = None  # engine uses its default (215)

    def _stage(height_cm, calibration_used):
        _pending["bmi"] = {
            "session_id": session_id, "height_cm": height_cm,
            "weight_kg": None, "calibration_used": calibration_used,
        }

    try:
        engine = BmiEngine(session_id=session_id, on_stage=_stage, distance_cm=distance_cm)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _bmi_engine = engine
    _pending.pop("bmi", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_curl_up(request: Request):
    global _curl_up_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    def _stage(reps):
        _pending["curl_up"] = {"session_id": session_id, "reps": reps, "duration_sec": 30}

    try:
        engine = CurlUpEngine(session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _curl_up_engine = engine
    _pending.pop("curl_up", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_pull_up(request: Request):
    global _pull_up_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    def _stage(reps, duration_sec):
        _pending["pull_up"] = {
            "session_id": session_id, "reps": reps, "duration_sec": duration_sec,
        }

    try:
        engine = PullUpEngine(session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _pull_up_engine = engine
    _pending.pop("pull_up", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_full_pull_up(request: Request):
    global _full_pull_up_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    def _stage(reps, duration_sec):
        _pending["full_pull_up"] = {
            "session_id": session_id, "reps": reps, "duration_sec": duration_sec,
        }

    try:
        engine = FullPullUpEngine(session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _full_pull_up_engine = engine
    _pending.pop("full_pull_up", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_wall_toss(request: Request):
    global _wall_toss_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    def _stage(catches, ball_colour):
        _pending["wall_toss"] = {
            "session_id": session_id, "catches": catches,
            "duration_sec": 30, "ball_colour": ball_colour,
        }

    try:
        engine = WallTossEngine(session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _wall_toss_engine = engine
    _pending.pop("wall_toss", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_sit_reach(request: Request):
    global _sit_reach_engine
    body = await _json_body(request)
    session_id = body.get("session_id")
    leg_cm = body.get("leg_cm")
    if leg_cm is None or str(leg_cm).strip() == "":
        return JSONResponse({"ok": False, "error": "Missing required parameter: Hip-to-ankle length (cm)"}, status_code=400)
    try:
        leg_cm = float(leg_cm)
    except ValueError:
        return JSONResponse({"ok": False, "error": "Hip-to-ankle length must be a number"}, status_code=400)

    def _stage(reach_cm, leg_cm_val, rating):
        _pending["sit_reach"] = {
            "session_id": session_id, "reach_cm": reach_cm,
            "leg_cm": leg_cm_val, "rating": rating,
        }

    try:
        engine = SitReachEngine(session_id=session_id, on_stage=_stage, leg_cm=leg_cm)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _sit_reach_engine = engine
    _pending.pop("sit_reach", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}

async def _launch_trunk_lift(request: Request):
    global _trunk_lift_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    # Optional: the protocol's stick is 30 cm, and that default is what most
    # coaches will use. Any known length works as long as it is entered here.
    stick_cm = body.get("stick_cm")
    if stick_cm is None or str(stick_cm).strip() == "":
        stick_cm = 30.0
    else:
        try:
            stick_cm = float(stick_cm)
        except ValueError:
            return JSONResponse({"ok": False, "error": "Stick length must be a number"}, status_code=400)
        if not 5.0 <= stick_cm <= 300.0:
            return JSONResponse({"ok": False, "error": "Stick length must be between 5 and 300 cm"}, status_code=400)

    def _stage(lift_cm, trial1_cm, trial2_cm, stick_cm_val, rating):
        _pending["trunk_lift"] = {
            "session_id": session_id, "lift_cm": lift_cm,
            "trial1_cm": trial1_cm, "trial2_cm": trial2_cm,
            "stick_cm": stick_cm_val, "rating": rating,
        }

    try:
        engine = TrunkLiftEngine(session_id=session_id, on_stage=_stage, stick_cm=stick_cm)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _trunk_lift_engine = engine
    _pending.pop("trunk_lift", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_reverse_curl(request: Request):
    global _reverse_curl_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    # The manual says "preferred arm", which the participant picks — so the
    # default is to work it out from who is actually curling. A coach can
    # pin it when the auto choice keeps landing on the wrong side.
    arm = (body.get("arm") or "auto")
    arm = str(arm).strip().lower() or "auto"
    if arm not in ("auto", "left", "right"):
        return JSONResponse({"ok": False, "error": "Arm must be auto, left or right"},
                            status_code=400)

    def _stage(passed, arm_val, peak_angle_deg, hold_s, rating):
        _pending["reverse_curl"] = {
            "session_id": session_id, "passed": passed, "arm": arm_val,
            "peak_angle_deg": peak_angle_deg, "hold_s": hold_s, "rating": rating,
        }

    try:
        engine = ReverseCurlEngine(session_id=session_id, on_stage=_stage, arm=arm)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _reverse_curl_engine = engine
    _pending.pop("reverse_curl", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_shoulder_stretch(request: Request):
    global _shoulder_stretch_engine
    body = await _json_body(request)
    session_id = body.get("session_id")

    # Nothing to configure: BOTH sides are scored inside one trial and the
    # arm over the shoulder is named per frame from the hands, so there is no
    # side to choose or pin.
    def _stage(passed, left_passed, right_passed, best_gap_ratio,
               best_gap_left, best_gap_right, rating):
        _pending["shoulder_stretch"] = {
            "session_id": session_id, "passed": passed,
            "left_passed": left_passed, "right_passed": right_passed,
            "best_gap_ratio": best_gap_ratio,
            "best_gap_left": best_gap_left, "best_gap_right": best_gap_right,
            "rating": rating,
        }

    try:
        engine = ShoulderStretchEngine(session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _shoulder_stretch_engine = engine
    _pending.pop("shoulder_stretch", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


# helper function for vertical jump.
async def _launch_vertical_jump(request: Request):
    """Runs the vertical jump camera loop in-process and streams it to the
    browser instead of opening a desktop OpenCV window."""
    global _vertical_jump_engine

    body = await _json_body(request)
    session_id = body.get("session_id")

    def _stage(jump_cm, average_jump_cm, standing_reach_cm, attempts):
        _pending["vertical_jump"] = {
            "session_id":        session_id,
            "jump_cm":           jump_cm,
            "average_jump_cm":   average_jump_cm,
            "standing_reach_cm": standing_reach_cm,
            "attempts":          attempts,
        }

    try:
        engine = VerticalJumpEngine(session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _vertical_jump_engine = engine
    _pending.pop("vertical_jump", None)  # clear stale unsaved result from a previous run
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_shuttle_run(request: Request):
    global _shuttle_engine
    if _shuttle_engine is not None and _shuttle_engine.is_running():
        _shuttle_engine.stop()

    body = await _json_body(request)
    session_id = body.get("session_id")
    cam_a = body.get("cam_a_url")
    cam_b = body.get("cam_b_url")
    touches = body.get("touches") or 4
    if not cam_a or not cam_b:
        return JSONResponse({"ok": False, "error": "cam_a_url and cam_b_url are required"}, status_code=400)

    def _stage(payload: dict):
        _pending["shuttle_run"] = payload

    try:
        engine = ShuttleRunEngine(cam_a, cam_b, touches=touches, session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _shuttle_engine = engine
    _pending.pop("shuttle_run", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_dash_50m(request: Request):
    global _dash50m_engine
    if _dash50m_engine is not None and _dash50m_engine.is_running():
        _dash50m_engine.stop()

    body = await _json_body(request)
    session_id = body.get("session_id")
    cam_a = body.get("finish_cam_url")
    cam_b = body.get("overhead_cam_url")
    lanes = body.get("lanes") or 4
    lane_width = body.get("lane_width") or 1.22
    if not cam_a or not cam_b:
        return JSONResponse({"ok": False, "error": "finish_cam_url and overhead_cam_url are required"}, status_code=400)

    def _stage(payload: dict):
        _pending["dash_50m"] = payload

    try:
        engine = Dash50mEngine(cam_a, cam_b, lanes=lanes, lane_width=lane_width,
                            session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _dash50m_engine = engine
    _pending.pop("dash_50m", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


async def _launch_run_600m(request: Request):
    global _run600m_engine
    if _run600m_engine is not None and _run600m_engine.is_running():
        _run600m_engine.stop()

    body = await _json_body(request)
    session_id = body.get("session_id")
    cam_a = body.get("camera_url") or "0"
    lane_colors = body.get("lane_colors") or []
    if isinstance(lane_colors, str):
        lane_colors = [c.strip() for c in lane_colors.split(",") if c.strip()]
    camera_distance_m = body.get("camera_distance_m") or 0.0
    if not lane_colors:
        return JSONResponse({"ok": False, "error": "lane_colors is required (comma-separated, e.g. Red,Blue,Green)"}, status_code=400)

    def _stage(payload: dict):
        _pending["run_600m"] = payload

    try:
        engine = Run600mEngine(cam_a, lane_colors=lane_colors, camera_distance_m=camera_distance_m,
                            session_id=session_id, on_stage=_stage)
        engine.start()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    _run600m_engine = engine
    _pending.pop("run_600m", None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


@app.get("/api/engine-status")
def engine_status():
    """Quick browser-visible health check — hit http://localhost:5000/api/engine-status
    to see whether the live-feed (browser, no-desktop-window) engines loaded
    correctly, without having to dig through the server console."""
    return {
        "web_engines_ok": WEB_ENGINES_IMPORT_ERROR is None,
        "web_engines_error": WEB_ENGINES_IMPORT_ERROR,
        "web_engines_covers": ["wingspan", "bmi", "curl_up", "pull_up", "full_pull_up",
                               "wall_toss",
                               "sit_reach", "trunk_lift", "reverse_curl",
                               "shoulder_stretch",
                               "vertical_jump"],
        "dual_feed_engines_ok": DUAL_FEED_IMPORT_ERROR is None,
        "dual_feed_engines_error": DUAL_FEED_IMPORT_ERROR,
        "dual_feed_engines_covers": ["shuttle_run", "dash_50m", "run_600m"],
    }


@app.get("/api/status/{test_id}")
def status(test_id: str):
    if test_id not in TESTS:
        return JSONResponse({"ok": False, "error": "Unknown test"}, status_code=404)
    return {"ok": True, "running": _is_running(test_id)}


@app.get("/api/status")
def status_all():
    return {tid: _is_running(tid) for tid in TESTS}


@app.post("/api/stop/{test_id}")
def stop(test_id: str):
    if test_id == "wingspan" and _wingspan_engine is not None:
        # Auto-stage the best available reading before stopping,
        # so screen3 always has something to review even if SAVE READING was never tapped.
        if not _pending.get("wingspan"):
            state = _wingspan_engine.get_state()
            session_id = _wingspan_engine.session_id
            best_cm = None
            if state.get("snap_phase") == "DONE" and state.get("snap_final_cm"):
                best_cm = state["snap_final_cm"]
            elif state.get("result_cm") is not None:
                best_cm = state["result_cm"]
            if best_cm:
                _pending["wingspan"] = {"session_id": session_id, "wingspan_cm": best_cm}
        _wingspan_engine.stop()
        return {"ok": True}
    if test_id == "vertical_jump" and _vertical_jump_engine is not None:
        _vertical_jump_engine.stop()  # engine auto-stages best-available result internally
        return {"ok": True}
    if test_id == "bmi" and _bmi_engine is not None:
        _bmi_engine.stop()
        return {"ok": True}
    if test_id == "curl_up" and _curl_up_engine is not None:
        if not _pending.get("curl_up") and _curl_up_engine.count > 0:
            _pending["curl_up"] = {
                "session_id": _curl_up_engine.session_id,
                "reps": _curl_up_engine.count, "duration_sec": 30,
            }
        _curl_up_engine.stop()
        return {"ok": True}
    if test_id == "pull_up" and _pull_up_engine is not None:
        # The test is untimed, so STOP is how it always ends — the engine's
        # _stage_on_stop() is what stages the rep count for screen3.
        _pull_up_engine.stop()
        return {"ok": True}
    if test_id == "full_pull_up" and _full_pull_up_engine is not None:
        # Untimed, like the modified pull-up: _stage_on_stop() stages the count
        # if the coach stops before the participant drops off the bar.
        _full_pull_up_engine.stop()
        return {"ok": True}
    if test_id == "wall_toss" and _wall_toss_engine is not None:
        if not _pending.get("wall_toss") and _wall_toss_engine.counter.toss_count > 0:
            _pending["wall_toss"] = {
                "session_id": _wall_toss_engine.session_id,
                "catches": _wall_toss_engine.counter.toss_count, "duration_sec": 30,
                "ball_colour": _wall_toss_engine.counter._preset_name,
            }
        _wall_toss_engine.stop()
        return {"ok": True}
    if test_id == "sit_reach" and _sit_reach_engine is not None:
        _sit_reach_engine.stop()
        return {"ok": True}
    if test_id == "trunk_lift" and _trunk_lift_engine is not None:
        # The engine stages the best trial captured so far from its own
        # stop() hook, so a half-finished test still reaches screen3.
        _trunk_lift_engine.stop()
        return {"ok": True}
    if test_id == "reverse_curl" and _reverse_curl_engine is not None:
        # Stops stage whatever the trial reached, so an unresolved one still
        # reaches screen3 (as not-passed) for the coach to confirm.
        _reverse_curl_engine.stop()
        return {"ok": True}
    if test_id == "shoulder_stretch" and _shoulder_stretch_engine is not None:
        # The engine stages from its own stop() hook, so an unresolved trial
        # still reaches screen3 as not-passed, carrying the closest gap the
        # fingers actually reached.
        _shoulder_stretch_engine.stop()
        return {"ok": True}
    if test_id == "shuttle_run" and _shuttle_engine is not None:
        _shuttle_engine.maybe_finish()  # stage whatever's finished so far
        _shuttle_engine.stop()
        return {"ok": True}
    if test_id == "dash_50m" and _dash50m_engine is not None:
        _dash50m_engine.maybe_finish()
        _dash50m_engine.stop()
        return {"ok": True}
    if test_id == "run_600m" and _run600m_engine is not None:
        _run600m_engine.maybe_finish()
        _run600m_engine.stop()
        return {"ok": True}

    proc = _running.get(test_id)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        del _running[test_id]
    return {"ok": True}


# ── Wingspan live-feed control (browser buttons replace keyboard) ─────────
#
#  GET  /api/wingspan/feed   — MJPEG stream of the live annotated camera feed
#  GET  /api/wingspan/state  — polled by screen4 to drive button/label state
#  POST /api/wingspan/action — {"action": "calibrate"|"measure"|"snapshot"|
#                                "save"|"cancel"|"enter_cm", "value": <float>}

class WingspanAction(BaseModel):
    action: str
    value: Optional[float] = None


@app.get("/api/wingspan/feed")
def wingspan_feed():
    if _wingspan_engine is None or not _wingspan_engine.is_running():
        return JSONResponse({"ok": False, "error": "Wingspan test isn't running"}, status_code=409)

    def gen():
        boundary = b"--frame"
        _FRAME_INTERVAL = 0.05   # 20 fps cap — browser can't usefully render faster
        _last_sent = None        # deduplicate: skip if engine hasn't produced a new frame
        _next_send = time.monotonic()
        while _wingspan_engine is not None and _wingspan_engine.is_running():
            now = time.monotonic()
            if now < _next_send:
                time.sleep(max(0, _next_send - now))
                continue
            jpg = _wingspan_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.005)
                continue
            _last_sent = jpg
            _next_send = now + _FRAME_INTERVAL
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/wingspan/state")
def wingspan_state():
    if _wingspan_engine is None:
        return {"ok": True, "running": False}
    state = _wingspan_engine.get_state()
    return {"ok": True, "running": _wingspan_engine.is_running(), **state}


@app.post("/api/wingspan/action")
def wingspan_action(b: WingspanAction):
    if _wingspan_engine is None or not _wingspan_engine.is_running():
        return JSONResponse({"ok": False, "error": "Wingspan test isn't running"}, status_code=409)

    act = b.action
    if act == "calibrate":
        return _wingspan_engine.action_calibrate()
    if act == "measure":
        return _wingspan_engine.action_measure()
    if act == "cancel":
        return _wingspan_engine.action_cancel()
    if act == "enter_cm":
        return _wingspan_engine.action_enter_cm(b.value)
    if act == "snapshot":
        return _wingspan_engine.action_snapshot()
    if act == "save":
        return _wingspan_engine.action_save()
    return JSONResponse({"ok": False, "error": f"Unknown action '{act}'"}, status_code=400)


@app.post("/api/wingspan/frame")
async def wingspan_frame(request: Request):
    """Receive one JPEG frame captured by the browser camera.

    The body is the raw JPEG bytes (Content-Type: image/jpeg). We hand it to
    the engine, which decodes + processes it and republishes the annotated
    result on /api/wingspan/feed. This is what replaced the server opening its
    own webcam via cv2.VideoCapture(0)."""
    if _wingspan_engine is None or not _wingspan_engine.is_running():
        return JSONResponse({"ok": False, "error": "Wingspan test isn't running"}, status_code=409)
    data = await request.body()
    facing = request.query_params.get("facing")
    ok = _wingspan_engine.push_frame(data, facing=facing)
    if not ok:
        return JSONResponse({"ok": False, "error": "Could not decode frame"}, status_code=400)
    return {"ok": True}


# ── Browser-camera frame intake for the other single-camera live tests ────
# Same model as /api/wingspan/frame: the browser captures the camera and
# POSTs JPEG frames here; the engine processes them and re-streams the
# annotated result on /api/<test>/feed. This is what replaced each engine
# opening the server's own webcam (cv2.VideoCapture(0)).
async def _receive_frame(engine, request: Request):
    if engine is None or not engine.is_running():
        return JSONResponse({"ok": False, "error": "Test isn't running"}, status_code=409)
    data = await request.body()
    # The browser appends ?facing=user|environment so the engine knows which
    # lens the frame came from. The back camera must NOT be selfie-mirrored;
    # every browser-camera engine exposes set_facing() to honour that (dual-cam
    # engines don't, so we skip them).
    facing = request.query_params.get("facing")
    if facing is not None and hasattr(engine, "set_facing"):
        engine.set_facing(facing)
    ok = engine.push_frame(data)
    if not ok:
        return JSONResponse({"ok": False, "error": "Could not decode frame"}, status_code=400)
    return {"ok": True}


@app.post("/api/bmi/frame")
async def bmi_frame(request: Request):
    return await _receive_frame(_bmi_engine, request)


@app.post("/api/curl_up/frame")
async def curl_up_frame(request: Request):
    return await _receive_frame(_curl_up_engine, request)


@app.post("/api/pull_up/frame")
async def pull_up_frame(request: Request):
    return await _receive_frame(_pull_up_engine, request)


@app.post("/api/full_pull_up/frame")
async def full_pull_up_frame(request: Request):
    return await _receive_frame(_full_pull_up_engine, request)


@app.post("/api/wall_toss/frame")
async def wall_toss_frame(request: Request):
    return await _receive_frame(_wall_toss_engine, request)


@app.post("/api/trunk_lift/frame")
async def trunk_lift_frame(request: Request):
    return await _receive_frame(_trunk_lift_engine, request)


@app.post("/api/reverse_curl/frame")
async def reverse_curl_frame(request: Request):
    return await _receive_frame(_reverse_curl_engine, request)


@app.post("/api/shoulder_stretch/frame")
async def shoulder_stretch_frame(request: Request):
    return await _receive_frame(_shoulder_stretch_engine, request)


@app.post("/api/sit_reach/frame")
async def sit_reach_frame(request: Request):
    return await _receive_frame(_sit_reach_engine, request)


@app.post("/api/vertical_jump/frame")
async def vertical_jump_frame(request: Request):
    return await _receive_frame(_vertical_jump_engine, request)


@app.post("/api/run_600m/frame")
async def run_600m_frame(request: Request):
    # 600m is single-camera, so it takes browser frames like the Group A
    # tests. shuttle_run/dash_50m stay on their two-IP-camera model.
    return await _receive_frame(_run600m_engine, request)



# ── Vertical jump live-feed control (browser buttons replace keyboard) ────
class VerticalJumpAction(BaseModel):
    action: str
    value: Optional[float] = None


@app.get("/api/vertical_jump/feed")
def vertical_jump_feed():
    if _vertical_jump_engine is None or not _vertical_jump_engine.is_running():
        return JSONResponse({"ok": False, "error": "Vertical jump test isn't running"}, status_code=409)

    def gen():
        boundary = b"--frame"
        _FRAME_INTERVAL = 0.05
        _last_sent = None
        _next_send = time.monotonic()
        while _vertical_jump_engine is not None and _vertical_jump_engine.is_running():
            now = time.monotonic()
            if now < _next_send:
                time.sleep(max(0, _next_send - now))
                continue
            jpg = _vertical_jump_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.005)
                continue
            _last_sent = jpg
            _next_send = now + _FRAME_INTERVAL
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/vertical_jump/state")
def vertical_jump_state():
    if _vertical_jump_engine is None:
        return {"ok": True, "running": False}
    state = _vertical_jump_engine.get_state()
    return {"ok": True, **state}


@app.post("/api/vertical_jump/action")
def vertical_jump_action(b: VerticalJumpAction):
    if _vertical_jump_engine is None or not _vertical_jump_engine.is_running():
        return JSONResponse({"ok": False, "error": "Vertical jump test isn't running"}, status_code=409)

    act = b.action
    if act == "capture_baseline":
        return _vertical_jump_engine.action_capture_baseline()
    if act == "calibrate":
        return _vertical_jump_engine.action_calibrate(b.value)
    if act == "start_attempt":
        return _vertical_jump_engine.action_start_attempt()
    if act == "cancel_attempt":
        return _vertical_jump_engine.action_cancel_attempt()
    if act == "reset":
        return _vertical_jump_engine.action_reset()
    if act == "save":
        return _vertical_jump_engine.action_save()
    return JSONResponse({"ok": False, "error": f"Unknown action '{act}'"}, status_code=400)


# ── Shared MJPEG generator for dual-feed engines (two cams -> two routes) ──
def _dual_feed_gen(get_engine, jpeg_attr: str):
    def gen():
        boundary = b"--frame"
        _FRAME_INTERVAL = 0.05
        _last_sent = None
        _next_send = time.monotonic()
        while True:
            engine = get_engine()
            if engine is None or not engine.is_running():
                break
            now = time.monotonic()
            if now < _next_send:
                time.sleep(max(0, _next_send - now))
                continue
            jpg = getattr(engine, jpeg_attr)()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.005)
                continue
            _last_sent = jpg
            _next_send = now + _FRAME_INTERVAL
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return gen


class DualFeedAction(BaseModel):
    action: str
    value: Optional[dict] = None


# ── Shuttle run live-feed control ──────────────────────────────────────
@app.get("/api/shuttle_run/feed_a")
def shuttle_run_feed_a():
    if _shuttle_engine is None or not _shuttle_engine.is_running():
        return JSONResponse({"ok": False, "error": "Shuttle run isn't running"}, status_code=409)
    return StreamingResponse(_dual_feed_gen(lambda: _shuttle_engine, "get_jpeg_a")(),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/shuttle_run/feed_b")
def shuttle_run_feed_b():
    if _shuttle_engine is None or not _shuttle_engine.is_running():
        return JSONResponse({"ok": False, "error": "Shuttle run isn't running"}, status_code=409)
    return StreamingResponse(_dual_feed_gen(lambda: _shuttle_engine, "get_jpeg_b")(),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/shuttle_run/state")
def shuttle_run_state():
    if _shuttle_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_shuttle_engine.get_state()}


@app.post("/api/shuttle_run/action")
def shuttle_run_action(b: DualFeedAction):
    if _shuttle_engine is None or not _shuttle_engine.is_running():
        return JSONResponse({"ok": False, "error": "Shuttle run isn't running"}, status_code=409)
    return _shuttle_engine.action(b.action, b.value)


# ── Dash 50m live-feed control ──────────────────────────────────────────
@app.get("/api/dash_50m/feed_a")
def dash_50m_feed_a():
    if _dash50m_engine is None or not _dash50m_engine.is_running():
        return JSONResponse({"ok": False, "error": "Dash 50m isn't running"}, status_code=409)
    return StreamingResponse(_dual_feed_gen(lambda: _dash50m_engine, "get_jpeg_a")(),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/dash_50m/feed_b")
def dash_50m_feed_b():
    if _dash50m_engine is None or not _dash50m_engine.is_running():
        return JSONResponse({"ok": False, "error": "Dash 50m isn't running"}, status_code=409)
    return StreamingResponse(_dual_feed_gen(lambda: _dash50m_engine, "get_jpeg_b")(),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/dash_50m/state")
def dash_50m_state():
    if _dash50m_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_dash50m_engine.get_state()}


@app.post("/api/dash_50m/action")
def dash_50m_action(b: DualFeedAction):
    if _dash50m_engine is None or not _dash50m_engine.is_running():
        return JSONResponse({"ok": False, "error": "Dash 50m isn't running"}, status_code=409)
    return _dash50m_engine.action(b.action, b.value)


# ── 600m run live-feed control (single camera; feed_b mirrors feed_a) ───
@app.get("/api/run_600m/feed_a")
def run_600m_feed_a():
    if _run600m_engine is None or not _run600m_engine.is_running():
        return JSONResponse({"ok": False, "error": "600m run isn't running"}, status_code=409)
    return StreamingResponse(_dual_feed_gen(lambda: _run600m_engine, "get_jpeg_a")(),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/run_600m/state")
def run_600m_state():
    if _run600m_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_run600m_engine.get_state()}


@app.post("/api/run_600m/action")
def run_600m_action(b: DualFeedAction):
    if _run600m_engine is None or not _run600m_engine.is_running():
        return JSONResponse({"ok": False, "error": "600m run isn't running"}, status_code=409)
    return _run600m_engine.action(b.action, b.value)


# ── BMI live-feed control ──────────────────────────────────────────────
class BmiAction(BaseModel):
    action: str
    value: Optional[float] = None

@app.get("/api/bmi/feed")
def bmi_feed():
    if _bmi_engine is None or not _bmi_engine.is_running():
        return JSONResponse({"ok": False, "error": "BMI test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _bmi_engine is not None and _bmi_engine.is_running():
            jpg = _bmi_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/bmi/state")
def bmi_state():
    if _bmi_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_bmi_engine.get_state()}

@app.post("/api/bmi/action")
def bmi_action(b: BmiAction):
    if _bmi_engine is None or not _bmi_engine.is_running():
        return JSONResponse({"ok": False, "error": "BMI test isn't running"}, status_code=409)
    if b.action == "scan":
        return _bmi_engine.action_scan()
    if b.action == "calibrate":
        return _bmi_engine.action_calibrate(b.value)
    if b.action == "reset":
        return _bmi_engine.action_reset()
    return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)


# ── Curl-up live-feed control ──────────────────────────────────────────
class CurlUpAction(BaseModel):
    action: str

@app.get("/api/curl_up/feed")
def curl_up_feed():
    if _curl_up_engine is None or not _curl_up_engine.is_running():
        return JSONResponse({"ok": False, "error": "Curl-up test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _curl_up_engine is not None and _curl_up_engine.is_running():
            jpg = _curl_up_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/curl_up/state")
def curl_up_state():
    if _curl_up_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_curl_up_engine.get_state()}

@app.post("/api/curl_up/action")
def curl_up_action(b: CurlUpAction):
    if _curl_up_engine is None or not _curl_up_engine.is_running():
        return JSONResponse({"ok": False, "error": "Curl-up test isn't running"}, status_code=409)
    if b.action == "start":
        return _curl_up_engine.action_start()
    if b.action == "reset":
        # Also drop any staged result from a prior run so screen3 doesn't save a
        # stale count after the coach resets and re-attempts (see action_reset).
        _pending.pop("curl_up", None)
        return _curl_up_engine.action_reset()
    return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)


# ── Modified pull-up live-feed control ─────────────────────────────────
class PullUpAction(BaseModel):
    action: str

@app.get("/api/pull_up/feed")
def pull_up_feed():
    if _pull_up_engine is None or not _pull_up_engine.is_running():
        return JSONResponse({"ok": False, "error": "Pull-up test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _pull_up_engine is not None and _pull_up_engine.is_running():
            jpg = _pull_up_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/pull_up/state")
def pull_up_state():
    if _pull_up_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_pull_up_engine.get_state()}

@app.post("/api/pull_up/action")
def pull_up_action(b: PullUpAction):
    if _pull_up_engine is None or not _pull_up_engine.is_running():
        return JSONResponse({"ok": False, "error": "Pull-up test isn't running"}, status_code=409)
    if b.action == "start":
        return _pull_up_engine.action_start()
    if b.action == "reset":
        # Also drop any staged result from a prior attempt so screen3 doesn't
        # save a stale count after the coach resets and re-attempts.
        _pending.pop("pull_up", None)
        return _pull_up_engine.action_reset()
    return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)


# ── Pull-up live-feed control ──────────────────────────────────────────
class FullPullUpAction(BaseModel):
    action: str

@app.get("/api/full_pull_up/feed")
def full_pull_up_feed():
    if _full_pull_up_engine is None or not _full_pull_up_engine.is_running():
        return JSONResponse({"ok": False, "error": "Pull-up test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _full_pull_up_engine is not None and _full_pull_up_engine.is_running():
            jpg = _full_pull_up_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/full_pull_up/state")
def full_pull_up_state():
    if _full_pull_up_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_full_pull_up_engine.get_state()}

@app.post("/api/full_pull_up/action")
def full_pull_up_action(b: FullPullUpAction):
    if _full_pull_up_engine is None or not _full_pull_up_engine.is_running():
        return JSONResponse({"ok": False, "error": "Pull-up test isn't running"}, status_code=409)
    if b.action == "start":
        return _full_pull_up_engine.action_start()
    if b.action == "reset":
        _pending.pop("full_pull_up", None)
        return _full_pull_up_engine.action_reset()
    return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)


# ── Wall-toss live-feed control ────────────────────────────────────────
class WallTossAction(BaseModel):
    action: str
    value: Optional[float] = None

@app.get("/api/wall_toss/feed")
def wall_toss_feed():
    if _wall_toss_engine is None or not _wall_toss_engine.is_running():
        return JSONResponse({"ok": False, "error": "Wall toss test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _wall_toss_engine is not None and _wall_toss_engine.is_running():
            jpg = _wall_toss_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/wall_toss/state")
def wall_toss_state():
    if _wall_toss_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_wall_toss_engine.get_state()}

@app.post("/api/wall_toss/action")
def wall_toss_action(b: WallTossAction):
    if _wall_toss_engine is None or not _wall_toss_engine.is_running():
        return JSONResponse({"ok": False, "error": "Wall toss test isn't running"}, status_code=409)
    if b.action == "start":
        return _wall_toss_engine.action_start()
    if b.action == "reset":
        return _wall_toss_engine.action_reset()
    if b.action == "set_preset":
        return _wall_toss_engine.action_set_preset(b.value)
    return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)


# ── Trunk lift live-feed control ───────────────────────────────────────
class TrunkLiftAction(BaseModel):
    action: str
    # calib_click carries {x, y}; stick_cm carries a number. Both arrive in
    # the same `value` field screen4 already sends for every engine action.
    value: Optional[Any] = None

@app.get("/api/trunk_lift/feed")
def trunk_lift_feed():
    if _trunk_lift_engine is None or not _trunk_lift_engine.is_running():
        return JSONResponse({"ok": False, "error": "Trunk lift test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _trunk_lift_engine is not None and _trunk_lift_engine.is_running():
            jpg = _trunk_lift_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/trunk_lift/state")
def trunk_lift_state():
    if _trunk_lift_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_trunk_lift_engine.get_state()}

@app.post("/api/trunk_lift/action")
def trunk_lift_action(b: TrunkLiftAction):
    if _trunk_lift_engine is None or not _trunk_lift_engine.is_running():
        return JSONResponse({"ok": False, "error": "Trunk lift test isn't running"}, status_code=409)
    # The engine validates the action name itself and reports unknown ones.
    return _trunk_lift_engine.action(b.action, b.value)

# ── Reverse curl live-feed control ─────────────────────────────────────
class ReverseCurlAction(BaseModel):
    action: str
    # `arm` carries "auto"/"left"/"right"; pass/fail/reset carry nothing. Both
    # arrive in the same `value` field screen4 already sends for every engine.
    value: Optional[Any] = None

@app.get("/api/reverse_curl/feed")
def reverse_curl_feed():
    if _reverse_curl_engine is None or not _reverse_curl_engine.is_running():
        return JSONResponse({"ok": False, "error": "Reverse curl test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _reverse_curl_engine is not None and _reverse_curl_engine.is_running():
            jpg = _reverse_curl_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/reverse_curl/state")
def reverse_curl_state():
    if _reverse_curl_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_reverse_curl_engine.get_state()}

@app.post("/api/reverse_curl/action")
def reverse_curl_action(b: ReverseCurlAction):
    if _reverse_curl_engine is None or not _reverse_curl_engine.is_running():
        return JSONResponse({"ok": False, "error": "Reverse curl test isn't running"}, status_code=409)
    # The engine validates the action name itself and reports unknown ones.
    return _reverse_curl_engine.action(b.action, b.value)

# ── Shoulder stretch live-feed control ─────────────────────────────────
class ShoulderStretchAction(BaseModel):
    action: str
    # `side` carries "auto"/"left"/"right"; end/reset carry nothing. Both arrive
    # in the same `value` field screen4 already sends for every engine. There is
    # no PASS/FAIL action — this engine decides the verdict itself.
    value: Optional[Any] = None

@app.get("/api/shoulder_stretch/feed")
def shoulder_stretch_feed():
    if _shoulder_stretch_engine is None or not _shoulder_stretch_engine.is_running():
        return JSONResponse({"ok": False, "error": "Shoulder stretch test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _shoulder_stretch_engine is not None and _shoulder_stretch_engine.is_running():
            jpg = _shoulder_stretch_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/shoulder_stretch/state")
def shoulder_stretch_state():
    if _shoulder_stretch_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_shoulder_stretch_engine.get_state()}

@app.post("/api/shoulder_stretch/action")
def shoulder_stretch_action(b: ShoulderStretchAction):
    if _shoulder_stretch_engine is None or not _shoulder_stretch_engine.is_running():
        return JSONResponse({"ok": False, "error": "Shoulder stretch test isn't running"}, status_code=409)
    # The engine validates the action name itself and reports unknown ones.
    return _shoulder_stretch_engine.action(b.action, b.value)

# ── Sit & reach live-feed control ──────────────────────────────────────
class SitReachAction(BaseModel):
    action: str

@app.get("/api/sit_reach/feed")
def sit_reach_feed():
    if _sit_reach_engine is None or not _sit_reach_engine.is_running():
        return JSONResponse({"ok": False, "error": "Sit & reach test isn't running"}, status_code=409)
    def gen():
        boundary = b"--frame"
        _last_sent = None
        while _sit_reach_engine is not None and _sit_reach_engine.is_running():
            jpg = _sit_reach_engine.get_jpeg()
            if jpg is None or jpg is _last_sent:
                time.sleep(0.02)
                continue
            _last_sent = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/sit_reach/state")
def sit_reach_state():
    if _sit_reach_engine is None:
        return {"ok": True, "running": False}
    return {"ok": True, **_sit_reach_engine.get_state()}

@app.post("/api/sit_reach/action")
def sit_reach_action(b: SitReachAction):
    if _sit_reach_engine is None or not _sit_reach_engine.is_running():
        return JSONResponse({"ok": False, "error": "Sit & reach test isn't running"}, status_code=409)
    if b.action == "mark":
        return _sit_reach_engine.action_mark()
    if b.action == "reset":
        return _sit_reach_engine.action_reset()
    return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)

# ── Database — session management ──────────────────────────────────────────

@app.post("/api/session")
def create_session(body: SessionCreate):
    """
    Create a student session before running tests.

    Body (JSON, all fields optional):
        student_name, age, gender ("M"/"F"/"Other"), school, grade, notes
    """
    sid = db.new_session(
        student_name = body.student_name,
        age          = body.age,
        gender       = body.gender,
        school       = body.school,
        grade        = body.grade,
        notes        = body.notes,
    )
    return {"ok": True, "session_id": sid}


@app.get("/api/session/{session_id}")
def get_session(session_id: int):
    row = db.get_session(session_id)
    if not row:
        return JSONResponse({"ok": False, "error": "Session not found"}, status_code=404)
    return {"ok": True, "session": row}


@app.get("/api/sessions")
def list_sessions():
    return {"ok": True, "sessions": db.list_sessions()}


# ── Pending results — AI scripts hand off here instead of auto-saving ───────
#
#  POST /api/stage/<test_id>   — AI script reports its raw result (not saved to DB yet)
#  GET  /api/pending/<test_id> — frontend polls/fetches the staged result for review
#  POST /api/pending/<test_id>/clear — frontend clears it after saving or discarding

@app.post("/api/stage/{test_id}")
async def stage_result(test_id: str, request: Request):
    if test_id not in TESTS:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    body = await _json_body(request)
    _pending[test_id] = body
    return {"ok": True}


@app.get("/api/pending/{test_id}")
def get_pending(test_id: str):
    if test_id not in TESTS:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    return {"ok": True, "pending": _pending.get(test_id)}


@app.post("/api/pending/{test_id}/clear")
def clear_pending(test_id: str):
    _pending.pop(test_id, None)
    return {"ok": True}


# ── Database — save results ────────────────────────────────────────────────
#
#  POST /api/save/<test_id>
#  Called by the test scripts or the frontend after a test completes.
#
#  All endpoints accept:
#    session_id  (required) — from POST /api/session
#    plus test-specific fields shown below.

@app.post("/api/save/bmi")
def save_bmi(b: BMISave):
    """Fields: session_id, height_cm, weight_kg (opt), calibration_used (opt bool)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_bmi(
        session_id       = b.session_id,
        height_cm        = b.height_cm,
        weight_kg        = b.weight_kg,
        calibration_used = bool(b.calibration_used),
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/wingspan")
def save_wingspan(b: WingspanSave):
    """Fields: session_id, wingspan_cm, height_cm (opt), calibration_distance_cm (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_wingspan(
        session_id              = b.session_id,
        wingspan_cm             = b.wingspan_cm,
        height_cm               = b.height_cm,
        calibration_distance_cm = b.calibration_distance_cm,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/curl_up")
def save_curl_up(b: CurlUpSave):
    """Fields: session_id, reps, duration_sec (opt, default 30), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_curl_up(
        session_id   = b.session_id,
        reps         = b.reps,
        duration_sec = b.duration_sec if b.duration_sec is not None else 30,
        rating       = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/pull_up")
def save_pull_up(b: PullUpSave):
    """Fields: session_id, reps, duration_sec (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_pull_up(
        session_id   = b.session_id,
        reps         = b.reps,
        duration_sec = b.duration_sec,
        rating       = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/full_pull_up")
def save_full_pull_up(b: FullPullUpSave):
    """Fields: session_id, reps, duration_sec (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_full_pull_up(
        session_id   = b.session_id,
        reps         = b.reps,
        duration_sec = b.duration_sec,
        rating       = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/wall_toss")
def save_wall_toss(b: WallTossSave):
    """Fields: session_id, catches, duration_sec (opt), ball_colour (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_wall_toss(
        session_id   = b.session_id,
        catches      = b.catches,
        duration_sec = b.duration_sec if b.duration_sec is not None else 30,
        ball_colour  = b.ball_colour,
        rating       = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/shuttle_run")
def save_shuttle_run(b: ShuttleRunSave):
    """Fields: session_id, time_sec, lane (opt, default 1), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_shuttle_run(
        session_id = b.session_id,
        time_sec   = b.time_sec,
        lane       = b.lane if b.lane is not None else 1,
        rating     = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/dash_50m")
def save_dash_50m(b: Dash50mSave):
    """Fields: session_id, time_sec, lane (opt), start_mode (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_dash_50m(
        session_id = b.session_id,
        time_sec   = b.time_sec,
        lane       = b.lane if b.lane is not None else 1,
        start_mode = b.start_mode or "whistle",
        rating     = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/sit_reach")
def save_sit_reach(b: SitReachSave):
    """Fields: session_id, reach_cm, leg_cm (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_sit_reach(
        session_id = b.session_id,
        reach_cm   = b.reach_cm,
        leg_cm     = b.leg_cm,
        rating     = b.rating,
    )
    return {"ok": True, "id": row_id}

@app.post("/api/save/trunk_lift")
def save_trunk_lift(b: TrunkLiftSave):
    """Fields: session_id, lift_cm, trial1_cm (opt), trial2_cm (opt), stick_cm (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_trunk_lift(
        session_id = b.session_id,
        lift_cm    = b.lift_cm,
        trial1_cm  = b.trial1_cm,
        trial2_cm  = b.trial2_cm,
        stick_cm   = b.stick_cm,
        rating     = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/reverse_curl")
def save_reverse_curl(b: ReverseCurlSave):
    """Fields: session_id, passed, arm (opt), peak_angle_deg (opt), hold_s (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    if b.passed is None:
        return JSONResponse({"ok": False, "error": "passed required (pass/fail test)"}, status_code=400)
    row_id = db.save_reverse_curl(
        session_id     = b.session_id,
        passed         = b.passed,
        arm            = b.arm,
        peak_angle_deg = b.peak_angle_deg,
        hold_s         = b.hold_s,
        rating         = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/shoulder_stretch")
def save_shoulder_stretch(b: ShoulderStretchSave):
    """Fields: session_id, passed (both sides), left_passed/right_passed (opt),
    best_gap_ratio / best_gap_left / best_gap_right (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    if b.passed is None:
        return JSONResponse({"ok": False, "error": "passed required (pass/fail test)"}, status_code=400)
    row_id = db.save_shoulder_stretch(
        session_id     = b.session_id,
        passed         = b.passed,
        left_passed    = b.left_passed,
        right_passed   = b.right_passed,
        best_gap_ratio = b.best_gap_ratio,
        best_gap_left  = b.best_gap_left,
        best_gap_right = b.best_gap_right,
        rating         = b.rating,
    )
    return {"ok": True, "id": row_id}


@app.post("/api/save/vertical_jump")
def save_vertical_jump(b: VerticalJumpSave):
    """Fields: session_id, jump_cm, average_jump_cm (opt), standing_reach_cm (opt), attempts (opt list)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_vertical_jump(
        session_id        = b.session_id,
        jump_cm           = b.jump_cm,
        average_jump_cm   = b.average_jump_cm,
        standing_reach_cm = b.standing_reach_cm,
        attempts          = b.attempts,
    )
    return {"ok": True, "id": row_id}

@app.post("/api/save/run_600m")
def save_run_600m(b: Run600mSave):
    """Fields: session_id, time_sec, lane (opt, default 1), lap_count (opt), rating (opt)"""
    if not b.session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    row_id = db.save_run_600m(
        session_id = b.session_id,
        time_sec   = b.time_sec,
        lane       = b.lane if b.lane is not None else 1,
        lap_count  = b.lap_count,
        rating     = b.rating,
    )
    return {"ok": True, "id": row_id}
# ── Database — fetch results ───────────────────────────────────────────────

@app.get("/api/results")
def results_all(student: Optional[str] = Query(default=None)):
    """Flat view across all tests. ?student=<name> to filter by student."""
    if student:
        rows = db.get_student_summary(student)
    else:
        rows = db.get_all_results()
    return {"ok": True, "results": rows}


@app.get("/api/results/{test_id}")
def results_by_test(test_id: str, session_id: Optional[int] = Query(default=None)):
    """All rows for one test table. ?session_id=<n> to filter."""
    if test_id not in db.VALID_TABLES:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    rows = db.get_all(test_id, session_id=session_id)
    return {"ok": True, "test": test_id, "results": rows}


@app.delete("/api/result/{test_id}/{row_id}")
def delete_result(test_id: str, row_id: int):
    """Delete a single test result row."""
    if test_id not in db.VALID_TABLES:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    deleted = db.delete_result(test_id, row_id)
    if not deleted:
        return JSONResponse({"ok": False, "error": "Record not found"}, status_code=404)
    return {"ok": True, "deleted": {"test": test_id, "row_id": row_id}}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: int):
    """Delete a session and all its associated test results (CASCADE)."""
    deleted = db.delete_session(session_id)
    if not deleted:
        return JSONResponse({"ok": False, "error": "Session not found"}, status_code=404)
    return {"ok": True, "deleted": {"session_id": session_id}}


@app.get("/api/export/{test_id}")
def export_csv(test_id: str):
    """Export a test table to CSV and return the file."""
    if test_id not in db.VALID_TABLES:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    rows = db.get_all(test_id)
    if not rows:
        return JSONResponse({"ok": False, "error": "No data yet"}, status_code=404)

    output = io.StringIO()
    w = csv.DictWriter(output, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{test_id}_results.csv"'},
    )


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n  GoForFit NEP FMS — starting …")
    print(f"  Tests directory : {TESTS_DIR}")
    print(f"  Database        : {BASE_DIR / 'goforfit.db'}")
    print(f" Live Site URL   : https://talentid.goforfit.in/fms/")
    print(f" API Docs        : https://talentid.goforfit.in/fms/docs\n")
    
    # Match port to your Apache reverse proxy (8080)
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)