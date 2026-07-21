"""
GoForFit NEP FMS — FastAPI Launcher  (with DB storage)
======================================================
Every test runs IN-PROCESS as a browser-controlled engine, streaming MJPEG
frames to <img> in screen4 and taking its commands from on-screen buttons.
The original desktop OpenCV scripts still live in tests/ and are still
runnable by hand, but the server no longer spawns them: it stopped being a
subprocess launchpad once the last test moved in-process.

Converted from Flask to FastAPI. Behavior, routes, and JSON response shapes
are preserved 1:1 with the original app. The Jinja templates (index.html,
screen2.html, screen3.html, screen4.html, results.html) are used unmodified —
a small `url_for` compatibility shim below lets them keep using Flask-style
calls like `url_for('screen2', category='All Tests')`.

Usage:
    pip install -r requirements.txt   (Python 3.10+ — see README)
    python app.py
    Then open  http://localhost:5001  in your browser.
    (Port 5001 because macOS AirPlay owns 5000; localhost-only by default —
    set GOFORFIT_HOST=0.0.0.0 to reach the app from a phone. Every route
    needs the access code printed at startup; see auth.py and the README.)

Interactive API docs (new, courtesy of FastAPI): http://localhost:5001/docs
"""

import os
# ── Force OpenCV into headless mode BEFORE cv2 is imported anywhere ────────
# Without this, OpenCV's Qt/GTK backend opens a blank window the moment
# VideoCapture is created, even when imshow() is never called.
os.environ.setdefault("DISPLAY", "")          # no X11 display → offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # Qt backend → offscreen
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # skip Windows Media Foundation GUI

import sys
import csv
import io
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import cv2
import jinja2
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from pydantic import BaseModel

from db import db  # unchanged database module
import auth as auth_mod

app = FastAPI(title="GoForFit NEP FMS")

# ── Access control ────────────────────────────────────────────────────────
# Resolved at import, NOT in the __main__ block below: the app is normally
# started with `uvicorn app:app`, which never runs __main__ — putting the
# gate there would have left every production launch wide open.
ACCESS_TOKEN, ACCESS_TOKEN_GENERATED = auth_mod.resolve_token()
app.add_middleware(auth_mod.TokenAuthMiddleware, token=ACCESS_TOKEN)
auth_mod.print_startup_banner(ACCESS_TOKEN, ACCESS_TOKEN_GENERATED)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
TESTS_DIR = BASE_DIR / "tests"
TEMPLATES_DIR = BASE_DIR / "templates"

# ── Browser-controlled live feeds (no desktop OpenCV window) ──────────────
# Wingspan went first — it needed several distinct in-test actions
# (calibrate / measure / snapshot / save) that don't work from a phone with
# no keyboard. Every other test has since followed, so all nine now run
# in-process and stream MJPEG frames to <img> in screen4.
sys.path.insert(0, str(TESTS_DIR))

# These hold the *actual* exception text when an engine module fails to
# import, so we can surface it to the browser instead of only ever
# printing it to a server console nobody is watching.
import traceback as _traceback

import camera as _camera_mod  # camera-source resolution, shared with the engines

WEB_ENGINES_IMPORT_ERROR: Optional[str] = None
DUAL_FEED_IMPORT_ERROR: Optional[str] = None

try:
    from web_engines import (
        BmiEngine, CurlUpEngine, WallTossEngine, SitReachEngine,
        WingspanEngine, VerticalJumpEngine,
    )
except Exception as _e:  # pragma: no cover — surfaced to the browser, see below
    BmiEngine = CurlUpEngine = WallTossEngine = SitReachEngine = None
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

# Per-test engine instances live in _ENGINES[test_id].instance — see the
# engine registry below the shared-state lock.

# ── Shared-state lock ──────────────────────────────────────────────────────
# FastAPI runs `def` endpoints in a threadpool and `async def` endpoints on
# the event loop, and engine worker threads call on_stage callbacks — so the
# engine globals above, `_pending`, and `_launching` are all mutated from
# genuinely concurrent threads. Every read-check-write of that state must
# happen under this lock.
#
# DEADLOCK RULE: engine worker threads call on_stage (which takes this lock)
# while holding the engine's own `_lock`. Therefore code holding _state_lock
# must never call engine.get_state()/get_jpeg() (they take the engine lock).
# Plain attribute reads (is_running(), session_id, .count, …) are safe.
_state_lock = threading.RLock()

# Test id currently mid-launch (guard set → engine slot assigned). Camera
# open takes seconds; this keeps a second launch out of that window.
# coach -> test_id they are mid-launch on. Per coach, not global: two coaches
# with two cameras must be able to be opening them at the same moment.
_launching: dict = {}

# (coach, test_id) -> engine. Was a single slot per test, which is what let
# one coach's STOP kill another's running test.
_RUNS: dict = {}

# camera key -> (coach, test_id). The one thing that genuinely cannot be
# shared: two engines opening the same device get a confusing failure or, on
# some platforms, one silently starves the other of frames.
_CAMERA_CLAIMS: dict = {}

# Camera keys claimed but whose engine hasn't been registered yet. Opening a
# camera takes seconds, and during that window the claim is real while the
# engine legitimately doesn't exist — pruning must not mistake that for a
# dead run and hand the device to someone else mid-open.
_LAUNCHING_CLAIMS: set = set()


# ── Engine registry ────────────────────────────────────────────────────────
# One slot per in-process camera engine. Everything the HTTP layer needs to
# serve a test — how to build it from a request body, which actions it
# accepts, which MJPEG feeds it exposes — lives here, so adding a test is a
# builder + one registry entry instead of edits scattered across the file.

class _EngineSlot:
    def __init__(self, test_id, label, cls, builder, actions, feeds):
        self.test_id = test_id
        self.label = label        # human name used in "<label> isn't running" errors
        self.cls = cls            # engine class, or None if its module failed to import
        self.builder = builder    # (body: dict) -> engine instance | JSONResponse error
        self.actions = actions    # {action_name: (engine, value) -> response}
        self.feeds = feeds        # subset of ("feed", "feed_a", "feed_b")
        self.instance = None      # current engine — read/write under _state_lock


def _coach(request: Request) -> Optional[str]:
    """Signed-in coach for this request. The middleware resolved it already;
    the getattr fallback keeps direct-call unit tests working."""
    return getattr(request.state, "coach", None) or auth_mod.coach_of(request)


def _pending_stage(coach: str, test_id: str, payload: dict) -> None:
    """on_stage sink for engine worker threads — stages a result for screen3.

    Keyed by (coach, test) rather than test alone. With one key per test, two
    coaches running the same test shared a slot: whoever finished first had
    their reading picked up by the other coach's screen3 and saved against
    the WRONG STUDENT, silently and with a plausible-looking number.
    """
    with _state_lock:
        _pending[(coach, test_id)] = payload


# ── Per-test builders: parse the launch body, wire the on_stage payload ────

def _build_wingspan(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")
    return WingspanEngine(
        session_id=session_id,
        camera_source=body.get('_camera', 0),
        on_stage=lambda cm: _pending_stage(coach, "wingspan", {"session_id": session_id, "wingspan_cm": cm}),
    )


def _build_vertical_jump(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")

    def _stage(jump_cm, average_jump_cm, standing_reach_cm, attempts):
        _pending_stage(coach, "vertical_jump", {
            "session_id":        session_id,
            "jump_cm":           jump_cm,
            "average_jump_cm":   average_jump_cm,
            "standing_reach_cm": standing_reach_cm,
            "attempts":          attempts,
        })

    return VerticalJumpEngine(session_id=session_id, on_stage=_stage,
                      camera_source=body.get('_camera', 0))


def _build_bmi(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")

    def _stage(height_cm, calibration_used):
        _pending_stage(coach, "bmi", {
            "session_id": session_id, "height_cm": height_cm,
            "weight_kg": None, "calibration_used": calibration_used,
        })

    return BmiEngine(session_id=session_id, on_stage=_stage,
                      camera_source=body.get('_camera', 0))


def _build_curl_up(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")
    return CurlUpEngine(
        session_id=session_id,
        camera_source=body.get('_camera', 0),
        on_stage=lambda reps: _pending_stage(coach, 
            "curl_up", {"session_id": session_id, "reps": reps, "duration_sec": 30}),
    )


def _build_wall_toss(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")

    def _stage(catches, ball_colour):
        _pending_stage(coach, "wall_toss", {
            "session_id": session_id, "catches": catches,
            "duration_sec": 30, "ball_colour": ball_colour,
        })

    return WallTossEngine(session_id=session_id, on_stage=_stage,
                      camera_source=body.get('_camera', 0))


def _build_sit_reach(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")
    leg_cm = body.get("leg_cm")
    if leg_cm is None or str(leg_cm).strip() == "":
        return JSONResponse({"ok": False, "error": "Missing required parameter: Hip-to-ankle length (cm)"}, status_code=400)
    try:
        leg_cm = float(leg_cm)
    except ValueError:
        return JSONResponse({"ok": False, "error": "Hip-to-ankle length must be a number"}, status_code=400)

    def _stage(reach_cm, leg_cm_val, rating):
        _pending_stage(coach, "sit_reach", {
            "session_id": session_id, "reach_cm": reach_cm,
            "leg_cm": leg_cm_val, "rating": rating,
        })

    return SitReachEngine(session_id=session_id, on_stage=_stage, leg_cm=leg_cm,
                          camera_source=body.get('_camera', 0))


def _build_shuttle_run(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")
    cam_a = body.get("cam_a_url")
    cam_b = body.get("cam_b_url")
    touches = body.get("touches") or 4
    if not cam_a or not cam_b:
        return JSONResponse({"ok": False, "error": "cam_a_url and cam_b_url are required"}, status_code=400)
    return ShuttleRunEngine(
        cam_a, cam_b, touches=touches, session_id=session_id,
        on_stage=lambda payload: _pending_stage(coach, "shuttle_run", payload),
    )


def _build_dash_50m(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")
    cam_a = body.get("finish_cam_url")
    cam_b = body.get("overhead_cam_url")
    lanes = body.get("lanes") or 4
    lane_width = body.get("lane_width") or 1.22
    if not cam_a or not cam_b:
        return JSONResponse({"ok": False, "error": "finish_cam_url and overhead_cam_url are required"}, status_code=400)
    return Dash50mEngine(
        cam_a, cam_b, lanes=lanes, lane_width=lane_width, session_id=session_id,
        on_stage=lambda payload: _pending_stage(coach, "dash_50m", payload),
    )


def _build_run_600m(body: dict):
    session_id = body.get("session_id")
    coach = body.get("_coach")
    cam_a = body.get("camera_url") or "0"
    lane_colors = body.get("lane_colors") or []
    if isinstance(lane_colors, str):
        lane_colors = [c.strip() for c in lane_colors.split(",") if c.strip()]
    camera_distance_m = body.get("camera_distance_m") or 0.0
    if not lane_colors:
        return JSONResponse({"ok": False, "error": "lane_colors is required (comma-separated, e.g. Red,Blue,Green)"}, status_code=400)
    return Run600mEngine(
        cam_a, lane_colors=lane_colors, camera_distance_m=camera_distance_m,
        session_id=session_id,
        on_stage=lambda payload: _pending_stage(coach, "run_600m", payload),
    )


# actions=None in a slot means the engine exposes a generic
# engine.action(name, value) dispatcher (the dual-feed engines do).
_ENGINES: dict[str, _EngineSlot] = {
    "wingspan": _EngineSlot(
        "wingspan", "Wingspan test", WingspanEngine, _build_wingspan,
        actions={
            "calibrate": lambda e, v: e.action_calibrate(),
            "measure":   lambda e, v: e.action_measure(),
            "cancel":    lambda e, v: e.action_cancel(),
            "enter_cm":  lambda e, v: e.action_enter_cm(v),
            "snapshot":  lambda e, v: e.action_snapshot(),
            "save":      lambda e, v: e.action_save(),
        },
        feeds=("feed",),
    ),
    "vertical_jump": _EngineSlot(
        "vertical_jump", "Vertical jump test", VerticalJumpEngine, _build_vertical_jump,
        actions={
            "capture_baseline": lambda e, v: e.action_capture_baseline(),
            "calibrate":        lambda e, v: e.action_calibrate(v),
            "start_attempt":    lambda e, v: e.action_start_attempt(),
            "reset":            lambda e, v: e.action_reset(),
            "save":             lambda e, v: e.action_save(),
        },
        feeds=("feed",),
    ),
    "bmi": _EngineSlot(
        "bmi", "BMI test", BmiEngine, _build_bmi,
        actions={
            "scan":      lambda e, v: e.action_scan(),
            "calibrate": lambda e, v: e.action_calibrate(v),
            "reset":     lambda e, v: e.action_reset(),
        },
        feeds=("feed",),
    ),
    "curl_up": _EngineSlot(
        "curl_up", "Curl-up test", CurlUpEngine, _build_curl_up,
        actions={
            "start": lambda e, v: e.action_start(),
            "reset": lambda e, v: e.action_reset(),
        },
        feeds=("feed",),
    ),
    "wall_toss": _EngineSlot(
        "wall_toss", "Wall toss test", WallTossEngine, _build_wall_toss,
        actions={
            "start":      lambda e, v: e.action_start(),
            "reset":      lambda e, v: e.action_reset(),
            "set_preset": lambda e, v: e.action_set_preset(v),
        },
        feeds=("feed",),
    ),
    "sit_reach": _EngineSlot(
        "sit_reach", "Sit & reach test", SitReachEngine, _build_sit_reach,
        actions={
            "mark":  lambda e, v: e.action_mark(),
            "reset": lambda e, v: e.action_reset(),
        },
        feeds=("feed",),
    ),
    "shuttle_run": _EngineSlot(
        "shuttle_run", "Shuttle run", ShuttleRunEngine, _build_shuttle_run,
        actions=None,  # None -> generic engine.action(name, value) dispatch
        feeds=("feed_a", "feed_b"),
    ),
    "dash_50m": _EngineSlot(
        "dash_50m", "Dash 50m", Dash50mEngine, _build_dash_50m,
        actions=None,
        feeds=("feed_a", "feed_b"),
    ),
    "run_600m": _EngineSlot(
        "run_600m", "600m run", Run600mEngine, _build_run_600m,
        actions=None,
        feeds=("feed_a",),  # single physical camera — no feed_b route
    ),
}


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
        "tip":    "Stand 192 cm (6.3 ft) from the camera for best accuracy.",
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
        "tip":    "Extend both arms fully, palms forward, fingers spread.",
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
    "wall_toss": {
        "id":          "wall_toss",
        "title":       "Wall Toss Coordination Test",
        "subtitle":    "Alternate-hand ball toss — 30 seconds",
        "icon":        "🎯",
        "category":    "Coordination",
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
                "required":    False,  # _build_shuttle_run defaults to 4
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
                "required":    False,  # _build_dash_50m defaults to 4
            },
            {
                "name":        "lane_width",
                "label":       "Lane width (metres)",
                "type":        "number",
                "min":         0.5,
                "max":         3,
                "placeholder": "e.g. 1.22",
                "arg":         "--lane-width",
                "required":    False,  # _build_dash_50m defaults to 1.22
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
    "vertical_jump": {
        "id":          "vertical_jump",
        "title":       "Vertical Jump Test",
        "subtitle":    "Standing reach + max jump height — 5 attempts",
        "icon":        "🐰",
        "category":    "Power",
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
                "required":    False,  # _build_run_600m defaults to index 0
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
                "required":    False,  # _build_run_600m defaults to 0.0
            },
        ],
        "timed":  True,
        "unit":   "s",
        "live_feed":   True,
    },
}

# ── Laravel sync field map ─────────────────────────────────────────────────
# Rendered into screen3.html as EXERCISE_SYNC_MAP (single source of truth —
# previously hand-synced in the template). exerciseType/scoreKey are the
# labels the Laravel parent window expects in the postMessage payload. Only
# "wingspan" has been confirmed against Bhawani's sample code — confirm the
# rest with the integration owner before relying on them.
LARAVEL_SYNC = {
    "bmi":           {"exerciseType": "BMI",          "scoreKey": "bmiScore"},
    "wingspan":      {"exerciseType": "Wingspan",     "scoreKey": "wingspanScore"},   # confirmed
    "curl_up":       {"exerciseType": "CurlUp",       "scoreKey": "curlUpScore"},
    "wall_toss":     {"exerciseType": "WallToss",     "scoreKey": "wallTossScore"},
    "shuttle_run":   {"exerciseType": "ShuttleRun",   "scoreKey": "shuttleRunScore"},
    "dash_50m":      {"exerciseType": "Dash50m",      "scoreKey": "dash50mScore"},
    "sit_reach":     {"exerciseType": "SitReach",     "scoreKey": "sitReachScore"},
    "vertical_jump": {"exerciseType": "VerticalJump", "scoreKey": "verticalJumpScore"},
    "run_600m":      {"exerciseType": "Run600m",      "scoreKey": "run600mScore"},
}

def _is_running(coach: str, test_id: str) -> bool:
    with _state_lock:
        return _is_running_locked(coach, test_id)


def _engine_locked(coach: str, test_id: str):
    """This coach's engine for this test, or None. Requires _state_lock."""
    return _RUNS.get((coach, test_id))


def _is_running_locked(coach: str, test_id: str) -> bool:
    """Requires _state_lock. Only touches lock-free engine attrs (see rule above)."""
    eng = _RUNS.get((coach, test_id))
    return eng is not None and eng.is_running()


def _active_test_id_locked(coach: str) -> Optional[str]:
    """The test THIS COACH is running (or mid-launch on), or None.

    Still one test at a time per coach — a coach has one pair of hands and
    one camera — but no longer one test at a time per SERVER, which is what
    made a second coach impossible. Requires _state_lock.
    """
    if coach in _launching:
        return _launching[coach]
    for candidate in TESTS:
        if _is_running_locked(coach, candidate):
            return candidate
    return None


def _prune_camera_claims_locked() -> None:
    """Drop claims whose engine is no longer running.

    A claim is taken at launch and released on stop — but an engine can also
    stop by ITSELF: its loop raises, the capture wedges and the watchdog
    reports it dead, or the coach closes the tab and never presses STOP.
    Without this the claim outlived the engine and the camera was locked out
    of the whole server until a restart, reported as "the webcam stopped
    working". Liveness is the source of truth; the claim is just a cache.
    Requires _state_lock.
    """
    for key, holder in list(_CAMERA_CLAIMS.items()):
        if key in _LAUNCHING_CLAIMS:
            continue  # mid-launch: the engine legitimately doesn't exist yet
        eng = _RUNS.get(holder)
        if eng is None or not eng.is_running():
            del _CAMERA_CLAIMS[key]
            _RUNS.pop(holder, None)


def _camera_conflict_locked(coach: str, sources) -> Optional[tuple]:
    """(other_coach, their_test) if someone else already holds one of these
    cameras, else None. Requires _state_lock."""
    _prune_camera_claims_locked()
    for src in sources:
        holder = _CAMERA_CLAIMS.get(_camera_mod.source_key(src))
        if holder is not None and holder[0] != coach:
            return holder
    return None


def _claim_cameras_locked(coach: str, test_id: str, sources) -> None:
    for src in sources:
        key = _camera_mod.source_key(src)
        _CAMERA_CLAIMS[key] = (coach, test_id)
        _LAUNCHING_CLAIMS.add(key)


def _claims_settled_locked(coach: str, test_id: str) -> None:
    """The engine is registered — its claims are now backed by a live run."""
    for key, holder in _CAMERA_CLAIMS.items():
        if holder == (coach, test_id):
            _LAUNCHING_CLAIMS.discard(key)


def _release_cameras_locked(coach: str, test_id: str) -> None:
    for key, holder in list(_CAMERA_CLAIMS.items()):
        if holder == (coach, test_id):
            del _CAMERA_CLAIMS[key]
            _LAUNCHING_CLAIMS.discard(key)

# ── Pending (staged-but-not-saved) AI results ──────────────────────────────
# AI test scripts no longer write straight to the DB. They POST their raw
# result here instead, where it sits until a human reviews it on screen3 and
# explicitly clicks "Save". Keyed by (coach, test_id) — see _pending_stage for
# why the coach half is not optional.
_pending: dict = {}


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

@app.get("/unlock", name="unlock")
def unlock_form(request: Request, next: str = "/"):
    if ACCESS_TOKEN is None:
        return RedirectResponse(auth_mod._safe_next(next), status_code=303)
    return templates.TemplateResponse(
        request, "unlock.html", {"next": auth_mod._safe_next(next), "error": None},
    )


@app.post("/unlock")
async def unlock_submit(request: Request):
    form = await request.form()
    supplied = (form.get("token") or "").strip()
    target = auth_mod._safe_next(form.get("next"))
    if ACCESS_TOKEN is None:
        return RedirectResponse(target, status_code=303)
    if not auth_mod.tokens_match(supplied, ACCESS_TOKEN):
        # 401, not a redirect: a wrong code must not look like success, and
        # the status makes a failed unlock visible in the server log.
        return templates.TemplateResponse(
            request, "unlock.html",
            {"next": target, "error": "That code isn't right — check the server console."},
            status_code=401,
        )
    response = RedirectResponse(target, status_code=303)
    auth_mod.set_token_cookie(response, ACCESS_TOKEN)
    return response


@app.get("/whoami", name="whoami")
def whoami_form(request: Request, next: str = "/"):
    return templates.TemplateResponse(
        request, "whoami.html",
        {"next": auth_mod._safe_next(next), "error": None,
         "coaches": db.list_coaches(), "camera": auth_mod.camera_of(request)},
    )


@app.post("/whoami")
async def whoami_submit(request: Request):
    form = await request.form()
    target = auth_mod._safe_next(form.get("next"))
    coach = auth_mod.clean_coach_name(form.get("coach"))
    camera = (form.get("camera") or "").strip() or auth_mod.DEFAULT_CAMERA

    if not coach:
        return templates.TemplateResponse(
            request, "whoami.html",
            {"next": target, "error": "Enter a name so your results can be told apart.",
             "coaches": db.list_coaches(), "camera": camera},
            status_code=400,
        )

    response = RedirectResponse(target, status_code=303)
    auth_mod.set_coach_cookie(response, coach)
    auth_mod.set_camera_cookie(response, camera)
    return response


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


@app.get("/screen3", name="screen3")
def screen3(request: Request, test: Optional[str] = None):
    test_id = test or "wall_toss"
    test_cfg = TESTS.get(test_id, TESTS["wall_toss"])
    return templates.TemplateResponse(
        request, "screen3.html",
        {"test": test_cfg, "sync_map": LARAVEL_SYNC},
    )


@app.get("/screen4", name="screen4")
def screen4(request: Request, test: Optional[str] = None):
    test_id = test or "wall_toss"
    test_cfg = TESTS.get(test_id, TESTS["wall_toss"])
    # Feed topology comes from the engine registry, so screen4's JS never
    # hardcodes which tests are dual-camera or what their stream URLs are.
    engine_feeds = {tid: list(slot.feeds) for tid, slot in _ENGINES.items()}
    return templates.TemplateResponse(
        request, "screen4.html",
        {"test": test_cfg, "engine_feeds": engine_feeds},
    )


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


def _launch_sources(test_id: str, body: dict) -> list:
    """Which camera(s) this launch will open. The dual-camera tests take two
    explicit URLs from the setup form; everything else drives the one camera
    the coach configured at sign-in."""
    if test_id == "shuttle_run":
        return [body.get("cam_a_url"), body.get("cam_b_url")]
    if test_id == "dash_50m":
        return [body.get("finish_cam_url"), body.get("overhead_cam_url")]
    if test_id == "run_600m":
        return [body.get("camera_url") or "0"]
    return [body.get("_camera", 0)]


@app.post("/api/launch/{test_id}")
async def launch(test_id: str, request: Request):
    body = await _json_body(request)  # read BEFORE touching shared state — no awaits after this

    # Who is launching, and on which camera. Both come from the signed-in
    # identity rather than the request body — a coach must not be able to
    # launch as someone else, deliberately or by a stale page.
    coach = _coach(request)
    body["_coach"] = coach
    body["_camera"] = auth_mod.camera_of(request)

    # Two guards, and they are different questions now.
    #
    #   1. Is THIS COACH already running something? One coach has one pair of
    #      hands; a second launch is a double-tapped START or a forgotten test.
    #   2. Is the camera they want already held by SOMEONE ELSE? This is the
    #      only genuinely unshareable resource. Two engines on one device get
    #      a confusing open failure, or one silently starves the other.
    #
    # The old code asked a third question — "is anything at all running on
    # this server?" — which is what made a second coach impossible.
    # The check and the `_launching` claim are one atomic step; the slow part
    # (opening the camera) then runs outside the lock.
    sources = _launch_sources(test_id, body)
    with _state_lock:
        active = _active_test_id_locked(coach) if test_id in TESTS else None
        if active is not None:
            detail = (
                "It is already running — press STOP / SAVE before starting it again."
                if active == test_id
                else f"'{TESTS[active]['title']}' is still running — stop it first."
            )
            return JSONResponse(
                {"ok": False, "error": detail, "running": active},
                status_code=409,
            )
        if test_id in TESTS:
            holder = _camera_conflict_locked(coach, sources)
            if holder is not None:
                other_coach, other_test = holder
                return JSONResponse(
                    {
                        "ok": False,
                        "error": (
                            f"{other_coach} is using that camera for "
                            f"'{TESTS[other_test]['title']}'. Point your camera "
                            "somewhere else on the sign-in screen, or wait for them to finish."
                        ),
                        "held_by": other_coach,
                    },
                    status_code=409,
                )
            _launching[coach] = test_id
            _claim_cameras_locked(coach, test_id, sources)
    try:
        result = _dispatch_launch(test_id, body)
        if isinstance(result, JSONResponse):  # launch failed — don't hold the camera
            with _state_lock:
                _release_cameras_locked(coach, test_id)
        return result
    except Exception:
        with _state_lock:
            _release_cameras_locked(coach, test_id)
        raise
    finally:
        with _state_lock:
            _launching.pop(coach, None)


def _launch_engine(slot: _EngineSlot, body: dict):
    """Common launch path for every in-process engine: build from the body,
    start (slow — camera open), register the instance, clear stale pending."""
    coach = body.get("_coach")
    engine = slot.builder(body)
    if isinstance(engine, JSONResponse):  # builder rejected the params
        return engine
    try:
        engine.start()
    except Exception as exc:
        _traceback.print_exc()  # str(exc) alone hides the real failure site
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    with _state_lock:
        _RUNS[(coach, slot.test_id)] = engine
        _claims_settled_locked(coach, slot.test_id)
        # Clear only THIS coach's stale result — another coach's staged
        # reading for the same test is theirs and must survive.
        _pending.pop((coach, slot.test_id), None)
    return {"ok": True, "pid": -1, "mode": "live_feed"}


def _dispatch_launch(test_id: str, body: dict):
    slot = _ENGINES.get(test_id)
    if slot is not None and slot.cls is not None:
        return _launch_engine(slot, body)

    cfg = TESTS.get(test_id)
    if cfg is None:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)

    # This test is supposed to run as an in-browser live feed (like wingspan) —
    # if we get this far it's because its engine class is None, i.e. its engine
    # module failed to import. Previously this silently fell through to
    # spawning the *original* desktop-window script instead, which is exactly
    # the "camera opens in another window, and Laravel never gets synced"
    # bug reports we've gotten — so refuse instead of silently degrading.
    # Every registered test is live_feed, i.e. served by an in-process engine.
    # Reaching here means the engine class is None — its module failed to
    # import. This used to fall through to spawning the ORIGINAL desktop-window
    # script as a subprocess, which is exactly the "camera opens in another
    # window and nothing syncs back" bug reports we got, so it refuses instead
    # of silently degrading.
    #
    # The whole subprocess launcher lived below this point and has been
    # deleted: with no non-live test in the registry it could never execute,
    # so it was untested surface that read as a working fallback. Bringing a
    # desktop-window test back means writing that path again deliberately,
    # with tests — not resurrecting code nobody had run.
    is_dual = test_id in ("shuttle_run", "dash_50m", "run_600m")
    detail = DUAL_FEED_IMPORT_ERROR if is_dual else WEB_ENGINES_IMPORT_ERROR
    return JSONResponse(
        {
            "ok": False,
            "error": (
                f"Live camera engine for '{test_id}' is unavailable "
                f"(module failed to import: {detail or 'unknown error — check server console'}). "
                "Fix the dependency shown above and restart the server."
            ),
        },
        status_code=503,
    )


@app.get("/api/engine-status")
def engine_status():
    """Quick browser-visible health check — hit http://localhost:5000/api/engine-status
    to see whether the live-feed (browser, no-desktop-window) engines loaded
    correctly, without having to dig through the server console."""
    return {
        "web_engines_ok": WEB_ENGINES_IMPORT_ERROR is None,
        "web_engines_error": WEB_ENGINES_IMPORT_ERROR,
        "web_engines_covers": ["wingspan", "bmi", "curl_up", "wall_toss", "sit_reach", "vertical_jump"],
        "dual_feed_engines_ok": DUAL_FEED_IMPORT_ERROR is None,
        "dual_feed_engines_error": DUAL_FEED_IMPORT_ERROR,
        "dual_feed_engines_covers": ["shuttle_run", "dash_50m", "run_600m"],
    }


@app.get("/api/status/{test_id}")
def status(test_id: str, request: Request):
    if test_id not in TESTS:
        return JSONResponse({"ok": False, "error": "Unknown test"}, status_code=404)
    return {"ok": True, "running": _is_running(_coach(request), test_id)}


@app.get("/api/status")
def status_all(request: Request):
    return {tid: _is_running(_coach(request), tid) for tid in TESTS}


@app.post("/api/stop/{test_id}")
def stop(test_id: str, request: Request):
    # Engines stage their own partial/best-available result during stop()
    # (_stage_on_stop / _maybe_stage / maybe_finish) — the HTTP layer just
    # stops whichever engine is registered for this test. engine.stop() runs
    # OUTSIDE _state_lock (see the deadlock rule at _state_lock's definition).
    # Only the coach who started it. Before this, either coach could stop the
    # other's run mid-measurement, and neither had any way to tell.
    coach = _coach(request)
    with _state_lock:
        eng = _RUNS.pop((coach, test_id), None)
        _release_cameras_locked(coach, test_id)
    if eng is not None:
        eng.stop()
    return {"ok": True}


# ── Live-feed control: one parameterized route set for every engine ───────
#
#  GET  /api/<test_id>/feed      — MJPEG stream (single-camera engines)
#  GET  /api/<test_id>/feed_a|b  — MJPEG streams (dual-camera engines)
#  GET  /api/<test_id>/state     — polled by screen4 to drive the button UI
#  POST /api/<test_id>/action    — {"action": "...", "value": ...}
#
# The paths are identical to the old hand-written per-test routes, so the
# frontend (screen4.html LIVE_ENGINES) needs no changes. Which feeds exist
# and which actions are legal come from the _ENGINES registry.

class EngineAction(BaseModel):
    action: str
    value: Any = None


def _slot_engine(test_id: str, request: Request):
    """Resolve (slot, THIS COACH's engine) or (None, 404 response).

    Scoped by coach so a feed, a state poll or an action can only ever reach
    the asker's own run — otherwise coach B polling /api/bmi/state would be
    watching coach A's test, and B's buttons would drive it.
    """
    slot = _ENGINES.get(test_id)
    if slot is None:
        return None, JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    with _state_lock:
        return slot, _engine_locked(_coach(request), test_id)


def _not_running(slot: _EngineSlot) -> JSONResponse:
    return JSONResponse({"ok": False, "error": f"{slot.label} isn't running"}, status_code=409)


def _mjpeg_response(eng, get_frame):
    """Stream eng's frames until it stops. Captures `eng` — a relaunch swaps
    the slot's instance, but an open stream must not silently switch engines."""
    def gen():
        boundary = b"--frame"
        FRAME_INTERVAL = 0.05   # 20 fps cap — browser can't usefully render faster
        last_seq = None         # dedupe on the engine's publish counter
        next_send = time.monotonic()
        while eng.is_running():
            now = time.monotonic()
            if now < next_send:
                time.sleep(max(0, next_send - now))
                continue
            jpg, seq = get_frame(eng)
            if jpg is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            next_send = now + FRAME_INTERVAL
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


_FEED_GETTERS = {
    "feed":   lambda e: e.get_frame(),
    "feed_a": lambda e: e.get_frame_a(),
    "feed_b": lambda e: e.get_frame_b(),
}


def _engine_feed(test_id: str, feed_name: str, request: Request):
    slot, eng = _slot_engine(test_id, request)
    if slot is None:
        return eng  # the 404
    if feed_name not in slot.feeds:
        return JSONResponse({"ok": False, "error": f"'{test_id}' has no {feed_name} stream"}, status_code=404)
    if eng is None or not eng.is_running():
        return _not_running(slot)
    return _mjpeg_response(eng, _FEED_GETTERS[feed_name])


@app.get("/api/{test_id}/feed")
def engine_feed(test_id: str, request: Request):
    return _engine_feed(test_id, "feed", request)


@app.get("/api/{test_id}/feed_a")
def engine_feed_a(test_id: str, request: Request):
    return _engine_feed(test_id, "feed_a", request)


@app.get("/api/{test_id}/feed_b")
def engine_feed_b(test_id: str, request: Request):
    return _engine_feed(test_id, "feed_b", request)


@app.get("/api/{test_id}/state")
def engine_state(test_id: str, request: Request):
    slot, eng = _slot_engine(test_id, request)
    if slot is None:
        return eng  # the 404
    if eng is None:
        return {"ok": True, "running": False}
    resp = {"ok": True, "running": eng.is_running()}
    resp.update(eng.get_state())  # engine-reported running (read under its lock) wins
    return resp


@app.post("/api/{test_id}/action")
def engine_action(test_id: str, b: EngineAction, request: Request):
    slot, eng = _slot_engine(test_id, request)
    if slot is None:
        return eng  # the 404
    if eng is None or not eng.is_running():
        return _not_running(slot)
    if slot.actions is None:  # dual-feed engines: generic dispatcher
        return eng.action(b.action, b.value)
    handler = slot.actions.get(b.action)
    if handler is None:
        return JSONResponse({"ok": False, "error": f"Unknown action '{b.action}'"}, status_code=400)
    return handler(eng, b.value)


# ── Database — session management ──────────────────────────────────────────

@app.post("/api/session")
def create_session(body: SessionCreate, request: Request):
    """
    Create a student session before running tests.

    Body (JSON, all fields optional):
        student_name, age, gender ("M"/"F"/"Other"), school, grade, notes

    The coach comes from the signed-in identity, never from the body — a
    measurement must be stamped with who actually took it, not with whatever
    the page claimed.
    """
    sid = db.new_session(
        student_name = body.student_name,
        age          = body.age,
        gender       = body.gender,
        school       = body.school,
        grade        = body.grade,
        notes        = body.notes,
        coach        = _coach(request),
    )
    return {"ok": True, "session_id": sid}


@app.get("/api/session/{session_id}")
def get_session(session_id: int):
    row = db.get_session(session_id)
    if not row:
        return JSONResponse({"ok": False, "error": "Session not found"}, status_code=404)
    return {"ok": True, "session": row}


@app.get("/api/sessions")
def list_sessions(request: Request, everyone: bool = Query(default=False)):
    """This coach's sessions. ?everyone=1 for the whole server's."""
    coach = None if everyone else _coach(request)
    return {"ok": True, "sessions": db.list_sessions(coach=coach), "coach": _coach(request)}


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
    with _state_lock:
        _pending[(_coach(request), test_id)] = body
    return {"ok": True}


@app.get("/api/pending/{test_id}")
def get_pending(test_id: str, request: Request):
    if test_id not in TESTS:
        return JSONResponse({"ok": False, "error": f"Unknown test '{test_id}'"}, status_code=404)
    # Only this coach's staged reading. This lookup is the exact line that
    # used to hand one coach the other's measurement, which screen3 then
    # saved against the wrong student.
    with _state_lock:
        return {"ok": True, "pending": _pending.get((_coach(request), test_id))}


@app.post("/api/pending/{test_id}/clear")
def clear_pending(test_id: str, request: Request):
    with _state_lock:
        _pending.pop((_coach(request), test_id), None)
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
def results_all(request: Request,
                student: Optional[str] = Query(default=None),
                everyone: bool = Query(default=False)):
    """Flat view across all tests, scoped to the signed-in coach by default.

    ?student=<name> filters by student; ?everyone=1 drops the coach scope for
    a whole-school view. Rows recorded before coach sign-in existed have
    coach NULL and only ever appear under ?everyone=1.
    """
    coach = _coach(request)
    if student:
        rows = db.get_student_summary(student)
        if not everyone:
            rows = [r for r in rows if r.get("coach") == coach]
    else:
        rows = db.get_all_results(coach=None if everyone else coach)
    return {"ok": True, "results": rows, "coach": coach, "scoped": not everyone}


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
# Defaults are deliberately safe: localhost only (defence in depth — every
# route also requires the shared access code, see auth.py), no auto-reload (a
# reload restarts the process mid-test, killing the camera thread and losing
# staged results).
# Port 5001 because macOS AirPlay squats on 5000.
#
# To use the app from a phone on the same network (the normal coach setup),
# opt in explicitly:   GOFORFIT_HOST=0.0.0.0 python app.py

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("GOFORFIT_HOST", "127.0.0.1")
    port = int(os.environ.get("GOFORFIT_PORT", "5001"))
    reload = os.environ.get("GOFORFIT_RELOAD") == "1"
    shown_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host

    print("\n  GoForFit NEP FMS — starting …")
    print(f"  Tests directory : {TESTS_DIR}")
    print(f"  Database        : {BASE_DIR / 'goforfit.db'}")
    print(f"  Open in browser : http://{shown_host}:{port}")
    print(f"  API docs        : http://{shown_host}:{port}/docs")
    if host == "0.0.0.0":
        print("  NOTE: serving on ALL interfaces — anyone on this network can reach the app.")
    print()
    uvicorn.run("app:app", host=host, port=port, reload=reload)