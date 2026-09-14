"""
tests/vertical_jump_web_engine.py
==================================
In-process, browser-controlled version of vertical_jump_test.py.

Instead of opening its own OpenCV desktop window, this engine runs the
camera capture + MediaPipe pose loop on a background thread inside the
FastAPI process, exposes the latest annotated frame as MJPEG, and accepts
actions (capture baseline / calibrate / start attempt / reset) from the
browser buttons on screen4.html — the same pattern wingspan_web_engine.py
uses for the wingspan test.
"""

import os
import threading
import time
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CAMERA_SOURCE = 0
ATTEMPTS_REQUIRED = 5
BASELINE_FRAMES = 30
JUMP_RESET_TOLERANCE = 12
DEFAULT_PIXEL_TO_CM = 0.35
# Which body point defines the jump measurement:
#   "hip"   center-of-mass (hip midpoint) rise — stable, works even when the
#           hands leave the top of the frame at the apex; needs only a resting
#           stance, no arms-up "standing reach" capture.
#   "reach" legacy fingertip reach differential (Sargent) — kept as a fallback.
MEASURE_METHOD = "hip"
# A landing only records an attempt after the body has stayed above the takeoff
# threshold for at least this many consecutive frames. Rejects single-frame
# landmark-noise spikes that would otherwise log a bogus attempt.
JUMP_MIN_AIR_FRAMES = 3
# Design B: the scale is a cm-per-pixel value learned from a known HEIGHT
# (exactly like the BMI engine), then reused to measure standing reach and jump
# automatically. A dedicated file (separate from BMI's, and from the old
# reach-based calibration.txt) so a stale value from the previous method is
# never misread as this one.
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration_vertical_jump_cmpx.txt")
HAND_EXTENSION_RATIO = 0.75
VISIBILITY_THRESHOLD = 0.55
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

# Height-calibration tuning (ported from the BMI engine's fixes 1–3).
CROWN_K          = 1.60   # crown ≈ 1.6 eye→mouth spans above the eyes
HEIGHT_MIN_VIS   = 0.60   # landmarks must be clearly seen for a height reading
HEIGHT_SMOOTH    = 20     # frames kept for the median pixel height
HEIGHT_STABLE_CV = 0.02   # recent heights must agree within 2% (std/mean)

NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30


def person_height_px(landmarks, width, height):
    """Build-independent standing height in pixels (crown→heel), or None if the
    pose isn't a clean full-body view. Identical approach to the BMI engine:
    crown estimated from facial proportions (eye→mouth), bottom from both heels,
    everything gated on high visibility."""
    def ok(idx):
        return landmarks[idx].visibility >= HEIGHT_MIN_VIS

    if not all(ok(i) for i in (NOSE, LEFT_EYE, RIGHT_EYE, MOUTH_LEFT, MOUTH_RIGHT)):
        return None
    if not (ok(LEFT_HEEL) and ok(RIGHT_HEEL)):
        return None

    eye_y   = (landmarks[LEFT_EYE].y + landmarks[RIGHT_EYE].y) * 0.5 * height
    mouth_y = (landmarks[MOUTH_LEFT].y + landmarks[MOUTH_RIGHT].y) * 0.5 * height
    face_span = mouth_y - eye_y
    if face_span <= 1:
        return None
    crown_y = eye_y - CROWN_K * face_span
    bottom_y = max(landmarks[LEFT_HEEL].y, landmarks[RIGHT_HEEL].y) * height
    span = bottom_y - crown_y
    return span if span > 20 else None

SHADOW_COLORS = [
    (80, 255, 120), (255, 220, 80), (255, 140, 80),
    (180, 120, 255), (80, 220, 255),
]


# ── Geometry / drawing helpers (ported unchanged from vertical_jump_test.py) ─

def landmark_to_point(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def get_body_points(landmarks, width, height):
    left_hip = landmark_to_point(landmarks[LEFT_HIP], width, height)
    right_hip = landmark_to_point(landmarks[RIGHT_HIP], width, height)
    left_ankle = landmark_to_point(landmarks[LEFT_ANKLE], width, height)
    right_ankle = landmark_to_point(landmarks[RIGHT_ANKLE], width, height)
    nose = landmark_to_point(landmarks[NOSE], width, height)
    left_eye = landmark_to_point(landmarks[LEFT_EYE], width, height)
    right_eye = landmark_to_point(landmarks[RIGHT_EYE], width, height)
    left_ear = landmark_to_point(landmarks[LEFT_EAR], width, height)
    right_ear = landmark_to_point(landmarks[RIGHT_EAR], width, height)

    hip_center = (left_hip + right_hip) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0
    head_y = min(nose[1], left_eye[1], right_eye[1], left_ear[1], right_ear[1])
    return hip_center, ankle_center, nose, head_y


def estimate_hand_tip(landmarks, wrist_idx, elbow_idx, width, height):
    wrist = landmarks[wrist_idx]
    elbow = landmarks[elbow_idx]
    if wrist.visibility < VISIBILITY_THRESHOLD or elbow.visibility < VISIBILITY_THRESHOLD:
        return None
    wrist_pt = landmark_to_point(wrist, width, height)
    elbow_pt = landmark_to_point(elbow, width, height)
    return wrist_pt + (wrist_pt - elbow_pt) * HAND_EXTENSION_RATIO


def get_reach_points(landmarks, width, height):
    left_tip = estimate_hand_tip(landmarks, LEFT_WRIST, LEFT_ELBOW, width, height)
    right_tip = estimate_hand_tip(landmarks, RIGHT_WRIST, RIGHT_ELBOW, width, height)
    candidates = [tip for tip in [left_tip, right_tip] if tip is not None]
    if not candidates:
        return None, None, None
    best_tip = min(candidates, key=lambda pt: pt[1])
    return best_tip, left_tip, right_tip


def draw_pose(frame, landmarks, width, height):
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28), (27, 31), (28, 32),
    ]
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        cv.line(frame, (int(start.x * width), int(start.y * height)),
                 (int(end.x * width), int(end.y * height)), (120, 255, 120), 2)

    key_points = [
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
    ]
    for idx in key_points:
        lm = landmarks[idx]
        if lm.visibility >= VISIBILITY_THRESHOLD:
            cv.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, (255, 240, 120), -1)

    left_tip = estimate_hand_tip(landmarks, LEFT_WRIST, LEFT_ELBOW, width, height)
    right_tip = estimate_hand_tip(landmarks, RIGHT_WRIST, RIGHT_ELBOW, width, height)
    for tip, wrist_idx in [(left_tip, LEFT_WRIST), (right_tip, RIGHT_WRIST)]:
        if tip is not None:
            wrist = landmarks[wrist_idx]
            wrist_pt = (int(wrist.x * width), int(wrist.y * height))
            tip_pt = (int(tip[0]), int(tip[1]))
            cv.line(frame, wrist_pt, tip_pt, (255, 190, 0), 2)
            cv.circle(frame, tip_pt, 6, (0, 220, 255), -1)


def draw_status(frame, lines):
    for index, text in enumerate(lines):
        cv.putText(frame, text, (20, 40 + index * 35), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def draw_height_shadows(frame, baseline_feet_y, baseline_reach_y, attempt_shadows, pixel_to_cm,
                        measure_label="Standing reach"):
    width = frame.shape[1]
    if baseline_feet_y is not None:
        cv.line(frame, (width - 260, int(baseline_feet_y)), (width - 20, int(baseline_feet_y)), (0, 215, 255), 2)
        cv.putText(frame, "Feet baseline", (width - 255, int(baseline_feet_y) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    if baseline_reach_y is not None:
        cv.line(frame, (width - 260, int(baseline_reach_y)), (width - 20, int(baseline_reach_y)), (255, 180, 0), 2)
        cv.putText(frame, measure_label, (width - 255, int(baseline_reach_y) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
    for attempt in attempt_shadows:
        reach_y = int(attempt["peak_reach_y"])
        label = f"A{attempt['attempt']}: reach {attempt['reach_cm']:.1f} cm | jump {attempt['jump_cm']:.1f} cm"
        cv.line(frame, (width - 260, reach_y), (width - 20, reach_y), attempt["color"], 3)
        cv.putText(frame, label, (width - 255, reach_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, attempt["color"], 2)
        if baseline_feet_y is not None:
            cv.line(frame, (width - 40, int(baseline_feet_y)), (width - 40, reach_y), attempt["color"], 2)


def load_calibration():
    """Return the saved cm-per-pixel scale, or None if not calibrated yet."""
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
                value = float(file.read().strip())
            if 0.01 < value < 5.0:
                return value
        except Exception:
            pass
    return None


def save_calibration(pixel_to_cm):
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
        file.write(str(round(pixel_to_cm, 6)))


# ── Engine ────────────────────────────────────────────────────────────────

class VerticalJumpEngine:
    """Runs the vertical-jump camera loop on a background thread and exposes
    MJPEG frames + JSON state to the browser, instead of an OpenCV window."""

    def __init__(self, session_id=None, on_stage=None):
        self.session_id = session_id
        self.on_stage = on_stage
        self._lock = threading.RLock()
        self._running = False
        self._thread = None
        self._cap = None
        self._model = None
        self._jpeg = None
        self._staged = False
        # Selfie-mirror the preview by default; turned off for the back camera
        # (set_facing) so a real-world back-camera view isn't shown flipped.
        self._mirror = True

        # Browser-sourced camera (see BaseWebEngine in web_engines.py for the
        # rationale): frames arrive via push_frame() from getUserMedia in the
        # browser instead of cv.VideoCapture on this server.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        _loaded = load_calibration()
        self._scale_calibrated = _loaded is not None
        self.pixel_to_cm = _loaded if _loaded is not None else DEFAULT_PIXEL_TO_CM
        # Live full-body height (pixels) used for the BMI-style calibration.
        self._cal_px_buf = deque(maxlen=HEIGHT_SMOOTH)
        self._cal_px_median = None
        self._cal_stable = False
        self.standing_reach_cm = None
        self.baseline_feet_y = None
        self.baseline_reach_y = None
        self.baseline_hip_y = None            # resting hip-centre y (hip method)
        self.baseline_feet_samples = []
        self.baseline_reach_samples = []
        self.baseline_hip_samples = []
        self.attempt_results_cm = []
        self.attempt_shadows = []
        self.attempt_active = False
        self.capture_baseline_mode = False
        self.peak_reach_y = None
        self.peak_jump_px = 0.0
        self.jump_in_air = False
        self.air_frames = 0                   # consecutive airborne frames (debounce)
        self.current_jump_cm = 0.0
        self.message = ""

    def _baseline_ready(self):
        """True once the resting stance needed to start an attempt is captured."""
        if MEASURE_METHOD == "hip":
            return self.baseline_hip_y is not None
        return (self.baseline_feet_y is not None
                and self.baseline_reach_y is not None
                and self.standing_reach_cm is not None)

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self):
        with self._lock:
            if self._running:
                return
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options, output_segmentation_masks=True, num_poses=1,
            )
            self._model = vision.PoseLandmarker.create_from_options(options)

            # No server-side camera: frames come from the browser via
            # push_frame(). The loop waits for the first one.
            with self._frame_lock:
                self._in_frame = None
            self._frame_event.clear()

            self._running = True
            self._staged = False
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        self._frame_event.set()   # wake the loop so it sees _running == False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        # Stage the best-available result on stop (the normal finish path —
        # there's no SAVE button; screen4 status even says "tap STOP / SAVE").
        # Without this the local screen3 review + /api/save path never receives
        # a vertical-jump result. _maybe_stage() is idempotent (the _staged
        # guard) so it's a no-op if action_save already staged it.
        self._maybe_stage()
        with self._frame_lock:
            self._in_frame = None
        if self._model:
            self._model.close()
            self._model = None

    def set_facing(self, facing):
        """Which browser camera is streaming ('user' = front, 'environment' =
        back). The back camera is a real-world view, so it must not be
        selfie-mirrored."""
        if facing is None:
            return
        with self._lock:
            self._mirror = (facing != "environment")

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame; keep only the newest."""
        if not self._running or not jpeg_bytes:
            return False
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv.imdecode(arr, cv.IMREAD_COLOR)
        if frame is None:
            return False
        with self._frame_lock:
            self._in_frame = frame
        self._frame_event.set()
        return True

    def is_running(self):
        return self._running

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    # ── state for the browser ───────────────────────────────────────────

    def get_state(self):
        with self._lock:
            best = max(self.attempt_results_cm) if self.attempt_results_cm else None
            avg = (sum(self.attempt_results_cm) / len(self.attempt_results_cm)
                   if self.attempt_results_cm else None)
            return {
                "running":            self._running,
                "method":             MEASURE_METHOD,
                "baseline_captured":  self._baseline_ready(),
                "capturing_baseline": self.capture_baseline_mode,
                "calibrated":         self._scale_calibrated,
                "pose_detected":      self._cal_px_median is not None,
                "cal_stable":         self._cal_stable,
                "standing_reach_cm":  round(self.standing_reach_cm, 1) if self.standing_reach_cm else None,
                "pixel_to_cm":        round(self.pixel_to_cm, 4),
                "attempt_active":     self.attempt_active,
                "current_jump_cm":    round(self.current_jump_cm, 1),
                "attempts":           [round(a, 1) for a in self.attempt_results_cm],
                "attempts_required":  ATTEMPTS_REQUIRED,
                "best_jump_cm":       round(best, 1) if best is not None else None,
                "average_jump_cm":    round(avg, 1) if avg is not None else None,
                "complete":           len(self.attempt_results_cm) >= ATTEMPTS_REQUIRED,
                "message":            self.message,
            }

    # ── actions (called from the FastAPI request thread) ────────────────

    def action_capture_baseline(self):
        with self._lock:
            if not self._scale_calibrated:
                return {"ok": False, "error": "Calibrate height first, then capture the standing reach."}
            self.baseline_feet_samples.clear()
            self.baseline_reach_samples.clear()
            self.baseline_hip_samples.clear()
            self.capture_baseline_mode = True
            self.message = ("Stand tall and still — capturing your stance..."
                            if MEASURE_METHOD == "hip"
                            else "Reach straight up and hold steady — capturing standing reach...")
        return {"ok": True}

    def action_calibrate(self, value):
        """Design B: calibrate the cm-per-pixel scale from a KNOWN HEIGHT (same
        as the BMI engine). Standing reach and jump are then measured
        automatically — no per-person reach entry needed."""
        with self._lock:
            if self.attempt_active:
                return {"ok": False, "error": "Finish the current attempt first."}
            if self._cal_px_median is None:
                return {"ok": False, "error": "Stand fully in frame — head and both feet must be visible"}
            if not self._cal_stable:
                return {"ok": False, "error": "Hold still — measurement still settling"}
            try:
                real_height_cm = float(value)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Invalid height value."}
            if not (50 <= real_height_cm <= 250):
                return {"ok": False, "error": "Height must be between 50 and 250 cm."}

            self.pixel_to_cm = real_height_cm / self._cal_px_median
            self._scale_calibrated = True
            save_calibration(self.pixel_to_cm)
            # If a standing-reach baseline was already captured, refresh its cm
            # value against the new scale.
            if self.baseline_feet_y is not None and self.baseline_reach_y is not None:
                span_px = max(1.0, self.baseline_feet_y - self.baseline_reach_y)
                self.standing_reach_cm = span_px * self.pixel_to_cm
            _next = "capture your stance" if MEASURE_METHOD == "hip" else "capture standing reach"
            self.message = f"Calibrated: {self.pixel_to_cm:.4f} cm/px — now {_next}."
        return {"ok": True}

    def action_start_attempt(self):
        with self._lock:
            if not self._scale_calibrated:
                return {"ok": False, "error": "Calibrate height first."}
            if not self._baseline_ready():
                return {"ok": False, "error": "Capture the standing stance first."}
            if self.attempt_active:
                return {"ok": False, "error": "An attempt is already in progress."}
            if len(self.attempt_results_cm) >= ATTEMPTS_REQUIRED:
                return {"ok": False, "error": "All attempts complete."}
            self.attempt_active = True
            self.peak_jump_px = 0.0
            self.peak_reach_y = None
            self.jump_in_air = False
            self.air_frames = 0
            self.message = f"Attempt {len(self.attempt_results_cm) + 1} in progress — jump!"
        return {"ok": True}

    def action_reset(self):
        with self._lock:
            self.pixel_to_cm = DEFAULT_PIXEL_TO_CM
            self._scale_calibrated = False
            if os.path.exists(CALIBRATION_FILE):
                try:
                    os.remove(CALIBRATION_FILE)
                except OSError:
                    pass
            self._cal_px_buf.clear()
            self._cal_px_median = None
            self._cal_stable = False
            self.standing_reach_cm = None
            self.baseline_feet_y = None
            self.baseline_reach_y = None
            self.baseline_hip_y = None
            self.baseline_feet_samples.clear()
            self.baseline_reach_samples.clear()
            self.baseline_hip_samples.clear()
            self.attempt_results_cm.clear()
            self.attempt_shadows.clear()
            self.attempt_active = False
            self.capture_baseline_mode = False
            self.peak_reach_y = None
            self.peak_jump_px = 0.0
            self.jump_in_air = False
            self.air_frames = 0
            self.current_jump_cm = 0.0
            self._staged = False
            self.message = "Reset."
        return {"ok": True}

    def action_cancel_attempt(self):
        """Abort the in-progress attempt without touching the calibration, the
        stance baseline, or previously recorded attempts — the escape hatch for
        a jump that never crossed the takeoff threshold."""
        with self._lock:
            if not self.attempt_active:
                return {"ok": False, "error": "No attempt in progress."}
            self.attempt_active = False
            self.jump_in_air = False
            self.air_frames = 0
            self.peak_jump_px = 0.0
            self.peak_reach_y = None
            self.current_jump_cm = 0.0
            self.message = "Attempt cancelled — tap START ATTEMPT to retry."
        return {"ok": True}

    def action_save(self):
        self._maybe_stage(force=True)
        return {"ok": True}

    # ── internal ─────────────────────────────────────────────────────────

    def _capture_baseline_sample(self, hip_y, reach_y, feet_y):
        """Accumulate resting-stance samples and finalize the baseline once the
        pose settles. Must be called with self._lock held, and only while
        capture_baseline_mode is True."""
        if MEASURE_METHOD == "hip":
            self.baseline_hip_samples.append(hip_y)
            self.baseline_feet_samples.append(feet_y)
            if len(self.baseline_hip_samples) < BASELINE_FRAMES:
                return
            hips = np.asarray(self.baseline_hip_samples[-BASELINE_FRAMES:], dtype=float)
            feet = np.asarray(self.baseline_feet_samples[-BASELINE_FRAMES:], dtype=float)
            # Gate on the feet→hip span (height above the floor) so someone
            # drifting toward/away from the camera can't false-settle.
            span = feet - hips
            span_mean = float(span.mean())
            stable = span_mean > 1 and (float(span.std()) / span_mean) <= HEIGHT_STABLE_CV
            if stable:
                self.baseline_hip_y = float(hips.mean())
                self.baseline_feet_y = float(feet.mean())
                self.standing_reach_cm = None       # not measured in hip mode
                self.baseline_hip_samples.clear()
                self.baseline_feet_samples.clear()
                self.capture_baseline_mode = False
                self.message = "Stance set — tap START ATTEMPT, then jump."
            else:
                self.baseline_hip_samples.pop(0)
                self.baseline_feet_samples.pop(0)
                self.message = "Hold still — settling..."
            return

        # Legacy reach method: needs the fingertip visible to sample.
        if reach_y is None:
            return
        self.baseline_feet_samples.append(feet_y)
        self.baseline_reach_samples.append(reach_y)
        if len(self.baseline_reach_samples) < BASELINE_FRAMES:
            return
        feet  = np.asarray(self.baseline_feet_samples[-BASELINE_FRAMES:], dtype=float)
        reach = np.asarray(self.baseline_reach_samples[-BASELINE_FRAMES:], dtype=float)
        span  = feet - reach            # standing reach in pixels
        span_mean = float(span.mean())
        stable = span_mean > 1 and (float(span.std()) / span_mean) <= HEIGHT_STABLE_CV
        if stable:
            self.baseline_feet_y = float(feet.mean())
            self.baseline_reach_y = float(reach.mean())
            self.standing_reach_cm = max(1.0, span_mean) * self.pixel_to_cm
            self.baseline_feet_samples.clear()
            self.baseline_reach_samples.clear()
            self.capture_baseline_mode = False
            self.message = (f"Standing reach: {self.standing_reach_cm:.1f} cm "
                            f"— tap START ATTEMPT and jump.")
        else:
            self.baseline_feet_samples.pop(0)
            self.baseline_reach_samples.pop(0)
            self.message = "Hold the reach steady — settling..."

    def _record_attempt(self, current_y):
        """Finalize the current jump attempt. Must be called with self._lock held."""
        jump_height_cm = max(0.0, self.peak_jump_px * self.pixel_to_cm)
        self.attempt_results_cm.append(jump_height_cm)
        attempt_index = len(self.attempt_results_cm)
        absolute_reach_cm = (self.standing_reach_cm or 0.0) + jump_height_cm
        self.attempt_shadows.append({
            "attempt": attempt_index,
            "jump_cm": jump_height_cm,
            "reach_cm": absolute_reach_cm,
            "peak_reach_y": self.peak_reach_y if self.peak_reach_y is not None else current_y,
            "color": SHADOW_COLORS[(attempt_index - 1) % len(SHADOW_COLORS)],
        })
        self.message = f"Attempt {attempt_index}: jump {jump_height_cm:.1f} cm"
        self.peak_jump_px = 0.0
        self.peak_reach_y = None
        self.jump_in_air = False
        self.air_frames = 0
        self.attempt_active = False
        if len(self.attempt_results_cm) == ATTEMPTS_REQUIRED:
            self.message = "All 5 attempts complete — tap STOP / SAVE."

    def _maybe_stage(self, force=False):
        with self._lock:
            if self._staged and not force:
                return
            if not self.attempt_results_cm:
                return
            if self.on_stage:
                self.on_stage(
                    jump_cm=round(max(self.attempt_results_cm), 2),
                    average_jump_cm=round(sum(self.attempt_results_cm) / len(self.attempt_results_cm), 2),
                    standing_reach_cm=self.standing_reach_cm,
                    attempts=[round(a, 2) for a in self.attempt_results_cm],
                )
            self._staged = True

    def _loop(self):
        while True:
            with self._lock:
                if not self._running:
                    break
            # Wait for the browser to push a frame (up to 1s), then take the
            # newest one. Replaces self._cap.read().
            if not self._frame_event.wait(timeout=1.0):
                continue
            self._frame_event.clear()
            with self._frame_lock:
                frame = self._in_frame
                self._in_frame = None
            if frame is None:
                continue

            if self._mirror:
                frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._model.detect(mp_image)

            with self._lock:
                self._process_frame(frame, width, height, results)
                ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self._jpeg = buf.tobytes()

    def _process_frame(self, frame, width, height, results):
        """Must be called with self._lock held."""
        status_lines = [f"Attempts: {len(self.attempt_results_cm)}/{ATTEMPTS_REQUIRED}"]
        current_reach_y = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            hip_center, ankle_center, nose, head_y = get_body_points(landmarks, width, height)
            reach_tip, left_tip, right_tip = get_reach_points(landmarks, width, height)
            draw_pose(frame, landmarks, width, height)

            # Live full-body height for the BMI-style scale calibration, with the
            # same smoothing + stability gate. Works whether arms are down (during
            # calibration) or raised (crown/heel are unaffected by arm position).
            ph = person_height_px(landmarks, width, height)
            if ph:
                self._cal_px_buf.append(ph)
                self._cal_px_median = float(np.median(self._cal_px_buf))
                if len(self._cal_px_buf) >= HEIGHT_SMOOTH:
                    arr = np.asarray(self._cal_px_buf, dtype=float)
                    m = float(arr.mean())
                    self._cal_stable = m > 0 and (float(arr.std()) / m) <= HEIGHT_STABLE_CV
                else:
                    self._cal_stable = False
            else:
                self._cal_px_buf.clear()
                self._cal_px_median = None
                self._cal_stable = False

            cv.circle(frame, tuple(hip_center.astype(int)), 7, (255, 0, 0), -1)
            cv.circle(frame, tuple(ankle_center.astype(int)), 7, (0, 255, 255), -1)
            cv.circle(frame, tuple(nose.astype(int)), 7, (0, 0, 255), -1)

            current_hip_y = float(hip_center[1])

            if reach_tip is not None:
                current_reach_y = float(reach_tip[1])
                tip_xy = (int(reach_tip[0]), int(reach_tip[1]))
                cv.circle(frame, tip_xy, 8, (0, 220, 255), -1)
                cv.line(frame, (tip_xy[0] - 16, tip_xy[1]), (tip_xy[0] + 16, tip_xy[1]), (0, 220, 255), 2)

            # The point that defines the jump: hip centre (default) or fingertip.
            if MEASURE_METHOD == "hip":
                current_y, baseline_y = current_hip_y, self.baseline_hip_y
            else:
                current_y, baseline_y = current_reach_y, self.baseline_reach_y

            if current_y is not None and baseline_y is not None:
                self.current_jump_cm = max(0.0, (baseline_y - current_y) * self.pixel_to_cm)
            else:
                self.current_jump_cm = 0.0

            # ── Capture the resting stance baseline ──────────────────────────
            if self.capture_baseline_mode:
                self._capture_baseline_sample(current_hip_y, current_reach_y, float(ankle_center[1]))

            # ── Track an in-progress jump ────────────────────────────────────
            if self.attempt_active and baseline_y is not None and current_y is not None:
                rise_px = baseline_y - current_y
                if rise_px > self.peak_jump_px:
                    self.peak_jump_px = rise_px
                    self.peak_reach_y = current_y

                if self.current_jump_cm > JUMP_RESET_TOLERANCE:
                    self.jump_in_air = True
                    self.air_frames += 1

                if self.jump_in_air and self.current_jump_cm < JUMP_RESET_TOLERANCE:
                    if self.air_frames >= JUMP_MIN_AIR_FRAMES:
                        self._record_attempt(current_y)
                    else:
                        # Too brief to be a real jump — treat as landmark noise,
                        # discard it, and keep the attempt open.
                        self.peak_jump_px = 0.0
                        self.peak_reach_y = None
                        self.jump_in_air = False
                        self.air_frames = 0
                        self.message = "Ignored a false blip — jump when ready."

        else:
            # No pose this frame — reset the calibration-height tracking so a
            # fresh, settled stance is required before calibrating again.
            self._cal_px_buf.clear()
            self._cal_px_median = None
            self._cal_stable = False

        status_lines.insert(1, f"Calibration: {self.pixel_to_cm:.4f} cm/px")
        if MEASURE_METHOD == "hip":
            measure_baseline_y, measure_label = self.baseline_hip_y, "Standing hip"
        else:
            measure_baseline_y, measure_label = self.baseline_reach_y, "Standing reach"
        draw_height_shadows(frame, self.baseline_feet_y, measure_baseline_y,
                             self.attempt_shadows, self.pixel_to_cm, measure_label=measure_label)
        draw_status(frame, status_lines)
