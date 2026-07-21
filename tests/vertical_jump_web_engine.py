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

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from stall_watchdog import StallWatchdogMixin

CAMERA_SOURCE = 0
ATTEMPTS_REQUIRED = 5
BASELINE_FRAMES = 30
JUMP_RESET_TOLERANCE = 12
DEFAULT_PIXEL_TO_CM = 0.35
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration_vertical_jump.txt")
HAND_EXTENSION_RATIO = 0.75
VISIBILITY_THRESHOLD = 0.55
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
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


def draw_height_shadows(frame, baseline_feet_y, baseline_reach_y, attempt_shadows, pixel_to_cm):
    width = frame.shape[1]
    if baseline_feet_y is not None:
        cv.line(frame, (width - 260, int(baseline_feet_y)), (width - 20, int(baseline_feet_y)), (0, 215, 255), 2)
        cv.putText(frame, "Feet baseline", (width - 255, int(baseline_feet_y) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    if baseline_reach_y is not None:
        cv.line(frame, (width - 260, int(baseline_reach_y)), (width - 20, int(baseline_reach_y)), (255, 180, 0), 2)
        cv.putText(frame, "Standing reach", (width - 255, int(baseline_reach_y) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
    for attempt in attempt_shadows:
        reach_y = int(attempt["peak_reach_y"])
        label = f"A{attempt['attempt']}: reach {attempt['reach_cm']:.1f} cm | jump {attempt['jump_cm']:.1f} cm"
        cv.line(frame, (width - 260, reach_y), (width - 20, reach_y), attempt["color"], 3)
        cv.putText(frame, label, (width - 255, reach_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, attempt["color"], 2)
        if baseline_feet_y is not None:
            cv.line(frame, (width - 40, int(baseline_feet_y)), (width - 40, reach_y), attempt["color"], 2)


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
                value = float(file.read().strip())
            if 0.01 < value < 5.0:
                return value
        except Exception:
            pass
    return DEFAULT_PIXEL_TO_CM


def save_calibration(pixel_to_cm):
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
        file.write(str(round(pixel_to_cm, 6)))


# ── Engine ────────────────────────────────────────────────────────────────

class VerticalJumpEngine(StallWatchdogMixin):
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
        self._frame_seq = 0   # bumped per publish — feed streams dedupe on this, not identity
        self._staged = False

        self.pixel_to_cm = load_calibration()
        self.standing_reach_cm = None
        self.baseline_feet_y = None
        self.baseline_reach_y = None
        self.baseline_feet_samples = []
        self.baseline_reach_samples = []
        self.attempt_results_cm = []
        self.attempt_shadows = []
        self.attempt_active = False
        self.capture_baseline_mode = False
        self.peak_reach_y = None
        self.peak_jump_px = 0.0
        self.jump_in_air = False
        self.current_jump_cm = 0.0
        self.message = ""

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

            self._cap = cv.VideoCapture(CAMERA_SOURCE)
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                self._model.close()
                self._model = None
                raise RuntimeError("Unable to open camera source.")

            self._running = True
            self._staged = False
            self._watchdog_reset()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        thread, self._thread = self._thread, None

        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
            if thread.is_alive():
                # _loop is still inside cap.read()/inference. Releasing the
                # capture now is a use-after-free in native OpenCV, which kills
                # the whole process with no Python traceback. Hand cleanup to a
                # reaper that waits for the loop to actually exit (same rule as
                # BaseWebEngine.stop()). Stage first — it only reads our own
                # attempt data under the lock.
                self._maybe_stage()
                threading.Thread(target=self._reap, args=(thread,), daemon=True).start()
                return

        self._reap(thread)
        self._maybe_stage()

    def _reap(self, thread):
        if thread is not None:
            thread.join()
        cap, self._cap = self._cap, None
        model, self._model = self._model, None
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        if model:
            try:
                model.close()
            except Exception:
                pass

    def is_running(self):
        return self._running

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_frame(self):
        """(jpeg_bytes|None, seq) — seq changes iff a new frame was published."""
        with self._lock:
            return self._jpeg, self._frame_seq

    # ── state for the browser ───────────────────────────────────────────

    def get_state(self):
        with self._lock:
            best = max(self.attempt_results_cm) if self.attempt_results_cm else None
            avg = (sum(self.attempt_results_cm) / len(self.attempt_results_cm)
                   if self.attempt_results_cm else None)
            return {
                "running":            self._running,
                "baseline_captured":  self.baseline_feet_y is not None,
                "capturing_baseline": self.capture_baseline_mode,
                "calibrated":         self.standing_reach_cm is not None,
                "standing_reach_cm":  self.standing_reach_cm,
                "pixel_to_cm":        round(self.pixel_to_cm, 4),
                "attempt_active":     self.attempt_active,
                "current_jump_cm":    round(self.current_jump_cm, 1),
                "attempts":           [round(a, 1) for a in self.attempt_results_cm],
                "attempts_required":  ATTEMPTS_REQUIRED,
                "best_jump_cm":       round(best, 1) if best is not None else None,
                "average_jump_cm":    round(avg, 1) if avg is not None else None,
                "complete":           len(self.attempt_results_cm) >= ATTEMPTS_REQUIRED,
                "message":            self.message,
                # This engine has no _error of its own; a wedged capture is
                # the only failure it can report to screen4.
                "error":              self._stall_error(),
            }

    # ── actions (called from the FastAPI request thread) ────────────────

    def action_capture_baseline(self):
        with self._lock:
            self.baseline_feet_samples.clear()
            self.baseline_reach_samples.clear()
            self.capture_baseline_mode = True
            self.message = "Collecting standing reach... hold the reach pose steady."
        return {"ok": True}

    def action_calibrate(self, value):
        with self._lock:
            if self.baseline_feet_y is None or self.baseline_reach_y is None:
                return {"ok": False, "error": "Capture standing reach first."}
            if self.attempt_active:
                return {"ok": False, "error": "Finish the current attempt first."}
            try:
                real_reach_cm = float(value)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Invalid standing reach value."}
            if not (50 <= real_reach_cm <= 300):
                return {"ok": False, "error": "Standing reach must be between 50 and 300 cm."}

            span_px = max(1.0, self.baseline_feet_y - self.baseline_reach_y)
            self.standing_reach_cm = real_reach_cm
            self.pixel_to_cm = real_reach_cm / span_px
            save_calibration(self.pixel_to_cm)
            self.message = f"Calibrated: {self.pixel_to_cm:.4f} cm/px"
        return {"ok": True}

    def action_start_attempt(self):
        with self._lock:
            if self.baseline_feet_y is None or self.baseline_reach_y is None or self.standing_reach_cm is None:
                return {"ok": False, "error": "Calibrate standing reach first."}
            if self.attempt_active:
                return {"ok": False, "error": "An attempt is already in progress."}
            if len(self.attempt_results_cm) >= ATTEMPTS_REQUIRED:
                return {"ok": False, "error": "All attempts complete."}
            self.attempt_active = True
            self.peak_jump_px = 0.0
            self.peak_reach_y = None
            self.jump_in_air = False
            self.message = f"Attempt {len(self.attempt_results_cm) + 1} in progress — jump!"
        return {"ok": True}

    def action_reset(self):
        with self._lock:
            self.pixel_to_cm = DEFAULT_PIXEL_TO_CM
            self.standing_reach_cm = None
            self.baseline_feet_y = None
            self.baseline_reach_y = None
            self.baseline_feet_samples.clear()
            self.baseline_reach_samples.clear()
            self.attempt_results_cm.clear()
            self.attempt_shadows.clear()
            self.attempt_active = False
            self.capture_baseline_mode = False
            self.peak_reach_y = None
            self.peak_jump_px = 0.0
            self.jump_in_air = False
            self.current_jump_cm = 0.0
            self._staged = False
            if os.path.exists(CALIBRATION_FILE):
                os.remove(CALIBRATION_FILE)
            self.message = "Reset."
        return {"ok": True}

    def action_save(self):
        self._maybe_stage(force=True)
        return {"ok": True}

    # ── internal ─────────────────────────────────────────────────────────

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
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.02)
                continue

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
                    self._frame_seq += 1
                    self._watchdog_mark()

    def _process_frame(self, frame, width, height, results):
        """Must be called with self._lock held."""
        status_lines = [f"Attempts: {len(self.attempt_results_cm)}/{ATTEMPTS_REQUIRED}"]
        current_reach_y = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            hip_center, ankle_center, nose, head_y = get_body_points(landmarks, width, height)
            reach_tip, left_tip, right_tip = get_reach_points(landmarks, width, height)
            draw_pose(frame, landmarks, width, height)

            cv.circle(frame, tuple(hip_center.astype(int)), 7, (255, 0, 0), -1)
            cv.circle(frame, tuple(ankle_center.astype(int)), 7, (0, 255, 255), -1)
            cv.circle(frame, tuple(nose.astype(int)), 7, (0, 0, 255), -1)

            if reach_tip is not None:
                current_reach_y = float(reach_tip[1])
                tip_xy = (int(reach_tip[0]), int(reach_tip[1]))
                cv.circle(frame, tip_xy, 8, (0, 220, 255), -1)
                cv.line(frame, (tip_xy[0] - 16, tip_xy[1]), (tip_xy[0] + 16, tip_xy[1]), (0, 220, 255), 2)

            if current_reach_y is not None and self.baseline_reach_y is not None:
                self.current_jump_cm = max(0.0, (self.baseline_reach_y - current_reach_y) * self.pixel_to_cm)
            else:
                self.current_jump_cm = 0.0

            if self.capture_baseline_mode and current_reach_y is not None:
                self.baseline_feet_samples.append(ankle_center[1])
                self.baseline_reach_samples.append(current_reach_y)
                if len(self.baseline_feet_samples) >= BASELINE_FRAMES:
                    self.baseline_feet_y = float(np.mean(self.baseline_feet_samples))
                    self.baseline_reach_y = float(np.mean(self.baseline_reach_samples))
                    self.baseline_feet_samples.clear()
                    self.baseline_reach_samples.clear()
                    self.capture_baseline_mode = False
                    self.message = "Standing reach captured — enter your reach height in cm."

            if (self.attempt_active and self.baseline_reach_y is not None
                    and self.standing_reach_cm is not None and current_reach_y is not None):
                if self.current_jump_cm > self.peak_jump_px * self.pixel_to_cm:
                    self.peak_jump_px = self.baseline_reach_y - current_reach_y
                    self.peak_reach_y = current_reach_y

                if self.current_jump_cm > JUMP_RESET_TOLERANCE:
                    self.jump_in_air = True

                if self.jump_in_air and self.current_jump_cm < JUMP_RESET_TOLERANCE:
                    jump_height_cm = max(0.0, self.peak_jump_px * self.pixel_to_cm)
                    absolute_reach_cm = self.standing_reach_cm + jump_height_cm
                    self.attempt_results_cm.append(jump_height_cm)
                    attempt_index = len(self.attempt_results_cm)
                    self.attempt_shadows.append({
                        "attempt": attempt_index,
                        "jump_cm": jump_height_cm,
                        "reach_cm": absolute_reach_cm,
                        "peak_reach_y": self.peak_reach_y if self.peak_reach_y is not None else current_reach_y,
                        "color": SHADOW_COLORS[(attempt_index - 1) % len(SHADOW_COLORS)],
                    })
                    self.message = f"Attempt {attempt_index}: jump {jump_height_cm:.1f} cm"

                    self.peak_jump_px = 0.0
                    self.peak_reach_y = None
                    self.jump_in_air = False
                    self.attempt_active = False

                    if len(self.attempt_results_cm) == ATTEMPTS_REQUIRED:
                        self.message = "All 5 attempts complete — tap STOP / SAVE."

        status_lines.insert(1, f"Calibration: {self.pixel_to_cm:.4f} cm/px")
        draw_height_shadows(frame, self.baseline_feet_y, self.baseline_reach_y,
                             self.attempt_shadows, self.pixel_to_cm)
        draw_status(frame, status_lines)
