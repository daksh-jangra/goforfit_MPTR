"""
GoForFit — Unified Browser Test Engine
=======================================
One file, one shared `BaseWebEngine` + one class ("section") per test that
runs the camera/MediaPipe loop in-process and streams annotated frames to
the browser via MJPEG — instead of opening a separate desktop OpenCV window
via subprocess. This is the same approach wingspan_web_engine.py and
vertical_jump_web_engine.py already used; this file generalizes it into a
shared base class and adds four more tests.

SECTIONS IN THIS FILE
----------------------
  0. BaseWebEngine      shared threading / MJPEG / lock plumbing
  1. BmiEngine           wraps bmi__3_.py         (height / BMI)
  2. CurlUpEngine        wraps parcurlup.py       (partial curl-up counter)
  3. WallTossEngine      wraps wtt.py             (alternate-hand wall toss)
  4. SitReachEngine      wraps snr.py             (sit & reach)
  5. WingspanEngine      re-exported, unchanged, from wingspan_web_engine.py
  6. VerticalJumpEngine  re-exported, unchanged, from vertical_jump_web_engine.py

Every engine exposes the same contract app.py drives:

    start() / stop() / is_running() / get_jpeg() / get_state()

plus a handful of `action_*(value=None)` methods that screen4.html's
on-screen buttons call over HTTP instead of sending key presses into a
desktop cv2 window. Each engine reuses the ORIGINAL test script's geometry /
drawing / state-machine code by importing that module and calling its
functions/classes directly (exactly how wingspan_web_engine.py reuses
wingspan2.py) — only the "loop that used to call cv2.imshow/waitKey" is
reimplemented here, per test.

NOT converted here: shuttle_run (shuttleruntest.py), dash_50m
(sprint_timer.py), and run_600m (600m_new/). All three use TWO cameras (the
first two also use a microphone for whistle detection), which needs a
different multi-feed UI in screen4.html than the single <img> MJPEG panel
used here. They keep running as desktop-window subprocesses for now.
"""

import os
import sys
import math
import time
import threading
from collections import deque

import cv2
import numpy as np

# Make sure sibling test scripts (bmi__3_.py, parcurlup.py, wtt.py, snr.py,
# wingspan2.py, wingspan_web_engine.py, ...) import cleanly regardless of the
# working directory the FastAPI app was started from.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from stall_watchdog import StallWatchdogMixin  # noqa: E402  (needs _TESTS_DIR on sys.path)
import camera  # noqa: E402  — shared camera-source resolution


def _ensure_pose_model(path, url=None):
    """Downloads a MediaPipe PoseLandmarker .task file if it isn't already
    present. bmi__3_.py, parcurlup.py and snr.py all use the same
    pose_landmarker_full.task, so this is shared instead of duplicated."""
    if os.path.exists(path):
        return
    import ssl
    import urllib.request
    url = url or (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task"
    )
    # Python on macOS ships without the system CA bundle wired into OpenSSL, so
    # a plain urlretrieve raises CERTIFICATE_VERIFY_FAILED. Verify against
    # certifi's bundle (a hard dependency of mediapipe) so the model downloads
    # on a fresh machine instead of failing at first launch.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as fh:
        fh.write(resp.read())


# ════════════════════════════════════════════════════════════════════════
#  SECTION 0 — SHARED BASE ENGINE
# ════════════════════════════════════════════════════════════════════════
def _release_capture(cap):
    """Release a VideoCapture, ignoring an already-released/None handle."""
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass


class BaseWebEngine(StallWatchdogMixin):
    """One live browser-controlled test session. Created on 'Start Test',
    destroyed on 'Stop'. Subclasses implement the hooks marked below;
    everything else (camera lifecycle, background thread, MJPEG frame
    buffer, thread-safe state, stall detection) is shared.
    """

    JPEG_QUALITY = 75
    FRAME_W = 960
    FRAME_H = 540
    STOP_JOIN_TIMEOUT = 2

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        self.session_id = session_id
        self.on_stage = on_stage  # callback(*args) — stages a result for screen3 review
        # Which camera this engine drives. Comes from the coach's own setting,
        # so two coaches with two phones can run tests simultaneously.
        self.camera_source = camera_source
        # False when the camera already mirrors (see camera.mirror)
        self.mirror_preview = mirror

        # RLock: run loops call _safe_stage() while already holding the lock,
        # and _safe_stage locks its _error write.
        self._lock = threading.RLock()
        self._jpeg = None
        self._frame_seq = 0   # bumped per publish — feed streams dedupe on this, not identity
        self._running = False
        self._thread = None
        self._error = None
        self._staged = False  # True once a result reached on_stage; see _stage_on_stop()
        self.cap = None

    def set_facing(self, facing):
        """Tell the engine which browser camera is streaming ('user' = front,
        'environment' = back). The back camera is a real-world view and must NOT
        be selfie-mirrored; the front (or unknown/laptop) camera keeps the
        mirror. Subclasses that need extra work (e.g. wall toss's handedness)
        override this and call super()."""
        if facing is None:
            return
        self.mirror_preview = (facing != "environment")

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._open_camera()
        # The camera is open from here on, so anything that fails below must
        # release it before propagating. Otherwise the capture leaks with no
        # reference left to clean it up, camera 0 stays busy for the life of
        # the process, and every later launch — of any test — fails with
        # "Cannot open camera (index 0)" as if the hardware were at fault.
        try:
            self._load_models()
            self._running = True
            self._error = None
            self._watchdog_reset()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._running = False
            self._thread = None
            cap, self.cap = self.cap, None
            _release_capture(cap)
            self._release_models()  # safe on a partially-loaded engine
            raise

    def stop(self):
        # Give the engine a chance to stage a partial/best-available result
        # before shutdown, if nothing reached on_stage yet — so screen3 always
        # has something to review. Never let staging block the stop.
        if not self._staged:
            try:
                self._stage_on_stop()
            except Exception:
                pass

        # self.cap must stay set until the loop thread is really gone —
        # _run_loop reads self.cap.read() every iteration.
        self._running = False
        thread, self._thread = self._thread, None

        if thread is not None and thread.is_alive():
            thread.join(timeout=self.STOP_JOIN_TIMEOUT)
            if thread.is_alive():
                # _run_loop is still inside cap.read()/inference. Releasing the
                # capture now is a use-after-free down in native OpenCV, which
                # takes the whole process down with no Python traceback. Hand
                # cleanup to a reaper that waits for the loop to actually exit;
                # the engine already reports as stopped either way.
                threading.Thread(target=self._reap, args=(thread,), daemon=True).start()
                return

        self._reap(thread)

    def _reap(self, thread):
        """Join the loop thread (if any), then release everything it was using."""
        if thread is not None:
            thread.join()
        cap, self.cap = self.cap, None
        _release_capture(cap)
        self._release_models()

    def is_running(self):
        return self._running

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_state(self):
        with self._lock:
            state = self._state_locked()
            state["running"] = self._running
            # A stall never overwrites a real failure — _error is the more
            # specific reason when both are true.
            stall = self._error or self._stall_error()
            if stall:
                state["error"] = stall
        return state

    # ── helpers used by subclasses ───────────────────────────────────────
    def _open_camera(self):
        self.cap = camera.open_capture(
            self.camera_source, width=self.FRAME_W, height=self.FRAME_H, fps=30)

    def _publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                self._frame_seq += 1
                self._watchdog_mark()

    def get_frame(self):
        """(jpeg_bytes|None, seq) — seq changes iff a new frame was published."""
        with self._lock:
            return self._jpeg, self._frame_seq

    def _safe_stage(self, *args):
        """Calls on_stage(*args) and records failures without killing the loop."""
        if self.on_stage:
            try:
                self.on_stage(*args)
                self._staged = True
            except Exception as e:
                with self._lock:
                    self._error = f"Stage failed: {e}"

    # ── hooks every subclass must/may override ──────────────────────────
    def _stage_on_stop(self):
        """Called by stop() when no result has been staged yet. Subclasses
        may stage a partial/best-available result via _safe_stage(). Reads
        only the engine's own attributes — runs on the caller's thread while
        the loop thread may still be live."""
        pass

    def _load_models(self):
        """Open MediaPipe detectors / build any per-test helper objects."""
        pass

    def _release_models(self):
        """Close whatever _load_models opened."""
        pass

    def _run_loop(self):
        """Background thread body: read frames, run detection, draw the
        HUD, call self._publish(frame) each iteration. Must set
        self._error and clear self._running on unrecoverable failure."""
        raise NotImplementedError

    def _state_locked(self):
        """Return the JSON-able dict screen4.html polls. Called with
        self._lock already held — read shared fields directly."""
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════
#  SECTION 1 — BMI / HEIGHT ESTIMATOR   (wraps bmi__3_.py)
# ════════════════════════════════════════════════════════════════════════
import bmi__3_ as bmi_mod  # noqa: E402  (reuse geometry / drawing / detector code)

bmi_mod.CALIBRATION_FILE = os.path.join(_TESTS_DIR, "calibration.txt")
_MODEL_BMI = os.path.join(_TESTS_DIR, bmi_mod.MODEL_FILENAME)

# ── BMI height-in-pixels tuning (fixes 1–3) ─────────────────────────────────
# These override bmi__3_.py's get_height_pixels() for the live web engine only;
# the desktop script is left untouched.
_BMI_MIN_VIS   = 0.60   # stricter than bmi_mod.MIN_VISIBILITY (0.55): a landmark
                        # must be clearly seen to count, so the reading is stable.
_BMI_CROWN_K   = 1.60   # crown sits ~1.6 eye→mouth spans above the eyes. This is
                        # a facial proportion (build-independent), replacing the
                        # old shoulder/hip-width head estimate that made broader
                        # people read taller.
_BMI_SMOOTH    = 20     # frames kept for the median (was 15) — more = steadier.
_BMI_STABLE_CV = 0.020  # the last _BMI_SMOOTH pixel heights must agree within
                        # ~2% (std/mean) before a reading is trusted for
                        # calibration or scanning. Rejects mid-move frames.

# MediaPipe Pose face landmark indices (not defined in bmi__3_.py).
_IDX_LEFT_EYE   = 2
_IDX_RIGHT_EYE  = 5
_IDX_MOUTH_L    = 9
_IDX_MOUTH_R    = 10


def _bmi_height_px(landmarks, frame_h, frame_w):
    """Build-independent standing height in pixels, or None if the pose isn't a
    clean, full-body view.

    Fix 1 — the crown (top of head) is estimated from FACE proportions
    (eye→mouth span) instead of shoulder/hip width, so body build no longer
    biases the height.
    Fix 2 — requires nose + both eyes + both mouth corners + BOTH heels at high
    visibility; otherwise returns None (no partial-body readings)."""
    def ok(idx):
        return landmarks[idx].visibility >= _BMI_MIN_VIS

    # Face must be clearly seen (crown estimate depends on it).
    if not all(ok(i) for i in (bmi_mod.IDX_NOSE, _IDX_LEFT_EYE, _IDX_RIGHT_EYE,
                               _IDX_MOUTH_L, _IDX_MOUTH_R)):
        return None
    # Both heels must be clearly seen (stable, consistent floor contact).
    if not (ok(bmi_mod.IDX_LEFT_HEEL) and ok(bmi_mod.IDX_RIGHT_HEEL)):
        return None

    eye_y   = (landmarks[_IDX_LEFT_EYE].y + landmarks[_IDX_RIGHT_EYE].y) * 0.5 * frame_h
    mouth_y = (landmarks[_IDX_MOUTH_L].y + landmarks[_IDX_MOUTH_R].y) * 0.5 * frame_h
    face_span = mouth_y - eye_y            # eyes → mouth, downward-positive
    if face_span <= 1:                     # face nearly edge-on / bad detection
        return None
    crown_y = eye_y - _BMI_CROWN_K * face_span

    bottom_y = max(landmarks[bmi_mod.IDX_LEFT_HEEL].y,
                   landmarks[bmi_mod.IDX_RIGHT_HEEL].y) * frame_h

    height = bottom_y - crown_y
    return height if height > 20 else None


class BmiEngine(BaseWebEngine):
    """Live BMI/height engine. Browser flow simplifies bmi__3_.py's
    type-the-digits-into-the-frame calibration into one CALIBRATE button +
    number field (same UX as wingspan's CALIBRATE), and auto-stages the
    result the moment a 5-second scan finishes (no separate SAVE button —
    matches the original script's own auto-save-on-scan behavior)."""

    # Calibration is stored as centimetres-per-pixel: the real height entered
    # during calibration divided by the subject's pixel height at that instant
    # (height_cm = pixel_height * cm_per_px). This is the same self-calibration
    # wingspan / vertical-jump / sit-reach already use — it needs no camera
    # distance and no lens field-of-view, so one calibration is correct at
    # whatever distance a school sets up, on whatever camera. A dedicated file
    # (not the old calibration.txt) is used so a stale scale-factor from the
    # previous distance+FOV method is never misread as a cm-per-pixel value.
    CALIB_FILE = os.path.join(_TESTS_DIR, "bmi_calibration_cmpx.txt")

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True,
                 distance_cm=None):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # distance_cm is still accepted so the launch endpoint's call signature
        # is unchanged, but it is no longer used for anything: known-height
        # calibration makes the camera distance irrelevant to the reading.
        self.distance_cm = distance_cm
        # Browser-sourced camera (works on phone AND laptop): the browser
        # captures the camera and POSTs JPEG frames to /api/bmi/frame ->
        # push_frame(); the loop consumes the newest one. No server-side
        # cv2.VideoCapture is opened, so BMI also runs on a hosted box with
        # no webcam. Same model wingspan already uses.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()
        self.landmarker = None
        self.cm_per_px = self._load_cmpx()   # None until a known height is entered
        self.px_buffer = []
        self.scan_active = False
        self.scan_start_time = None
        self.scan_readings = []
        self.smoothed_height = None
        self.px_median = None
        self.stable = False          # True once the pixel-height reading settles
        self.final_height = None
        self.last_save = None
        self.status_msg = ""

    @classmethod
    def _load_cmpx(cls):
        """Return the saved cm-per-pixel scale, or None if not calibrated."""
        try:
            with open(cls.CALIB_FILE) as f:
                value = float(f.read().strip())
            return value if value > 0 else None
        except (OSError, ValueError):
            return None

    @classmethod
    def _save_cmpx(cls, value):
        with open(cls.CALIB_FILE, "w") as f:
            f.write(str(round(value, 6)))

    def _open_camera(self):
        # Override BaseWebEngine._open_camera: no server webcam is opened.
        # Frames arrive from the browser via push_frame(), so self.cap stays
        # None (_release_capture(None) is a safe no-op on stop). This is what
        # lets BMI run on a phone and on a server with no camera.
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept; older unprocessed frames are dropped
        so the annotated feed stays near real-time. Returns True on decode."""
        if not self._running or not jpeg_bytes:
            return False
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._frame_lock:
            self._in_frame = frame
        self._frame_event.set()
        return True

    def _load_models(self):
        _ensure_pose_model(_MODEL_BMI, bmi_mod.MODEL_URL)
        options = bmi_mod.PoseLandmarkerOpts(
            base_options=bmi_mod.BaseOptions(model_asset_path=_MODEL_BMI),
            running_mode=bmi_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.70,
            min_pose_presence_confidence=0.70,
            min_tracking_confidence=0.60,
        )
        self.landmarker = bmi_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def _state_locked(self):
        scan_secs_left = 0.0
        if self.scan_active and self.scan_start_time:
            scan_secs_left = max(0.0, bmi_mod.SCAN_SECONDS - (time.time() - self.scan_start_time))
        return {
            "calibrated": self.cm_per_px is not None,
            "scale_factor": round(self.cm_per_px, 4) if self.cm_per_px else None,
            "pose_detected": self.px_median is not None,
            "stable": self.stable,
            "smoothed_cm": round(self.smoothed_height, 1) if self.smoothed_height else None,
            "scanning": self.scan_active,
            "scan_secs_left": round(scan_secs_left, 1),
            "final_height_cm": self.final_height,
            "last_save": self.last_save,
            "status_msg": self.status_msg,
        }

    # ── actions (buttons in screen4.html) ──────────────────────────────
    def action_scan(self):
        with self._lock:
            if self.scan_active:
                return {"ok": False, "error": "Scan already running"}
            if self.cm_per_px is None:
                return {"ok": False, "error": "Calibrate first — enter a known height"}
            if not self.smoothed_height:
                return {"ok": False, "error": "No pose detected — stand in frame first"}
            self.scan_active = True
            self.scan_start_time = time.time()
            self.scan_readings = []
            self.final_height = None
            return {"ok": True}

    def action_calibrate(self, real_height_cm):
        with self._lock:
            if not self.px_median:
                return {"ok": False, "error": "Stand fully in frame — head and both feet must be visible"}
            if not self.stable:
                return {"ok": False, "error": "Hold still — measurement still settling"}
            try:
                real_height_cm = float(real_height_cm)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Enter a number"}
            if not (50 < real_height_cm < 250):
                return {"ok": False, "error": "Enter a real height between 50 and 250 cm"}
            # cm-per-pixel learned directly from a known real height in the frame.
            self.cm_per_px = real_height_cm / self.px_median
            self._save_cmpx(self.cm_per_px)
            self.status_msg = f"Calibrated! {self.cm_per_px:.4f} cm/px"
            return {"ok": True, "scale_factor": round(self.cm_per_px, 4)}

    def action_reset(self):
        with self._lock:
            self.cm_per_px = None
            if os.path.exists(self.CALIB_FILE):
                os.remove(self.CALIB_FILE)
            self.status_msg = "Calibration reset"
            return {"ok": True}

    # ── background loop ─────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                # Frames come from the browser (push_frame), not a server
                # camera. Wait up to 1s for the newest one; on timeout loop
                # back so stop() (which flips self._running) stays responsive.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                fh, fw = frame.shape[:2]
                frame = camera.mirror(frame, self.mirror_preview)
                mp_image = bmi_mod.mp.Image(
                    image_format=bmi_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                result = self.landmarker.detect_for_video(mp_image, ts)

                with self._lock:
                    cm_per_px = self.cm_per_px
                    scan_active = self.scan_active

                px_median = None
                smoothed = None
                stable = False
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    bmi_mod.draw_skeleton(frame, landmarks, fh, fw)
                    # Fixes 1 & 2: build-independent crown + full-body/high-vis gate.
                    px_h = _bmi_height_px(landmarks, fh, fw)
                    if px_h:
                        # Pure known-height calibration: the subject's on-screen
                        # pixel height maps to real cm through cm_per_px, learned
                        # once from a person of known height. No camera distance
                        # and no lens FOV are involved, so the reading holds at
                        # any distance the camera is set up — provided the subject
                        # stands where the calibration was done.
                        self.px_buffer.append(px_h)
                        if len(self.px_buffer) > _BMI_SMOOTH:
                            self.px_buffer.pop(0)
                        px_median = float(np.median(self.px_buffer))
                        # Fix 3: only trust the reading once the buffer is full and
                        # the recent samples agree within _BMI_STABLE_CV (person is
                        # standing still). Rejects mid-move / settling frames.
                        if len(self.px_buffer) >= _BMI_SMOOTH:
                            arr = np.asarray(self.px_buffer, dtype=float)
                            mean = float(arr.mean())
                            if mean > 0 and float(arr.std()) / mean <= _BMI_STABLE_CV:
                                stable = True
                        if cm_per_px:
                            smoothed = px_median * cm_per_px
                            # Feed the scan only with settled frames.
                            if scan_active and stable:
                                with self._lock:
                                    self.scan_readings.append(smoothed)
                    else:
                        # Pose incomplete (partial body / low visibility) → drop
                        # stale samples so a fresh, clean stance must re-settle
                        # before it's trusted again.
                        self.px_buffer.clear()

                with self._lock:
                    self.px_median = px_median
                    self.smoothed_height = smoothed
                    self.stable = stable

                scan_secs_left = 0
                with self._lock:
                    if self.scan_active:
                        elapsed = time.time() - self.scan_start_time
                        scan_secs_left = max(0.0, bmi_mod.SCAN_SECONDS - elapsed)
                        if scan_secs_left <= 0:
                            self.scan_active = False
                            if len(self.scan_readings) >= 5:
                                # Median over the settled frames — robust to the
                                # occasional outlier frame.
                                final = round(float(np.median(self.scan_readings)), 1)
                                self.final_height = final
                                self.last_save = final
                                self.status_msg = f"Done! Final height: {final:.1f} cm"
                                self._safe_stage(final, self.cm_per_px is not None)
                            else:
                                self.status_msg = "Scan failed — hold still, full body in frame. Try again."
                            self.scan_readings = []

                bmi_mod.draw_hud(
                    frame, smoothed, [], self.status_msg,
                    cm_per_px if cm_per_px else 1.0,
                    False, "", scan_active, scan_secs_left, self.final_height,
                )
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PARTIAL CURL-UP COUNTER   (wraps parcurlup.py)
# ════════════════════════════════════════════════════════════════════════
import parcurlup as curlup_mod  # noqa: E402

_MODEL_CURLUP = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class CurlUpEngine(BaseWebEngine):
    """Live curl-up counter. Fullscreen stretching + the F key are dropped
    (the browser panel already fills the screen); everything else — the
    30s timer, hip-angle state machine, and draw_ui() HUD — is reused
    unchanged from parcurlup.py."""

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera (works on phone AND laptop): the browser
        # captures the camera and POSTs JPEG frames to /api/curl_up/frame ->
        # push_frame(); the loop consumes the newest one. No server-side
        # cv2.VideoCapture is opened, so curl-up also runs on a hosted box with
        # no webcam. Same model BMI/wingspan already use.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()
        self.landmarker = None
        self.count = 0
        self.phase = "up"
        self.start_time = None
        self.test_started = False
        self.test_done = False
        self.angle_buf = deque(maxlen=curlup_mod.ANGLE_BUFFER_LEN)
        self.shoulder_buf = deque(maxlen=curlup_mod.ANGLE_BUFFER_LEN)
        self.last_save = None

    def _open_camera(self):
        # Override BaseWebEngine._open_camera: no server webcam is opened.
        # Frames arrive from the browser via push_frame(), so self.cap stays
        # None (_release_capture(None) is a safe no-op on stop). Same approach
        # as BmiEngine — lets curl-up run on a phone and on a server with no
        # camera.
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept; older unprocessed frames are dropped
        so the annotated feed stays near real-time. Returns True on decode."""
        if not self._running or not jpeg_bytes:
            return False
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._frame_lock:
            self._in_frame = frame
        self._frame_event.set()
        return True

    def _load_models(self):
        _ensure_pose_model(_MODEL_CURLUP)
        base_opts = curlup_mod.mp_python.BaseOptions(model_asset_path=_MODEL_CURLUP)
        opts = curlup_mod.mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=curlup_mod.mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = curlup_mod.mp_vision.PoseLandmarker.create_from_options(opts)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def _state_locked(self):
        elapsed = (time.time() - self.start_time) if (self.test_started and self.start_time) else 0.0
        remaining = max(0.0, curlup_mod.TEST_DURATION_SEC - elapsed)
        return {
            "count": self.count,
            "phase": self.phase,
            "test_started": self.test_started,
            "test_done": self.test_done,
            "remaining_sec": round(remaining, 1),
            "last_save": self.last_save,
        }

    def action_start(self):
        with self._lock:
            if self.test_started or self.test_done:
                return {"ok": False, "error": "Test already started"}
            self.start_time = time.time()
            self.test_started = True
            return {"ok": True}

    def action_reset(self):
        with self._lock:
            self.count = 0
            self.phase = "up"
            self.start_time = None
            self.test_started = False
            self.test_done = False
            self.angle_buf.clear()
            self.shoulder_buf.clear()
            self.last_save = None
            # Clear the "a result already reached on_stage" flag too — otherwise
            # a partial second attempt stopped before the 30s timer would skip
            # _stage_on_stop() and leave the previous run's count staged.
            self._staged = False
            return {"ok": True}

    def _stage_on_stop(self):
        # Stopped before the 30s timer staged a result — stage the partial rep
        # count so screen3 has something to review.
        if self.count > 0:
            self._safe_stage(self.count)

    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                # Frames come from the browser (push_frame), not a server
                # camera. Wait up to 1s for the newest one; on timeout loop
                # back so stop() (which flips self._running) stays responsive.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so landmarks/overlays live in the same
                # flipped space — every self-facing test previews selfie-style
                # (subject's right on screen right), matching bmi/wingspan/etc.
                # Mirror-safe here: the count uses L/R landmark midpoints + angles.
                frame = camera.mirror(frame, self.mirror_preview)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = curlup_mod.mp.Image(image_format=curlup_mod.mp.ImageFormat.SRGB, data=rgb)
                # detect_for_video needs strictly increasing timestamps; browser
                # frame arrival is wall-clock, so derive ts from elapsed time and
                # bump on collision (mirrors BmiEngine).
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                result = self.landmarker.detect_for_video(mp_image, ts)

                angle = 0.0
                shoulder_rise = 0.0
                body_visible = False

                with self._lock:
                    test_started = self.test_started
                    test_done = self.test_done
                    start_time = self.start_time
                elapsed = (time.time() - start_time) if (test_started and start_time) else 0.0

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    # Side-view gate: a curl-up is filmed from the SIDE, so the
                    # far half of the body is occluded and those landmarks score
                    # low. Requiring BOTH sides (the old all-6 gate) can never
                    # pass a side view. Instead pick the NEAR side — whichever of
                    # left/right has its shoulder+hip+knee more visible — and gate
                    # on that side alone. Angle is then computed from that side's
                    # joints directly (not an L/R midpoint, which also needs both).
                    LEFT = (curlup_mod.IDX["L_SHOULDER"], curlup_mod.IDX["L_HIP"], curlup_mod.IDX["L_KNEE"])
                    RIGHT = (curlup_mod.IDX["R_SHOULDER"], curlup_mod.IDX["R_HIP"], curlup_mod.IDX["R_KNEE"])
                    side_min = lambda idxs: min(lms[i].visibility for i in idxs)
                    l_vis, r_vis = side_min(LEFT), side_min(RIGHT)
                    s_i, h_i, k_i = (LEFT if l_vis >= r_vis else RIGHT)
                    near_vis = max(l_vis, r_vis)
                    body_visible = near_vis > 0.5

                    if body_visible:
                        shoulder = curlup_mod.lm_px(lms, s_i, w, h)
                        hip = curlup_mod.lm_px(lms, h_i, w, h)
                        knee = curlup_mod.lm_px(lms, k_i, w, h)

                        self.angle_buf.append(curlup_mod.calc_angle(shoulder, hip, knee))
                        angle = float(np.median(list(self.angle_buf)))

                        self.shoulder_buf.append(shoulder[1] / h)
                        if len(self.shoulder_buf) == curlup_mod.ANGLE_BUFFER_LEN:
                            shoulder_rise = float(
                                np.max(list(self.shoulder_buf)) - np.min(list(self.shoulder_buf))
                            )

                        with self._lock:
                            if test_started and not test_done and elapsed < curlup_mod.TEST_DURATION_SEC:
                                if self.phase == "up" and angle <= curlup_mod.CURL_DOWN_ANGLE:
                                    self.phase = "down"
                                elif self.phase == "down" and angle >= curlup_mod.CURL_UP_ANGLE:
                                    if shoulder_rise >= curlup_mod.MIN_SHOULDER_RISE_FRAC:
                                        self.count += 1
                                    self.phase = "up"

                    curlup_mod.draw_pose(frame, lms, w, h)

                with self._lock:
                    if test_started and elapsed >= curlup_mod.TEST_DURATION_SEC and not self.test_done:
                        self.test_done = True
                        self.last_save = self.count
                        self._safe_stage(self.count)
                    count = self.count
                    phase = self.phase
                    test_done = self.test_done

                curlup_mod.draw_ui(frame, count, phase, angle, elapsed, shoulder_rise,
                                    body_visible, test_started, test_done, browser_mode=True)
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ALTERNATE-HAND WALL TOSS   (wraps wtt.py)
# ════════════════════════════════════════════════════════════════════════
import wtt as wtt_mod  # noqa: E402


class WallTossEngine(BaseWebEngine):
    """Live wall-toss counter. Reuses wtt.WallTossCounter wholesale — its
    ball detection, hand tracking, 5-state toss machine, and _draw() HUD
    all work unchanged since none of them call cv2.imshow/waitKey
    themselves. Click-to-calibrate ball colour (which needs a mouse
    position over the desktop window) is replaced by choosing one of the
    5 built-in colour presets from a button row."""

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        self.counter = None
        self.last_save = None
        # Browser-sourced camera (works on phone AND laptop): the browser
        # captures the camera and POSTs JPEG frames to /api/wall_toss/frame ->
        # push_frame(); the loop consumes the newest one. No server-side
        # cv2.VideoCapture is opened, so wall toss also runs on a hosted box
        # with no webcam. Same model bmi/curl_up/wingspan already use.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

    def _open_camera(self):
        # Override BaseWebEngine._open_camera: no server webcam is opened.
        # Frames arrive from the browser via push_frame(), so self.cap stays
        # None (_release_capture(None) is a safe no-op on stop).
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept; older unprocessed frames are dropped
        so the annotated feed stays near real-time. Returns True on decode."""
        if not self._running or not jpeg_bytes:
            return False
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._frame_lock:
            self._in_frame = frame
        self._frame_event.set()
        return True

    def set_facing(self, facing):
        # Also update the live counter so its handedness (Left/Right) swap
        # matches the new mirror state — the counter reads self.mirror per frame,
        # so no rebuild is needed.
        super().set_facing(facing)
        if self.counter is not None:
            self.counter.mirror = self.mirror_preview

    def _load_models(self):
        self.counter = wtt_mod.WallTossCounter(mirror=self.mirror_preview)

    def _release_models(self):
        if self.counter is not None:
            try:
                self.counter._detector.close()
            except Exception:
                pass

    def _state_locked(self):
        c = self.counter
        frames = getattr(self, "_diag_frames", 0)
        hits = getattr(self, "_diag_hits", 0)
        t0 = getattr(self, "_diag_t0", None)
        elapsed = (time.time() - t0) if t0 else 0.0
        proc_fps = round(frames / elapsed, 1) if elapsed > 0.5 else None
        hit_rate = round(hits / frames, 2) if frames else None
        left_seen = round(getattr(self, "_diag_left", 0) / frames, 2) if frames else None
        right_seen = round(getattr(self, "_diag_right", 0) / frames, 2) if frames else None
        return {
            "state": c.state,
            "toss_count": c.toss_count,
            "active": c.active,
            "done": c.done,
            "final_score": c.final_score,
            "ball_colour": c._preset_name,
            "status_msg": c.status_msg,
            "last_save": self.last_save,
            "proc_fps": proc_fps,
            "ball_hit_rate": hit_rate,
            "left_seen_rate": left_seen,
            "right_seen_rate": right_seen,
        }

    def action_start(self):
        with self._lock:
            self.counter.start()
            self.last_save = None
            # Diagnostics: measure the real processing rate + ball-detection hit
            # rate during the test, to see whether misses come from frame rate or
            # from detection dropouts.
            self._diag_frames = 0
            self._diag_hits = 0
            self._diag_left = 0
            self._diag_right = 0
            self._diag_t0 = time.time()
            return {"ok": True}

    def action_reset(self):
        with self._lock:
            self.counter.reset()
            self.last_save = None
            return {"ok": True}

    def action_set_preset(self, preset_num):
        with self._lock:
            try:
                preset_num = int(preset_num)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Invalid colour preset"}
            if preset_num not in wtt_mod.BALL_PRESETS:
                return {"ok": False, "error": "Invalid colour preset"}
            self.counter.set_preset(preset_num)
            return {"ok": True, "ball_colour": self.counter._preset_name}

    def _stage_on_stop(self):
        # Stopped before the 30s timer staged a result — stage the partial
        # catch count so screen3 has something to review.
        c = self.counter
        if c is not None and c.toss_count > 0:
            self._safe_stage(c.toss_count, c._preset_name)

    def _run_loop(self):
        try:
            while self._running:
                # Frames come from the browser (push_frame), not a server
                # camera. Wait up to 1s for the newest one; on timeout loop
                # back so stop() (which flips self._running) stays responsive.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue
                frame = camera.mirror(frame, self.mirror_preview)
                c = self.counter
                # Match the desktop's pixel-threshold tuning to this frame's
                # actual width (browser runs ~960px vs the 1280px desktop
                # capture the constants were tuned for).
                c.set_frame_scale(frame.shape[1])

                bc, br = c._detect_ball(frame)
                # Prediction-bridged update: keeps tracking a fast, motion-blurred
                # ball through brief detection dropouts (see WallTossCounter.ingest_ball).
                c.ingest_ball(bc, br)

                # Diagnostics: count processed frames + real ball detections.
                if c.active:
                    self._diag_frames = getattr(self, "_diag_frames", 0) + 1
                    if bc is not None:
                        self._diag_hits = getattr(self, "_diag_hits", 0) + 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                mp_res = c._process_hands(rgb, frame.shape)

                # Diagnostics: how often each hand is actually detected — a toss
                # can only be counted when the catching hand is visible.
                if c.active:
                    if "Left" in c.hands:
                        self._diag_left = getattr(self, "_diag_left", 0) + 1
                    if "Right" in c.hands:
                        self._diag_right = getattr(self, "_diag_right", 0) + 1

                with self._lock:
                    c._update_state()

                # c._draw() is what actually flips c.done / sets c.final_score
                # when the 30s timer expires, so the completion check has to run
                # AFTER it — checking before draw (as this did) never sees the
                # done edge, so the result was never auto-staged on natural
                # completion. self.last_save is the "already staged" guard; it's
                # reset to None on start()/reset() and set exactly once here.
                c._draw(frame, mp_res)

                with self._lock:
                    if c.done and self.last_save is None:
                        self.last_save = c.final_score
                        self._safe_stage(c.final_score, c._preset_name)

                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SIT & REACH TEST   (wraps snr.py)
# ════════════════════════════════════════════════════════════════════════
import snr as snr_mod  # noqa: E402

_MODEL_SNR = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class SitReachEngine(BaseWebEngine):
    """Live sit-and-reach engine. `leg_cm` (hip-to-ankle length) is a
    required launch parameter — same as the desktop version's --leg-cm
    flag / startup prompt — passed in via the constructor. One MARK
    button mirrors the original's dual-purpose SPACE key: first press
    sets the start position, second press locks the final result."""

    def __init__(self, session_id=None, on_stage=None, leg_cm=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera (works on phone AND laptop): the browser
        # captures the camera and POSTs JPEG frames to /api/sit_reach/frame ->
        # push_frame(); the loop consumes the newest one. No server-side
        # cv2.VideoCapture is opened, so sit & reach also runs on a hosted box
        # with no webcam. Same model bmi/curl_up/wingspan already use.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()
        self.leg_cm = leg_cm
        self.landmarker = None
        self.lx_sm = snr_mod.RollingMean(6)
        self.ly_sm = snr_mod.RollingMean(6)
        self.rx_sm = snr_mod.RollingMean(6)
        self.ry_sm = snr_mod.RollingMean(6)
        self.cal_sm = snr_mod.RollingMean(20)
        self.state = snr_mod.State.IDLE
        self.start_left = None
        self.start_right = None
        self.rvec_locked = np.array([1.0, 0.0])
        self._cur_rvec = np.array([1.0, 0.0])
        self.max_disp_px = None
        self.max_disp_cm = None
        self.result_disp_px = 0.0
        self.result_disp_cm = None
        self.px_per_cm = None
        self.last_save = None

    def _open_camera(self):
        # Override BaseWebEngine._open_camera: no server webcam is opened.
        # Frames arrive from the browser via push_frame(), so self.cap stays
        # None (_release_capture(None) is a safe no-op on stop). Same approach
        # as BmiEngine/CurlUpEngine — lets sit & reach run on a phone and on a
        # server with no camera.
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept; older unprocessed frames are dropped
        so the annotated feed stays near real-time. Returns True on decode."""
        if not self._running or not jpeg_bytes:
            return False
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._frame_lock:
            self._in_frame = frame
        self._frame_event.set()
        return True

    def _load_models(self):
        _ensure_pose_model(_MODEL_SNR)
        options = snr_mod.PoseLandmarkerOptions(
            base_options=snr_mod.BaseOptions(model_asset_path=_MODEL_SNR),
            running_mode=snr_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = snr_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def _state_locked(self):
        return {
            "state": self.state.name,
            "leg_cm": self.leg_cm,
            "calibrated": self.px_per_cm is not None,
            "disp_cm": round(self.result_disp_cm, 1) if self.result_disp_cm is not None else None,
            "max_disp_cm": round(self.max_disp_cm, 1) if self.max_disp_cm is not None else None,
            "last_save": self.last_save,
        }

    def action_mark(self):
        """SPACE-equivalent: IDLE → sets start position. MEASURING → locks
        the final result and stages it for review."""
        with self._lock:
            if self.state == snr_mod.State.IDLE:
                has_left = self.lx_sm.ready
                has_right = self.rx_sm.ready
                if not (has_left or has_right):
                    return {"ok": False, "error": "No hands visible yet — move into frame"}
                self.start_left = np.array([self.lx_sm.value, self.ly_sm.value]) if has_left else None
                self.start_right = np.array([self.rx_sm.value, self.ry_sm.value]) if has_right else None
                self.rvec_locked = self._cur_rvec.copy()
                self.max_disp_px = None
                self.max_disp_cm = None
                self.result_disp_px = 0.0
                self.result_disp_cm = None
                self.state = snr_mod.State.MEASURING
                return {"ok": True}

            if self.state == snr_mod.State.MEASURING:
                self.result_disp_px = self.max_disp_px if self.max_disp_px is not None else 0.0
                self.result_disp_cm = self.max_disp_cm
                self.state = snr_mod.State.LOCKED
                final_cm = self.result_disp_cm
                if final_cm is not None:
                    if final_cm >= 15:
                        rating = "Excellent"
                    elif final_cm >= 5:
                        rating = "Above average"
                    elif final_cm >= 0:
                        rating = "Average"
                    elif final_cm >= -10:
                        rating = "Below average"
                    else:
                        rating = "Poor — more stretching recommended"
                    self.last_save = round(final_cm, 1)
                    self._safe_stage(round(final_cm, 1), self.leg_cm, rating)
                return {"ok": True, "reach_cm": self.last_save}

            return {"ok": False, "error": "Already locked — tap RESET for a new trial"}

    def action_reset(self):
        with self._lock:
            self.state = snr_mod.State.IDLE
            self.start_left = None
            self.start_right = None
            self.max_disp_px = None
            self.max_disp_cm = None
            self.result_disp_px = 0.0
            self.result_disp_cm = None
            self.lx_sm.reset()
            self.ly_sm.reset()
            self.rx_sm.reset()
            self.ry_sm.reset()
            self.last_save = None
            return {"ok": True}

    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                # Frames come from the browser (push_frame), not a server
                # camera. Wait up to 1s for the newest one; on timeout loop
                # back so stop() (which flips self._running) stays responsive.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so landmarks/overlays live in the same
                # flipped space (selfie-style preview, consistent with the other
                # self-facing tests). Mirror-safe: both fingertips are tracked
                # symmetrically and the reach vector is locked in this same
                # flipped coordinate space at measurement start.
                frame = camera.mirror(frame, self.mirror_preview)
                fh, fw = frame.shape[:2]
                mp_image = snr_mod.mp.Image(
                    image_format=snr_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)

                left_pt = right_pt = None
                calib_h = calib_a = None
                with self._lock:
                    disp_px = self.result_disp_px
                    disp_cm = self.result_disp_cm
                    state = self.state

                if results.pose_landmarks:
                    lms = results.pose_landmarks[0]

                    seg_px = snr_mod.hip_ankle_px(lms, fw, fh)
                    if seg_px is not None and seg_px > 0 and self.leg_cm:
                        with self._lock:
                            self.px_per_cm = self.cal_sm.push(seg_px / self.leg_cm)
                    calib_h, calib_a = snr_mod.best_calib_segment(lms, fw, fh)

                    raw_left, raw_right, cur_rvec = snr_mod.fingers_raw(lms, fw, fh)
                    self._cur_rvec = cur_rvec
                    if raw_left is not None:
                        slx = self.lx_sm.push(raw_left[0])
                        sly = self.ly_sm.push(raw_left[1])
                        left_pt = (int(slx), int(sly))
                    if raw_right is not None:
                        srx = self.rx_sm.push(raw_right[0])
                        sry = self.ry_sm.push(raw_right[1])
                        right_pt = (int(srx), int(sry))

                    if state == snr_mod.State.MEASURING:
                        candidates = []
                        with self._lock:
                            start_left = self.start_left
                            start_right = self.start_right
                            rvec_locked = self.rvec_locked
                        if start_left is not None and self.lx_sm.ready:
                            cur = np.array([self.lx_sm.value, self.ly_sm.value])
                            candidates.append(snr_mod.signed_disp_2d(start_left, cur, rvec_locked))
                        if start_right is not None and self.rx_sm.ready:
                            cur = np.array([self.rx_sm.value, self.ry_sm.value])
                            candidates.append(snr_mod.signed_disp_2d(start_right, cur, rvec_locked))
                        if candidates:
                            disp_px = max(candidates)
                            disp_cm = (disp_px / self.px_per_cm) if self.px_per_cm else None
                            with self._lock:
                                if self.max_disp_px is None or disp_px > self.max_disp_px:
                                    self.max_disp_px = disp_px
                                    self.max_disp_cm = disp_cm

                    for i, j in snr_mod._POSE_CONNECTIONS:
                        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                            p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                            p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)

                start_left_pt = (
                    (int(self.start_left[0]), int(self.start_left[1]))
                    if self.start_left is not None else None
                )
                start_right_pt = (
                    (int(self.start_right[0]), int(self.start_right[1]))
                    if self.start_right is not None else None
                )

                snr_mod.draw_hud(
                    frame, state, self.leg_cm or 0.0, self.px_per_cm or 0.0,
                    calibrated=(self.px_per_cm is not None),
                    left_pt=left_pt, right_pt=right_pt,
                    start_left=start_left_pt, start_right=start_right_pt,
                    disp_px=disp_px, disp_cm=disp_cm,
                    max_disp_cm=self.max_disp_cm,
                    calib_hip=calib_h, calib_ank=calib_a,
                    frame_no=0, debug=False,
                )
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 5 / 6 — re-export the two engines that already existed, so
#  every live-feed engine in the app can be imported from ONE module.
#  Their code is untouched (still lives in its own file) to avoid any
#  regression risk to the two tests that already work in production.
# ════════════════════════════════════════════════════════════════════════
from wingspan_web_engine import WingspanEngine          # noqa: E402,F401
from vertical_jump_web_engine import VerticalJumpEngine  # noqa: E402,F401

__all__ = [
    "BaseWebEngine",
    "BmiEngine",
    "CurlUpEngine",
    "WallTossEngine",
    "SitReachEngine",
    "WingspanEngine",
    "VerticalJumpEngine",
]
