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
  2B. PullUpEngine       wraps pullup.py          (modified pull-up counter)
  2C. FullPullUpEngine   wraps pullup_full.py     (pull-up counter)
  3. WallTossEngine      wraps wtt.py             (alternate-hand wall toss)
  4. SitReachEngine      wraps snr.py             (sit & reach)
  4B. TrunkLiftEngine    wraps trunk_lift.py      (trunk lift height)
  4C. ReverseCurlEngine  wraps reverse_curl.py    (reverse curl pass/fail)
  4D. ShoulderStretchEngine  wraps shoulder_stretch.py  (shoulder stretch pass/fail)
  4E. IsometricPushUpEngine  wraps isometric_pushup.py  (isometric push-up hold time)
  4F. SeatedPushUpEngine     wraps seated_pushup.py     (seated push-up hold time)
  4G. TargetStretchEngine    wraps target_stretch.py    (target stretch, 0-2 per side)
  4H. DumbbellPressEngine    wraps dumbbell_press.py    (dumbbell press counter)
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
#  SECTION 2B — MODIFIED PULL-UP COUNTER   (wraps pullup.py)
# ════════════════════════════════════════════════════════════════════════
import pullup as pullup_mod  # noqa: E402

_MODEL_PULLUP = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class PullUpEngine(BaseWebEngine):
    """Live modified pull-up counter. There is no countdown: the FMS protocol
    is one trial scored on total correct reps, which ends by itself when the
    participant pauses for two seconds (pullup.PullUpCounter raises `stopped`,
    and the result is staged right then). The coach can still tap STOP / SAVE
    to end it early. Rep scoring and HUD are reused from pullup.py."""

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera (works on phone AND laptop): the browser
        # captures the camera and POSTs JPEG frames to /api/pull_up/frame ->
        # push_frame(); the loop consumes the newest one. Same model the other
        # single-camera tests use.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()
        self.landmarker = None
        self.counter = pullup_mod.PullUpCounter()
        self.start_time = None
        self.test_started = False
        self.last_save = None

    @property
    def count(self):
        return self.counter.count

    def _open_camera(self):
        # No server webcam — frames arrive from the browser via push_frame(),
        # so self.cap stays None (_release_capture(None) is a safe no-op).
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
        _ensure_pose_model(_MODEL_PULLUP)
        base_opts = pullup_mod.mp_python.BaseOptions(model_asset_path=_MODEL_PULLUP)
        opts = pullup_mod.mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=pullup_mod.mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = pullup_mod.mp_vision.PoseLandmarker.create_from_options(opts)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def _state_locked(self):
        elapsed = (time.time() - self.start_time) if (self.test_started and self.start_time) else 0.0
        return {
            "count": self.counter.count,
            "phase": self.counter.phase,
            "test_started": self.test_started,
            "test_done": self.counter.stopped,
            "elapsed_sec": round(elapsed, 1),
            "last_save": self.last_save,
        }

    def action_start(self):
        with self._lock:
            if self.test_started:
                return {"ok": False, "error": "Test already started"}
            self.start_time = time.time()
            self.test_started = True
            return {"ok": True}

    def action_reset(self):
        with self._lock:
            self.counter.reset()
            self.start_time = None
            self.test_started = False
            self.last_save = None
            # Clear the "already staged" flag so a second attempt stages its own
            # count instead of leaving the previous run's result staged.
            self._staged = False
            return {"ok": True}

    _hud = pullup_mod  # module providing draw_ui(); overridden by FullPullUpEngine

    def _score_pose(self, lms, w, h, test_started):
        """Feed one pose to the counter; returns whether the body was visible
        enough to score."""
        # Side-view gate: a modified pull-up is filmed from the SIDE, so the
        # far arm is occluded and scores low. near_arm() picks whichever arm
        # the camera sees best.
        (s_i, e_i, wr_i), near_vis = pullup_mod.near_arm(lms)
        # The arm carries the bar line, the elbow angle and the shoulder lift,
        # so it alone decides whether a frame can be scored. A frame without a
        # readable face still goes to the counter with chin=None: no rep can be
        # credited while the chin — half the rule — can't be seen, but the test
        # runs on instead of the blackout reading as a tracking dropout and
        # voiding the rep in progress.
        face_ok = all(lms[pullup_mod.IDX[k]].visibility > 0.5
                      for k in ("NOSE", "MOUTH_L", "MOUTH_R"))
        if near_vis <= 0.5:
            return False
        shoulder = pullup_mod.lm_px(lms, s_i, w, h)
        elbow = pullup_mod.lm_px(lms, e_i, w, h)
        wrist = pullup_mod.lm_px(lms, wr_i, w, h)
        angle = pullup_mod.calc_angle(shoulder, elbow, wrist)
        with self._lock:
            self.counter.update(
                angle,
                pullup_mod.chin_y(lms) if face_ok else None,
                pullup_mod.bar_y(lms, wr_i),
                lms[s_i].y,
                # Second signal: the chin is projected from the face, so it
                # doesn't decide a rep alone — the shoulders must also have
                # come up past the elbows.
                pullup_mod.shoulder_lift(shoulder, elbow),
                counting=test_started,
            )
        return True

    def _stage_on_stop(self):
        # Reached when the coach stops the test before the counter's own end
        # rule fired; the loop stages it itself when that rule ends the test
        # normally. A count of zero is staged like any other: a participant who
        # manages no pull-up has scored 0, and leaving it unstaged forces the
        # coach to either type a number the kid didn't earn or record nothing.
        # Only a trial that actually started counts — stopping a test that was
        # never begun must not stage a 0 nobody attempted.
        if self.test_started:
            with self._lock:
                self.last_save = self.counter.count
                elapsed = (time.time() - self.start_time) if self.start_time else 0.0
            self._safe_stage(self.counter.count, round(elapsed, 1))

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
                # flipped space. Mirror-safe: the count uses one near-side arm's
                # angle plus a vertical nose displacement, neither of which a
                # horizontal flip changes.
                frame = camera.mirror(frame, self.mirror_preview)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = pullup_mod.mp.Image(image_format=pullup_mod.mp.ImageFormat.SRGB, data=rgb)
                # detect_for_video needs strictly increasing timestamps; browser
                # frame arrival is wall-clock, so derive ts from elapsed time and
                # bump on collision (mirrors the other browser-camera engines).
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                result = self.landmarker.detect_for_video(mp_image, ts)

                body_visible = False

                with self._lock:
                    test_started = self.test_started
                    start_time = self.start_time
                elapsed = (time.time() - start_time) if (test_started and start_time) else 0.0

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    body_visible = self._score_pose(lms, w, h, test_started)
                    pullup_mod.draw_pose(frame, lms, w, h)

                with self._lock:
                    count = self.counter.count
                    phase = self.counter.phase
                    angle = self.counter.angle
                    gap = self.counter.gap
                    reached = self.counter.reached
                    stopped = self.counter.stopped
                    # The counter's own end rule (two-second pause here, dropping
                    # off the bar for the full pull-up) is how the test normally
                    # ends — stage the result the moment it fires, so screen3 has
                    # it even if the coach walks away without tapping STOP.
                    newly_stopped = stopped and self.last_save is None
                    if newly_stopped:
                        self.last_save = count
                if newly_stopped:
                    self._safe_stage(count, round(elapsed, 1))

                # Each HUD takes one extra argument the other's doesn't: the
                # modified pull-up reports whether they're hanging on the bar
                # yet, the full pull-up how far the shoulders are above the
                # elbows. Each counter only carries its own, so the other stays
                # absent rather than being passed an argument it can't take.
                extra = {}
                hung = getattr(self.counter, "hung", None)
                if hung is not None:
                    # Both of these are the modified pull-up counter's, so they
                    # travel together off the one check.
                    extra["hung"] = hung
                    extra["lifted"] = self.counter.lifted
                lift = getattr(self.counter, "lift", None)
                if lift is not None:
                    extra["lift"] = lift
                    extra["back_view"] = self.counter.back_view
                self._hud.draw_ui(frame, count, phase, angle, elapsed, gap, reached,
                                  body_visible, test_started, stopped=stopped,
                                  browser_mode=True, **extra)
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 2C — PULL-UP COUNTER   (wraps pullup_full.py)
# ════════════════════════════════════════════════════════════════════════
import pullup_full as pullup_full_mod  # noqa: E402


class FullPullUpEngine(PullUpEngine):
    """Live (full-hang) pull-up counter. Same browser-camera plumbing, START /
    RESET flow and staging as the modified pull-up; only the scoring differs:
    reps run full hang -> shoulders above the elbows -> full hang, and the test
    ends by itself when the participant drops off the bar (pullup_full raises
    `stopped`). Rep scoring and HUD are reused from pullup_full.py."""

    _hud = pullup_full_mod

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        self.counter = pullup_full_mod.FullPullUpCounter()

    def _score_pose(self, lms, w, h, test_started):
        visible, sample = pullup_full_mod.pose_sample(lms, w, h)
        if visible:
            with self._lock:
                self.counter.update(sample, counting=test_started)
        return visible


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
#  SECTION 4B — TRUNK LIFT   (wraps trunk_lift.py)
# ════════════════════════════════════════════════════════════════════════
import trunk_lift as tl_mod  # noqa: E402

_MODEL_TRUNK = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class TrunkLiftEngine(BaseWebEngine):
    """Live trunk-lift engine. The coach taps the two ends of the measuring
    stick on the video (`calib_click`), which fixes both the scale and the
    floor line; from then on every frame reports the chin height above the
    mat. Trials capture themselves when the height holds steady, and the
    MARK button captures manually when the coach would rather call it.

    All the geometry and the trial state machine live in trunk_lift.py —
    this class only replaces the desktop cv2.imshow/waitKey loop, the same
    way SitReachEngine wraps snr.py."""

    def __init__(self, session_id=None, on_stage=None, stick_cm=None,
                 trials=tl_mod.DEFAULT_TRIALS, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera, like sit & reach: frames are POSTed to
        # /api/trunk_lift/frame -> push_frame(), so no server webcam is
        # opened and the test runs on a phone or a camera-less host.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.stick_cm = float(stick_cm) if stick_cm else tl_mod.DEFAULT_STICK_CM
        self.landmarker = None
        self.tracker = tl_mod.TrunkLiftTracker(trials=trials)
        self.cx_sm = tl_mod.RollingMean(tl_mod.SMOOTH_FRAMES)
        self.cy_sm = tl_mod.RollingMean(tl_mod.SMOOTH_FRAMES)

        self.calib = None          # tl_mod.FloorCalibration once both ends are tapped
        self._calib_points = []    # raw taps, in frame pixels, like the shuttle-run engine
        self.height_cm = None      # latest chin height above the mat
        self.pose_detected = False
        self.last_save = None

    # ── camera plumbing (identical contract to SitReachEngine) ──────────
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept so the annotated feed stays live."""
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
        """Switching cameras flips the preview horizontally, which moves the
        stick to different pixels — the stored taps would then describe a
        floor line that is no longer under the participant. Drop the
        calibration and ask for two fresh taps."""
        was = self.mirror_preview
        super().set_facing(facing)
        if was != self.mirror_preview:
            with self._lock:
                self._clear_calibration_locked(
                    "Camera changed — tap RESET, then the stick ends again"
                    if self.tracker.trials else
                    "Camera changed — tap the stick ends again")

    def _load_models(self):
        _ensure_pose_model(_MODEL_TRUNK)
        options = tl_mod.PoseLandmarkerOptions(
            base_options=tl_mod.BaseOptions(model_asset_path=_MODEL_TRUNK),
            running_mode=tl_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = tl_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        best = self.tracker.best_cm
        return {
            "state": tl_mod.State.CALIBRATE.name if self.calib is None else self.tracker.state.name,
            "calibrated": self.calib is not None,
            "calib_points": len(self._calib_points),
            "px_per_cm": round(self.calib.px_per_cm, 2) if self.calib else None,
            "stick_cm": self.stick_cm,
            "pose_detected": self.pose_detected,
            "height_cm": round(self.height_cm, 1) if self.height_cm is not None else None,
            "height_in": round(tl_mod.to_inches(self.height_cm), 1) if self.height_cm is not None else None,
            "hold_pct": int(round(self.tracker.hold_progress() * 100)),
            "trials": [round(t, 1) for t in self.tracker.trials],
            "trials_required": self.tracker.trials_required,
            # The manual records to the nearest centimetre; the decimals stay
            # in `trials` so the coach can see what the cap or the rounding did.
            # best_exact_cm carries the unrounded value so screen4 can show
            # "11.7 cm -> records as 12 cm" — otherwise a best that doesn't
            # match either trial reads as an arithmetic bug.
            "best_cm": round(best) if best is not None else None,
            "best_exact_cm": round(best, 1) if best is not None else None,
            "best_in": round(tl_mod.to_inches(best), 1) if best is not None else None,
            "capped": bool(best is not None and best >= tl_mod.MAX_CM - 1e-6),
            "status_msg": self.tracker.status_msg,
            "last_save": self.last_save,
        }

    # ── actions (screen4 buttons / taps on the feed) ────────────────────
    def _refuse_locked(self, msg):
        """screen4 rewrites its status box from the polled state every tick,
        which wipes an action's error almost immediately. status_msg is part
        of that state, so put the refusal there too — it then stays on screen
        until the next capture or undo."""
        self.tracker.status_msg = msg
        return {"ok": False, "error": msg}

    def _clear_calibration_locked(self, msg=""):
        self.calib = None
        self._calib_points = []
        self.height_cm = None
        self.tracker.clear_hold()
        self.tracker.status_msg = msg

    def action_calib_click(self, value=None):
        """One tap on the video, in frame pixels — {x, y}. Two taps make the
        stick line: the first sets one end, the second completes it."""
        x = (value or {}).get("x")
        y = (value or {}).get("y")
        if x is None or y is None:
            return {"ok": False, "error": "calib_click needs {x, y}"}
        with self._lock:
            # Two taps never land in exactly the same pixels, so a fresh
            # calibration is a slightly different ruler — and the trials
            # already recorded were measured with the old one. Same rule as
            # action_stick_cm: make the coach clear them deliberately.
            if self.tracker.trials:
                return self._refuse_locked(
                    "Trials already recorded on this scale — tap RESET to recalibrate")
            if self.calib is not None:
                self._clear_calibration_locked()
            self._calib_points.append((float(x), float(y)))
            if len(self._calib_points) < 2:
                return {"ok": True, "count": 1}
            try:
                self.calib = tl_mod.FloorCalibration(
                    self._calib_points[0], self._calib_points[1], self.stick_cm)
            except ValueError as exc:
                self._calib_points = []
                return {"ok": False, "error": f"{exc} — tap both ends again"}
            self.tracker.status_msg = f"Calibrated: {self.calib.px_per_cm:.1f} px/cm"
            return {"ok": True, "count": 2, "px_per_cm": round(self.calib.px_per_cm, 2)}

    def action_calib_reset(self, value=None):
        with self._lock:
            if self.tracker.trials:
                return self._refuse_locked(
                    "Trials already recorded on this scale — tap RESET to recalibrate")
            self._clear_calibration_locked("Tap both ends of the stick")
            return {"ok": True}

    def action_stick_cm(self, value=None):
        """Change the reference length (the coach's stick isn't 30 cm).
        Rebuilds the calibration from the taps already made rather than
        making the coach tap again."""
        try:
            cm = float(value)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Stick length must be a number"}
        if not 5.0 <= cm <= 300.0:
            return {"ok": False, "error": "Stick length must be between 5 and 300 cm"}
        with self._lock:
            # Trials already captured were measured against the old scale.
            # Rebuilding it now would leave one test holding two different
            # units, so make the coach clear the trials deliberately.
            if self.tracker.trials:
                return self._refuse_locked(
                    "Trials already recorded on the old scale — tap RESET first")
            self.stick_cm = cm
            if len(self._calib_points) == 2:
                try:
                    self.calib = tl_mod.FloorCalibration(
                        self._calib_points[0], self._calib_points[1], cm)
                    # This is the one path that swaps the scale without ever
                    # passing through calib=None, so the loop never feeds a
                    # None height and never clears the window itself.
                    self.tracker.clear_hold()
                except ValueError as exc:
                    self._clear_calibration_locked(str(exc))
            return {"ok": True, "stick_cm": cm}

    def action_mark(self, value=None):
        """Capture the current height as a trial, for when the automatic
        hold doesn't fire but the coach saw a good one."""
        with self._lock:
            if self.calib is None:
                return {"ok": False, "error": "Tap both ends of the measuring stick first"}
            # tracker.mark() returns None in DONE without touching
            # status_msg, which still holds the last capture's success
            # line — reporting that as the error reads as a failure.
            if self.tracker.complete:
                return self._refuse_locked(
                    f"All {self.tracker.trials_required} trials recorded — "
                    "tap UNDO TRIAL to redo one, or RESET")
            captured = self.tracker.mark()
            if captured is None:
                return {"ok": False, "error": self.tracker.status_msg or "Nothing to capture"}
            self._maybe_stage_locked()
            return {"ok": True, "trial_cm": round(captured, 1),
                    "best_cm": round(self.tracker.best_cm)}

    def action_undo(self, value=None):
        with self._lock:
            if not self.tracker.undo():
                return {"ok": False, "error": "No trials to undo"}
            # The staged result described the trial that just went away, and
            # screen3 reads that staged copy — so correct it NOW rather than
            # at stop(), or the coach can save the value they just undid.
            self._staged = False
            self.last_save = None
            self._maybe_stage_locked(force=True)
            return {"ok": True, "trials": [round(t, 1) for t in self.tracker.trials]}

    def action_reset(self, value=None):
        with self._lock:
            self.tracker.reset()
            self.cx_sm.reset()
            self.cy_sm.reset()
            self.height_cm = None
            self.last_save = None
            self._staged = False
            return {"ok": True}

    def action(self, name, value=None):
        """Generic dispatcher, so app.py can forward screen4's {action, value}
        payload straight through the way the dual-feed engines do."""
        handler = {
            "calib_click": self.action_calib_click,
            "calib_reset": self.action_calib_reset,
            "stick_cm":    self.action_stick_cm,
            "mark":        self.action_mark,
            "undo":        self.action_undo,
            "reset":       self.action_reset,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _maybe_stage_locked(self, force=False):
        """Stage the best trial once the protocol's two are in. `force` also
        stages a shorter (or empty) set, which undo needs so the staged copy
        stops describing a trial that no longer exists. Called with
        self._lock held."""
        if not (self.tracker.complete or force):
            return
        best = self.tracker.best_cm
        if best is None:
            if force:
                # Every trial was undone: replace the staged result with an
                # empty one instead of leaving the old number for review.
                self.last_save = None
                self._safe_stage(None, None, None, self.stick_cm, None)
            return
        trials = [round(t, 1) for t in self.tracker.trials]
        self.last_save = round(best)
        # Rounded to the nearest cm, as the manual records it. No rating:
        # the age-banded zones aren't confirmed yet, so screen3 shows the
        # number alone rather than a label the coach would have to undo.
        self._safe_stage(self.last_save,
                         trials[0] if len(trials) > 0 else None,
                         trials[1] if len(trials) > 1 else None,
                         self.stick_cm, None)

    def _stage_on_stop(self):
        """Stopped mid-test: one good trial still beats sending screen3
        nothing, so stage whatever was captured."""
        with self._lock:
            if self._staged or not self.tracker.trials:
                return
            best = self.tracker.best_cm
            trials = [round(t, 1) for t in self.tracker.trials]
            self.last_save = round(best)
            self._safe_stage(self.last_save,
                             trials[0] if len(trials) > 0 else None,
                             trials[1] if len(trials) > 1 else None,
                             self.stick_cm, None)

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                # Frames come from the browser; a 1s timeout keeps stop()
                # responsive when the phone stops sending.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so landmarks, the coach's taps and
                # the overlay all live in the same pixel space.
                frame = camera.mirror(frame, self.mirror_preview)
                fh, fw = frame.shape[:2]
                mp_image = tl_mod.mp.Image(
                    image_format=tl_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)

                chin_pt = None
                height_cm = None
                pose_detected = bool(results.pose_landmarks)

                if pose_detected:
                    lms = results.pose_landmarks[0]
                    raw_chin = tl_mod.chin_point(lms, fw, fh)
                    if raw_chin is not None:
                        chin_pt = np.array([self.cx_sm.push(raw_chin[0]),
                                            self.cy_sm.push(raw_chin[1])])
                    else:
                        # Face lost: drop the smoothing history so the chin
                        # doesn't creep back from stale frames on re-detect.
                        self.cx_sm.reset()
                        self.cy_sm.reset()

                    for i, j in tl_mod._POSE_CONNECTIONS:
                        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                            p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                            p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)

                with self._lock:
                    calib = self.calib
                    if calib is not None and chin_pt is not None:
                        height_cm = calib.height_cm(chin_pt)
                    self.height_cm = height_cm
                    self.pose_detected = pose_detected
                    captured = self.tracker.update(height_cm)
                    if captured is not None:
                        self._maybe_stage_locked()
                    calib_pts = list(self._calib_points) if calib is None else None
                    stick_cm = self.stick_cm
                    # Drawn inside the lock: the HUD walks tracker.trials, and
                    # an UNDO arriving mid-draw would shrink that list between
                    # the length check and the index, killing the loop.
                    tl_mod.draw_hud(frame, self.tracker, calib, chin_pt, height_cm,
                                    calib_points=calib_pts, stick_cm=stick_cm)
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4C — REVERSE CURL   (wraps reverse_curl.py)
# ════════════════════════════════════════════════════════════════════════
import reverse_curl as rc_mod  # noqa: E402

_MODEL_REVERSE_CURL = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class ReverseCurlEngine(BaseWebEngine):
    """Live reverse-curl engine. Pass/fail on a single trial: the elbow must
    flex to rc_mod.UP_ANGLE_DEG or less, hold there for 2 seconds, and the
    weight must come back down to the thigh.

    Nothing to calibrate — the resting elbow angle is picked up automatically
    once the participant sits still with the weight on the thigh, so the coach
    only has to press Start. The camera decides PASS and FAIL by itself and
    nobody can overrule it. Stopping mid-attempt is a FAIL (one trial);
    stopping before any real attempt leaves it NOT COMPLETED, which screen3
    will not save.

    All the geometry and the state machine live in reverse_curl.py — this
    class only replaces the desktop cv2.imshow/waitKey loop, the same way
    TrunkLiftEngine wraps trunk_lift.py."""

    def __init__(self, session_id=None, on_stage=None, arm=rc_mod.ARM_AUTO,
                 up_angle=rc_mod.UP_ANGLE_DEG, hold_s=rc_mod.HOLD_S,
                 camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera, like trunk lift: frames are POSTed to
        # /api/reverse_curl/frame -> push_frame(), so no server webcam is
        # opened and the test runs on a phone or a camera-less host.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.landmarker = None
        self.tracker = rc_mod.ReverseCurlTracker(up_angle=up_angle, hold_s=hold_s)
        # requested_arm is the PERSON's side, as the coach chose it; the
        # selector works in MediaPipe's naming. See _to_person().
        self.requested_arm = arm
        self.selector = rc_mod.ArmSelector(forced=self._to_landmark(arm))
        self.smooth = rc_mod.MedianSmoother(rc_mod.SMOOTH_FRAMES)
        self._tracked_arm = None   # arm the tracker was last fed (MediaPipe naming)
        self.result_arm = None     # the person's side the verdict was about

        self.angle_deg = None
        self.pose_detected = False
        self.last_save = None

    # ── camera plumbing (identical contract to TrunkLiftEngine) ─────────
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept so the annotated feed stays live."""
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
        """Switching lens is a different view of the participant and flips the
        mirror, which swaps which MediaPipe side is the working arm. A rest
        baseline or an arm lock taken through the old lens describes nothing
        in the new one, so an undecided trial starts over. A decided one keeps
        its verdict — result_arm still names the right arm."""
        was = self.mirror_preview
        super().set_facing(facing)
        if was == self.mirror_preview:
            return
        with self._lock:
            self.selector = rc_mod.ArmSelector(forced=self._to_landmark(self.requested_arm))
            self._tracked_arm = None
            self.smooth.reset()
            if not self.tracker.resolved:
                self.tracker.reset()

    # MediaPipe names each side from the picture it is handed. Front-camera
    # frames are mirrored BEFORE inference (so the feed reads like a mirror),
    # which makes MediaPipe call the person's right arm "left". Everything
    # inside — the selector, the landmark indices — stays in MediaPipe's
    # naming; these convert at the edge, so the arm shown and saved — and one
    # pinned at launch — always means the person's own arm. (Wall toss corrects
    # its handedness for the same reason.)
    @staticmethod
    def _swap(arm):
        return {rc_mod.ARM_LEFT: rc_mod.ARM_RIGHT,
                rc_mod.ARM_RIGHT: rc_mod.ARM_LEFT}.get(arm, arm)

    def _to_person(self, arm):
        return self._swap(arm) if self.mirror_preview else arm

    def _to_landmark(self, arm):
        return self._swap(arm) if self.mirror_preview else arm

    def _person_arm(self):
        """The arm to show and save. Once the trial is decided this is the
        arm the verdict was about, even if the lens changes afterwards."""
        if self.tracker.resolved and self.result_arm is not None:
            return self.result_arm
        return self._to_person(self.selector.arm)

    def _load_models(self):
        _ensure_pose_model(_MODEL_REVERSE_CURL)
        options = rc_mod.PoseLandmarkerOptions(
            base_options=rc_mod.BaseOptions(model_asset_path=_MODEL_REVERSE_CURL),
            running_mode=rc_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = rc_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        tr = self.tracker
        return {
            "state": tr.state.name,
            "arm": self._person_arm(),
            "arm_locked": self.selector.locked,
            "requested_arm": self.requested_arm,
            "pose_detected": self.pose_detected,
            "angle_deg": round(self.angle_deg, 1) if self.angle_deg is not None else None,
            "baseline_deg": round(tr.baseline_deg, 1) if tr.baseline_deg is not None else None,
            "target_angle_deg": tr.up_angle,
            "hold_s": round(tr.hold_elapsed, 1),
            "hold_best_s": round(tr.hold_best_s, 1),
            "required_hold_s": tr.hold_s,
            "hold_pct": int(round(tr.hold_progress() * 100)),
            "peak_angle_deg": tr.best_peak_deg,
            "attempts": len(tr.attempts),
            "resolved": tr.resolved,
            "passed": bool(tr.passed) if tr.resolved else None,
            "verdict_by": tr.verdict_by,
            "status_msg": tr.status_msg,
            "last_save": self.last_save,
        }

    # ── actions (screen4 buttons) ───────────────────────────────────────
    def action_reset(self, value=None):
        with self._lock:
            self.tracker.reset()
            # Rebuilt, not reset: a forced arm must be re-translated for the
            # lens in use now, not the one it was chosen under.
            self.selector = rc_mod.ArmSelector(forced=self._to_landmark(self.requested_arm))
            self._tracked_arm = None
            self.smooth.reset()
            self.angle_deg = None
            self.result_arm = None
            self.last_save = None
            self._staged = False
            return {"ok": True}

    def action(self, name, value=None):
        """Generic dispatcher, so app.py can forward screen4's {action, value}
        payload straight through the way the other engines do."""
        handler = {
            "reset": self.action_reset,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _maybe_stage_locked(self):
        """Stage the camera's verdict once the trial resolves. Called with
        self._lock held."""
        if not self.tracker.resolved:
            return
        res = self.tracker.result()
        self.last_save = "Pass" if res["passed"] else "Fail"
        self.result_arm = self._to_person(self.selector.arm)
        self._safe_stage(res["passed"], self.result_arm, res["peak_angle_deg"],
                         res["hold_s"], self.last_save)

    def _stage_on_stop(self):
        """Stopped before the camera reached a verdict. tracker.finish()
        applies the manual's one-trial rule: a real attempt under way is a
        FAIL, a completed hold with the weight already back down is a PASS.
        If no real attempt was seen it stages NOT COMPLETED (passed=None) —
        most often the camera couldn't see the arm — which screen3 shows but
        refuses to save, so the test is run again."""
        with self._lock:
            if self._staged:
                return
            if self.tracker.finish() is not None:
                self._maybe_stage_locked()
                return
            res = self.tracker.result()
            self.result_arm = self._to_person(self.selector.arm)
            self.last_save = "Not completed"
            self._safe_stage(None, self.result_arm, res["peak_angle_deg"],
                             res["hold_s"], self.last_save)

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                # Frames come from the browser; a 1s timeout keeps stop()
                # responsive when the phone stops sending.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so the landmarks and the overlay
                # live in the same pixel space.
                frame = camera.mirror(frame, self.mirror_preview)
                fh, fw = frame.shape[:2]
                mp_image = rc_mod.mp.Image(
                    image_format=rc_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)

                arm_pts = None
                lms = None
                raw = {rc_mod.ARM_LEFT: None, rc_mod.ARM_RIGHT: None}
                pose_detected = bool(results.pose_landmarks)

                if pose_detected:
                    lms = results.pose_landmarks[0]
                    for i, j in rc_mod._POSE_CONNECTIONS:
                        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                            p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                            p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)

                    raw = {a: rc_mod.elbow_angle(lms, a, fw, fh)
                           for a in (rc_mod.ARM_LEFT, rc_mod.ARM_RIGHT)}

                with self._lock:
                    # Arm choice, smoothing and verdict in one shared step
                    # (rc_mod.step_arm), under the lock: RESET and a lens
                    # change replace the selector from other threads.
                    arm, angle, outcome = rc_mod.step_arm(
                        self.selector, self.tracker, self.smooth, raw, self._tracked_arm)
                    self._tracked_arm = arm
                    if angle is not None and lms is not None:
                        arm_pts = [rc_mod.lm_px(lms, i, fw, fh) for i in rc_mod.ARM_IDX[arm]]
                    self.angle_deg = angle
                    self.pose_detected = pose_detected
                    if outcome is not None:
                        self._maybe_stage_locked()
                    # Drawn inside the lock: the HUD reads tracker fields that
                    # a PASS/FAIL/RESET action could change mid-draw.
                    rc_mod.draw_hud(frame, self.tracker, arm=self._person_arm(),
                                    angle=angle, arm_pts=arm_pts)
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4D — SHOULDER STRETCH   (wraps shoulder_stretch.py)
# ════════════════════════════════════════════════════════════════════════
import shoulder_stretch as ss_mod  # noqa: E402

_MODEL_SHOULDER_POSE = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")
_MODEL_SHOULDER_HAND = os.path.join(_TESTS_DIR, "hand_landmarker.task")


class ShoulderStretchEngine(BaseWebEngine):
    """Live Brockport shoulder-stretch engine (figure 5.27). One trial,
    pass/fail: one arm reaches over the shoulder and down the back, the other
    comes up the back, and the item PASSES if the fingertips touch.

    The camera goes BEHIND the participant — the hands meet between the
    shoulder blades, so there is nothing to see from the front. Frames are
    POSTed from the browser to /api/shoulder_stretch/frame, the same as trunk
    lift and reverse curl, so no server webcam is opened.

    Unlike reverse curl there are no PASS / FAIL buttons: the engine decides
    this test. The contact metric (minimum fingertip-pair distance over the
    hands' own palm length) is measured and self-calibrating, so the only
    controls are the LEFT / RIGHT side pin and RESET. Stopping the test closes
    the single trial — an untouched one stages as a FAIL carrying the closest
    gap it reached, so screen3 can see how near it came.

    This engine owns TWO MediaPipe detectors (pose + hands), the way
    wingspan_web_engine.py does. All the geometry, the ownership gate and the
    state machine live in shoulder_stretch.py — this class only replaces the
    desktop cv2.imshow/waitKey loop."""

    def __init__(self, session_id=None, on_stage=None, side=ss_mod.SIDE_AUTO,
                 pass_ratio=ss_mod.PASS_RATIO, touch_frames=ss_mod.TOUCH_FRAMES,
                 camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera, like reverse curl.
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.landmarker = None
        self.hand_det = None
        self.tracker = ss_mod.ShoulderStretchTracker(pass_ratio=pass_ratio,
                                                     touch_frames=touch_frames)
        # The ONE side selector. It is created here and only ever changed in
        # place (pin / reset) — never replaced — and it is the single source
        # of "side" for the UI, the tracker and the staged row (C5).
        self.selector = ss_mod.SideVoter()
        # HUD-only smoother: the tracker is fed the RAW ratio (M9).
        self.smooth = ss_mod.MedianSmoother(ss_mod.SMOOTH_FRAMES)
        # Whose body the trial is measured against: seeds only when exactly
        # one body is in frame (the student alone at the start), then follows
        # that body — a coach stepping in is never measured (C1/C2/M8).
        self.participant_lock = ss_mod.ParticipantLock()

        self.gap_ratio = None       # latest smoothed ratio, in palm lengths
        self.pose_detected = False
        self.hands_detected = 0      # raw hands the detector returned
        self.hands_measurable = False  # both hands owned AND big enough to measure
        self.last_save = None

    # ── camera plumbing (identical contract to ReverseCurlEngine) ───────
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept so the annotated feed stays live."""
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
        """This test is NEVER mirrored, whichever camera sends the frames: it
        films the participant's BACK, not a selfie, and the side is read from
        where the raised elbow is in the picture (right arm = right of the
        picture from behind). Selfie-mirroring the laptop/front camera used to
        swap LEFT and RIGHT."""
        super().set_facing(facing)
        self.mirror_preview = False

    # ── models ──────────────────────────────────────────────────────────
    def _load_models(self):
        _ensure_pose_model(_MODEL_SHOULDER_POSE)
        # _ensure_pose_model's default URL is the pose model, so the hand
        # model is fetched by passing shoulder_stretch's own URL for it. Both
        # .task files already ship in tests/; this only matters on a fresh
        # checkout. (wingspan2.ensure_models would do it too, but it calls
        # sys.exit(1) on failure, which would take the server down.)
        _ensure_pose_model(_MODEL_SHOULDER_HAND,
                           url=ss_mod.MODEL_URLS["hand_landmarker.task"])

        pose_options = ss_mod.PoseLandmarkerOptions(
            base_options=ss_mod.BaseOptions(model_asset_path=_MODEL_SHOULDER_POSE),
            running_mode=ss_mod.VisionRunningMode.VIDEO,
            # 2, not 1: with num_poses=1 MediaPipe locks onto whichever body
            # it likes (measured: the coach on 98-100% of two-person frames).
            # ParticipantLock picks the participant out of these.
            num_poses=ss_mod.NUM_POSES,
            # 0.45, following wingspan2.build_pose_detector's production
            # values — NOT reverse_curl's 0.55. Seen from behind with the arms
            # folded up the back, 0.55 drops the pose on good frames. Do not
            # "fix" this back to 0.55 by analogy with the other pass/fail test.
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        self.landmarker = ss_mod.PoseLandmarker.create_from_options(pose_options)

        hand_options = ss_mod.HandLandmarkerOptions(
            base_options=ss_mod.BaseOptions(model_asset_path=_MODEL_SHOULDER_HAND),
            running_mode=ss_mod.VisionRunningMode.VIDEO,
            num_hands=2,
            # wingspan2.build_hand_detector's production values.
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.hand_det = ss_mod.HandLandmarker.create_from_options(hand_options)

    def _release_models(self):
        # Runs on the reaper thread after a timed-out stop() as well, so each
        # handle is guarded and nulled separately.
        for name in ("landmarker", "hand_det"):
            det = getattr(self, name, None)
            if det is not None:
                try:
                    det.close()
                except Exception:
                    pass
                setattr(self, name, None)

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        tr = self.tracker
        return {
            "state": tr.state.name,
            # The arm over the shoulder RIGHT NOW; it follows the student
            # when they switch, because BOTH sides must be banked.
            "side": self.selector.side,
            "left_done": tr.side_done(ss_mod.SIDE_LEFT),
            "right_done": tr.side_done(ss_mod.SIDE_RIGHT),
            "no_attempt": tr.no_attempt,
            "seconds_left": (round(tr.cap_remaining(), 1)
                             if tr.cap_remaining() is not None else None),
            "trial_cap_s": tr.cap_s,
            # False until a lone body has been seen; screen4 shows "Only the
            # student in frame to start" while it is False.
            "locked": self.participant_lock.locked,
            "pose_detected": self.pose_detected,
            # hands_detected is what the detector returned; hands_measurable
            # is whether those hands passed the ownership gate and were big
            # enough in frame to measure. Dropouts are normal in this pose, so
            # screen4 shows the difference rather than a bare "no hands".
            "hands_detected": self.hands_detected,
            "hands_measurable": self.hands_measurable,
            # Named gap_ratio (not "ratio") to match best_gap_ratio and the
            # column the row is saved under — screen4 renders both together.
            "gap_ratio": round(self.gap_ratio, 3) if self.gap_ratio is not None else None,
            "best_gap_ratio": (round(tr.best_gap_ratio, 3)
                               if tr.best_gap_ratio is not None else None),
            "threshold_ratio": tr.pass_ratio,
            "touch_frames": tr.touch_count,
            "required_frames": tr.touch_frames,
            "touch_pct": int(round(tr.touch_progress() * 100)),
            "frames_seen": tr.frames_seen,
            "resolved": tr.resolved,
            "passed": bool(tr.passed) if tr.resolved else None,
            "verdict_by": tr.verdict_by,
            "status_msg": tr.status_msg,
            "last_save": self.last_save,
        }

    # ── actions (screen4 buttons) ───────────────────────────────────────
    def action_side(self, value=None):
        """Kept so an old screen4 button (or a saved layout) does not 400 —
        but there is nothing to pin any more: BOTH sides must be banked
        inside one trial, so the arm is named per frame from the hands and
        follows the student when they switch."""
        return {"ok": False,
                "error": "This test scores both sides automatically - "
                         "no side to pin"}

    def action_end(self, value=None):
        """Close the single trial now (the desktop script's E key). This is
        NOT a coach verdict — the result is whatever the measurement earned,
        so an untouched trial ends as a FAIL carrying its closest gap. Stop
        does the same thing via _stage_on_stop; this lets the coach end the
        trial without tearing the feed down."""
        with self._lock:
            if self.tracker.resolved:
                return {"ok": False, "error": "Trial already decided — press RESET first"}
            self._maybe_stage_locked(force=True)
            return {"ok": True, "passed": bool(self.tracker.passed)}

    def action_reset(self, value=None):
        with self._lock:
            self.tracker.reset()
            self.selector.reset()
            self.smooth.reset()
            self.participant_lock.reset()
            self.gap_ratio = None
            self.hands_detected = 0
            self.hands_measurable = False
            self.last_save = None
            self._staged = False
            return {"ok": True}

    def action(self, name, value=None):
        """Generic dispatcher, so app.py can forward screen4's {action, value}
        payload straight through the way the other engines do. There is no
        pass/fail action: the engine decides this test. `end` closes the trial
        on whatever it measured; stopping the test does the same."""
        handler = {
            "side":  self.action_side,
            # `arm` accepted as an alias so a screen4 button block copied from
            # reverse curl still pins the side instead of 400-ing.
            "arm":   self.action_side,
            "end":   self.action_end,
            "reset": self.action_reset,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _maybe_stage_locked(self, force=False):
        """Stage the verdict once the trial resolves. Called with self._lock
        held. `force` closes an unresolved trial first, so what gets staged is
        always a real verdict rather than a half-finished state."""
        if not self.tracker.resolved:
            if not force:
                return
            self.tracker.end_trial()
        # The selector is the only source of "side": copy it onto the tracker
        # so result(), the HUD, the UI and the staged row can't disagree — a
        # pin or un-pin made since the last frame would otherwise be lost (C5).
        # Voting stops at the verdict, so this is the side AT the verdict.
        res = self.tracker.result()
        # Nothing was ever measurable: stage it as NOT COMPLETED (passed=None)
        # the way reverse curl does, so screen3 shows it and refuses to save a
        # fail the camera never actually saw.
        if self.tracker.no_attempt:
            self.last_save = "Not completed"
            self._safe_stage(None, None, None, None, None, None, self.last_save)
            return
        self.last_save = "Pass" if res["passed"] else "Fail"
        # Positional contract with app.py's _stage(passed, left_passed,
        # right_passed, best_gap_ratio, best_gap_left, best_gap_right, rating)
        # closure — keep the two in lockstep.
        self._safe_stage(res["passed"], res["left_passed"], res["right_passed"],
                         res.get("best_gap_ratio"), res.get("best_gap_left"),
                         res.get("best_gap_right"), self.last_save)

    def _stage_on_stop(self):
        """Stopped mid-test: send screen3 what the trial actually achieved
        rather than nothing. An unresolved trial stages as NOT passed, with
        the closest fingertip gap reached, so the coach can see how close it
        came — and so the 0.20 threshold can be retuned later from rows that
        include the near misses, not just the passes."""
        with self._lock:
            if self._staged:
                return
            self._maybe_stage_locked(force=True)

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                # Frames come from the browser; a 1s timeout keeps stop()
                # responsive when the phone stops sending.
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so the landmarks and the overlay
                # live in the same pixel space. With the back camera
                # (facing='environment') this is a no-op, which is the
                # intended setup for a shot taken from behind.
                # Never mirrored (see set_facing): the side is read from the
                # picture as filmed from behind.
                fh, fw = frame.shape[:2]
                mp_image = ss_mod.mp.Image(
                    image_format=ss_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                # One image, both graphs, the SAME timestamp — each keeps its
                # own strictly-increasing check in VIDEO mode.
                results = self.landmarker.detect_for_video(mp_image, ts)
                h_result = self.hand_det.detect_for_video(mp_image, ts)

                now = time.time()
                ratio = None        # RAW per-frame ratio -> the tracker
                shown = None        # smoothed ratio -> HUD / gap_ratio only
                tip_pts = None
                vote = None
                hand_lms_list = h_result.hand_landmarks or []
                hands_detected = len(hand_lms_list)

                # Whose body: matched to the participant lock (seeded only on
                # a lone body), or the last one for up to BODY_HOLD_S when the
                # pose detector returns nobody — hands are often found on
                # frames where the pose is not (M7). Under self._lock because
                # RESET clears the lock from the request thread.
                with self._lock:
                    fresh = self.participant_lock.update(
                        results.pose_landmarks, fw, fh, now)
                    body = self.participant_lock.body_for(now)
                pose_detected = body is not None

                if body is not None:
                    # Student on camera: arms the backstop that closes a trial
                    # where nothing is ever measurable. The 15 s trial clock
                    # still waits for a real attempt.
                    with self._lock:
                        self.tracker.mark_visible(now)
                    for i, j in ss_mod._POSE_CONNECTIONS:
                        if body[i].visibility >= 0.45 and body[j].visibility >= 0.45:
                            p1 = (int(body[i].x * fw), int(body[i].y * fh))
                            p2 = (int(body[j].x * fw), int(body[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)
                    if hand_lms_list:
                        ss_mod.draw_hand_skeleton(frame, hand_lms_list)
                    reading = ss_mod.frame_gap_ratio(hand_lms_list, body, fw, fh)
                    # Side votes only while the stretch itself is in view (a
                    # valid reading), and only from a body matched on THIS
                    # frame — warming up or facing the camera must not name it.
                    if reading is not None:
                        # The down-pointing hand came over the shoulder, so it
                        # names the arm; the elbows are only a fallback.
                        vote = ss_mod.side_from_hands(hand_lms_list, body, fw, fh)
                        if vote is None and fresh is not None:
                            vote = ss_mod.detect_side(fresh, fw, fh)
                    if reading is not None:
                        ratio = reading.ratio
                        shown = self.smooth.push(reading.ratio, now)
                        tip_pts = (reading.point_a, reading.point_b)
                # No smooth.reset() on an unmeasurable frame (M9): dropouts are
                # normal from behind, and the smoother's samples expire by age.

                with self._lock:
                    self.gap_ratio = shown
                    self.pose_detected = pose_detected
                    self.hands_detected = hands_detected
                    self.hands_measurable = ratio is not None
                    if not self.tracker.resolved:   # side frozen at the verdict
                        self.selector.update(vote)
                    side = self.selector.side
                    outcome = self.tracker.update(ratio, now=now, side=side)
                    if outcome is not None:
                        self._maybe_stage_locked()
                    # Drawn inside the lock: the HUD reads tracker fields that
                    # a SIDE/RESET action could change mid-draw.
                    ss_mod.draw_hud(frame, self.tracker, side=side,
                                    ratio=shown, tip_pts=tip_pts)
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4E — ISOMETRIC PUSH-UP   (wraps isometric_pushup.py)
# ════════════════════════════════════════════════════════════════════════
import isometric_pushup as ipu_mod  # noqa: E402

_MODEL_ISOMETRIC_PUSHUP = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class IsometricPushUpEngine(BaseWebEngine):
    """Live isometric push-up engine (Brockport fig. 5.14). One trial: the
    participant holds the UP position of a push-up and the score is the time
    held, to the nearest second, up to 40 s.

    Nothing to calibrate and nothing to press between Start and Stop: the
    clock starts once the camera has seen the correct position held still for
    a second, and stops by itself on a bend, sag, sway or the 40 s cap. The
    coach can correct the seconds on screen3. Stopping mid-hold ends the
    clock at the last moment the camera saw the position; stopping before
    the clock ever started stages nothing (NOT COMPLETED, not a zero).

    All the geometry and the state machine live in isometric_pushup.py —
    this class only replaces the desktop cv2.imshow/waitKey loop."""

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera: frames are POSTed to
        # /api/isometric_push_up/frame -> push_frame().
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.landmarker = None
        self.tracker = ipu_mod.IsometricPushUpTracker()
        self.pose_detected = False
        self.last_save = None

    # ── camera plumbing (identical contract to ReverseCurlEngine) ───────
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept so the annotated feed stays live."""
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
        """Switching lens flips the mirror, which swaps the sides and the
        forward/back direction the sway is measured in — a starting shape
        taken through the old lens describes nothing in the new one. So an
        undecided trial starts over; a finished one keeps its score."""
        was = self.mirror_preview
        super().set_facing(facing)
        if was == self.mirror_preview:
            return
        with self._lock:
            if not self.tracker.resolved:
                self.tracker.reset()

    def _load_models(self):
        _ensure_pose_model(_MODEL_ISOMETRIC_PUSHUP)
        options = ipu_mod.PoseLandmarkerOptions(
            base_options=ipu_mod.BaseOptions(model_asset_path=_MODEL_ISOMETRIC_PUSHUP),
            running_mode=ipu_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = ipu_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        tr = self.tracker
        m, b = tr.current, tr.baseline

        def r(d, k, n=0):
            return round(d[k], n) if d is not None else None

        return {
            "state": tr.state.name,
            "pose_detected": self.pose_detected,
            "clock_running": tr.state == ipu_mod.State.HOLDING,
            "elapsed_s": round(tr.elapsed_s, 1),
            "max_hold_s": tr.max_hold_s,
            "hold_pct": int(round(100 * tr.elapsed_s / tr.max_hold_s)) if tr.max_hold_s else 0,
            "elbow_deg": r(m, "elbow"), "elbow_start_deg": r(b, "elbow"),
            "bend_deg": r(m, "bend"), "bend_start_deg": r(b, "bend"),
            "knee_deg": r(m, "knee"), "knee_start_deg": r(b, "knee"),
            "resolved": tr.resolved,
            "seconds": tr.score_s if tr.resolved else None,
            "end_reason": tr.end_reason,
            "status_msg": tr.status_msg,
            "last_save": self.last_save,
        }

    # ── actions (screen4 buttons) ───────────────────────────────────────
    def action_reset(self, value=None):
        with self._lock:
            self.tracker.reset()
            self.last_save = None
            self._staged = False
            return {"ok": True}

    def action(self, name, value=None):
        """Generic dispatcher, so app.py can forward screen4's {action, value}
        payload straight through the way the other engines do."""
        handler = {
            "reset": self.action_reset,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _maybe_stage_locked(self):
        """Stage the score once the hold is over. Called with self._lock held."""
        if not self.tracker.resolved:
            return
        res = self.tracker.result()
        self.last_save = f"{res['hold_sec']} s"
        self._safe_stage(res["hold_sec"], res["held_exact_s"], res["end_reason"])

    def _stage_on_stop(self):
        """STOP pressed. A hold under way ends where the camera last saw the
        position; no hold started stages nothing, so screen3 shows no reading
        rather than a zero the student never earned."""
        with self._lock:
            if self._staged:
                return
            if self.tracker.finish() is not None:
                self._maybe_stage_locked()

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so the landmarks and the overlay
                # live in the same pixel space.
                frame = camera.mirror(frame, self.mirror_preview)
                fh, fw = frame.shape[:2]
                mp_image = ipu_mod.mp.Image(
                    image_format=ipu_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                now = time.time()
                ts = int((now - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)

                lms = results.pose_landmarks[0] if results.pose_landmarks else None
                if lms is not None:
                    for i, j in ipu_mod._POSE_CONNECTIONS:
                        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                            p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                            p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)
                sides = ipu_mod.read_sides(lms, fw, fh)

                with self._lock:
                    # Under the lock: RESET and a lens change replace tracker
                    # state from other threads.
                    outcome = self.tracker.update(sides, now=now)
                    self.pose_detected = lms is not None
                    if outcome is not None:
                        self._maybe_stage_locked()
                    ipu_mod.draw_hud(frame, self.tracker,
                                     ipu_mod.tracked_points(self.tracker, lms, fw, fh))
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4F — SEATED PUSH-UP   (wraps seated_pushup.py)
# ════════════════════════════════════════════════════════════════════════
import seated_pushup as spu_mod  # noqa: E402

_MODEL_SEATED_PUSHUP = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class SeatedPushUpEngine(BaseWebEngine):
    """Live seated push-up engine (Brockport fig. 5.21). One trial: the
    participant pushes down on blocks, armrests or wheels until the body is
    off the seat, and the score is the time held, to the nearest second, up
    to 20 s.

    Nothing to calibrate and nothing to press between Start and Stop: the
    camera learns where the hips sit, the clock starts once they are lifted
    and steady for a second, and it stops by itself when they come back down,
    an elbow bends, or at the 20 s cap. The coach can correct the seconds on
    screen3. Stopping mid-hold ends the clock at the last moment the camera
    saw the student up; stopping before the clock ever started stages
    nothing (NOT COMPLETED, not a zero).

    All the geometry and the state machine live in seated_pushup.py — this
    class only replaces the desktop cv2.imshow/waitKey loop."""

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # Browser-sourced camera: frames are POSTed to
        # /api/seated_push_up/frame -> push_frame().
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.landmarker = None
        self.tracker = spu_mod.SeatedPushUpTracker()
        self.pose_detected = False
        self.last_save = None

    # ── camera plumbing (identical contract to IsometricPushUpEngine) ───
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept so the annotated feed stays live."""
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
        """Switching lens flips the mirror, which swaps the left and right
        arm — starting elbow angles taken through the old lens would be
        compared with the wrong arm. So an undecided trial starts over; a
        finished one keeps its score."""
        was = self.mirror_preview
        super().set_facing(facing)
        if was == self.mirror_preview:
            return
        with self._lock:
            if not self.tracker.resolved:
                self.tracker.reset()

    def _load_models(self):
        _ensure_pose_model(_MODEL_SEATED_PUSHUP)
        options = spu_mod.PoseLandmarkerOptions(
            base_options=spu_mod.BaseOptions(model_asset_path=_MODEL_SEATED_PUSHUP),
            running_mode=spu_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = spu_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        tr = self.tracker
        m, b = tr.current, tr.baseline

        def r(d, k, n=0):
            return round(d[k], n) if d is not None and d.get(k) is not None else None

        return {
            "state": tr.state.name,
            "pose_detected": self.pose_detected,
            "body_tracked": m is not None,
            "clock_running": tr.state == spu_mod.State.HOLDING,
            "elapsed_s": round(tr.elapsed_s, 1),
            "max_hold_s": tr.max_hold_s,
            "hold_pct": int(round(100 * tr.elapsed_s / tr.max_hold_s)) if tr.max_hold_s else 0,
            "seat_found": tr.seat_found,
            "lift": round(tr.lift, 2) if tr.lift is not None else None,
            "lift_needed": spu_mod.LIFT_MIN_TORSO,
            "lift_stop": round(tr.sink_line, 2) if tr.sink_line is not None else None,
            "elbow_left_deg": r(m, "elbow_left"), "elbow_left_start_deg": r(b, "elbow_left"),
            "elbow_right_deg": r(m, "elbow_right"), "elbow_right_start_deg": r(b, "elbow_right"),
            "resolved": tr.resolved,
            "seconds": tr.score_s if tr.resolved else None,
            "end_reason": tr.end_reason,
            "status_msg": tr.status_msg,
            "last_save": self.last_save,
        }

    # ── actions (screen4 buttons) ───────────────────────────────────────
    def action_reset(self, value=None):
        with self._lock:
            self.tracker.reset()
            self.last_save = None
            self._staged = False
            return {"ok": True}

    def action(self, name, value=None):
        """Generic dispatcher, so app.py can forward screen4's {action, value}
        payload straight through the way the other engines do."""
        handler = {
            "reset": self.action_reset,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _maybe_stage_locked(self):
        """Stage the score once the hold is over. Called with self._lock held."""
        if not self.tracker.resolved:
            return
        res = self.tracker.result()
        self.last_save = f"{res['hold_sec']} s"
        self._safe_stage(res["hold_sec"], res["held_exact_s"], res["end_reason"])

    def _stage_on_stop(self):
        """STOP pressed. A hold under way ends where the camera last saw the
        student up; no hold started stages nothing, so screen3 shows no
        reading rather than a zero the student never earned."""
        with self._lock:
            if self._staged:
                return
            if self.tracker.finish() is not None:
                self._maybe_stage_locked()

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so the landmarks and the overlay
                # live in the same pixel space.
                frame = camera.mirror(frame, self.mirror_preview)
                fh, fw = frame.shape[:2]
                mp_image = spu_mod.mp.Image(
                    image_format=spu_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                now = time.time()
                ts = int((now - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)

                lms = results.pose_landmarks[0] if results.pose_landmarks else None
                if lms is not None:
                    for i, j in spu_mod._POSE_CONNECTIONS:
                        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                            p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                            p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)
                body = spu_mod.body_metrics(lms, fw, fh)

                with self._lock:
                    # Under the lock: RESET and a lens change replace tracker
                    # state from other threads.
                    outcome = self.tracker.update(body, now=now)
                    self.pose_detected = lms is not None
                    if outcome is not None:
                        self._maybe_stage_locked()
                    spu_mod.draw_hud(frame, self.tracker,
                                     spu_mod.tracked_points(self.tracker, lms, fw, fh))
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4G — TARGET STRETCH   (wraps target_stretch.py)
# ════════════════════════════════════════════════════════════════════════
import target_stretch as tst_mod  # noqa: E402

_MODEL_TST_POSE = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")
_MODEL_TST_HAND = os.path.join(_TESTS_DIR, "hand_landmarker.task")


class TargetStretchEngine(BaseWebEngine):
    """Live Brockport target stretch engine (form 5.1). The coach picks ONE
    joint before START; the run measures it on the LEFT side and then the
    RIGHT side. Each side's score is the best angle held for 1 second inside
    its 10 seconds, rounded to 15 degrees and scored 0/1/2 from table 5.3.

    The engine decides the scores - there is no coach override. The only
    controls are NEXT (end the side under way now) and RESET. Stopping
    mid-run stages whatever sides were measured; nothing measured stages
    nothing.

    Owns two MediaPipe detectors (pose + hands), like ShoulderStretchEngine;
    the hand detector is only created for the joints that need it (wrist
    extension, supination, pronation). All the geometry and the state machine
    live in target_stretch.py - this class only replaces the desktop loop."""

    def __init__(self, session_id=None, on_stage=None, joint=tst_mod.DEFAULT_JOINT,
                 camera_source=0, mirror=True):
        super().__init__(session_id, on_stage, camera_source, mirror)
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.joint = tst_mod.JOINTS[joint]
        self.landmarker = None
        self.hand_det = None
        self.tracker = tst_mod.TargetStretchTracker(joint)
        self.meter = tst_mod.JointMeter(joint)
        self.pose_detected = False
        self.last_save = None

    # ── camera plumbing (same contract as IsometricPushUpEngine) ────────
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
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
        """Switching lens flips the mirror, and the mirror decides which side
        of the picture is the student's left. An undecided run starts over;
        a finished one keeps its scores."""
        was = self.mirror_preview
        super().set_facing(facing)
        if was == self.mirror_preview:
            return
        with self._lock:
            if not self.tracker.resolved:
                self.tracker.reset()
                self.meter.reset()

    def _load_models(self):
        _ensure_pose_model(_MODEL_TST_POSE)
        self.landmarker = tst_mod.PoseLandmarker.create_from_options(
            tst_mod.PoseLandmarkerOptions(
                base_options=tst_mod.BaseOptions(model_asset_path=_MODEL_TST_POSE),
                running_mode=tst_mod.VisionRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            ))
        if self.joint.hand:
            _ensure_pose_model(_MODEL_TST_HAND,
                               url=tst_mod.MODEL_URLS["hand_landmarker.task"])
            self.hand_det = tst_mod.HandLandmarker.create_from_options(
                tst_mod.HandLandmarkerOptions(
                    base_options=tst_mod.BaseOptions(model_asset_path=_MODEL_TST_HAND),
                    running_mode=tst_mod.VisionRunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                ))

    def _release_models(self):
        for name in ("landmarker", "hand_det"):
            det = getattr(self, name, None)
            if det is not None:
                try:
                    det.close()
                except Exception:
                    pass
                setattr(self, name, None)

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        tr = self.tracker
        j = self.joint

        def rnd(v):
            return None if v is None else int(round(v))

        best = tst_mod.side_result(j, tr.best) if tr.best is not None else None
        left = tr.time_left()
        state = {
            "state": tr.phase.name,
            "joint": j.key,
            "joint_title": j.title,
            "view": j.view,
            "straight": j.straight,
            "score2_deg": j.score2,
            "score1_deg": j.score1,
            "side": tr.side,
            "side_num": tr.side_idx + 1 if tr.side else None,
            "pose_detected": self.pose_detected,
            "current_deg": rnd(tr.current),
            "held_deg": rnd(tr.held),
            "best_deg": best.deg if best else None,
            "best_score": best.score if best else None,
            "seconds_left": None if left is None else int(math.ceil(left)),
            "side_window_s": int(tr.side_window_s),
            "resolved": tr.resolved,
            "status_msg": tr.status_msg,
            "last_save": self.last_save,
        }
        for s in tst_mod.SIDES:
            r = tr.results[s]
            state[f"{s}_deg"] = r.deg if r else None
            state[f"{s}_score"] = r.score if r else None
            state[f"{s}_done"] = tr.end_reasons[s] is not None
        return state

    # ── actions (screen4 buttons) ───────────────────────────────────────
    def action_reset(self, value=None):
        with self._lock:
            self.tracker.reset()
            self.meter.reset()
            self.last_save = None
            self._staged = False
            return {"ok": True}

    def action_next(self, value=None):
        """End the side under way now, on the best it has held so far."""
        with self._lock:
            if self.tracker.resolved:
                return {"ok": False, "error": "Both sides are already done"}
            if self.tracker.next_side() is not None:
                self._maybe_stage_locked()
            return {"ok": True}

    def action(self, name, value=None):
        handler = {
            "reset": self.action_reset,
            "next": self.action_next,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _maybe_stage_locked(self):
        """Stage once the run is over and at least one side was measured.
        Called with self._lock held."""
        if not self.tracker.resolved or not self.tracker.measured_any:
            return
        res = self.tracker.result()
        self.last_save = " / ".join(
            f"{s[0].upper()} {res[s + '_score']}" if res[s + "_score"] is not None
            else f"{s[0].upper()} -" for s in tst_mod.SIDES)
        self._safe_stage(res)

    def _stage_on_stop(self):
        """STOP pressed: the side under way closes on what it has held, a side
        not reached stays unmeasured. Nothing measured stages nothing."""
        with self._lock:
            if self._staged:
                return
            if self.tracker.finish() is not None:
                self._maybe_stage_locked()

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so the landmarks and the overlay
                # share one pixel space. The meter is told which, because the
                # mirror decides which side of the picture is the student's left.
                mirrored = self.mirror_preview
                frame = camera.mirror(frame, mirrored)
                fh, fw = frame.shape[:2]
                mp_image = tst_mod.mp.Image(
                    image_format=tst_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                now = time.time()
                ts = int((now - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)
                lms = results.pose_landmarks[0] if results.pose_landmarks else None
                hands = []
                if self.hand_det is not None:
                    hands = self.hand_det.detect_for_video(mp_image, ts).hand_landmarks or []

                tst_mod.draw_pose(frame, lms)
                tst_mod.draw_hand_points(frame, hands)

                with self._lock:
                    # Under the lock: RESET, NEXT and a lens change touch the
                    # tracker from other threads.
                    self.pose_detected = lms is not None
                    segments = []
                    if not self.tracker.resolved:
                        rd = self.meter.measure(lms, hands, fw, fh, self.tracker.side, mirrored,
                                                now=now)
                        segments = rd.segments
                        if self.tracker.update(rd.extent, now=now, problem=rd.problem) is not None:
                            self._maybe_stage_locked()
                    tst_mod.draw_hud(frame, self.tracker, segments)
                self._publish(frame)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4H — DUMBBELL PRESS   (wraps dumbbell_press.py)
# ════════════════════════════════════════════════════════════════════════
import dumbbell_press as dp_mod  # noqa: E402

_MODEL_DUMBBELL_PRESS = os.path.join(_TESTS_DIR, "pose_landmarker_full.task")


class DumbbellPressEngine(BaseWebEngine):
    """Live dumbbell-press counter. One trial, max 50 reps: the coach taps
    START on "GO", each press that gets the hand well above the head counts
    (hand height, not the elbow angle — see dumbbell_press.py), and the test ends by itself at 50 reps or after a 4 s
    rest (dp_mod.DumbbellPressTest raises the end and the count is staged
    right then). STOP ends it early. In AUTO both arms are watched until the
    first rep locks the working one; the coach can pin LEFT / RIGHT (the
    student's own hand) instead, so a free hand raised first can't take
    over. Rep logic and HUD live in dumbbell_press.py."""

    ARM_CHOICES = (dp_mod.ARM_AUTO, dp_mod.ARM_LEFT, dp_mod.ARM_RIGHT)

    def __init__(self, session_id=None, on_stage=None, camera_source=0, mirror=True,
                 arm=dp_mod.ARM_AUTO):
        super().__init__(session_id, on_stage, camera_source, mirror)
        # The PERSON's side, as the coach picked it. Translated to MediaPipe's
        # naming on every frame, so a lens change can't swap it.
        self.requested_arm = arm if arm in self.ARM_CHOICES else dp_mod.ARM_AUTO
        # Browser-sourced camera: frames are POSTed to
        # /api/dumbbell_press/frame -> push_frame().
        self._frame_lock = threading.Lock()
        self._in_frame = None
        self._frame_event = threading.Event()

        self.landmarker = None
        self.test = dp_mod.DumbbellPressTest()
        self.pose_detected = False
        self.arm_tracked = False
        self.last_save = None

    @property
    def count(self):
        return self.test.count

    # ── camera plumbing (identical contract to ReverseCurlEngine) ───────
    def _open_camera(self):
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

    def push_frame(self, jpeg_bytes):
        """Accept one browser-captured JPEG frame and hand it to the loop.
        Only the newest frame is kept so the annotated feed stays live."""
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

    # Front-camera frames are mirrored before inference, which makes
    # MediaPipe call the person's right arm "left". The test works in
    # MediaPipe's naming; _person() converts at the edge, so the arm shown
    # and saved is the person's own.
    @staticmethod
    def _swap(arm):
        return {dp_mod.ARM_LEFT: dp_mod.ARM_RIGHT,
                dp_mod.ARM_RIGHT: dp_mod.ARM_LEFT}.get(arm, arm)

    def _person(self, arm):
        return self._swap(arm) if self.mirror_preview else arm

    def set_facing(self, facing):
        """Switching lens flips the mirror, so each arm's MediaPipe name
        swaps. Swap the test's per-arm state with it, so a lens change
        mid-set keeps the count on the same physical arm."""
        was = self.mirror_preview
        super().set_facing(facing)
        if was == self.mirror_preview:
            return
        with self._lock:
            t = self.test
            t.counters = {a: t.counters[self._swap(a)] for a in dp_mod.ARMS}
            for c in t.counters.values():
                c.lost()
            t.arm = self._swap(t.arm)

    def _load_models(self):
        _ensure_pose_model(_MODEL_DUMBBELL_PRESS)
        options = dp_mod.PoseLandmarkerOptions(
            base_options=dp_mod.BaseOptions(model_asset_path=_MODEL_DUMBBELL_PRESS),
            running_mode=dp_mod.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self.landmarker = dp_mod.PoseLandmarker.create_from_options(options)

    def _release_models(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    # ── state for screen4 ───────────────────────────────────────────────
    def _state_locked(self):
        t = self.test
        arm = t.shown_arm()
        c = t.counters[arm] if arm else None
        return {
            "count": t.count,
            "max_reps": t.max_reps,
            "test_started": t.started,
            "test_done": t.stopped,
            "end_reason": t.end_reason,
            "elapsed_sec": round(t.elapsed(), 1),
            "arm": self._person(arm),
            "arm_locked": t.arm is not None,
            "requested_arm": self.requested_arm,
            "phase": c.phase if c else None,
            "wrist_up": round(c.wrist_up, 2) if (c and c.wrist_up is not None) else None,
            "wrist_up_needed": dp_mod.WRIST_UP_UA,
            "rest_s": round(t.rest_elapsed, 1),
            "rest_limit_s": t.rest_s,
            "short_presses": t.short_presses,
            "pose_detected": self.pose_detected,
            "arm_tracked": self.arm_tracked,
            "status_msg": t.status_msg,
            "turned": t.turned,
            "turn_deg": round(t.turn_deg) if t.turn_deg is not None else None,
            "no_room": t.no_room,
            "room_ua": round(t.room_ua, 2) if t.room_ua is not None else None,
            "last_save": self.last_save,
        }

    # ── actions (screen4 buttons) ───────────────────────────────────────
    def action_start(self, value=None):
        with self._lock:
            if not self.test.start():
                return {"ok": False, "error": "Test already started"}
            return {"ok": True}

    def action_arm(self, value=None):
        """The coach pins the arm (or goes back to auto). Only before START:
        changing it mid-trial would move a count between hands. RESET first."""
        arm = str(value or "").strip().lower()
        if arm not in self.ARM_CHOICES:
            return {"ok": False, "error": "Pick auto, left or right"}
        with self._lock:
            if self.test.started:
                return {"ok": False, "error": "Press RESET before changing the arm"}
            self.requested_arm = arm
            return {"ok": True}

    def action_reset(self, value=None):
        with self._lock:
            self.test.reset()
            self.last_save = None
            # A second attempt stages its own count.
            self._staged = False
            return {"ok": True}

    def action(self, name, value=None):
        handler = {
            "start": self.action_start,
            "reset": self.action_reset,
            "arm": self.action_arm,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown action: {name}"}
        return handler(value)

    # ── staging ─────────────────────────────────────────────────────────
    def _stage_locked(self):
        # The loop refreshes the pin every frame; do it here too, so a 0
        # staged before any frame still records the coach's arm.
        pinned = self._person(self.requested_arm)
        self.test.pinned = pinned if pinned in dp_mod.ARMS else None
        res = self.test.result()
        self.last_save = res["reps"]
        self._safe_stage(res["reps"], self._person(res["arm"]),
                         res["duration_sec"], res["end_reason"])

    def _stage_on_stop(self):
        # Coach stopped before the test ended by itself. A 0 is staged like
        # any other count (a student who can't press the weight scored 0),
        # but only for a trial that actually started.
        with self._lock:
            if self._staged or not self.test.started:
                return
            self.test.stop(dp_mod.END_STOPPED)
            self._stage_locked()

    # ── loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        stream_start = time.time()
        last_ts = -1
        try:
            while self._running:
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                # Mirror BEFORE inference so the landmarks and the overlay
                # live in the same pixel space.
                frame = camera.mirror(frame, self.mirror_preview)
                fh, fw = frame.shape[:2]
                mp_image = dp_mod.mp.Image(
                    image_format=dp_mod.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                ts = int((time.time() - stream_start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                results = self.landmarker.detect_for_video(mp_image, ts)

                lms = None
                readings = {dp_mod.ARM_LEFT: None, dp_mod.ARM_RIGHT: None}
                pose_detected = bool(results.pose_landmarks)
                if pose_detected:
                    lms = results.pose_landmarks[0]
                    for i, j in dp_mod._POSE_CONNECTIONS:
                        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                            p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                            p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)
                    readings = {a: dp_mod.arm_reading(lms, a, fw, fh) for a in dp_mod.ARMS}
                turn = (dp_mod.body_turn_deg(results.pose_world_landmarks[0], lms)
                        if lms is not None and results.pose_world_landmarks else None)

                with self._lock:
                    # Front-on matters for the score; warn the coach if not.
                    self.test.note_turn(turn)
                    # The coach's arm is the PERSON's side; MediaPipe's name
                    # for it depends on the mirror (person -> MediaPipe is the
                    # same swap), so it is re-translated every frame.
                    pinned = self._person(self.requested_arm)
                    self.test.pinned = pinned if pinned in dp_mod.ARMS else None
                    ended = self.test.update(readings)
                    # Room above the head for a full press; warn if not.
                    self.test.note_room(dp_mod.shoulder_top_px(lms, fh) if lms is not None else None)
                    if ended and not self._staged:
                        # The test's own end (50 reps / 4 s rest) is how it
                        # normally finishes — stage now, so screen3 has the
                        # count even if nobody taps STOP.
                        self._stage_locked()
                    arm = self.test.shown_arm()
                    self.pose_detected = pose_detected
                    self.arm_tracked = bool(arm and readings.get(arm) is not None)
                    arm_pts = None
                    if self.arm_tracked and lms is not None:
                        arm_pts = [dp_mod.lm_px(lms, i, fw, fh) for i in dp_mod.ARM_IDX[arm]]
                    # Drawn inside the lock: START/RESET change the test.
                    dp_mod.draw_hud(frame, self.test, arm_label=self._person(arm),
                                    arm_pts=arm_pts)
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
    "PullUpEngine",
    "FullPullUpEngine",
    "WallTossEngine",
    "SitReachEngine",
    "TrunkLiftEngine",
    "ReverseCurlEngine",
    "ShoulderStretchEngine",
    "IsometricPushUpEngine",
    "SeatedPushUpEngine",
    "TargetStretchEngine",
    "WingspanEngine",
    "VerticalJumpEngine",
]
