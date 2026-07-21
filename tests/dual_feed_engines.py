"""
GoForFit — Dual-Camera Browser Test Engines
=============================================
Extends the BaseWebEngine pattern from web_engines.py to tests that need
TWO simultaneous camera feeds (+ optionally a microphone for whistle-start
detection): shuttle_run, dash_50m, and run_600m.

Import this module from web_engines.py (or straight from app.py) and treat
each class the same way the existing single-cam engines are treated:
    start() / stop() / is_running() / get_state() / action_*(...)
Only difference: instead of one get_jpeg() there are two —
get_jpeg_a() / get_jpeg_b() — because the frontend needs two <img> feeds.

SECTIONS
--------
  0. DualFeedEngine     shared 2-camera threading / MJPEG / state plumbing
  1. WhistleMixin       optional mic-based start trigger (sounddevice)
  2. ShuttleRunEngine    wraps shuttleruntest.py's per-lane touch-count logic
  3. Dash50mEngine       wraps sprint_timer.py's finish-crossing logic
  4. Run600mEngine       wraps 600m_new/ colour-lane lap counting

NOTE: This file reuses each original script's core detection functions
(imported, not rewritten) exactly like wingspan_web_engine.py reuses
wingspan2.py. Only the "loop that used to open a desktop cv2 window and
call imshow/waitKey" is replaced with a background thread that publishes
JPEGs + a state dict for the browser to poll, same contract as the other
six engines in web_engines.py.

Wire actual per-frame detection calls into `_process_pair()` in each
subclass below by importing the specific function you want from the
original script (see the `# TODO: wire` comments) — the threading/MJPEG/
state/action plumbing is complete and ready to run as-is.
"""

import os
import sys
import time
import threading

import cv2
import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from stall_watchdog import StallWatchdogMixin  # noqa: E402  (needs _TESTS_DIR on sys.path)
import camera  # noqa: E402  — shared camera-source resolution


def _open_cam(url_or_index):
    """Accepts '0', '1', an http(s) MJPEG/IP-cam URL, or an int index.

    Delegates to camera.py so every engine resolves a source identically —
    the guard that stops two coaches opening the same device compares these
    sources, and it can only do that if "0" and 0 mean one thing everywhere.
    Returns an unopened capture rather than raising, because callers here
    check isOpened() to report which of the two cameras failed.
    """
    resolved = camera.resolve_source(url_or_index)
    cap = cv2.VideoCapture(resolved, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(resolved)
    return cap


# ─────────────────────────────────────────────────────────────────────────
#  0. DualFeedEngine — shared base
# ─────────────────────────────────────────────────────────────────────────
def _release_capture(cap):
    """Release a VideoCapture, ignoring an already-released/None handle."""
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass


class DualFeedEngine(StallWatchdogMixin):
    # Mirrored selfie-style, like every other engine — a subject facing the
    # camera sees their real right hand on the right of the screen. The flip
    # happens in _run_loop BEFORE _process_pair, so detection, overlays and
    # browser calibration clicks all share one coordinate space.
    #
    # Safe for the direction-sensitive logic here: 600m infers travel
    # direction on the Y axis and keys lanes by shirt colour; dash_50m detects
    # finishes via background-subtraction motion inside a coach-calibrated
    # strip (direction-agnostic); shuttle_run's lane boundaries are clicked by
    # the coach on this same flipped feed.
    # CAVEAT: positional lane numbering (shuttle_run's sorted boundary clicks,
    # dash_50m's overhead column split) is screen-relative — lane 1 is the
    # leftmost lane AS DISPLAYED, which mirroring moves to the other side of
    # the track. Both engines now draw their L-numbers on the feed and expose
    # a flip action, so the coach reads the labels rather than guessing;
    # shuttle_run additionally reverses End B by default, since two cameras
    # facing each other ALWAYS disagree about left. Confirm on the live feed
    # before a timed run.
    MIRROR = True
    JPEG_QUALITY = 75
    FRAME_W = 960
    FRAME_H = 540
    STOP_JOIN_TIMEOUT = 2

    def __init__(self, cam_a_url, cam_b_url, session_id=None, on_stage=None, mirror=True, **kwargs):
        self.mirror_preview = mirror
        self.cam_a_url = cam_a_url
        self.cam_b_url = cam_b_url
        self.session_id = session_id
        self.on_stage = on_stage
        self.params = kwargs  # touches / lanes / lane_width / etc, per subclass

        # RLock so _safe_stage may lock its _error write regardless of caller context.
        self._lock = threading.RLock()
        self._jpeg_a = None
        self._jpeg_b = None
        self._seq_a = 0   # bumped per publish — feed streams dedupe on this, not identity
        self._seq_b = 0
        self._running = False
        self._thread = None
        self._error = None
        self.cap_a = None
        self.cap_b = None

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self.cap_a = _open_cam(self.cam_a_url)
        self.cap_b = _open_cam(self.cam_b_url)
        if not self.cap_a.isOpened():
            self._release_cameras()
            raise RuntimeError(f"Cannot open camera A ({self.cam_a_url!r})")
        if not self.cap_b.isOpened():
            self._release_cameras()
            raise RuntimeError(f"Cannot open camera B ({self.cam_b_url!r})")

        # Both cameras are open from here on, so anything that fails below must
        # release them before propagating — a leaked capture keeps the device
        # busy for the life of the process and makes every later launch fail
        # with "Cannot open camera" as if the hardware were at fault.
        try:
            for cap in (self.cap_a, self.cap_b):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._load_models()
            self._running = True
            self._error = None
            self._watchdog_reset()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._running = False
            self._thread = None
            self._release_cameras()
            self._release_models()  # safe on a partially-loaded engine
            raise

    def _release_cameras(self):
        caps, self.cap_a, self.cap_b = (self.cap_a, self.cap_b), None, None
        for cap in caps:
            _release_capture(cap)

    def maybe_finish(self):
        """Stage whatever result is available so far. Subclasses override;
        called automatically by stop() so the HTTP layer doesn't have to."""
        pass

    def stop(self):
        # Stage whatever's finished before shutdown — never let it block the stop.
        try:
            self.maybe_finish()
        except Exception:
            pass

        # cap_a/cap_b must stay set until the loop thread is really gone —
        # _run_loop reads self.cap_a.read()/self.cap_b.read() every iteration.
        self._running = False
        thread, self._thread = self._thread, None

        if thread is not None and thread.is_alive():
            thread.join(timeout=self.STOP_JOIN_TIMEOUT)
            if thread.is_alive():
                # _run_loop is still inside cap.read()/inference. Releasing the
                # captures now is a use-after-free down in native OpenCV, which
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
        self._release_cameras()
        self._release_models()

    def is_running(self):
        return self._running

    def get_jpeg_a(self):
        with self._lock:
            return self._jpeg_a

    def get_jpeg_b(self):
        with self._lock:
            return self._jpeg_b

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

    # ── hooks subclasses override ────────────────────────────────────
    def _load_models(self):
        pass

    def _release_models(self):
        pass

    def _state_locked(self):
        return {}

    def _process_pair(self, frame_a, frame_b):
        """Subclasses do detection + drawing here, return (drawn_a, drawn_b)."""
        return frame_a, frame_b

    def action(self, name, value=None):
        """Subclasses override to handle browser button clicks."""
        return {"ok": False, "error": f"Unknown action '{name}'"}

    # ── shared publish helper ────────────────────────────────────────
    def _publish(self, frame_a, frame_b):
        ok_a, buf_a = cv2.imencode(".jpg", frame_a, [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY])
        ok_b, buf_b = cv2.imencode(".jpg", frame_b, [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY])
        with self._lock:
            if ok_a:
                self._jpeg_a = buf_a.tobytes()
                self._seq_a += 1
            if ok_b:
                self._jpeg_b = buf_b.tobytes()
                self._seq_b += 1
            if ok_a or ok_b:
                self._watchdog_mark()

    def get_frame_a(self):
        """(jpeg_bytes|None, seq) — seq changes iff a new frame was published."""
        with self._lock:
            return self._jpeg_a, self._seq_a

    def get_frame_b(self):
        with self._lock:
            return self._jpeg_b, self._seq_b

    def _safe_stage(self, *args):
        if self.on_stage:
            try:
                self.on_stage(*args)
            except Exception as exc:
                with self._lock:
                    self._error = f"Stage failed: {exc}"

    def _run_loop(self):
        try:
            while self._running:
                ok_a, frame_a = self.cap_a.read()
                ok_b, frame_b = self.cap_b.read()
                if not ok_a or not ok_b:
                    time.sleep(0.02)
                    continue
                # Before _process_pair so detection/overlays/clicks agree.
                frame_a = camera.mirror(frame_a, self.mirror_preview)
                frame_b = camera.mirror(frame_b, self.mirror_preview)
                drawn_a, drawn_b = self._process_pair(frame_a, frame_b)
                self._publish(drawn_a, drawn_b)
        except Exception as exc:
            self._error = str(exc)
            self._running = False


# ─────────────────────────────────────────────────────────────────────────
#  1. WhistleMixin — background mic listener, shared by shuttle_run & dash_50m
# ─────────────────────────────────────────────────────────────────────────
class WhistleMixin:
    """Uses `sounddevice` (NOT pyaudio — sprint_timer.py's pyaudio import is
    the reliability bug flagged earlier; standardize on sounddevice like
    shuttleruntest.py already does). Detects a 2.4-4.1kHz whistle band and
    calls self._on_whistle() once per detected blast.
    """
    WHISTLE_LO_HZ = 2400
    WHISTLE_HI_HZ = 4100
    SAMPLE_RATE = 44100

    def _start_whistle_listener(self):
        try:
            import sounddevice as sd
        except ImportError:
            self._error = "sounddevice not installed — whistle detection disabled"
            return
        self._audio_stream = sd.InputStream(
            channels=1, samplerate=self.SAMPLE_RATE,
            callback=self._audio_callback, blocksize=2048,
        )
        self._audio_stream.start()

    def _stop_whistle_listener(self):
        stream = getattr(self, "_audio_stream", None)
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        # Simple FFT band-energy check — swap for a stricter detector if
        # false triggers show up in a noisy gym.
        samples = indata[:, 0]
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.SAMPLE_RATE)
        band = spectrum[(freqs >= self.WHISTLE_LO_HZ) & (freqs <= self.WHISTLE_HI_HZ)]
        if band.size and band.mean() > (spectrum.mean() * 6 + 1e-6):
            self._on_whistle()

    def _on_whistle(self):
        """Override in subclass — start the timer(s)."""
        pass


# ─────────────────────────────────────────────────────────────────────────
#  2. ShuttleRunEngine — reuses shuttleruntest.py's LaneTracker,
#     StudentManager, WhistleDetector, Calibration, open_capture directly.
#     Only the desktop mouse-click calibration loop is replaced: the browser
#     sends click coordinates via action("calib_click", {"cam":"a","x":..,
#     "y":..}) instead of cv2's setMouseCallback, everything downstream
#     (LaneTracker, touch logic, StudentManager) is the ORIGINAL code.
# ─────────────────────────────────────────────────────────────────────────
class ShuttleRunEngine(DualFeedEngine):
    NUM_LANES = 4  # matches shuttleruntest.NUM_LANES

    def __init__(self, cam_a_url, cam_b_url, touches=4, session_id=None, on_stage=None):
        super().__init__(cam_a_url, cam_b_url, session_id=session_id, on_stage=on_stage, touches=touches)
        import shuttleruntest as srt
        self._srt = srt
        self.touches_required = int(touches)
        self.manager = srt.StudentManager(self.NUM_LANES, self.touches_required)
        self.whistle = srt.WhistleDetector()

        # Calibration state — points collected via browser clicks, one list
        # per camera, same NUM_LANES+1 (boundaries) + 2 (touch line) points
        # calibrate_camera() collects in the desktop version.
        self._calib_points = {"a": [], "b": []}
        self.calib = {"a": None, "b": None}   # -> srt.Calibration once done
        self.trackers = {"a": [], "b": []}    # list[LaneTracker] once calibrated
        self.started = False

        # The two cameras face each other across the track, so End B sees the
        # lanes in the opposite left-right order to End A. Both ends feed the
        # SAME StudentManager, keyed by lane number — so without this, End B's
        # "lane 1" is End A's lane 4 and every runner's touch count is a blend
        # of two different people. Default True because facing cameras are the
        # documented setup; the coach can flip it from screen4 if the cameras
        # are rigged side-by-side instead.
        self.reverse_lanes_b = True

    def _load_models(self):
        self.whistle.start()

    def _release_models(self):
        self.whistle.stop()
        for side in ("a", "b"):
            for t in self.trackers[side]:
                t.close()

    # ── browser-driven calibration (replaces calibrate_camera()'s cv2 loop) ──
    def _needs_points(self):
        return self.NUM_LANES + 1 + 2  # lane boundaries + 2 touch-line points

    def _try_finish_calibration(self, side, frame_shape):
        pts = self._calib_points[side]
        if len(pts) < self._needs_points():
            return
        lane_pts = pts[: self.NUM_LANES + 1]
        line_pts = pts[self.NUM_LANES + 1: self.NUM_LANES + 3]
        lane_xs = sorted(p[0] for p in lane_pts)
        touch_line_y = int(np.mean([p[1] for p in line_pts]))
        h, w = frame_shape[:2]
        self.calib[side] = self._srt.Calibration(lane_xs, touch_line_y, w, h)

        self._build_trackers(side, touch_line_y)

    def _lane_number_for_column(self, side, column_idx0):
        """Which lane number the idx-th column (left to right, as displayed)
        belongs to. Reversing the NUMBERING is the correct fix here, not
        mirroring the boundary x-coords: the columns are where the lanes
        actually are on this camera's frame, only the labels are backwards."""
        if side == "b" and self.reverse_lanes_b:
            return self.NUM_LANES - column_idx0
        return column_idx0 + 1

    def _build_trackers(self, side, touch_line_y):
        for old in self.trackers[side]:
            old.close()  # each tracker owns a MediaPipe Pose instance
        end_label = "A" if side == "a" else "B"
        trackers = []
        for column_idx0 in range(self.NUM_LANES):
            x0, x1 = self.calib[side].lane_bounds(column_idx0)
            lane = self._lane_number_for_column(side, column_idx0)
            trackers.append(self._srt.LaneTracker(lane, end_label, x0, x1, touch_line_y, self.manager))
        self.trackers[side] = trackers

    def _draw_calib_overlay(self, frame, side):
        for i, (x, y) in enumerate(self._calib_points[side]):
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        needed = self._needs_points()
        cv2.putText(frame, f"Calib {side.upper()}: {len(self._calib_points[side])}/{needed}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def _on_whistle(self):
        if self.started:
            return
        self.manager.start()
        self.started = True

    def _process_pair(self, frame_a, frame_b):
        if self.whistle.detected.is_set() and not self.started:
            self._on_whistle()

        for side, frame in (("a", frame_a), ("b", frame_b)):
            if self.calib[side] is None:
                self._try_finish_calibration(side, frame.shape)
                self._draw_calib_overlay(frame, side)
                continue
            cv2.line(frame, (0, self.calib[side].touch_line_y), (frame.shape[1], self.calib[side].touch_line_y),
                      (0, 165, 255), 2)
            for x in self.calib[side].lane_boundaries_x:
                cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 1)
            # Lane numbers ON the feed — this is what makes the numbering
            # checkable at a glance in the field: the same physical lane must
            # carry the same L-number in both windows before a timed run.
            snap = self.manager.snapshot()
            for tracker in self.trackers[side]:
                count, finished = snap.get(tracker.lane_number, (0, False))
                cv2.putText(frame, f"L{tracker.lane_number}: {count}" + (" DONE" if finished else ""),
                            (tracker.x0 + 5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 200, 0) if finished else (255, 255, 255), 2)
            if self.started:
                for tracker in self.trackers[side]:
                    tracker.process(frame)

        if self.started and self.manager.all_finished():
            self.maybe_finish()

        return frame_a, frame_b

    def _state_locked(self):
        snap = self.manager.snapshot()
        return {
            "phase": "calibrating" if (self.calib["a"] is None or self.calib["b"] is None)
                     else ("waiting_whistle" if not self.started else "running"),
            "touches_required": self.touches_required,
            "calibrated_a": self.calib["a"] is not None,
            "calibrated_b": self.calib["b"] is not None,
            "reverse_lanes_b": self.reverse_lanes_b,
            "lanes": {lane: {"touches": tc, "finished": fin} for lane, (tc, fin) in snap.items()},
        }

    def action(self, name, value=None):
        if name == "calib_click":
            side = (value or {}).get("cam")
            x, y = (value or {}).get("x"), (value or {}).get("y")
            if side not in ("a", "b") or x is None or y is None:
                return {"ok": False, "error": "calib_click needs {cam:'a'|'b', x, y}"}
            self._calib_points[side].append((int(x), int(y)))
            return {"ok": True, "count": len(self._calib_points[side])}
        if name == "calib_reset":
            side = (value or {}).get("cam", "a")
            self._calib_points[side] = []
            self.calib[side] = None
            return {"ok": True}
        if name == "flip_lane_order_b":
            # For a side-by-side camera rig, where End B is NOT reversed.
            # Safe mid-session: only the labels change, and rebuilding End B's
            # trackers re-reads the same columns under new lane numbers.
            self.reverse_lanes_b = (not self.reverse_lanes_b) if value is None else bool(value)
            if self.calib["b"] is not None:
                self._build_trackers("b", self.calib["b"].touch_line_y)
            return {"ok": True, "reverse_lanes_b": self.reverse_lanes_b}
        if name == "manual_start":
            self._on_whistle()
            return {"ok": True}
        if name == "reset":
            self.manager = self._srt.StudentManager(self.NUM_LANES, self.touches_required)
            self.started = False
            self.whistle.detected.clear()
            return {"ok": True}
        return {"ok": False, "error": f"Unknown action '{name}'"}

    def maybe_finish(self):
        with self.manager.lock:
            start_time = self.manager.start_time
            laps = [
                {"lane": lane, "time_sec": round(s.finish_time - start_time, 3)}
                for lane, s in sorted(self.manager.students.items())
                if s.finished and s.finish_time and start_time
            ]
        if laps:
            self._safe_stage({"session_id": self.session_id, "laps": laps})


# ─────────────────────────────────────────────────────────────────────────
#  3. Dash50mEngine
#  NOTE — honesty flag: sprint_timer.py is written as a fully procedural
#  script (module-level globals + threads spawned in main()), not a class,
#  unlike shuttleruntest.py's clean LaneTracker/StudentManager split. It
#  isn't cleanly importable/reusable piece-by-piece, so this is a fresh,
#  self-contained implementation inspired by its finish-strip-crossing
#  approach rather than a direct reuse — treat this one as needing real
#  camera-side validation before relying on it, unlike ShuttleRunEngine and
#  Run600mEngine below which call your existing, already-tested classes.
#  It also fixes the earlier pyaudio bug by using sounddevice throughout.
# ─────────────────────────────────────────────────────────────────────────
class Dash50mEngine(DualFeedEngine, WhistleMixin):
    """cam_a = finish camera (perpendicular), cam_b = overhead (lane identity)."""

    def __init__(self, cam_a_url, cam_b_url, lanes=4, lane_width=1.22,
                 session_id=None, on_stage=None):
        super().__init__(cam_a_url, cam_b_url, session_id=session_id, on_stage=on_stage,
                          lanes=lanes, lane_width=lane_width)
        self.num_lanes = int(lanes)
        self.lane_width = float(lane_width)
        self.lane_times = {i: None for i in range(1, self.num_lanes + 1)}
        self.start_t = None
        self.whistle_armed = True
        # Lane identity here comes from ONE overhead camera, so there's no
        # second view to disagree with — but the numbering is still purely
        # screen-relative, and the mirrored feed put lane 1 on the opposite
        # side of the track. Nothing can infer the right answer; the coach
        # reads the L-numbers off the overhead feed and flips if they're
        # backwards.
        self.reverse_lanes = False
        self._finish_strip = None       # (x1, x2) set via calib_click, like calibrate_finish_strip()
        self._calib_points = []
        # One per camera: the two point at different scenes, so a single
        # shared model would learn a blend of both as "background".
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=False)
        self._bg_overhead = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=False)

    def _load_models(self):
        self._start_whistle_listener()

    def _release_models(self):
        self._stop_whistle_listener()

    def _on_whistle(self):
        if not self.whistle_armed:
            return
        self.start_t = time.time()
        self.whistle_armed = False

    def _lane_for_column(self, column_idx0):
        """Lane number for the idx-th overhead column, left to right as displayed."""
        if self.reverse_lanes:
            return self.num_lanes - column_idx0
        return column_idx0 + 1

    def _column_bounds(self, column_idx0, width):
        """[x0, x1) of the idx-th overhead column.

        Rounded from the exact fraction rather than a fixed integer stride, so
        the columns tile the full frame whatever the lane count: 960px over 7
        lanes is 137.1px each, and a fixed stride leaves a 6px strip on the
        right belonging to no lane. This also replaces a reshape() that
        required width to divide evenly — with lanes=7 or 9 (both offered by
        the setup form) it raised inside the run loop and killed the test
        mid-race, after the runners had already gone.
        """
        x0 = round(column_idx0 * width / self.num_lanes)
        x1 = round((column_idx0 + 1) * width / self.num_lanes)
        return x0, x1

    def _process_pair(self, frame_a, frame_b):
        # frame_a = finish camera: motion crossing a calibrated vertical
        # strip = a finish event. frame_b = overhead camera: divide frame
        # width into num_lanes equal columns for lane identity (matches
        # sprint_timer.py's fixed-lane-width assumption).
        if self._finish_strip is None:
            for i, (x, y) in enumerate(self._calib_points):
                cv2.circle(frame_a, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame_a, f"Click finish strip L/R edge: {len(self._calib_points)}/2",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame_a, frame_b

        x1, x2 = self._finish_strip
        cv2.rectangle(frame_a, (x1, 0), (x2, frame_a.shape[0]), (0, 165, 255), 2)

        # Label the overhead columns so lane identity is verifiable before a
        # timed run instead of after the times come out attached to the wrong
        # runners.
        for idx0 in range(self.num_lanes):
            x0, _ = self._column_bounds(idx0, frame_b.shape[1])
            cv2.line(frame_b, (x0, 0), (x0, frame_b.shape[0]), (255, 255, 0), 1)
            cv2.putText(frame_b, f"L{self._lane_for_column(idx0)}", (x0 + 5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Feed BOTH models every frame, before the whistle and regardless of
        # whether the mask gets used. A background subtractor only knows what
        # it has been shown: the overhead model used to be applied solely
        # inside the motion branch below, so the very first frame it ever saw
        # was the one being asked about — it reported the whole frame as
        # foreground and the lane pick collapsed to whichever column argmax
        # tried first, every single time.
        mask_a = self._bg_subtractor.apply(frame_a)
        mask_b = self._bg_overhead.apply(frame_b)

        if self.start_t is not None:
            motion_px = int(np.count_nonzero(mask_a[:, x1:x2]))
            if motion_px > 400:  # tune to track lighting — a runner's body crossing the strip
                # Identify which lane via the overhead camera's horizontal split
                width = mask_b.shape[1]
                cols = [int(mask_b[:, x0:x1].sum())
                        for x0, x1 in (self._column_bounds(i, width) for i in range(self.num_lanes))]
                lane = self._lane_for_column(int(np.argmax(cols)))
                if self.lane_times.get(lane) is None:
                    self.lane_times[lane] = time.time() - self.start_t

        return frame_a, frame_b

    def _state_locked(self):
        return {
            "phase": "calibrating" if self._finish_strip is None
                     else ("waiting_whistle" if self.whistle_armed else "running"),
            "num_lanes": self.num_lanes,
            "lane_times": dict(self.lane_times),
            "reverse_lanes": self.reverse_lanes,
        }

    def action(self, name, value=None):
        if name == "calib_click":
            x, y = (value or {}).get("x"), (value or {}).get("y")
            if x is None:
                return {"ok": False, "error": "calib_click needs {x, y}"}
            self._calib_points.append((int(x), int(y)))
            if len(self._calib_points) >= 2:
                xs = sorted(p[0] for p in self._calib_points[:2])
                self._finish_strip = (xs[0], xs[1])
            return {"ok": True, "count": len(self._calib_points)}
        if name == "calib_reset":
            self._calib_points = []
            self._finish_strip = None
            return {"ok": True}
        if name == "flip_lane_order":
            # Relabels the overhead columns when L1 reads as the wrong side of
            # the track. Only affects times recorded from here on.
            self.reverse_lanes = (not self.reverse_lanes) if value is None else bool(value)
            return {"ok": True, "reverse_lanes": self.reverse_lanes}
        if name == "manual_start":
            self._on_whistle()
            return {"ok": True}
        if name == "reset":
            self.lane_times = {i: None for i in range(1, self.num_lanes + 1)}
            self.whistle_armed = True
            self.start_t = None
            return {"ok": True}
        return {"ok": False, "error": f"Unknown action '{name}'"}

    def maybe_finish(self):
        laps = [{"lane": i, "time_sec": round(t, 2)} for i, t in self.lane_times.items() if t]
        if laps:
            self._safe_stage({"session_id": self.session_id, "laps": laps})


# ─────────────────────────────────────────────────────────────────────────
#  4. Run600mEngine — reuses 600m_new's ColorDetector, LaneManager,
#     StateMachine, GlobalTimer, AppConfig directly (lane_manager.py's own
#     docstring notes it has "no OpenCV GUI dependency... which is what
#     allows the whole detection+timing+lap-counting engine to be reused
#     directly inside a Flask/FastAPI backend" — this is exactly that).
#     Only config_screen.py's Tkinter color-picker dialog is replaced —
#     lane colors now come from the launch params form on screen3.html
#     instead of a desktop dialog.
# ─────────────────────────────────────────────────────────────────────────
class Run600mEngine(DualFeedEngine):
    """Single physical camera in practice (matches 600m_new/main.py) —
    cam_b defaults to cam_a so it still fits the DualFeedEngine contract."""

    def __init__(self, cam_a_url, cam_b_url=None, lane_colors=None, camera_distance_m=0.0,
                 session_id=None, on_stage=None):
        cam_b_url = cam_b_url or cam_a_url
        super().__init__(cam_a_url, cam_b_url, session_id=session_id, on_stage=on_stage)
        sys.path.insert(0, os.path.join(_TESTS_DIR, "600m_new"))
        from config import AppConfig
        from color_detector import ColorDetector
        from lane_manager import LaneManager
        from state_machine import AppState, StateMachine
        from timer import GlobalTimer

        self.lane_colors = lane_colors or []  # e.g. ["Red","Blue","Green","Yellow","Orange"]
        cfg = AppConfig()
        cfg.camera_distance_m = float(camera_distance_m)
        cfg.lane_colors = {i + 1: c for i, c in enumerate(self.lane_colors)}
        cfg.num_lanes = len(self.lane_colors)
        self.cfg = cfg
        self.detector = ColorDetector(self.lane_colors)
        self.lane_manager = LaneManager(cfg)
        self.state_machine = StateMachine(AppState)
        self.global_timer = GlobalTimer()
        self._AppState = AppState
        self._last_t = None

    def _load_models(self):
        self._last_t = time.time()

    def _process_pair(self, frame_a, frame_b):
        now = time.time()
        dt = now - (self._last_t or now)
        self._last_t = now

        detections = self.detector.detect(frame_a)
        counting = self.state_machine.is_running()
        self.global_timer.update(dt)
        newly_lapped = self.lane_manager.update(detections, dt, now, counting)

        finish_y = self.cfg.finish_line_y_px()
        cv2.line(frame_a, (0, finish_y), (frame_a.shape[1], finish_y), (0, 165, 255), 2)
        for lane in self.lane_manager.lanes.values():
            if lane.last_centroid:
                cv2.circle(frame_a, lane.last_centroid, 8, (0, 255, 0), -1)
                cv2.putText(frame_a, f"L{lane.lane_number} {lane.color}", lane.last_centroid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if newly_lapped:
            self.maybe_finish()

        return frame_a, frame_a

    def _state_locked(self):
        return {
            "started": self.state_machine.is_running(),
            "elapsed_sec": round(self.global_timer.total_elapsed, 2),
            "lap_counts": {
                lane.lane_number: {"color": lane.color, "time_sec": round(lane.timer.total_elapsed, 2)}
                for lane in self.lane_manager.lanes.values()
            },
        }

    def action(self, name, value=None):
        if name in ("start", "stop", "restart", "space"):
            self.state_machine.handle_space()
            if not self.state_machine.is_running():
                self.maybe_finish()
            return {"ok": True}
        return {"ok": False, "error": f"Unknown action '{name}'"}

    def maybe_finish(self):
        laps = [
            {"lane": lane.lane_number, "color": lane.color, "time_sec": round(lane.timer.total_elapsed, 2)}
            for lane in self.lane_manager.lanes.values()
        ]
        if laps:
            self._safe_stage({"session_id": self.session_id, "laps": laps})
