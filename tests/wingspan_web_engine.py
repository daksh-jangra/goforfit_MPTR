"""
WINGSPAN ENGINE — browser-controlled version
=============================================
This wraps the exact same MediaPipe hand/pose detection and drawing logic
that lives in wingspan2.py, but removes every cv2.imshow()/cv2.waitKey()
dependency so the test can be run entirely from a phone/tablet browser:

  • The camera-capture + processing loop runs in a background thread.
  • The latest annotated frame is exposed as JPEG bytes for an MJPEG stream.
  • All control actions (calibrate / measure / snapshot / save / cancel)
    arrive as plain method calls triggered by on-screen buttons instead of
    key presses inside a desktop OpenCV window.

wingspan2.py is imported (not copied) so all the geometry/drawing code stays
in one place — this file only re-implements the orchestration loop that used
to live inside wingspan2.main().
"""

import os
import csv
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import wingspan2 as ws  # reuse geometry / drawing / detector-building code

# ── Stability gate (ported from the BMI engine's fix 3) ─────────────────────
# A fingertip-to-fingertip span is only trusted once the last _WS_STABLE_WIN
# frames agree within _WS_STABLE_CV (std/mean). This rejects the jittery frames
# where a hand landmark wobbles, so calibration locks onto a settled hold and a
# snapshot averages only steady readings. Hand landmarks jitter a little more
# than pose, so the band is slightly looser than BMI's 2%.
_WS_STABLE_WIN = 15
_WS_STABLE_CV  = 0.025
_WS_MIN_SNAP_SAMPLES = 5   # a snapshot needs at least this many settled frames

# ── Wrist fallback ──────────────────────────────────────────────────────────
# The span is normally measured fingertip-to-fingertip (hand landmarks). Those
# fingertips are exactly what MediaPipe's hand detector loses at the left/right
# frame edges when the arms are fully spread, which clears the stability buffer
# and resets calibration. The pose model's WRISTS, however, stay tracked
# robustly even at the edges. So on frames where a fingertip is missing but both
# wrists are visible, we reconstruct the fingertip span from the wrist span:
#   fingertip_span ≈ wrist_span * (fingertip_span / wrist_span)
# The ratio (roughly the two hands' contribution) is a near-constant of the
# T-pose, so we learn it as an EMA from frames where BOTH the fingertips AND both
# wrists are visible, then apply it to bridge the frames where a hand blinks out.
# This keeps every buffered span in the same fingertip basis the scale was
# calibrated in — so it adds robustness without shifting the measurement.
_WS_LEFT_WRIST_IDX  = 15   # MediaPipe pose landmark: left wrist
_WS_RIGHT_WRIST_IDX = 16   # MediaPipe pose landmark: right wrist
_WS_WRIST_VIS_THRESH = 0.30  # min pose-landmark visibility to trust a wrist
_WS_RATIO_ALPHA = 0.10     # EMA weight for new fingertip/wrist ratio samples
_WS_RATIO_MIN, _WS_RATIO_MAX = 1.0, 1.6  # sane band; wrists are inside fingertips

# ── make calibration + log files resolve to the tests/ folder regardless
#    of the current working directory the FastAPI app was started from ──
_TESTS_DIR = os.path.dirname(os.path.abspath(ws.__file__))
ws.CALIBRATION_FILE = os.path.join(_TESTS_DIR, "wingspan_calibration.json")
ws.LOG_FILE = os.path.join(_TESTS_DIR, "wingspan_log.csv")


def _draw_calib_overlay_web(frame, cs, fw, fh, left_tip=None, right_tip=None):
    """Calibration overlay rewritten for the browser flow — the real text
    entry happens in an HTML input box under the video, not on the frame."""
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (frame.shape[1], frame.shape[0]), (8, 8, 50), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

    W, H = frame.shape[1], frame.shape[0]
    cx = W // 2
    phase = cs.get("phase", "SPREAD")
    avg = cs.get("avg_px", 0.0)

    ws.txt(frame, "CALIBRATION MODE", (cx - 185, 60), 0.95, ws.YELLOW, 2)

    if phase == "SPREAD":
        if left_tip and right_tip:
            lx, ly = int(left_tip[0] * fw), int(left_tip[1] * fh)
            rx, ry = int(right_tip[0] * fw), int(right_tip[1] * fh)
            cv2.line(frame, (lx, ly), (rx, ry), ws.ORANGE, 2, cv2.LINE_AA)
            for pt in [(lx, ly), (rx, ry)]:
                cv2.circle(frame, pt, 11, ws.ORANGE, -1)
                cv2.circle(frame, pt, 14, ws.WHITE, 2)

        msg = "Spread both arms wide and hold still"
        (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ws.txt(frame, msg, (cx - mw // 2, 130), 0.62, ws.ORANGE, 2)

        if cs.get("px_buf"):
            elapsed = time.time() - cs["hold_start"] if cs.get("hold_start") else 0
            bar_w = int(min(elapsed / 2.0, 1.0) * 300)
            cv2.rectangle(frame, (cx - 150, 165), (cx + 150, 165 + 16), (60, 60, 60), -1)
            cv2.rectangle(frame, (cx - 150, 165), (cx - 150 + bar_w, 165 + 16), ws.GREEN, -1)

        span_lbl = f"Avg pixel span: {avg:.0f} px" if avg else "Avg pixel span: --"
        ws.txt(frame, span_lbl, (cx - 110, 210), 0.54, ws.CYAN)

        if cs.get("settling"):
            hold_msg = "Hold still — settling..."
            (hw, _), _ = cv2.getTextSize(hold_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            ws.txt(frame, hold_msg, (cx - hw // 2, 240), 0.56, ws.YELLOW, 2)

    elif phase == "ENTER_CM":
        ws.txt(frame, f"Span locked: {avg:.0f} px", (cx - 120, 130), 0.6, ws.CYAN, 2)
        ws.txt(frame, "Enter your real wingspan in the box below", (cx - 240, 165), 0.58, ws.WHITE, 1)
        ws.txt(frame, "then tap Confirm", (cx - 90, 195), 0.54, (200, 200, 200), 1)

    ws.txt(frame, "Tap Cancel to exit calibration", (cx - 160, H - 30), 0.5, (160, 160, 160), 1)


def _draw_align_warning(frame):
    W = frame.shape[1]
    warn = "Align arms to 90 deg to measure"
    (ww, wh), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    wx = (W - ww) // 2
    wy = 120
    cv2.rectangle(frame, (wx - 10, wy - wh - 8), (wx + ww + 10, wy + 8), (20, 20, 20), -1)
    cv2.rectangle(frame, (wx - 10, wy - wh - 8), (wx + ww + 10, wy + 8), ws.RED, 1)
    cv2.putText(frame, warn, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.68, ws.RED, 2, cv2.LINE_AA)


class WingspanEngine:
    """One live wingspan session. Created on 'Start Test', destroyed on 'Stop'."""

    def __init__(self, session_id=None, on_stage=None, distance_cm=None):
        self.session_id = session_id
        self.on_stage = on_stage  # callback(wingspan_cm) — stages result for screen3 review
        # Measurement distance entered by the coach on screen3. Wingspan's scale
        # is cm-per-pixel learned at the CALIBRATION distance, so a reading taken
        # at a different distance is off by (calib_distance / measure_distance).
        # Storing the real measurement distance lets us correct for that, so one
        # calibration still works at a venue that can't fit the exact calibration
        # distance. None -> no correction (measuring at the calibration distance).
        self.measure_distance_cm = float(distance_cm) if distance_cm else None

        self._lock = threading.Lock()
        self._jpeg = None
        self._running = False
        self._thread = None
        self._error = None

        # ── browser-sourced camera ──────────────────────────────────────
        # The video no longer comes from a camera physically attached to
        # this server (cv2.VideoCapture(0) — which fails on a hosted box
        # with no webcam, the "Cannot open camera (index 0)" error). The
        # browser captures the phone/laptop camera and POSTs JPEG frames to
        # /api/wingspan/frame, which calls push_frame(); the loop below
        # consumes the most recent one.
        self._frame_lock = threading.Lock()
        self._in_frame = None          # newest decoded BGR frame from the browser
        self._frame_event = threading.Event()

        self.hand_det = None
        self.pose_det = None

        # mirrors the locals that used to live inside wingspan2.main()
        self.calibration = ws.load_calibration()
        self.mode = "MEASURE"
        self.result_cm = None
        self.history = []
        self.snap = None
        self.cs = {}
        self.last_save = None
        self._arms_ok = False   # True only when both arms are in the 88-92° window
        # Stability gate: recent pixel-span samples + the settled flag.
        self._recent_px = deque(maxlen=_WS_STABLE_WIN)
        self._stable = False
        # Wrist fallback: learned EMA of fingertip_span / wrist_span, used to
        # reconstruct the fingertip span from the wrists when a hand blinks out.
        self._ft_wrist_ratio = None
        # Mirror the feed like a selfie. Correct for the FRONT camera, but the
        # BACK camera already shows the world un-mirrored, so it's turned off
        # when the browser reports it's streaming the back ('environment') cam.
        self._mirror = True

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return

        models = ws.ensure_models(_TESTS_DIR)
        self.hand_det = ws.build_hand_detector(models["hand_landmarker.task"])
        self.pose_det = ws.build_pose_detector(models["pose_landmarker_lite.task"])

        # No server-side camera is opened any more — frames arrive from the
        # browser via push_frame(). The loop simply waits for the first one.
        with self._frame_lock:
            self._in_frame = None
        self._frame_event.clear()

        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._frame_event.set()   # wake the loop so it can see _running == False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        with self._frame_lock:
            self._in_frame = None
        for det in (self.hand_det, self.pose_det):
            try:
                if det is not None:
                    det.close()
            except Exception:
                pass

    def is_running(self):
        return self._running

    def set_facing(self, facing):
        """Which browser camera is streaming ('user' = front, 'environment' =
        back). The back camera must not be selfie-mirrored. The frame endpoint
        calls this before push_frame(); push_frame still accepts a facing arg for
        any direct callers."""
        if facing is None:
            return
        self._mirror = (facing != "environment")

    def push_frame(self, jpeg_bytes, facing=None):
        """Accept one JPEG frame captured by the browser camera and hand it to
        the processing loop. Returns True if the bytes decoded to an image.

        Only the most recent frame is kept — if the browser uploads faster
        than MediaPipe can process, older frames are dropped rather than
        queued, so the annotated feed always reflects near-real-time input.

        `facing` is the browser's camera facingMode ('user' = front,
        'environment' = back). The back camera is not mirrored; anything else
        (front, or unknown/laptop) keeps the selfie mirror.
        """
        if not self._running or not jpeg_bytes:
            return False
        if facing is not None:
            self._mirror = (facing != "environment")
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._frame_lock:
            self._in_frame = frame
        self._frame_event.set()
        return True

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_state(self):
        with self._lock:
            return {
                "mode": self.mode,
                "phase": self.cs.get("phase") if self.mode == "CALIBRATE" else None,
                "calibrated": self.calibration is not None,
                "calib_cm": (
                    round(self.calibration["real_wingspan_cm"], 1) if self.calibration else None
                ),
                "result_cm": round(self.result_cm, 1) if self.result_cm is not None else None,
                "snap_phase": self.snap["phase"] if self.snap else None,
                "snap_final_cm": (
                    round(self.snap["final_cm"], 1)
                    if self.snap and self.snap.get("final_cm") is not None
                    else None
                ),
                "avg_px": (
                    round(self.cs.get("avg_px", 0)) if self.mode == "CALIBRATE" else None
                ),
                "arms_ok": self._arms_ok,
                "stable": self._stable,
                "history": [round(v, 1) for v in self.history[-6:]],
                "last_save": self.last_save,
                "error": self._error,
            }

    # ── actions — these replace the old key presses ───────────────────
    def action_calibrate(self):
        with self._lock:
            self.mode = "CALIBRATE"
            self.cs = {"phase": "SPREAD", "px_buf": [], "hold_start": None, "avg_px": 0.0}
        return {"ok": True}

    def action_measure(self):
        with self._lock:
            self.mode = "MEASURE"
            self.cs = {}
            self.snap = None
        return {"ok": True}

    def action_cancel(self):
        with self._lock:
            self.mode = "MEASURE"
            self.cs = {}
        return {"ok": True}

    def action_enter_cm(self, value):
        with self._lock:
            if self.mode != "CALIBRATE":
                return {"ok": False, "error": "Not in calibration mode"}
            try:
                real_cm = float(value)
                if real_cm <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                return {"ok": False, "error": "Enter a positive number of cm"}

            avg_px = self.cs.get("avg_px", 0.0)
            if not avg_px:
                return {"ok": False, "error": "Spread your arms wide first"}

            sf = ws.compute_scale(avg_px, real_cm)
            self.calibration = {
                "real_wingspan_cm": real_cm,
                "pixel_span": avg_px,
                "scale_factor": sf,
                "distance_cm": ws.PERSON_DISTANCE_CM,
                "timestamp": datetime.now().isoformat(),
            }
            ws.save_calibration(self.calibration)
            self.mode = "MEASURE"
            self.cs = {}
            return {"ok": True}

    def action_snapshot(self):
        with self._lock:
            if self.mode != "MEASURE":
                return {"ok": False, "error": "Switch to Measure mode first"}
            if not self.calibration:
                return {"ok": False, "error": "Calibrate first"}
            self.snap = {
                "phase": "COUNTING",
                "start_time": time.time(),
                "samples": [],
                "final_cm": None,
            }
            return {"ok": True}

    def action_save(self):
        with self._lock:
            val = None
            if self.snap is not None and self.snap.get("phase") == "DONE":
                final = self.snap["final_cm"]
                # final_cm is 0.0 when the snapshot window collected zero
                # angle-gated samples — arms were never in the 88-92° range.
                if final is None or final == 0.0:
                    return {
                        "ok": False,
                        "error": "No valid reading — keep arms at 90° and snapshot again",
                    }
                val = final
            elif self.result_cm is not None and self._arms_ok:
                # Only accept a live reading when arms are currently in range.
                val = self.result_cm

            if val is None:
                return {
                    "ok": False,
                    "error": "No valid reading — align arms to 90° first",
                }

            self.history.append(val)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            newfile = not os.path.exists(ws.LOG_FILE)
            with open(ws.LOG_FILE, "a", newline="") as f:
                w = csv.writer(f)
                if newfile:
                    w.writerow(["timestamp", "wingspan_cm", "distance_cm"])
                w.writerow([ts, f"{val:.2f}", ws.PERSON_DISTANCE_CM])

            saved_cm = round(val, 2)
            if self.on_stage:
                try:
                    self.on_stage(saved_cm)
                except Exception as e:
                    self._error = f"Stage failed: {e}"

            self.last_save = saved_cm
            self.snap = None
            return {"ok": True, "saved_cm": saved_cm}

    # ── background capture / processing loop ───────────────────────────
    def _loop(self):
        # Use real wall-clock ms so MediaPipe video-mode tracking works correctly
        _start_wall = time.monotonic()
        # Frame-skip counter: run MediaPipe every N frames, reuse last results
        # otherwise. The browser already throttles its upload rate (~12fps), so
        # process every frame we actually receive rather than dropping half.
        _SKIP = 1
        _frame_count = 0
        _last_h_result = None
        _last_p_result = None
        try:
            while self._running:
                # Wait for the browser to push a new frame (up to 1s), then
                # grab the newest one. This replaces self.cap.read().
                if not self._frame_event.wait(timeout=1.0):
                    continue
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._in_frame
                    self._in_frame = None
                if frame is None:
                    continue

                if self._mirror:
                    frame = cv2.flip(frame, 1)
                fw, fh = frame.shape[1], frame.shape[0]
                frame_ts_ms = int((time.monotonic() - _start_wall) * 1000)

                _frame_count += 1
                run_mp = (_frame_count % _SKIP == 0)

                if run_mp:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = ws.Image(image_format=ws.ImageFormat.SRGB, data=rgb)
                    _last_h_result = self.hand_det.detect_for_video(mp_img, frame_ts_ms)
                    _last_p_result = self.pose_det.detect_for_video(mp_img, frame_ts_ms)

                h_result = _last_h_result
                p_result = _last_p_result

                if h_result is None or p_result is None:
                    # Haven't had a first inference yet — just stream raw frame
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ok:
                        with self._lock:
                            self._jpeg = buf.tobytes()
                    continue

                if h_result.hand_landmarks:
                    ws.draw_hand_skeleton(frame, h_result.hand_landmarks)
                ws.draw_pose_skeleton(frame, p_result)

                # Handedness-independent endpoints: collect every detected hand's
                # middle fingertip and use the two most separated horizontally
                # (screen-leftmost and screen-rightmost). Works from the front,
                # the BACK, or any orientation — MediaPipe's Left/Right guess
                # (unreliable when it sees the backs of the hands) is not used.
                left_tip = right_tip = None
                if h_result.hand_landmarks:
                    tips = [(lms[ws.MIDDLE_TIP_IDX].x, lms[ws.MIDDLE_TIP_IDX].y)
                            for lms in h_result.hand_landmarks]
                    if len(tips) >= 2:
                        tips.sort(key=lambda t: t[0])   # by x: screen-left → right
                        left_tip, right_tip = tips[0], tips[-1]

                span_px = None
                if left_tip and right_tip:
                    lx, ly = left_tip[0] * fw, left_tip[1] * fh
                    rx, ry = right_tip[0] * fw, right_tip[1] * fh
                    span_px = ws.px_dist((lx, ly), (rx, ry))

                # ── wrist fallback ─────────────────────────────────────────
                # Extract the two wrists from the pose (robust at frame edges).
                # Sort by x so they line up with left_tip/right_tip (screen
                # left → right), independent of MediaPipe's L/R guess.
                l_wrist = r_wrist = None
                if p_result.pose_landmarks:
                    plms = p_result.pose_landmarks[0]
                    wrists = []
                    for idx in (_WS_LEFT_WRIST_IDX, _WS_RIGHT_WRIST_IDX):
                        if idx < len(plms) and plms[idx].visibility >= _WS_WRIST_VIS_THRESH:
                            wrists.append((plms[idx].x, plms[idx].y))
                    if len(wrists) >= 2:
                        wrists.sort(key=lambda w: w[0])
                        l_wrist, r_wrist = wrists[0], wrists[-1]

                wrist_span_px = None
                if l_wrist and r_wrist:
                    wrist_span_px = ws.px_dist(
                        (l_wrist[0] * fw, l_wrist[1] * fh),
                        (r_wrist[0] * fw, r_wrist[1] * fh),
                    )

                if span_px and wrist_span_px and wrist_span_px > 1e-3:
                    # Both bases visible → learn the fingertip/wrist ratio.
                    ratio = span_px / wrist_span_px
                    if _WS_RATIO_MIN <= ratio <= _WS_RATIO_MAX:
                        self._ft_wrist_ratio = (
                            ratio if self._ft_wrist_ratio is None
                            else (1 - _WS_RATIO_ALPHA) * self._ft_wrist_ratio
                                 + _WS_RATIO_ALPHA * ratio
                        )
                elif span_px is None and wrist_span_px and self._ft_wrist_ratio:
                    # A fingertip blinked out but the wrists are still tracked —
                    # reconstruct the fingertip span in the same basis the scale
                    # was calibrated in, so the buffer/timer don't reset.
                    span_px = wrist_span_px * self._ft_wrist_ratio
                    if left_tip is None or right_tip is None:
                        # keep the on-screen line sensible during the bridge
                        left_tip = left_tip or l_wrist
                        right_tip = right_tip or r_wrist

                # ── stability gate: only trust the span once the recent
                #    frames agree within _WS_STABLE_CV (person holding still).
                if span_px:
                    self._recent_px.append(span_px)
                else:
                    self._recent_px.clear()   # lost a hand → must re-settle
                stable = False
                if len(self._recent_px) >= _WS_STABLE_WIN:
                    arr = np.asarray(self._recent_px, dtype=float)
                    mean = float(arr.mean())
                    if mean > 0 and float(arr.std()) / mean <= _WS_STABLE_CV:
                        stable = True
                with self._lock:
                    self._stable = stable

                # ── read shared state (lock held briefly) ──────────────
                with self._lock:
                    mode = self.mode
                    calibration = self.calibration

                # ── all heavy work (drawing, encoding) outside the lock ──
                if mode == "MEASURE":
                    l_ang, r_ang = ws.draw_arm_angles(frame, p_result, fw, fh)
                    arms_ok = ws._angle_ok(l_ang) and ws._angle_ok(r_ang)
                    result_cm = ws.pixels_to_cm(span_px, calibration) if (span_px and calibration and arms_ok) else None
                    # Distance correction: the scale was calibrated at
                    # calibration["distance_cm"]. If the coach measures at a
                    # different distance (entered on screen3), rescale by
                    # measure_distance / calib_distance so the reading is correct
                    # at the venue's actual distance. No distance entered -> no
                    # change (assumes measurement at the calibration distance).
                    if result_cm is not None and self.measure_distance_cm:
                        calib_d = calibration.get("distance_cm") or ws.PERSON_DISTANCE_CM
                        result_cm *= self.measure_distance_cm / calib_d

                    ws.draw_live_measurement(frame, result_cm, left_tip, right_tip, fw, fh)
                    if not arms_ok:
                        _draw_align_warning(frame)

                    with self._lock:
                        history_snap = list(self.history[-6:])
                    ws.draw_hud(frame, mode, calibration, result_cm, history_snap, fw, fh)

                    # write state + update snap
                    with self._lock:
                        self.result_cm = result_cm
                        self._arms_ok = arms_ok
                        if self.snap is not None and self.snap["phase"] == "COUNTING":
                            # Only collect settled frames (arms straight AND the
                            # span has stopped wobbling).
                            if result_cm is not None and arms_ok and stable:
                                self.snap["samples"].append(result_cm)
                            elapsed = time.time() - self.snap["start_time"]
                            if elapsed >= ws.SNAPSHOT_SECS:
                                if len(self.snap["samples"]) >= _WS_MIN_SNAP_SAMPLES:
                                    self.snap["final_cm"] = ws.compute_stable_median(self.snap["samples"])
                                else:
                                    # Too few steady frames — signal a retry
                                    # instead of reporting a shaky number.
                                    self.snap["final_cm"] = 0.0
                                self.snap["phase"] = "DONE"
                        snap_copy = dict(self.snap) if self.snap is not None else None

                    if snap_copy is not None:
                        ws.draw_snapshot_overlay(frame, snap_copy)

                elif mode == "CALIBRATE":
                    with self._lock:
                        phase = self.cs.get("phase", "SPREAD")
                        if phase == "SPREAD":
                            if span_px:
                                self.cs.setdefault("px_buf", []).append(span_px)
                                if self.cs.get("hold_start") is None:
                                    self.cs["hold_start"] = time.time()
                                elapsed = time.time() - self.cs["hold_start"]
                                self.cs["avg_px"] = float(np.mean(self.cs["px_buf"][-90:]))
                                self.cs["settling"] = not stable
                                # Advance only after a held, SETTLED spread — the
                                # recent span must agree within _WS_STABLE_CV, so
                                # the locked scale isn't taken from a wobbly frame.
                                if elapsed >= 2.0 and len(self.cs["px_buf"]) >= 10 and stable:
                                    self.cs["avg_px"] = float(np.mean(self.cs["px_buf"][-90:]))
                                    self.cs["phase"] = "ENTER_CM"
                            else:
                                self.cs["hold_start"] = None
                                self.cs["settling"] = True
                        cs_copy = dict(self.cs)
                    _draw_calib_overlay_web(frame, cs_copy, fw, fh, left_tip, right_tip)

                # encode outside the lock; only the pointer swap needs it
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    jpeg_bytes = buf.tobytes()
                    with self._lock:
                        self._jpeg = jpeg_bytes

        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False