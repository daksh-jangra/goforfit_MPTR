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
from datetime import datetime

import cv2
import numpy as np

import wingspan2 as ws  # reuse geometry / drawing / detector-building code

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

    def __init__(self, session_id=None, on_stage=None):
        self.session_id = session_id
        self.on_stage = on_stage  # callback(wingspan_cm) — stages result for screen3 review

        self._lock = threading.Lock()
        self._jpeg = None
        self._running = False
        self._thread = None
        self._error = None

        self.cap = None
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

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return

        models = ws.ensure_models(_TESTS_DIR)
        self.hand_det = ws.build_hand_detector(models["hand_landmarker.task"])
        self.pose_det = ws.build_pose_detector(models["pose_landmarker_lite.task"])

        self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not self.cap.isOpened():
            # fallback: try default index without backend hint
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera (index 0)")
        # 640×480 gives ~4× fewer pixels than 1280×720 → dramatically faster MediaPipe
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Ask the driver for 30fps and minimal internal buffer (reduces latency)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        for det in (self.hand_det, self.pose_det):
            try:
                if det is not None:
                    det.close()
            except Exception:
                pass

    def is_running(self):
        return self._running

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
        # Frame-skip counter: run MediaPipe every N frames, reuse last results otherwise
        _SKIP = 2          # process 1 out of every 2 frames (~15 inference fps at 30fps cap)
        _frame_count = 0
        _last_h_result = None
        _last_p_result = None
        try:
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

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

                left_tip = right_tip = None
                if h_result.hand_landmarks and h_result.handedness:
                    for lms, handedness in zip(h_result.hand_landmarks, h_result.handedness):
                        label = handedness[0].category_name
                        tip = lms[ws.MIDDLE_TIP_IDX]
                        if label == "Left":
                            left_tip = (tip.x, tip.y)
                        else:
                            right_tip = (tip.x, tip.y)

                span_px = None
                if left_tip and right_tip:
                    lx, ly = left_tip[0] * fw, left_tip[1] * fh
                    rx, ry = right_tip[0] * fw, right_tip[1] * fh
                    span_px = ws.px_dist((lx, ly), (rx, ry))

                # ── read shared state (lock held briefly) ──────────────
                with self._lock:
                    mode = self.mode
                    calibration = self.calibration

                # ── all heavy work (drawing, encoding) outside the lock ──
                if mode == "MEASURE":
                    l_ang, r_ang = ws.draw_arm_angles(frame, p_result, fw, fh)
                    arms_ok = ws._angle_ok(l_ang) and ws._angle_ok(r_ang)
                    result_cm = ws.pixels_to_cm(span_px, calibration) if (span_px and calibration and arms_ok) else None

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
                            if result_cm is not None and arms_ok:
                                self.snap["samples"].append(result_cm)
                            elapsed = time.time() - self.snap["start_time"]
                            if elapsed >= ws.SNAPSHOT_SECS:
                                if self.snap["samples"]:
                                    self.snap["final_cm"] = ws.compute_stable_median(self.snap["samples"])
                                else:
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
                                if elapsed >= 2.0 and len(self.cs["px_buf"]) >= 10:
                                    self.cs["avg_px"] = float(np.mean(self.cs["px_buf"][-90:]))
                                    self.cs["phase"] = "ENTER_CM"
                            else:
                                self.cs["hold_start"] = None
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