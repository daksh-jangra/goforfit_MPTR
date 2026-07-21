"""
Shuttle Run Test (10 m) — Dual Camera + Whistle Start
======================================================

WHAT THIS DOES
---------------
- Two cameras face each other across a 10 m track, one at each end.
- Up to 4 students run in 4 fixed lanes (side by side), same lane at both ends.
- A whistle (2.4-4.1 kHz) detected via microphone starts the timer for everyone.
- MediaPipe Pose tracks ankle position in each lane to detect "touches" of the
  line at each end. Touches must alternate ends (far, near, far, near, ...).
- Each student needs N touches (you set N, default 4) to finish.
- Results (per-touch timestamps + total time per student) are saved to a
  timestamped CSV file, plus a raw JSON event log.

IMPORTANT ASSUMPTIONS (read this before running)
--------------------------------------------------
1. Students run in fixed, non-crossing lanes. This is what lets us match
   identity between the two independent cameras (true cross-camera person
   re-identification without lane discipline is unreliable and out of scope
   here). Mark 4 lanes on the ground if you don't already have them.
2. Camera 1 = "End A" (e.g. start line). Camera 2 = "End B" (far end, 10 m away).
3. You calibrate ONCE per session by clicking on each camera's preview image:
   - 5 points marking the 4 lane boundaries (left to right)
   - 1 point + 1 point marking a horizontal "touch line" (the line the
     student's foot must cross to register a touch)
   Camera 2 usually sees the lanes mirrored left-right relative to Camera 1
   (because it faces the opposite direction) — the script asks you to confirm
   this at calibration time so lane numbers line up correctly.
4. This uses ankle landmarks from MediaPipe Pose, run separately per lane crop
   (so 4 lanes x 2 cameras = 8 lightweight Pose instances). This is fine for
   4 people but is not a substitute for a proper multi-person tracker if you
   later want crossing lanes / crowds.

REQUIRED PACKAGES
-------------------
pip install opencv-python mediapipe numpy sounddevice

HARDWARE
---------
- 2 USB/webcams, one at each end of the track, mounted so the full lane width
  and the touch line are visible in frame.
- A microphone connected/visible to `sounddevice` for whistle detection
  (a laptop mic near the track, or a dedicated USB mic, works).

RUN
----
python shuttleruntest.py --cam-a 0 --cam-b 1

Press 'q' in either video window to abort early. Press 'c' during calibration
to confirm each calibration step.

GoForFit integration
---------------------
When launched by the GoForFit web app, a `--session-id` argument is passed
in. If present, finishing times for each lane are POSTed to the FastAPI
backend's `/api/stage/shuttle_run` endpoint for human review/save on
screen3 (same flow as every other AI test in GoForFit), in addition to the
CSV/JSON files always written locally.
"""

import argparse
import csv
import json
import os
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

import mp_tasks  # Tasks-API detector/drawing helpers (see mp_tasks.py)

try:
    import sounddevice as sd
except OSError as e:
    sd = None
    print(f"[WARN] sounddevice failed to import audio backend: {e}")
    print("       Whistle auto-detection will be unavailable; you can fall back "
          "to pressing SPACE to start manually.")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

NUM_LANES = 4
TRACK_LENGTH_M = 10.0

WHISTLE_FREQ_MIN = 2400.0     # Hz
WHISTLE_FREQ_MAX = 4100.0     # Hz
WHISTLE_MIN_MAGNITUDE = 40.0  # tune this to your mic's sensitivity/noise floor
WHISTLE_CONSEC_FRAMES = 3     # consecutive audio blocks needed to confirm a whistle
AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 2048

TOUCH_COOLDOWN_SEC = 0.6      # min time between two touches of the SAME lane
POSE_MODEL_COMPLEXITY = 0     # 0 = fastest, good enough for ankle position
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

RESULTS_DIR = "shuttle_run_results"


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass
class Student:
    lane: int                                   # 1-based lane number
    touches_required: int
    touch_timestamps: List[float] = field(default_factory=list)
    last_end: Optional[str] = None              # "A" or "B" — which end last touched
    finished: bool = False
    finish_time: Optional[float] = None

    def expected_next_end(self) -> str:
        """Students must start by running towards End B first."""
        if self.last_end is None:
            return "B"
        return "A" if self.last_end == "B" else "B"

    def register_touch(self, end: str, t: float) -> bool:
        """Attempt to register a touch at `end` at time `t`. Returns True if accepted."""
        if self.finished:
            return False
        if end != self.expected_next_end():
            return False  # out-of-sequence touch, ignore (likely lingering near line)
        if self.touch_timestamps:
            if t - self.touch_timestamps[-1] < TOUCH_COOLDOWN_SEC:
                return False
        self.touch_timestamps.append(t)
        self.last_end = end
        if len(self.touch_timestamps) >= self.touches_required:
            self.finished = True
            self.finish_time = t
        return True

    @property
    def touch_count(self) -> int:
        return len(self.touch_timestamps)


class StudentManager:
    """Thread-safe holder for all students' state."""

    def __init__(self, num_lanes: int, touches_required: int):
        self.lock = threading.Lock()
        self.students = {lane: Student(lane, touches_required) for lane in range(1, num_lanes + 1)}
        self.start_time: Optional[float] = None
        self.event_log = []  # raw log for JSON dump

    def start(self):
        with self.lock:
            self.start_time = time.time()

    def register_touch(self, lane: int, end: str) -> bool:
        t = time.time()
        with self.lock:
            if self.start_time is None:
                return False
            student = self.students[lane]
            accepted = student.register_touch(end, t)
            if accepted:
                self.event_log.append({
                    "lane": lane, "end": end, "time": t,
                    "elapsed": t - self.start_time,
                    "touch_number": student.touch_count
                })
            return accepted

    def all_finished(self) -> bool:
        with self.lock:
            return all(s.finished for s in self.students.values())

    def snapshot(self):
        with self.lock:
            return {lane: (s.touch_count, s.finished) for lane, s in self.students.items()}


# ----------------------------------------------------------------------------
# Whistle detection (microphone -> FFT -> frequency band check)
# ----------------------------------------------------------------------------

class WhistleDetector:
    """Listens on the default microphone and sets `detected` Event when a
    sustained tone in [WHISTLE_FREQ_MIN, WHISTLE_FREQ_MAX] Hz is heard."""

    def __init__(self):
        self.detected = threading.Event()
        self._consec = 0
        self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if self.detected.is_set():
            return
        samples = indata[:, 0].astype(np.float64)
        windowed = samples * np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / AUDIO_SAMPLE_RATE)

        band_mask = (freqs >= WHISTLE_FREQ_MIN) & (freqs <= WHISTLE_FREQ_MAX)
        if not np.any(band_mask):
            return
        band_energy = spectrum[band_mask]
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        peak_mag = spectrum[peak_idx]

        in_band_and_loud = (
            WHISTLE_FREQ_MIN <= peak_freq <= WHISTLE_FREQ_MAX
            and peak_mag >= WHISTLE_MIN_MAGNITUDE
            and band_energy.max() >= WHISTLE_MIN_MAGNITUDE
        )

        if in_band_and_loud:
            self._consec += 1
        else:
            self._consec = 0

        if self._consec >= WHISTLE_CONSEC_FRAMES:
            print(f"[WHISTLE] Detected tone ~{peak_freq:.0f} Hz (mag={peak_mag:.1f}) -> START")
            self.detected.set()

    def start(self):
        if sd is None:
            print("[WHISTLE] Audio backend unavailable — use manual start (SPACE).")
            return
        self._stream = sd.InputStream(
            channels=1,
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=AUDIO_BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        print(f"[WHISTLE] Listening for whistle in {WHISTLE_FREQ_MIN:.0f}-{WHISTLE_FREQ_MAX:.0f} Hz...")

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


# ----------------------------------------------------------------------------
# Calibration: click lane boundaries + touch line per camera
# ----------------------------------------------------------------------------

class Calibration:
    def __init__(self, lane_boundaries_x: List[int], touch_line_y: int, frame_w: int, frame_h: int):
        self.lane_boundaries_x = lane_boundaries_x  # length NUM_LANES+1, sorted ascending
        self.touch_line_y = touch_line_y
        self.frame_w = frame_w
        self.frame_h = frame_h
        # True on the camera that sees the lanes right-to-left (End B, facing
        # End A across the track). Both ends feed one StudentManager keyed by
        # lane number, so without this End B's lane 1 is End A's lane 4.
        self.reverse_lane_numbering = False

    def lane_bounds(self, lane_idx_0based: int) -> Tuple[int, int]:
        return self.lane_boundaries_x[lane_idx_0based], self.lane_boundaries_x[lane_idx_0based + 1]

    def lane_number_for_column(self, column_idx_0based: int) -> int:
        """Lane number for the idx-th column, left to right as displayed."""
        n = len(self.lane_boundaries_x) - 1
        return n - column_idx_0based if self.reverse_lane_numbering else column_idx_0based + 1


import urllib.request


class MJPEGStreamReader:
    """Manually reads an MJPEG-over-HTTP stream (as served by DroidCam / IP Webcam)
    and decodes frames with OpenCV, bypassing cv2.VideoCapture's network layer.
    This is needed because OpenCV's built-in HTTP/FFMPEG backend frequently fails
    to open these multipart streams on Windows even when the URL works fine in a
    browser. Exposes the same isOpened()/read()/release() interface as
    cv2.VideoCapture so it's a drop-in replacement."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self._opened = False
        self._bytes_buffer = b""
        try:
            self._stream = urllib.request.urlopen(url, timeout=timeout)
            self._opened = True
        except Exception as e:
            print(f"[MJPEG] Failed to open stream {url}: {e}")
            self._stream = None

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._stream is None:
            return False, None
        read_start = time.time()
        max_read_seconds = 2.0  # never block the caller longer than this
        try:
            while True:
                if time.time() - read_start > max_read_seconds:
                    return False, None  # give up this attempt, let caller retry / refresh GUI
                chunk = self._stream.read(4096)
                if not chunk:
                    return False, None
                self._bytes_buffer += chunk
                if len(self._bytes_buffer) > 2_000_000:
                    # Safety valve: never let the buffer grow unbounded if we can't
                    # find valid JPEG markers (corrupted/unexpected stream format).
                    self._bytes_buffer = b""
                    return False, None
                start = self._bytes_buffer.find(b'\xff\xd8')  # JPEG start marker
                end = self._bytes_buffer.find(b'\xff\xd9')    # JPEG end marker
                if start != -1 and end != -1 and end > start:
                    jpg = self._bytes_buffer[start:end + 2]
                    self._bytes_buffer = self._bytes_buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
        except Exception as e:
            print(f"[MJPEG] Stream read error: {e}")
            self._opened = False
            return False, None

    def release(self):
        self._opened = False
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass


def resolve_camera_source(source: str):
    """Allow --cam-a/--cam-b to be either a plain USB index ('0', '1', ...)
    or a phone/IP camera stream URL (e.g. http://192.168.1.5:8080/video)."""
    if source.isdigit():
        return int(source)
    return source  # cv2.VideoCapture accepts URL strings directly


def open_capture(source):
    """Open a camera source. USB indices use cv2.VideoCapture directly.
    URL sources use our manual MJPEGStreamReader, since cv2.VideoCapture's
    network backend is unreliable with DroidCam/IP-Webcam streams on Windows.

    For local indices, try a short list of capture backends in turn — Windows
    webcams are inconsistent about which one (DirectShow, Media Foundation,
    or the OS default) actually opens them."""
    if isinstance(source, str):
        return MJPEGStreamReader(source)
    else:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(source)  # last resort, will report not-opened upstream


def calibrate_camera(cap: cv2.VideoCapture, window_name: str) -> Calibration:
    """Interactive calibration: click NUM_LANES+1 x-coordinates left->right for
    lane boundaries, then click 2 points for the touch line (same y ideally)."""
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    print(f"\n[CALIBRATE] {window_name}")
    print(f"  Step 1: click {NUM_LANES + 1} points left-to-right marking the {NUM_LANES} lane boundaries.")
    print(f"  Step 2: click 2 points marking the touch line (roughly same height).")
    print(f"  Press 'r' to reset points, 'c' to confirm once all points are placed.")

    total_points_needed = (NUM_LANES + 1) + 2
    frame = None
    last_good_frame = None
    consecutive_failures = 0
    max_consecutive_failures = 150  # ~15 seconds at ~10 reads/sec before giving up

    while True:
        ret, new_frame = cap.read()

        if ret and new_frame is not None:
            frame = new_frame
            last_good_frame = new_frame
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if last_good_frame is not None:
                frame = last_good_frame.copy()
                cv2.putText(frame, "WARNING: lost frame, still trying to reconnect...",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                # Never got a single frame yet — show a placeholder instead of freezing.
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera stream...", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if consecutive_failures >= max_consecutive_failures:
                cv2.destroyWindow(window_name)
                sys.exit(
                    f"\n[FAIL] {window_name}: no frames received for ~15 seconds.\n"
                    f"  The camera/stream is not delivering video. Checklist:\n"
                    f"  1. Is the phone screen on and DroidCam actively streaming?\n"
                    f"  2. Is the phone still on the same WiFi (check it didn't drop/switch to mobile data)?\n"
                    f"  3. Re-test with: python test_single_camera.py --source <same-url-or-index>\n"
                    f"  4. If using PC webcam, is another app (Zoom/Teams/Camera app) already using it?"
                )

        display = frame.copy()

        for i, p in enumerate(points):
            cv2.circle(display, p, 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(display, f"Points: {len(points)}/{total_points_needed}  ('r'=reset 'c'=confirm)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            points = []
        elif key == ord('c') and len(points) >= total_points_needed:
            break
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            sys.exit("Calibration aborted by user.")

    cv2.destroyWindow(window_name)

    lane_pts = points[:NUM_LANES + 1]
    line_pts = points[NUM_LANES + 1: NUM_LANES + 3]

    lane_xs = sorted([p[0] for p in lane_pts])
    touch_line_y = int(np.mean([p[1] for p in line_pts]))

    h, w = frame.shape[:2]
    return Calibration(lane_xs, touch_line_y, w, h)


# ----------------------------------------------------------------------------
# Per-lane touch detection using MediaPipe Pose on lane crops
# ----------------------------------------------------------------------------

class LaneTracker:
    """Tracks a single lane in a single camera: runs Pose on the lane crop,
    watches the ankle y-position relative to the touch line, and fires a
    touch event via callback when the line is crossed (with cooldown)."""

    def __init__(self, lane_number: int, end_label: str, x0: int, x1: int, touch_line_y: int,
                 manager: StudentManager):
        self.lane_number = lane_number
        self.end_label = end_label  # "A" or "B"
        self.x0, self.x1 = x0, x1
        self.touch_line_y = touch_line_y
        self.manager = manager

        # Tasks API in VIDEO mode — the replacement for solutions.Pose(),
        # which was removed in mediapipe 0.10.30. Each lane gets its own
        # detector AND its own clock: VIDEO mode rejects a timestamp that
        # doesn't advance, and 4 lanes x 2 cameras share nothing.
        self.pose = mp_tasks.pose_landmarker(
            num_poses=1,
            detection_conf=MIN_DETECTION_CONFIDENCE,
            tracking_conf=MIN_TRACKING_CONFIDENCE,
            model=("pose_landmarker_full.task" if POSE_MODEL_COMPLEXITY >= 2
                   else "pose_landmarker_lite.task"),
        )
        self._clock = mp_tasks.VideoClock()
        self._prev_side: Optional[str] = None  # "near"/"far" relative to line
        self._last_touch_time = 0.0

    def process(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Process this lane's crop of `frame`. Returns ankle (x,y) in full-frame
        coords if a person was detected, else None. Also fires touch events."""
        crop = frame[:, self.x0:self.x1]
        if crop.size == 0:
            return None
        result = self.pose.detect_for_video(mp_tasks.to_mp_image(crop), self._clock.tick())
        if not result.pose_landmarks:
            return None

        # Tasks returns one landmark list PER detected pose; num_poses=1 so
        # there is at most one, where solutions returned a single object.
        lm = result.pose_landmarks[0]
        L_ANKLE, R_ANKLE = mp_tasks.LEFT_ANKLE, mp_tasks.RIGHT_ANKLE
        crop_h, crop_w = crop.shape[:2]

        candidates = []
        for lmk_idx in (L_ANKLE, R_ANKLE):
            lmk = lm[lmk_idx]
            if lmk.visibility >= 0.4:
                candidates.append(lmk.y)
        if not candidates:
            return None

        ankle_y_norm = max(candidates)  # the lower (larger y) of the two visible ankles = grounded foot
        ankle_y_px = int(ankle_y_norm * crop_h)
        ankle_x_px = int(0.5 * crop_w)  # only y matters for line-crossing; x is nominal (lane center)

        # Determine side relative to touch line. In image coords, y grows downward.
        # We treat "far" = above the line (smaller y), "near" = below/at the line.
        side = "far" if ankle_y_px < self.touch_line_y else "near"

        if self._prev_side is not None and side != self._prev_side and side == "near":
            # Crossed from far -> near => touched the line
            now = time.time()
            if now - self._last_touch_time >= TOUCH_COOLDOWN_SEC:
                accepted = self.manager.register_touch(self.lane_number, self.end_label)
                if accepted:
                    self._last_touch_time = now

        self._prev_side = side
        return (self.x0 + ankle_x_px, ankle_y_px)

    def close(self):
        self.pose.close()


# ----------------------------------------------------------------------------
# Camera worker thread: capture, run lane trackers, draw overlay, push to queue
# ----------------------------------------------------------------------------

class CameraWorker(threading.Thread):
    def __init__(self, cam_index, end_label: str, calib: Calibration,
                 manager: StudentManager, display_queue: "queue.Queue", stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.end_label = end_label
        self.calib = calib
        self.manager = manager
        self.display_queue = display_queue
        self.stop_event = stop_event

        self.cap = open_capture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {cam_index} for End {end_label}")

        self.lane_trackers = []
        for lane_idx0 in range(NUM_LANES):
            x0, x1 = calib.lane_bounds(lane_idx0)
            self.lane_trackers.append(
                LaneTracker(calib.lane_number_for_column(lane_idx0), end_label,
                            x0, x1, calib.touch_line_y, manager)
            )

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            for i, tracker in enumerate(self.lane_trackers):
                ankle_pos = tracker.process(frame)
                x0, x1 = tracker.x0, tracker.x1
                cv2.rectangle(frame, (x0, 0), (x1, frame.shape[0]), (80, 80, 80), 1)
                if ankle_pos:
                    cv2.circle(frame, ankle_pos, 6, (0, 0, 255), -1)

            cv2.line(frame, (0, self.calib.touch_line_y), (frame.shape[1], self.calib.touch_line_y),
                      (0, 255, 255), 2)

            snap = self.manager.snapshot()
            for lane_idx0 in range(NUM_LANES):
                lane_num = self.calib.lane_number_for_column(lane_idx0)
                x0, x1 = self.calib.lane_bounds(lane_idx0)
                count, finished = snap[lane_num]
                label = f"L{lane_num}: {count}"
                if finished:
                    label += " DONE"
                color = (0, 200, 0) if finished else (255, 255, 255)
                cv2.putText(frame, label, (x0 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.putText(frame, f"End {self.end_label}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            if not self.display_queue.full():
                self.display_queue.put((self.end_label, frame))

        self.cap.release()
        for t in self.lane_trackers:
            t.close()


# ----------------------------------------------------------------------------
# Results saving
# ----------------------------------------------------------------------------

def save_results(manager: StudentManager, touches_required: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"shuttle_run_{ts_str}.csv")
    json_path = os.path.join(RESULTS_DIR, f"shuttle_run_{ts_str}_events.json")

    with manager.lock:
        start_time = manager.start_time
        rows = []
        for lane, student in sorted(manager.students.items()):
            row = {"lane": lane, "touches_recorded": student.touch_count,
                   "finished": student.finished}
            for i in range(touches_required):
                key = f"touch_{i+1}_time_s"
                if i < len(student.touch_timestamps) and start_time:
                    row[key] = round(student.touch_timestamps[i] - start_time, 3)
                else:
                    row[key] = ""
            if student.finished and student.finish_time and start_time:
                row["total_time_s"] = round(student.finish_time - start_time, 3)
            else:
                row["total_time_s"] = ""
            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            fieldnames = ["lane", "touches_recorded", "finished"] + \
                         [f"touch_{i+1}_time_s" for i in range(touches_required)] + \
                         ["total_time_s"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(json_path, "w") as f:
            json.dump({
                "track_length_m": TRACK_LENGTH_M,
                "touches_required": touches_required,
                "start_time_epoch": start_time,
                "events": manager.event_log,
            }, f, indent=2)

    print(f"\n[SAVED] Results CSV : {csv_path}")
    print(f"[SAVED] Raw events  : {json_path}")


def stage_shuttle_run_results(session_id: Optional[int], manager: StudentManager):
    """POSTs each lane's finishing time to the GoForFit FastAPI backend's
    staging endpoint instead of writing to the database directly — the
    result sits as 'pending' until a human reviews and confirms it on
    screen3, matching the rest of GoForFit's AI-test result flow."""
    if session_id is None:
        return
    try:
        import requests
        with manager.lock:
            start_time = manager.start_time
            laps_payload = []
            for lane, student in sorted(manager.students.items()):
                if student.finished and student.finish_time and start_time:
                    laps_payload.append({
                        "lane": lane,
                        "time_sec": round(student.finish_time - start_time, 3),
                    })
        if not laps_payload:
            print("[DB] No finished lanes to stage.")
            return
        requests.post(
            "http://localhost:5000/api/stage/shuttle_run",
            json={"session_id": session_id, "laps": laps_payload},
            timeout=2,
        )
        print(f"[DB] shuttle_run staged for review  session={session_id}  "
              f"lanes={[l['lane'] for l in laps_payload]}")
    except Exception as e:
        print(f"[DB] Staging failed (is the GoForFit server running?): {e}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dual-camera 10m Shuttle Run Test")
    parser.add_argument("--cam-a", type=str, default="http://192.168.1.74:4747/video",
                         help="Camera for End A: a USB index (e.g. 0) or a phone stream URL "
                              "(e.g. http://192.168.1.74:4747/video)")
    parser.add_argument("--cam-b", type=str, default="1",
                         help="Camera for End B: a USB index (e.g. 1) or a phone stream URL "
                              "(e.g. http://192.168.1.6:8080/video)")
    parser.add_argument("--touches", type=int, default=None,
                         help="Number of line touches required to finish (default: asked interactively)")
    parser.add_argument("--manual-start", action="store_true",
                         help="Skip whistle detection; press SPACE in a video window to start instead")
    parser.add_argument("--session-id", type=int, default=None,
                         help="GoForFit session id — stages results for review on "
                              "screen3 instead of CSV-only")
    parser.add_argument("--mirror-b", type=str, default=None, choices=["y", "n"],
                         help="Skip the interactive mirror-check prompt: 'y' if End B's "
                              "camera view is mirrored left-right vs End A, else 'n'")
    args = parser.parse_args()

    touches_required = args.touches
    if touches_required is None:
        if sys.stdin.isatty():
            try:
                touches_required = int(input(f"How many line touches to complete the test? [default 4]: ") or 4)
            except ValueError:
                touches_required = 4
        else:
            touches_required = 4
            print("[CONFIG] No terminal attached and --touches not given — defaulting to 4.")
    print(f"[CONFIG] Track length: {TRACK_LENGTH_M} m | Lanes: {NUM_LANES} | Touches required: {touches_required}")

    manager = StudentManager(NUM_LANES, touches_required)

    cam_a_src = resolve_camera_source(args.cam_a)
    cam_b_src = resolve_camera_source(args.cam_b)

    cap_a = open_capture(cam_a_src)
    cap_b = open_capture(cam_b_src)
    if not cap_a.isOpened():
        sys.exit(f"Could not open camera A ({cam_a_src}).\n"
                  f"  1. Open that URL in a browser on this PC — you must see live video.\n"
                  f"  2. Make sure the DroidCam app is open and 'streaming' on the phone.\n"
                  f"  3. Make sure phone and PC are on the same WiFi network (not phone data).\n"
                  f"  4. Try the alternate path: http://192.168.1.74:4747/mjpegfeed?640x480\n"
                  f"  5. Temporarily disable PC firewall/VPN, which can block LAN connections.")
    if not cap_b.isOpened():
        sys.exit(f"Could not open camera B ({cam_b_src}).\n"
                  f"  Same checklist as camera A above — verify the URL loads in a browser first.")

    print("\nCalibrating End A camera (start line)...")
    calib_a = calibrate_camera(cap_a, "Calibrate End A")
    print("Calibrating End B camera (10m end)...")
    calib_b = calibrate_camera(cap_b, "Calibrate End B")

    if args.mirror_b is not None:
        mirror_check = args.mirror_b.strip().lower()
        print(f"[CONFIG] End B mirrored: {mirror_check} (from --mirror-b)")
    elif sys.stdin.isatty():
        mirror_check = input("Does End B's camera view show lanes MIRRORED left-right "
                              "compared to End A? [y/N]: ").strip().lower()
    else:
        # Launched non-interactively (e.g. by the GoForFit web app) — stdin is
        # closed, so input() would raise EOFError and crash right after
        # calibration. Default to "no mirroring"; pass --mirror-b y explicitly
        # if End B's camera really does face the opposite direction.
        mirror_check = "n"
        print("[CONFIG] No terminal attached — defaulting End B mirror to 'n'. "
              "Pass --mirror-b y next time if End B's view is flipped left-right.")
    # Reverse the lane NUMBERING, not the boundary coordinates. The previous
    # version mirrored the x-values (frame_w - x, re-sorted), which is a no-op
    # for a centred lane layout — [100,200,300,400,500] in a 600px frame maps
    # to itself — and for an off-centre one it slides the crop windows onto
    # positions where no lane is. The columns are already where the lanes are;
    # only the labels are backwards.
    calib_b.reverse_lane_numbering = (mirror_check == "y")

    cap_a.release()
    cap_b.release()

    stop_event = threading.Event()
    display_queue: "queue.Queue" = queue.Queue(maxsize=4)

    worker_a = CameraWorker(cam_a_src, "A", calib_a, manager, display_queue, stop_event)
    worker_b = CameraWorker(cam_b_src, "B", calib_b, manager, display_queue, stop_event)

    whistle = WhistleDetector()
    if not args.manual_start:
        whistle.start()
    else:
        print("[START] Manual start mode: press SPACE in a video window to begin.")

    worker_a.start()
    worker_b.start()

    print("\n[READY] Waiting for start signal (whistle or SPACE)...")
    started = False
    frames = {}

    try:
        while not stop_event.is_set():
            try:
                label, frame = display_queue.get(timeout=0.5)
                frames[label] = frame
            except queue.Empty:
                pass

            for label, frame in frames.items():
                cv2.imshow(f"End {label}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[ABORT] User quit.")
                break
            if not started and (key == ord(' ') or (not args.manual_start and whistle.detected.is_set())):
                manager.start()
                started = True
                print("[START] Timer started!")

            if started and manager.all_finished():
                print("[DONE] All students finished.")
                time.sleep(1.0)
                break
    finally:
        stop_event.set()
        whistle.stop()
        worker_a.join(timeout=2)
        worker_b.join(timeout=2)
        cv2.destroyAllWindows()

    if manager.start_time is not None:
        save_results(manager, touches_required)
        stage_shuttle_run_results(args.session_id, manager)
    else:
        print("[INFO] Test never started — nothing to save.")


if __name__ == "__main__":
    main()
