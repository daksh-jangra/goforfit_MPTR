import cv2
import numpy as np
import time
import csv
import argparse
import os
import threading
from datetime import datetime

# ── tuneable constants ─────────────────────────────────────────────────────────
COOLDOWN_S      = 0.001
DETECTION_BAND  = 65     # wide band — distant centroids are imprecise
MIN_BLOB_AREA   = 150    # very small — catch distant figures in blob fallback
PROC_SCALE      = 1.0    # full resolution — don't shrink; far-off people need every pixel
SKIP_FRAMES     = 1      # every other frame — better tracking continuity for multiple people
MAX_POSES       = 8      # maximum humans to detect at once
DISPLAY_WIDTH   = 960    # resize preview window only; detection still uses camera frames

# ── human validation thresholds ───────────────────────────────────────────────
# For DISTANT / SMALL people MediaPipe returns few, low-confidence landmarks.
# We use a 2-tier system:
#   • RELAXED  — fires when the person is small (bounding box < SMALL_PERSON_PX²)
#   • STRICT   — fires when the person is close / large
#
# This avoids filtering out far-off swimmers while still rejecting background noise.

SMALL_PERSON_PX     = 80    # bounding-box side (px) below which "relaxed" rules apply

# Relaxed rules (distant / small)
MIN_LM_RELAXED      = 2     # need at least 2 visible landmarks
VIS_THRESH_RELAXED  = 0.05  # very low — trust MediaPipe's pose detection more than scores
MIN_TORSO_RELAXED   = 8     # shoulder or hip pair spread (px); tiny at distance

# Strict rules (close / large)
MIN_LM_STRICT       = 4     # need 4+ landmarks for a confident close detection
VIS_THRESH_STRICT   = 0.15  # modest — still lenient to avoid missing turned swimmers
MIN_TORSO_STRICT    = 18    # shoulder/hip spread must be visible

# How many frames a tracked person can be "missing" before their ID is retired.
TRACKER_GRACE_FRAMES = 15   # raised — distant detections are noisier / intermittent

# Whistle detection
WHISTLE_FREQ_LOW  = 2000
WHISTLE_FREQ_HIGH = 4500
WHISTLE_THRESHOLD = 0.45
WHISTLE_HOLD_S    = 0.45
AUDIO_CHUNK       = 1024
AUDIO_RATE        = 16000

_ID_PALETTE = [
    (0,   220, 255),   # 0  vivid yellow
    (0,   80,  255),   # 1  red-orange
    (60,  255, 60),    # 2  lime green
    (255, 60,  200),   # 3  hot pink / magenta
    (255, 200, 0),     # 4  sky blue
    (0,   165, 255),   # 5  orange
    (200, 50,  255),   # 6  violet
    (50,  255, 220),   # 7  spring green
    (0,   255, 128),   # 8  seafoam
    (255, 0,   80),    # 9  electric blue
    (0,   200, 255),   # 10 amber
    (180, 255, 0),     # 11 chartreuse
    (255, 130, 0),     # 12 cobalt
    (100, 0,   255),   # 13 deep red
    (0,   255, 0),     # 14 pure green
    (128, 0,   255),   # 15 crimson
    (255, 255, 0),     # 16 aqua
    (0,   128, 255),   # 17 gold
    (200, 100, 255),   # 18 orchid
    (50,  220, 180),   # 19 olive-teal
]


# ── helpers ────────────────────────────────────────────────────────────────────
def format_time(s):
    m  = int(s) // 60
    sc = int(s) % 60
    ms = int((s - int(s)) * 1000)
    return f"{m:02d}:{sc:02d}.{ms:03d}"


def point_line_dist(pt, p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return np.hypot(pt[0] - p1[0], pt[1] - p1[1])
    return abs((pt[1] - p1[1]) * dx - (pt[0] - p1[0]) * dy) / length


def which_side(pt, p1, p2):
    val = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
    return 1 if val >= 0 else -1


def detect_white_line(frame):
    """Finds the white painted finish line; returns (p1, p2) or (None, None)."""
    h, w = frame.shape[:2]
    roi_x1, roi_x2 = int(w * 0.20), int(w * 0.80)
    roi = frame[:, roi_x1:roi_x2]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 55, 255]))
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, k, iterations=2)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180,
                            threshold=60,
                            minLineLength=int(h * 0.25),
                            maxLineGap=int(h * 0.12))
    if lines is None:
        return None, None
    good = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        if y1 == y2:
            continue
        angle = abs(np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1))))
        if angle <= 30:
            good.append((x1, y1, x2, y2))
    if not good:
        return None, None
    x_mids = [(x1 + x2) // 2 for x1, y1, x2, y2 in good]
    best, visited = [], [False] * len(good)
    for i, xm in enumerate(x_mids):
        if visited[i]:
            continue
        cluster = [i]
        for j, xm2 in enumerate(x_mids):
            if i != j and not visited[j] and abs(xm - xm2) < 40:
                cluster.append(j)
                visited[j] = True
        if len(cluster) > len(best):
            best = cluster
    segs = [good[i] for i in best]
    pts  = np.array([(x + roi_x1, y) for x1, y1, x2, y2 in segs
                     for x, y in [(x1, y1), (x2, y2)]], dtype=np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, cx, cy = float(vx), float(vy), float(cx), float(cy)
    if abs(vy) < 1e-6:
        return (int(cx), 0), (int(cx), h - 1)
    tT = (0 - cy) / vy
    tB = (h - 1 - cy) / vy
    xT = max(0, min(w - 1, int(cx + tT * vx)))
    xB = max(0, min(w - 1, int(cx + tB * vx)))
    return (xT, 0), (xB, h - 1)


# ── whistle detector ──────────────────────────────────────────────────────────
class WhistleDetector:
    def __init__(self):
        self.triggered   = False
        self._stop       = False
        self._hold_start = None
        self._thread     = None
        self._available  = False
        self._start()

    def _start(self):
        try:
            import pyaudio  # noqa
            self._available = True
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()
            print("[Whistle] Microphone listener started.")
        except ImportError:
            print("[Whistle] PyAudio not installed — whistle detection disabled.")

    def _listen(self):
        import pyaudio
        pa     = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE,
                         input=True, frames_per_buffer=AUDIO_CHUNK)
        freqs  = np.fft.rfftfreq(AUDIO_CHUNK, d=1.0 / AUDIO_RATE)
        band   = (freqs >= WHISTLE_FREQ_LOW) & (freqs <= WHISTLE_FREQ_HIGH)
        while not self._stop:
            try:
                raw  = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                pcm  = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                spec = np.abs(np.fft.rfft(pcm))
                ratio = spec[band].sum() / (spec.sum() + 1e-9)
                if ratio >= WHISTLE_THRESHOLD:
                    if self._hold_start is None:
                        self._hold_start = time.time()
                    elif time.time() - self._hold_start >= WHISTLE_HOLD_S:
                        if not self.triggered:
                            self.triggered = True
                            print("[Whistle] Detected — timer starting!")
                else:
                    self._hold_start = None
            except Exception:
                pass
        stream.stop_stream(); stream.close(); pa.terminate()

    def reset(self):
        self.triggered = False; self._hold_start = None

    def stop(self):
        self._stop = True


# ── MediaPipe Tasks pose tracker ───────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = "pose_landmarker_full.task"


def _download_model(dest: str):
    """Download the .task model file if not already present."""
    import urllib.request, os
    if os.path.exists(dest):
        return
    print(f"[MediaPipe] Downloading lite model → {dest}  (~5 MB) …")
    try:
        urllib.request.urlretrieve(MODEL_URL, dest)
        print("[MediaPipe] Download complete.")
    except Exception as e:
        print(f"[MediaPipe] Download failed: {e}")
        raise


class PoseTracker:
    """
    Wraps the MediaPipe Tasks PoseLandmarker (VIDEO mode).

    Two-tier human validation:
      • RELAXED  — for small/distant detections (bounding box < SMALL_PERSON_PX)
      • STRICT   — for close/large detections

    Uses ALL 33 body landmarks (not just upper body) so that swimmers partly
    occluded by water still get detected via legs/feet.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_poses: int = MAX_POSES):
        self._available  = False
        self._landmarker = None
        self._load(model_path, max_poses)

    def _load(self, model_path: str, max_poses: int):
        try:
            import mediapipe as mp
            _download_model(model_path)

            BaseOptions           = mp.tasks.BaseOptions
            PoseLandmarker        = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode     = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=max(1, max_poses),
                # Low thresholds — we do our own validation per detection
                min_pose_detection_confidence=0.10,
                min_pose_presence_confidence=0.10,
                min_tracking_confidence=0.10,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._mp         = mp
            self._available  = True
            print(f"[MediaPipe] PoseLandmarker loaded for up to {max_poses} humans.")
        except ImportError:
            print("[MediaPipe] Not installed — human-only crossing detection disabled.")
            print("            Install with:  pip install mediapipe")
        except Exception as e:
            print(f"[MediaPipe] Failed to load model: {e}")
            print("            Human-only crossing detection disabled.")

    # All 33 MediaPipe Pose landmarks (use everything visible)
    _ALL_LM = list(range(33))

    # Torso pairs for coherence check (index_a, index_b)
    _TORSO_PAIRS = [
        (11, 12),   # shoulders
        (23, 24),   # hips
        (11, 23),   # left side
        (12, 24),   # right side
    ]

    def get_centroids(self, frame_bgr, frame_w: int, frame_h: int,
                      timestamp_ms: int):
        """
        Returns list of (cx, cy) in pixel coords — one per confirmed human.
        timestamp_ms must be strictly increasing.
        """
        if not self._available or self._landmarker is None:
            return []
        try:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            )
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return []

        centroids = []

        for pose_lms in result.pose_landmarks:
            n_lm = len(pose_lms)

            # ── collect ALL landmarks that have any presence ──────────────────
            # We gather two sets:
            #   all_pts  — every landmark with presence > near-zero (for bounding box)
            #   good_pts — landmarks above the tier-appropriate visibility threshold
            all_pts  = []
            raw_lm   = {}   # idx → (x, y)

            for idx in self._ALL_LM:
                if idx >= n_lm:
                    continue
                lm       = pose_lms[idx]
                presence = getattr(lm, "presence", 1.0)
                if presence < 0.02:
                    continue
                x = int(lm.x * frame_w)
                y = int(lm.y * frame_h)
                if 0 <= x < frame_w and 0 <= y < frame_h:
                    all_pts.append((x, y))
                    raw_lm[idx] = (x, y, getattr(lm, "visibility", 1.0))

            if not all_pts:
                continue

            # ── bounding box → decide which tier to use ───────────────────────
            xs_all = [p[0] for p in all_pts]
            ys_all = [p[1] for p in all_pts]
            bbox_w = max(xs_all) - min(xs_all)
            bbox_h = max(ys_all) - min(ys_all)
            bbox_side = max(bbox_w, bbox_h)

            if bbox_side < SMALL_PERSON_PX:
                # RELAXED tier — distant / small swimmer
                vis_thresh = VIS_THRESH_RELAXED
                min_lm     = MIN_LM_RELAXED
                min_torso  = MIN_TORSO_RELAXED
            else:
                # STRICT tier — close / large person
                vis_thresh = VIS_THRESH_STRICT
                min_lm     = MIN_LM_STRICT
                min_torso  = MIN_TORSO_STRICT

            # ── filter landmarks by tier visibility threshold ─────────────────
            good_pts  = []
            good_lm   = {}   # idx → (x, y)

            for idx, (x, y, vis) in raw_lm.items():
                if vis >= vis_thresh:
                    good_pts.append((x, y))
                    good_lm[idx] = (x, y)

            if len(good_pts) < min_lm:
                continue

            # ── torso coherence: at least one pair must be spread enough ──────
            torso_ok = False
            for idx_a, idx_b in self._TORSO_PAIRS:
                if idx_a in good_lm and idx_b in good_lm:
                    xa, ya = good_lm[idx_a]
                    xb, yb = good_lm[idx_b]
                    if np.hypot(xb - xa, yb - ya) >= min_torso:
                        torso_ok = True
                        break

            # For very distant swimmers (relaxed tier, tiny bbox) skip torso
            # check entirely — a single visible landmark cluster is enough.
            if not torso_ok and bbox_side >= SMALL_PERSON_PX:
                continue

            # ── centroid from good landmarks ──────────────────────────────────
            cx = int(np.median([p[0] for p in good_pts]))
            cy = int(np.median([p[1] for p in good_pts]))

            if 0 <= cx < frame_w and 0 <= cy < frame_h:
                centroids.append((cx, cy))

        return centroids

    def close(self):
        if self._landmarker:
            self._landmarker.close()


# ── stable multi-person blob tracker ─────────────────────────────────────────
class BlobTracker:
    def __init__(self, max_dist: int = 100, grace_frames: int = TRACKER_GRACE_FRAMES):
        self.max_dist     = max_dist
        self.grace_frames = grace_frames
        self.next_id      = 0

        self.positions:  dict[int, tuple]        = {}  # bid → (cx, cy)
        self.prev_sides: dict[int, int | None]   = {}  # bid → last side
        self.colors:     dict[int, tuple]        = {}  # bid → BGR colour
        self._missing:   dict[int, int]          = {}  # bid → frames since last seen

    def update(self, new_centroids: list[tuple]) -> list[tuple]:
        # 1. Age all existing tracks
        for bid in list(self._missing.keys()):
            self._missing[bid] += 1

        # 2. Greedy nearest-neighbour matching
        unmatched_new = list(range(len(new_centroids)))
        matched_bids  = set()

        if self.positions and new_centroids:
            pairs = []
            for bid in list(self.positions.keys()):
                ox, oy = self.positions[bid]
                for ni in unmatched_new:
                    cx, cy = new_centroids[ni]
                    d = np.hypot(cx - ox, cy - oy)
                    if d <= self.max_dist:
                        pairs.append((d, bid, ni))
            pairs.sort(key=lambda x: x[0])

            used_ni = set()
            for d, bid, ni in pairs:
                if bid in matched_bids or ni in used_ni:
                    continue
                matched_bids.add(bid)
                used_ni.add(ni)
                cx, cy = new_centroids[ni]
                self.positions[bid] = (cx, cy)
                self._missing[bid]  = 0
            unmatched_new = [ni for ni in unmatched_new if ni not in used_ni]

        # 3. Spawn new tracks for unmatched detections
        for ni in unmatched_new:
            bid = self.next_id
            self.next_id += 1
            cx, cy = new_centroids[ni]
            self.positions[bid]  = (cx, cy)
            self.prev_sides[bid] = None
            self._missing[bid]   = 0
            self.colors[bid]     = _ID_PALETTE[bid % len(_ID_PALETTE)]

        # 4. Evict stale tracks
        for bid in list(self._missing.keys()):
            if self._missing[bid] > self.grace_frames:
                self._evict(bid)

        # 5. Return all active tracks
        return [
            (bid, self.positions[bid][0], self.positions[bid][1],
             self.prev_sides.get(bid))
            for bid in list(self.positions.keys())
        ]

    def _evict(self, bid: int):
        self.positions.pop(bid, None)
        self.prev_sides.pop(bid, None)
        self._missing.pop(bid, None)
        # Intentionally keep colors so IDs never get the same colour after wrap

    def set_side(self, bid: int, side: int):
        if bid in self.positions:
            self.prev_sides[bid] = side

    def clear(self):
        self.positions.clear()
        self.prev_sides.clear()
        self._missing.clear()
        self.next_id = 0
        self.colors.clear()


# ── main application ──────────────────────────────────────────────────────────
def _autosave_dash_50m(session_id, laps):
    """Stages each lap's result on the server for human review/save on
    screen3 — does NOT write to the database directly."""
    try:
        import requests
        lap_list = [
            {"lane": n, "time_sec": round(cumulative, 3), "start_mode": "whistle"}
            for n, split, cumulative in laps
        ]
        requests.post(
            "http://localhost:8080/api/stage/dash_50m",
            json={"session_id": session_id, "laps": lap_list},
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] dash_50m staged for review  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


class LapTimer:
    def __init__(self, source, use_whistle=True, use_mediapipe=True,
                 model_path=DEFAULT_MODEL_PATH, allow_blob_fallback=False,
                 process_scale=PROC_SCALE, skip_frames=SKIP_FRAMES,
                 max_poses=MAX_POSES, display_width=DISPLAY_WIDTH,
                 camera_width=0, camera_height=0):

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if camera_width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_width)
        if camera_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        self.p1 = None; self.p2 = None
        self.roi = None; self.roi_drawing = False; self.roi_temp = None
        self.dragging = False; self.drag_temp = None
        # freehand line: store both endpoints while dragging
        self._drag_start = None   # (x, y) where the mouse went down
        self._drag_end   = None   # (x, y) current mouse position while dragging
        self._frame_h = 480; self._frame_w = 640
        self.process_scale         = min(1.0, max(0.25, float(process_scale)))
        self.skip_frames           = max(0, int(skip_frames))
        self.display_width         = max(0, int(display_width))
        self._allow_blob_fallback  = bool(allow_blob_fallback)
        self._last_pose_warning_at = 0.0
        self._last_ts_ms           = 0

        # Timer state
        self.running     = False
        self.start_time  = None
        self.laps        = []
        self.last_lap_at = 0.0

        self._blob_cross_time: dict = {}
        self._blob_armed:      dict = {}

        # Detection
        # max_dist=150 — distant swimmers may shift more pixels per frame;
        # wider search radius keeps the same ID across skipped frames.
        self.tracker = BlobTracker(max_dist=150, grace_frames=TRACKER_GRACE_FRAMES)
        self.bg      = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)
        self.k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        # MediaPipe
        self.pose = PoseTracker(model_path, max_poses=max_poses) if use_mediapipe else None
        self._use_mediapipe = use_mediapipe and (
            self.pose is not None and self.pose._available)
        if not self._use_mediapipe and not self._allow_blob_fallback:
            print("[Detection] Human-only mode needs MediaPipe. "
                  "No crossings counted until it loads.")

        # Frame-skip counter
        self._skip_count = 0
        self._last_blobs: list = []

        # Lap flash
        self.flash_until = 0.0

        # Whistle
        self.whistle = WhistleDetector() if use_whistle else None

        # Status message
        self.msg   = "Click and drag to draw finish line anywhere"
        self.msg_t = time.time()

        cv2.namedWindow("Lap Timer", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Lap Timer", self._mouse)

        # FPS tracking
        self._fps_t = time.time()
        self._fps_n = 0
        self._fps   = 0.0

    # ── mouse ─────────────────────────────────────────────────────────────────
    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.p1 is not None and self.p2 is not None:
                return          # line already set; use L to clear first
            self.dragging    = True
            self._drag_start = (x, y)
            self._drag_end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self._drag_end = (x, y)
            if self.roi_drawing:
                self.roi_temp = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0 = self._drag_start
            # Accept only if the drag was long enough to be intentional (>10 px)
            if np.hypot(x - x0, y - y0) > 10:
                self.p1 = (x0, y0)
                self.p2 = (x,  y)
                self.set_msg("Line set  |  L to clear  |  right-click drag for ROI")
            else:
                self.set_msg("Drag was too short — try again")
            self._drag_start = None
            self._drag_end   = None
            self.tracker.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.roi is not None:
                return
            self.roi_drawing = True
            self.roi_temp    = (x, y)
            self._roi_start  = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and self.roi_drawing:
            self.roi_drawing = False
            x1 = min(self._roi_start[0], x); y1 = min(self._roi_start[1], y)
            x2 = max(self._roi_start[0], x); y2 = max(self._roi_start[1], y)
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.roi = (x1, y1, x2, y2)
                self.tracker.clear()
                self.set_msg("ROI set  |  C to clear ROI")

    # ── helpers ───────────────────────────────────────────────────────────────
    def set_msg(self, text):
        self.msg = text; self.msg_t = time.time()

    def elapsed(self):
        if not self.running or self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def record_lap(self, bid=None):
        now   = self.elapsed()
        split = now - self.last_lap_at
        n     = len(self.laps) + 1
        tag   = f" [obj {bid}]" if bid is not None else ""
        self.laps.append((n, split, now))
        self.last_lap_at = now
        self.flash_until = time.time() + 0.4
        print(f"LAP {n}{tag}   split {format_time(split)}   total {format_time(now)}")

    def save_csv(self):
        fname = f"laps_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Lap", "Split", "Cumulative"])
            for n, sp, cu in self.laps:
                w.writerow([n, format_time(sp), format_time(cu)])
        self.set_msg(f"Saved → {fname}")
        print(f"Saved {fname}")

    # ── detection ─────────────────────────────────────────────────────────────
    def process(self, frame):
        """Runs detection; updates self._last_blobs. Returns None."""

        # Whistle trigger
        if self.whistle and self.whistle.triggered and not self.running:
            self.running    = True
            self.start_time = time.time()
            self.set_msg("Whistle detected — timer started!")
            print("Timer STARTED (whistle)")
            self.whistle.reset()

        self._skip_count = (self._skip_count + 1) % (self.skip_frames + 1)
        if self._skip_count != 0:
            # Even on skipped frames, re-check crossings using last known positions
            if self.p1 is not None and self.running:
                self._check_crossings(self._last_blobs)
            return None

        h, w = frame.shape[:2]
        sh   = int(h * self.process_scale)
        sw   = int(w * self.process_scale)

        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)

        centroids = []
        used_pose = False

        if self._use_mediapipe:
            ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms
            raw       = self.pose.get_centroids(small, sw, sh, ts_ms)
            centroids = [(int(cx / self.process_scale), int(cy / self.process_scale))
                         for cx, cy in raw]
            used_pose = bool(centroids)

        # Human-only mode: skip raw motion blobs unless fallback is enabled
        if not centroids and not self._allow_blob_fallback:
            if self._use_mediapipe:
                _ = self.bg.apply(small)   # keep background model warm
            else:
                now = time.time()
                if now - self._last_pose_warning_at > 5.0:
                    self.set_msg("Install mediapipe for human-only detection")
                    self._last_pose_warning_at = now
            blobs = self.tracker.update([])
            self._last_blobs = blobs
            return None

        if not centroids:
            fg = self.bg.apply(small)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.k_open)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)

            if self.p1 is None or self.p2 is None:
                return fg

            contours, _ = cv2.findContours(
                fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_BLOB_AREA * (self.process_scale ** 2):
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int((M["m10"] / M["m00"]) / self.process_scale)
                cy = int((M["m01"] / M["m00"]) / self.process_scale)
                centroids.append((cx, cy))
        else:
            if used_pose:
                _ = self.bg.apply(small)

        # ROI filter
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            centroids = [(cx, cy) for cx, cy in centroids
                         if rx1 <= cx <= rx2 and ry1 <= cy <= ry2]

        blobs = self.tracker.update(centroids)
        self._last_blobs = blobs

        if self.running:
            self._check_crossings(blobs)

        return None

    def _check_crossings(self, blobs):
        if self.p1 is None or self.p2 is None:
            return
        now_w        = time.time()
        APPROACH_BAND = DETECTION_BAND * 2

        for bid, cx, cy, prev_side in blobs:
            side = which_side((cx, cy), self.p1, self.p2)
            self.tracker.set_side(bid, side)
            dist = point_line_dist((cx, cy), self.p1, self.p2)

            if dist <= APPROACH_BAND:
                if bid not in self._blob_armed:
                    self._blob_armed[bid] = now_w

            if prev_side is not None and prev_side != side:
                armed_at  = self._blob_armed.get(bid)
                in_band   = dist <= DETECTION_BAND
                in_window = (armed_at is not None and
                             (now_w - armed_at) <= COOLDOWN_S)

                if in_band or in_window:
                    last = self._blob_cross_time.get(bid, 0.0)
                    if now_w - last >= COOLDOWN_S:
                        self._blob_cross_time[bid] = now_w
                        self._blob_armed.pop(bid, None)
                        self.record_lap(bid)

    # ── drawing ───────────────────────────────────────────────────────────────
    def draw(self, frame):
        h, w = frame.shape[:2]

        # Lap flash
        if time.time() < self.flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 220, 60), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # ROI overlay
        roi_draw = self.roi
        if self.roi_drawing and hasattr(self, '_roi_start') and self.roi_temp:
            x1 = min(self._roi_start[0], self.roi_temp[0])
            y1 = min(self._roi_start[1], self.roi_temp[1])
            x2 = max(self._roi_start[0], self.roi_temp[0])
            y2 = max(self._roi_start[1], self.roi_temp[1])
            roi_draw = (x1, y1, x2, y2)
        if roi_draw:
            rx1, ry1, rx2, ry2 = roi_draw
            dark = (frame * 0.45).astype(np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
            frame[mask == 0] = dark[mask == 0]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
            cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        # Tracked person dots
        for bid, cx, cy, _ in self._last_blobs:
            col = self.tracker.colors.get(bid, (255, 255, 255))
            cv2.circle(frame, (cx, cy), 14, (0, 0, 0), 3)
            cv2.circle(frame, (cx, cy), 12, col, -1)
            cv2.circle(frame, (cx, cy - 3), 4, (255, 255, 255), -1)
            label = str(bid)
            lx, ly = cx + 16, cy + 5
            cv2.putText(frame, label, (lx + 1, ly + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

        # Finish line + detection band
        p1d = self.p1; p2d = self.p2
        if self.dragging and self._drag_start and self._drag_end:
            p1d = self._drag_start
            p2d = self._drag_end
        if p1d and p2d:
            color   = (0, 255, 0) if self.running else (255, 255, 255)
            overlay = frame.copy()
            dx      = p2d[0] - p1d[0]; dy = p2d[1] - p1d[1]
            length  = max(np.hypot(dx, dy), 1)
            nx, ny  = -dy / length, dx / length
            b       = DETECTION_BAND
            pts = np.array([
                [p1d[0] + nx * b, p1d[1] + ny * b],
                [p2d[0] + nx * b, p2d[1] + ny * b],
                [p2d[0] - nx * b, p2d[1] - ny * b],
                [p1d[0] - nx * b, p1d[1] - ny * b],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (200, 200, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.line(frame, p1d, p2d, color, 3)
            mid = ((p1d[0] + p2d[0]) // 2,
                   (p1d[1] + p2d[1]) // 2 + h // 2)
            cv2.putText(frame, "FINISH", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Big timer
        t_str       = format_time(self.elapsed())
        (tw, th), _ = cv2.getTextSize(t_str, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
        tx = (w - tw) // 2
        cv2.rectangle(frame, (tx - 12, 10), (tx + tw + 12, 18 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, t_str, (tx, 14 + th),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

        # Status indicators
        dot_col = (0, 220, 60) if self.running else (0, 100, 255)
        status  = "RUNNING" if self.running else "STOPPED"
        cv2.circle(frame, (18, 60), 7, dot_col, -1)
        cv2.putText(frame, status, (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_col, 2, cv2.LINE_AA)

        if self.whistle and self.whistle._available and not self.running:
            cv2.putText(frame, "MIC: ON", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 60), 1, cv2.LINE_AA)

        n_tracked = len(self._last_blobs)
        if self._use_mediapipe:
            mp_label = f"Human pose: ON  ({n_tracked} tracked)"
        elif self._allow_blob_fallback:
            mp_label = "Human pose: OFF (blob fallback)"
        else:
            mp_label = "Human pose: OFF"
        mp_col = (0, 200, 100) if self._use_mediapipe else (100, 100, 200)
        cv2.putText(frame, mp_label, (30, 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, mp_col, 1, cv2.LINE_AA)

        cv2.putText(frame, f"Laps: {len(self.laps)}", (18, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (18, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 200, 140), 1, cv2.LINE_AA)

        # Last 5 laps
        by = h - 12
        for n, sp, cu in reversed(self.laps[-5:]):
            txt = f"Lap {n}   {format_time(sp)}   ({format_time(cu)})"
            cv2.putText(frame, txt, (12, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
            by -= 22

        # Line mode & controls
        keys = ["D detect line", "L clear line",
                "C clear ROI", "SPACE start/stop", "R reset", "S save", "Q quit"]
        bx, by2 = w - 200, h - 12
        for k in reversed(keys):
            cv2.putText(frame, k, (bx, by2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 130), 1, cv2.LINE_AA)
            by2 -= 18

        # Status message
        if time.time() - self.msg_t < 4.0:
            col = (0, 220, 100) if any(w_ in self.msg.lower()
                                       for w_ in ("detected", "set", "started", "saved")) \
                  else (80, 180, 255)
            cv2.putText(frame, self.msg, (12, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("End of stream.")
                break

            self._frame_h = frame.shape[0]
            self._frame_w = frame.shape[1]

            self.process(frame)
            self.draw(frame)

            display = frame
            if self.display_width and frame.shape[1] > self.display_width:
                scale   = self.display_width / frame.shape[1]
                display = cv2.resize(
                    frame,
                    (self.display_width, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow("Lap Timer", display)

            # FPS counter
            self._fps_n += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self._fps   = self._fps_n / (now - self._fps_t)
                self._fps_n = 0
                self._fps_t = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty("Lap Timer", cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord(' '):
                if not self.running:
                    self.running    = True
                    self.start_time = time.time()
                    self.set_msg("Timer started")
                    print("Timer STARTED (manual)")
                else:
                    self.running = False
                    self.set_msg("Timer stopped — press R to reset")
                    print("Timer STOPPED")

            elif key in (ord('r'), ord('R')):
                self.running    = False
                self.start_time = None
                self.laps       = []
                self.last_lap_at = 0.0
                self._blob_cross_time = {}
                self._blob_armed      = {}
                self._drag_start = None
                self._drag_end   = None
                self.p1 = None; self.p2 = None; self.roi = None
                self._last_blobs = []
                self.tracker.clear()
                if self.whistle:
                    self.whistle.reset()
                self.set_msg("Reset — draw finish line, then right-click drag for ROI")
                print("RESET")

            elif key in (ord('l'), ord('L')):
                self.p1 = None; self.p2 = None
                self.tracker.clear(); self._last_blobs = []
                self.set_msg("Line cleared — draw or press D to detect")

            elif key in (ord('c'), ord('C')):
                self.roi = None
                self.tracker.clear(); self._last_blobs = []
                self.set_msg("ROI cleared — right-click drag to set a new one")

            elif key in (ord('d'), ord('D')):
                p1, p2 = detect_white_line(frame)
                if p1 and p2:
                    self.p1 = p1; self.p2 = p2
                    self.tracker.clear()
                    self.set_msg("White line detected ✓")
                    print(f"Line detected: {p1} → {p2}")
                else:
                    self.set_msg("Line not found — click on the video to draw manually")

            elif key in (ord('s'), ord('S')):
                if self.laps:
                    self.save_csv()
                else:
                    self.set_msg("No laps to save yet")

        # ── cleanup ───────────────────────────────────────────────────────────
        self.cap.release()
        cv2.destroyAllWindows()
        if self.whistle:
            self.whistle.stop()
        if self.pose:
            self.pose.close()

        if self.laps:
            print("\n=== Results ===")
            print(f"{'Lap':>4}  {'Split':>10}  {'Cumulative':>12}")
            print("-" * 32)
            for n, sp, cu in self.laps:
                print(f"{n:>4}  {format_time(sp):>10}  {format_time(cu):>12}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              default=0,
                    help="Camera index (0, 1…) or path to video file")
    ap.add_argument("--no-whistle",          action="store_true",
                    help="Disable microphone whistle detection")
    ap.add_argument("--no-mediapipe",        action="store_true",
                    help="Disable MediaPipe human pose detection")
    ap.add_argument("--allow-blob-fallback", action="store_true",
                    help="Allow motion blobs to count laps if no human pose found")
    ap.add_argument("--model",               default=DEFAULT_MODEL_PATH,
                    help=f"Path to .task file (default: {DEFAULT_MODEL_PATH})")
    ap.add_argument("--process-scale",       type=float, default=PROC_SCALE,
                    help="Detection scale 0.25–1.0; lower is faster")
    ap.add_argument("--skip-frames",         type=int,   default=SKIP_FRAMES,
                    help="Skip N frames between detections; higher is faster")
    ap.add_argument("--max-poses",           type=int,   default=MAX_POSES,
                    help="Maximum humans to detect at once")
    ap.add_argument("--display-width",       type=int,   default=DISPLAY_WIDTH,
                    help="Preview window width; 0 disables resizing")
    ap.add_argument("--camera-width",        type=int,   default=0,
                    help="Requested capture width; 0 keeps camera default")
    ap.add_argument("--camera-height",       type=int,   default=0,
                    help="Requested capture height; 0 keeps camera default")
    args = ap.parse_args()

    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass

    LapTimer(
        src,
        use_whistle=not args.no_whistle,
        use_mediapipe=not args.no_mediapipe,
        model_path=args.model,
        allow_blob_fallback=args.allow_blob_fallback,
        process_scale=args.process_scale,
        skip_frames=args.skip_frames,
        max_poses=args.max_poses,
        display_width=args.display_width,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    ).run()