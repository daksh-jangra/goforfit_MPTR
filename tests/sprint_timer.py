import cv2
import numpy as np
import pyaudio
import time
import threading
import queue
import platform
import math
import argparse
from collections import deque

if platform.system() == "Windows":
    KEY_UP    = 72
    KEY_DOWN  = 80
    KEY_LEFT  = 75
    KEY_RIGHT = 77
else:
    KEY_UP    = 82
    KEY_DOWN  = 84
    KEY_LEFT  = 81
    KEY_RIGHT = 83

# ─────────────────────────────────────────────
# CONFIGURATION — defaults, overridable from the command line (see parse_args())
# ─────────────────────────────────────────────
FINISH_CAM_URL   = "http://192.168.137.33:4747/video"  # perpendicular camera (timing)
OVERHEAD_CAM_URL = "http://192.168.137.6:4747/video"   # overhead camera (lane identity)

# ── Session parameters — filled in by parse_args() before anything else runs ──
NUM_LANES    = None
LANE_WIDTH_M = None

# GoForFit web-app session id (set via --session-id). When present, results
# are staged to the FastAPI backend for review on screen3 instead of only
# being written to a local CSV.
SESSION_ID = None

# Strip detection (perpendicular / timing camera)
# Vertical strip — detects left-to-right movement as runners cross the line.
STRIP_WIDTH        = 40
FG_PIXEL_THRESHOLD = 600    # white pixels needed to trigger a crossing
COOLDOWN_SEC       = 1.0    # min seconds between crossings ON THE SAME LANE

# Lane detection (overhead / identity camera)
# Top-down view — no perspective distortion, lanes are simple vertical columns.
LANE_FG_THRESHOLD     = 300   # white pixels needed to mark a lane "occupied"
LANE_MORPH_KSIZE      = 5     # morphological kernel size to kill noise blobs
LANE_MIN_BLOB_AREA    = 150   # minimum contour area (px²) to count as real object
LANE_OCCUPANCY_FRAMES = 3     # consecutive frames lane must be occupied to confirm
LANE_HISTORY_MS       = 250   # rolling look-back window (ms) — bridges async gap

# Whistle detection
WHISTLE_RATE       = 44100
WHISTLE_CHUNK      = 1024
WHISTLE_FREQ_LOW   = 2000
WHISTLE_FREQ_HIGH  = 4000
WHISTLE_ENERGY_THR = 0.15

# MOG2 background subtraction
MOG2_HISTORY       = 100
MOG2_VAR_THRESHOLD = 40

# Display
WINDOW_NAME = "GoForFit — 50m Sprint (Multi-Lane)"
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
def parse_args():
    """
    CLI arguments — this is what lets the GoForFit web app launch this
    script as a subprocess (stdin is closed/DEVNULL when launched that way,
    so the old input()-based prompts would just crash with EOFError).

    Run standalone from a terminal, every argument is optional and falls
    back to the module-level defaults above, so `python sprint_timer.py`
    still works exactly as before.
    """
    parser = argparse.ArgumentParser(description="GoForFit — 50m Sprint Timer")
    parser.add_argument("--finish-cam", type=str, default=FINISH_CAM_URL,
                         help="Finish (perpendicular) camera URL")
    parser.add_argument("--overhead-cam", type=str, default=OVERHEAD_CAM_URL,
                         help="Overhead camera URL")
    parser.add_argument("--lanes", type=int, default=4,
                         help="Number of lanes (default: 4)")
    parser.add_argument("--lane-width", type=float, default=1.22,
                         help="Lane width in metres (default: 1.22)")
    parser.add_argument("--session-id", type=int, default=None,
                         help="GoForFit session id — stages results for "
                              "review on screen3 instead of CSV-only")
    return parser.parse_args()


def print_session_banner():
    """Prints the same setup summary the old interactive prompt showed,
    just sourced from CLI args / globals instead of input()."""
    print("\n" + "═" * 60)
    print("  GoForFit — 50m Sprint Timer  |  Session Setup")
    print("═" * 60)
    print(f"  Lanes: {NUM_LANES}   Lane width: {LANE_WIDTH_M} m   "
          f"Total width: {NUM_LANES * LANE_WIDTH_M:.2f} m")
    print("─" * 60)
    print("  Camera setup:")
    print(f"    FINISH CAM   — {FINISH_CAM_URL}")
    print(f"    OVERHEAD CAM — {OVERHEAD_CAM_URL}")
    if SESSION_ID is not None:
        print(f"  Session ID: {SESSION_ID}  (results will be staged for review)")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
CLR_GREEN  = (0, 220, 80)
CLR_RED    = (0, 60, 220)
CLR_YELLOW = (0, 210, 255)
CLR_WHITE  = (255, 255, 255)
CLR_GRAY   = (160, 160, 160)
CLR_BG     = (30, 30, 30)
CLR_BLUE   = (220, 140, 0)

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
event_queue = queue.Queue()

# Finish (perpendicular) camera — vertical strip
strip_x           = None   # left edge of detection strip

# Overhead camera — lane column dividers
# Simple list of N+1 x-positions dividing the frame into N equal columns.
# Top-down view has no perspective distortion so no trapezoids needed.
lane_dividers     = None   # [x0, x1, ..., xN] in overhead frame pixels

finish_calibrated   = False
overhead_calibrated = False

# Race state
race_start_time = None
lane_running    = {}
lap_times: list = []   # [(elapsed_sec, lane_id), ...]

# Overhead camera lane occupancy — shared between overhead thread and finish thread
front_lane_history  = {}   # lane_id -> deque[(timestamp, bool)]
front_lane_occupied = {}   # lane_id -> bool (most recent confirmed frame)
front_lane_lock     = threading.Lock()
_lane_consec        = {}   # lane_id -> int (debounce counter)


def init_lane_state():
    global lane_running, front_lane_occupied, front_lane_history, _lane_consec, lap_times
    lap_times = []
    for i in range(1, NUM_LANES + 1):
        lane_running[i]        = False
        front_lane_occupied[i] = False
        front_lane_history[i]  = deque()
        _lane_consec[i]        = 0


# ══════════════════════════════════════════════
# WHISTLE DETECTION THREAD
# ══════════════════════════════════════════════
def whistle_thread():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=WHISTLE_RATE,
        input=True,
        frames_per_buffer=WHISTLE_CHUNK
    )
    print("[Whistle] Listening...")
    last_trigger = 0
    while True:
        try:
            raw = stream.read(WHISTLE_CHUNK, exception_on_overflow=False)
        except Exception:
            continue
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        fft_mag        = np.abs(np.fft.rfft(samples))
        freqs          = np.fft.rfftfreq(len(samples), d=1.0 / WHISTLE_RATE)
        total_energy   = np.sum(fft_mag) + 1e-9
        whistle_mask   = (freqs >= WHISTLE_FREQ_LOW) & (freqs <= WHISTLE_FREQ_HIGH)
        whistle_energy = np.sum(fft_mag[whistle_mask]) / total_energy
        now = time.perf_counter()
        if whistle_energy > WHISTLE_ENERGY_THR and (now - last_trigger) > 2.0:
            last_trigger = now
            event_queue.put(("WHISTLE", now))
            print(f"[Whistle] Detected  energy={whistle_energy:.3f}")
    stream.stop_stream()
    stream.close()
    p.terminate()


# ══════════════════════════════════════════════
# OVERHEAD CAMERA THREAD — lane identity
# ══════════════════════════════════════════════
def overhead_cam_thread():
    """
    Reads the overhead camera stream (phone mounted above finish line, looking down).

    Because the view is top-down there is NO perspective distortion — each lane
    is simply a vertical column of equal width.  Detection is therefore much
    simpler than the old front-camera trapezoid approach:

      1. Divide the frame into N equal vertical columns (one per lane).
      2. Run MOG2 on each column independently.
      3. Morphological open+close + blob-area filter to reject noise.
      4. Temporal debounce: require LANE_OCCUPANCY_FRAMES consecutive frames.
      5. Rolling history deque so the finish camera can look back LANE_HISTORY_MS.
    """
    global lane_dividers, overhead_calibrated
    global front_lane_occupied, front_lane_history, _lane_consec

    PROC_W = 320
    PROC_H = 240

    cap = cv2.VideoCapture(OVERHEAD_CAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[OverheadCam] ERROR: Cannot open stream at {OVERHEAD_CAM_URL}")
        event_queue.put(("FRONT_CAM_ERROR", None))
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    lane_mogs = {
        i: cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False
        ) for i in range(1, NUM_LANES + 1)
    }
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (LANE_MORPH_KSIZE, LANE_MORPH_KSIZE)
    )

    frame_w = None
    frame_h = None
    last_queue_put = 0.0

    while True:
        for _ in range(5):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            time.sleep(0.05)
            cap.open(OVERHEAD_CAM_URL)
            continue

        if frame_w is None:
            frame_h, frame_w = frame.shape[:2]
            if lane_dividers is None:
                # Default: equal columns across full frame width
                lane_dividers = [int(i * frame_w / NUM_LANES) for i in range(NUM_LANES + 1)]

        if not overhead_calibrated:
            now = time.perf_counter()
            if now - last_queue_put >= 0.033:
                event_queue.put(("FRONT_FRAME", frame.copy()))
                last_queue_put = now
            else:
                time.sleep(0.005)
            continue

        # ── Per-lane column detection ────────────────────────────────────────
        sx = PROC_W / frame_w
        small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)

        now = time.perf_counter()
        lane_px = {}

        for i in range(1, NUM_LANES + 1):
            # Scale column x-bounds to processing resolution
            x_left  = max(0,      int(lane_dividers[i - 1] * sx))
            x_right = min(PROC_W, int(lane_dividers[i]     * sx))

            if x_right <= x_left:
                lane_px[i] = 0
                continue

            crop     = small[:, x_left:x_right]   # full height, column width
            mask_mog = lane_mogs[i].apply(crop)

            mask_clean = cv2.morphologyEx(mask_mog, cv2.MORPH_OPEN,  morph_kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, morph_kernel)

            raw_count = int(np.count_nonzero(mask_clean))
            if raw_count > LANE_FG_THRESHOLD:
                contours, _ = cv2.findContours(
                    mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                valid_area = sum(
                    cv2.contourArea(c) for c in contours
                    if cv2.contourArea(c) >= LANE_MIN_BLOB_AREA
                )
                lane_px[i] = int(valid_area)
            else:
                lane_px[i] = raw_count

        # ── Temporal debounce + rolling history ──────────────────────────────
        with front_lane_lock:
            cutoff = now - LANE_HISTORY_MS / 1000.0
            for i in range(1, NUM_LANES + 1):
                raw_occupied = lane_px[i] > LANE_FG_THRESHOLD
                if raw_occupied:
                    _lane_consec[i] = min(_lane_consec[i] + 1, LANE_OCCUPANCY_FRAMES)
                else:
                    _lane_consec[i] = max(_lane_consec[i] - 1, 0)
                confirmed = _lane_consec[i] >= LANE_OCCUPANCY_FRAMES
                front_lane_occupied[i] = confirmed
                front_lane_history[i].append((now, confirmed))
                while front_lane_history[i] and front_lane_history[i][0][0] < cutoff:
                    front_lane_history[i].popleft()

        if now - last_queue_put >= 0.033:
            event_queue.put(("FRONT_FRAME", (frame.copy(), dict(lane_px),
                                             list(lane_dividers))))
            last_queue_put = now


# ══════════════════════════════════════════════
# FINISH (PERPENDICULAR) CAMERA THREAD — timing
# ══════════════════════════════════════════════
def finish_cam_thread():
    """
    Reads the perpendicular (side-mounted) camera stream.
    MOG2 on a vertical strip detects the precise crossing timestamp.
    Lane identity is resolved via the rolling overhead-camera occupancy history.
    """
    global strip_x, finish_calibrated

    cap = cv2.VideoCapture(FINISH_CAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[FinishCam] ERROR: Cannot open stream at {FINISH_CAM_URL}")
        event_queue.put(("FINISH_CAM_ERROR", None))
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False
    )

    frame_w        = None
    last_trigger   = 0.0
    last_queue_put = 0.0
    prev_occupied  = {i: False for i in range(1, NUM_LANES + 1)}

    while True:
        for _ in range(5):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            time.sleep(0.05)
            cap.open(FINISH_CAM_URL)
            continue

        if frame_w is None:
            frame_w = frame.shape[1]
            if strip_x is None:
                strip_x = frame_w // 2 - STRIP_WIDTH // 2

        if not finish_calibrated:
            now = time.perf_counter()
            if now - last_queue_put >= 0.033:
                event_queue.put(("FINISH_FRAME", frame.copy()))
                last_queue_put = now
            else:
                time.sleep(0.005)
            continue

        x1 = max(0, strip_x)
        x2 = min(frame_w, strip_x + STRIP_WIDTH)
        strip_crop = frame[:, x1:x2]

        fg_mask  = mog2.apply(strip_crop)
        white_px = int(np.count_nonzero(fg_mask))

        now = time.perf_counter()

        if white_px > FG_PIXEL_THRESHOLD and (now - last_trigger) > 0.15:
            cutoff = now - LANE_HISTORY_MS / 1000.0
            with front_lane_lock:
                recently_occupied = {
                    lane_id: any(occ for ts, occ in hist if ts >= cutoff)
                    for lane_id, hist in front_lane_history.items()
                }
                snap_occupied = dict(front_lane_occupied)

            newly_crossed = []
            for lane_id in range(1, NUM_LANES + 1):
                candidate = recently_occupied.get(lane_id, False) or \
                            snap_occupied.get(lane_id, False)
                rising = candidate and not prev_occupied.get(lane_id, False)
                if rising and lane_running.get(lane_id, False):
                    newly_crossed.append(lane_id)

            if newly_crossed:
                last_trigger = now
                event_queue.put(("CROSSING", (now, newly_crossed)))
                print(f"[FinishCam] Crossing t={now:.4f}  lanes={newly_crossed}  px={white_px}")

            prev_occupied = {k: recently_occupied.get(k, False) or snap_occupied.get(k, False)
                             for k in range(1, NUM_LANES + 1)}

        if now - last_queue_put >= 0.033:
            event_queue.put(("FINISH_FRAME", (frame.copy(), fg_mask.copy(), white_px, x1, x2)))
            last_queue_put = now


# ══════════════════════════════════════════════
# CALIBRATION — PERPENDICULAR CAMERA (vertical strip)
# ══════════════════════════════════════════════
def calibrate_finish_strip(initial_frame):
    global strip_x, finish_calibrated

    frame_h, frame_w = initial_frame.shape[:2]
    if strip_x is None:
        strip_x = frame_w // 2 - STRIP_WIDTH // 2

    local_strip_x = strip_x
    local_width   = STRIP_WIDTH
    step          = 5

    print("\n[Calibration 1/2] PERPENDICULAR camera — align strip with finish line.")
    print("  LEFT/RIGHT : move   +/- : resize   SPACE : confirm\n")

    frame = initial_frame.copy()
    while True:
        try:
            evt = event_queue.get_nowait()
            if evt[0] == "FINISH_FRAME":
                frame = evt[1] if isinstance(evt[1], np.ndarray) else evt[1][0]
        except queue.Empty:
            pass

        display = frame.copy()
        h, w    = display.shape[:2]
        x1 = max(0, local_strip_x)
        x2 = min(w, local_strip_x + local_width)

        overlay = display.copy()
        cv2.rectangle(overlay, (x1, 0), (x2, h), CLR_YELLOW, -1)
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        cv2.line(display, (x1, 0), (x1, h), CLR_YELLOW, 2)
        cv2.line(display, (x2, 0), (x2, h), CLR_YELLOW, 2)

        cv2.rectangle(display, (0, 0), (w, 50), CLR_BG, -1)
        cv2.putText(display,
            "STEP 1/2: PERPENDICULAR cam — align strip with finish line  |  ARROWS  +/-  SPACE=confirm",
            (10, 30), FONT, 0.46, CLR_WHITE, 1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if   key in (KEY_LEFT,  ord('a')): local_strip_x = max(0, local_strip_x - step)
        elif key in (KEY_RIGHT, ord('d')): local_strip_x = min(w - local_width, local_strip_x + step)
        elif key in (ord('+'), ord('=')):  local_width = min(100, local_width + 5)
        elif key == ord('-'):              local_width = max(10,  local_width - 5)
        elif key == ord(' '):
            strip_x = local_strip_x
            finish_calibrated = True
            print(f"[Calibration] Strip locked: x={strip_x}  width={local_width}\n")
            return
        elif key == 27:
            raise SystemExit


# ══════════════════════════════════════════════
# CALIBRATION — OVERHEAD CAMERA (vertical column dividers)
# ══════════════════════════════════════════════
def calibrate_lane_dividers(initial_frame):
    """
    Calibrate lane column dividers for the overhead camera.

    Because the camera looks straight down there is no perspective distortion —
    lanes are simply N vertical columns separated by N+1 draggable lines.

    Mouse drag any divider line left/right to align it with the lane marking
    visible on the ground in the overhead view.  SPACE or ENTER to confirm.
    """
    global lane_dividers, overhead_calibrated

    frame_h, frame_w = initial_frame.shape[:2]

    if lane_dividers is None:
        lane_dividers = [int(i * frame_w / NUM_LANES) for i in range(NUM_LANES + 1)]

    local_div   = list(lane_dividers)
    drag_state  = [None]   # None | index of divider being dragged
    HANDLE_R    = 12
    LINE_HIT    = 10

    def _mouse_cb(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, xd in enumerate(local_div):
                if abs(mx - xd) <= LINE_HIT:
                    drag_state[0] = idx
                    return
        elif event == cv2.EVENT_MOUSEMOVE:
            if drag_state[0] is None:
                return
            idx = drag_state[0]
            # Keep dividers in order with at least 10 px between neighbours
            lo = (local_div[idx - 1] + 10) if idx > 0            else 0
            hi = (local_div[idx + 1] - 10) if idx < NUM_LANES    else frame_w
            local_div[idx] = max(lo, min(hi, mx))
        elif event == cv2.EVENT_LBUTTONUP:
            drag_state[0] = None

    cv2.setMouseCallback(WINDOW_NAME, _mouse_cb)

    def _draw(display):
        h, w = display.shape[:2]
        for i in range(NUM_LANES):
            xl = local_div[i]
            xr = local_div[i + 1]
            tint = CLR_BLUE if i % 2 == 0 else CLR_GREEN
            ov   = display.copy()
            cv2.rectangle(ov, (xl, 0), (xr, h), tint, -1)
            cv2.addWeighted(ov, 0.22, display, 0.78, 0, display)
            cx = (xl + xr) // 2
            cv2.putText(display, f"L{i + 1}", (cx - 14, h // 2 + 6),
                        FONT, 0.8, CLR_WHITE, 2)

        for idx, xd in enumerate(local_div):
            active = drag_state[0] == idx
            cv2.line(display, (xd, 0), (xd, h),
                     CLR_YELLOW if active else CLR_WHITE, 2)
            cv2.circle(display, (xd, h // 2), HANDLE_R,
                       CLR_YELLOW if active else CLR_WHITE, -1)
            cv2.circle(display, (xd, h // 2), HANDLE_R, (0, 0, 0), 2)

        cv2.rectangle(display, (0, 0), (w, 52), CLR_BG, -1)
        cv2.putText(display,
            "STEP 2/2: OVERHEAD cam — drag divider lines onto lane markings",
            (8, 20), FONT, 0.46, CLR_WHITE, 1)
        cv2.putText(display,
            f"SPACE or ENTER = confirm & start  |  lanes={NUM_LANES}  lane_w={LANE_WIDTH_M}m",
            (8, 42), FONT, 0.40, CLR_GREEN, 1)

    print("\n[Calibration 2/2] OVERHEAD cam — drag vertical dividers onto lane lines.")
    print("  Drag any white divider line left/right.")
    print("  SPACE or ENTER to confirm.\n")

    frame = initial_frame.copy()
    while True:
        try:
            while True:
                evt = event_queue.get_nowait()
                if evt[0] == "FRONT_FRAME":
                    frame = evt[1] if isinstance(evt[1], np.ndarray) else evt[1][0]
        except queue.Empty:
            pass

        display = frame.copy()
        _draw(display)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(16) & 0xFF
        if key in (ord(' '), 13):
            lane_dividers[:] = sorted(local_div)
            print(f"[Calibration] Overhead camera locked.\n"
                  f"  Dividers: {lane_dividers}\n")
            break
        elif key == 27:
            raise SystemExit

    cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)
    overhead_calibrated = True


# ══════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════
def fmt_time(seconds):
    if seconds is None:
        return "--:--.--"
    m  = int(seconds) // 60
    s  = int(seconds) % 60
    cs = int((seconds - int(seconds)) * 100)
    return f"{m:02d}:{s:02d}.{cs:02d}"


def draw_strip_overlay(display, x1, x2, white_px):
    """Vertical strip overlay on the perpendicular (finish) camera feed."""
    h      = display.shape[0]
    active = white_px > FG_PIXEL_THRESHOLD
    colour = CLR_RED if active else CLR_GREEN
    overlay = display.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, h), colour, -1)
    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
    cv2.line(display, (x1, 0), (x1, h), colour, 2)
    cv2.line(display, (x2, 0), (x2, h), colour, 2)


def draw_lane_overlay(display, dividers, lane_px, lane_running):
    """
    Draw straight vertical column overlays on the overhead camera feed.
    No trapezoids — top-down view has no perspective distortion.
    """
    h, w = display.shape[:2]
    for i in range(1, NUM_LANES + 1):
        xl = dividers[i - 1]
        xr = dividers[i]
        occupied = lane_px.get(i, 0) > LANE_FG_THRESHOLD
        running  = lane_running.get(i, False)
        colour   = CLR_RED if occupied else (CLR_GREEN if running else CLR_GRAY)

        overlay = display.copy()
        cv2.rectangle(overlay, (xl, 0), (xr, h), colour, -1)
        cv2.addWeighted(overlay, 0.22, display, 0.78, 0, display)

        cx = (xl + xr) // 2
        label = f"L{i}" + (" RUN" if running else "")
        cv2.putText(display, label, (cx - 12, h // 2), FONT, 0.55, CLR_WHITE, 1)

    for xd in dividers:
        cv2.line(display, (xd, 0), (xd, h), CLR_WHITE, 1)


# ══════════════════════════════════════════════
# SCOREBOARD PANEL
# ══════════════════════════════════════════════
BOARD_W = 340

CLR_PANEL_BG   = (20, 20, 20)
CLR_ROW_DARK   = (35, 35, 35)
CLR_HEADER     = (50, 50, 50)
CLR_DONE       = (0, 180, 60)
CLR_RUNNING_TM = (0, 210, 255)
CLR_LAP        = (160, 160, 160)
CLR_RANK_GOLD  = (0, 200, 220)
CLR_RANK_SIL   = (180, 180, 180)
CLR_RANK_BRZ   = (30, 120, 200)

RANK_COLOURS = [CLR_RANK_GOLD, CLR_RANK_SIL, CLR_RANK_BRZ]


def draw_scoreboard(lane_running, lap_times, race_start_time):
    now = time.perf_counter()

    LAP_ROW_H = 32
    HEADER_H  = 56
    COL_H     = 36
    FOOTER_H  = 30
    PADDING   = 8
    MIN_ROWS  = 8

    n_laps  = len(lap_times)
    rows    = max(n_laps, MIN_ROWS)
    total_h = HEADER_H + COL_H + rows * LAP_ROW_H + FOOTER_H

    img = np.full((total_h, BOARD_W, 3), CLR_PANEL_BG, dtype=np.uint8)

    # ── Race-clock header ─────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (BOARD_W, HEADER_H), CLR_HEADER, -1)

    any_running  = any(lane_running.values())
    race_started = race_start_time is not None
    all_done     = race_started and not any_running and n_laps >= NUM_LANES

    if race_started:
        elapsed   = now - race_start_time
        clock_str = fmt_time(elapsed)
        state_str = "RUNNING" if any_running else ("FINISHED" if all_done else "PAUSED")
        state_clr = CLR_RUNNING_TM if any_running else (CLR_DONE if all_done else CLR_YELLOW)
    else:
        clock_str = "--:--.--"
        state_str = "WAITING FOR WHISTLE"
        state_clr = CLR_GRAY

    cv2.putText(img, clock_str,   (PADDING + 2,   38), FONT, 1.05, CLR_WHITE,  2, cv2.LINE_AA)
    cv2.putText(img, state_str,   (PADDING + 185, 24), FONT, 0.44, state_clr,  1, cv2.LINE_AA)
    cv2.putText(img, "RACE CLOCK",(PADDING + 185, 42), FONT, 0.34, CLR_GRAY,   1, cv2.LINE_AA)
    cv2.line(img, (0, HEADER_H - 1), (BOARD_W, HEADER_H - 1), CLR_GRAY, 1)

    # ── Column header ─────────────────────────────────────────────────────────
    y = HEADER_H
    cv2.rectangle(img, (0, y), (BOARD_W, y + COL_H), CLR_HEADER, -1)
    cv2.putText(img, " LAP",  (PADDING,       y + 24), FONT, 0.50, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, "LANE",  (PADDING + 68,  y + 24), FONT, 0.50, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, "TIME",  (PADDING + 140, y + 24), FONT, 0.50, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.line(img, (0, y + COL_H - 1), (BOARD_W, y + COL_H - 1), CLR_GRAY, 1)
    y += COL_H

    # ── Lap rows ──────────────────────────────────────────────────────────────
    for idx in range(rows):
        bg = CLR_ROW_DARK if idx % 2 == 0 else CLR_PANEL_BG
        cv2.rectangle(img, (0, y), (BOARD_W, y + LAP_ROW_H), bg, -1)

        if idx < n_laps:
            lap_t, lane_id = lap_times[idx]
            rank_clr = RANK_COLOURS[idx] if idx < 3 else CLR_DONE
            cv2.putText(img, f"{idx + 1:2d}",   (PADDING + 4,   y + 22), FONT, 0.60, rank_clr,  2, cv2.LINE_AA)
            cv2.putText(img, f"L{lane_id}",      (PADDING + 68,  y + 22), FONT, 0.58, CLR_WHITE,  1, cv2.LINE_AA)
            cv2.putText(img, fmt_time(lap_t),    (PADDING + 130, y + 22), FONT, 0.62, CLR_DONE,   1, cv2.LINE_AA)

        y += LAP_ROW_H
        cv2.line(img, (0, y - 1), (BOARD_W, y - 1), CLR_HEADER, 1)

    # ── Live elapsed row ──────────────────────────────────────────────────────
    if race_started and any_running:
        elapsed = now - race_start_time
        pulse   = (int(now * 2) % 2 == 0)
        dot_c   = CLR_RUNNING_TM if pulse else CLR_PANEL_BG
        live_y  = HEADER_H + COL_H + n_laps * LAP_ROW_H
        if live_y + LAP_ROW_H <= total_h - FOOTER_H:
            cv2.rectangle(img, (0, live_y), (BOARD_W, live_y + LAP_ROW_H), CLR_PANEL_BG, -1)
            cv2.circle(img, (PADDING + 12, live_y + 16), 6, dot_c, -1, cv2.LINE_AA)
            cv2.putText(img, fmt_time(elapsed),
                        (PADDING + 130, live_y + 22), FONT, 0.62, CLR_RUNNING_TM, 1, cv2.LINE_AA)

    # ── Footer ────────────────────────────────────────────────────────────────
    fy = total_h - FOOTER_H
    cv2.rectangle(img, (0, fy), (BOARD_W, total_h), CLR_HEADER, -1)
    cv2.putText(img, "q=save+quit  r=reset  s+N=arm lane  a=toggle arm-all  c=CSV",
                (PADDING, fy + 19), FONT, 0.30, CLR_GRAY, 1, cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════
def save_results_csv(lap_times, path="race_results.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lap", "lane", "time_sec", "timestamp"])
        writer.writeheader()
        for lap_n, (t, lane_id) in enumerate(lap_times, 1):
            writer.writerow({
                "lap":       lap_n,
                "lane":      lane_id,
                "time_sec":  f"{t:.3f}",
                "timestamp": time.strftime("%H:%M:%S"),
            })
    print(f"[Main] Results saved to {path}")


def stage_dash_50m_results(session_id, lap_times):
    """
    POSTs each lane's finishing time to the FastAPI backend's staging
    endpoint instead of writing to the database directly — the result sits
    as 'pending' until a human reviews and confirms it on screen3, matching
    the rest of GoForFit's AI-test result flow.
    """
    if session_id is None or not lap_times:
        return
    try:
        import requests
        laps_payload = [
            {"lane": lane_id, "time_sec": round(t, 3), "start_mode": "whistle"}
            for t, lane_id in lap_times
        ]
        requests.post(
            "http://localhost:8080/api/stage/dash_50m",
            json={"session_id": session_id, "laps": laps_payload},
            timeout=2,
        ).raise_for_status()
        print(f"[Main] dash_50m staged for review  session={session_id}  "
              f"lanes={[l['lane'] for l in laps_payload]}")
    except Exception as e:
        print(f"[Main] Staging failed (is the GoForFit server running?): {e}")


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
def main():
    global race_start_time, lane_running, lap_times
    global FINISH_CAM_URL, OVERHEAD_CAM_URL, NUM_LANES, LANE_WIDTH_M, SESSION_ID

    args = parse_args()
    FINISH_CAM_URL   = args.finish_cam
    OVERHEAD_CAM_URL = args.overhead_cam
    NUM_LANES        = args.lanes
    LANE_WIDTH_M     = args.lane_width
    SESSION_ID       = args.session_id

    print_session_banner()
    init_lane_state()

    t_whistle  = threading.Thread(target=whistle_thread,      daemon=True)
    t_overhead = threading.Thread(target=overhead_cam_thread, daemon=True)
    t_finish   = threading.Thread(target=finish_cam_thread,   daemon=True)
    t_whistle.start()
    t_overhead.start()
    t_finish.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 560)

    print("[Main] Waiting for camera streams...")
    finish_frame   = None
    overhead_frame = None
    while finish_frame is None or overhead_frame is None:
        try:
            evt = event_queue.get(timeout=5)
            if evt[0] == "FINISH_FRAME" and finish_frame is None:
                finish_frame = evt[1] if isinstance(evt[1], np.ndarray) else evt[1][0]
            elif evt[0] == "FRONT_FRAME" and overhead_frame is None:
                overhead_frame = evt[1] if isinstance(evt[1], np.ndarray) else evt[1][0]
            elif evt[0] in ("FINISH_CAM_ERROR", "FRONT_CAM_ERROR"):
                print(f"[Main] {evt[0]} — check URL and WiFi connection.")
                return
        except queue.Empty:
            print("[Main] Timed out waiting for camera frames. Check both URLs.")
            return

    calibrate_finish_strip(finish_frame)
    calibrate_lane_dividers(overhead_frame)

    finish_display   = finish_frame
    overhead_display = overhead_frame
    white_px         = 0
    strip_x1, strip_x2 = strip_x, strip_x + STRIP_WIDTH
    lane_px          = {i: 0 for i in range(1, NUM_LANES + 1)}

    if lane_dividers is None:
        raise RuntimeError("[Main] Overhead camera dividers not calibrated — aborting.")

    arm_all_on_whistle = True

    print("[Main] System armed. Blow whistle to start all lanes.")
    print("       s+N = arm lane N individually  |  a = toggle arm-all")
    print("       r = reset all  |  q = save+quit  |  c = save CSV  |  ESC = quit\n")

    while True:
        # ── Drain event queue ─────────────────────────────────────────────────
        try:
            while True:
                evt = event_queue.get_nowait()

                if evt[0] == "WHISTLE":
                    race_start_time = evt[1]
                    if arm_all_on_whistle:
                        for i in lane_running:
                            lane_running[i] = True
                    print(f"[Race] START  t={race_start_time:.4f}  "
                          f"lanes={[l for l, r in lane_running.items() if r]}")

                elif evt[0] == "CROSSING":
                    crossing_time, lanes = evt[1]
                    if race_start_time is not None:
                        race_time = crossing_time - race_start_time
                        for lane_id in lanes:
                            if lane_running.get(lane_id, False):
                                lap_times.append((race_time, lane_id))
                                lane_running[lane_id] = False
                                print(f"[Result] Lap {len(lap_times)}  Lane {lane_id} "
                                      f"= {fmt_time(race_time)}")

                elif evt[0] == "FINISH_FRAME":
                    data = evt[1]
                    if isinstance(data, np.ndarray):
                        finish_display = data
                    else:
                        finish_display, _, white_px, strip_x1, strip_x2 = data

                elif evt[0] == "FRONT_FRAME":
                    data = evt[1]
                    if isinstance(data, np.ndarray):
                        overhead_display = data
                    else:
                        overhead_display, lane_px, _ = data

        except queue.Empty:
            pass

        # ── Build display ─────────────────────────────────────────────────────
        CAM_H = 480

        def resize_to_h(img, h):
            oh, ow = img.shape[:2]
            return cv2.resize(img, (int(ow * h / oh), h))

        f_disp = finish_display.copy()
        draw_strip_overlay(f_disp, strip_x1, strip_x2, white_px)
        cv2.putText(f_disp, "FINISH CAM (perpendicular)", (6, f_disp.shape[0] - 8),
                    FONT, 0.45, CLR_WHITE, 1, cv2.LINE_AA)
        f_disp = resize_to_h(f_disp, CAM_H)

        ov_disp = overhead_display.copy()
        draw_lane_overlay(ov_disp, lane_dividers, lane_px, lane_running)
        cv2.putText(ov_disp, "OVERHEAD CAM (lane identity)", (6, ov_disp.shape[0] - 8),
                    FONT, 0.45, CLR_WHITE, 1, cv2.LINE_AA)
        ov_disp = resize_to_h(ov_disp, CAM_H)

        board = draw_scoreboard(lane_running, lap_times, race_start_time)
        bh = board.shape[0]
        if bh < CAM_H:
            pad   = np.full((CAM_H - bh, BOARD_W, 3), CLR_PANEL_BG, dtype=np.uint8)
            board = np.vstack([board, pad])
        elif bh > CAM_H:
            board = board[:CAM_H]

        combined = np.hstack([f_disp, ov_disp, board])
        cv2.imshow(WINDOW_NAME, combined)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('q'):
            save_results_csv(lap_times)
            break
        elif key == ord('c'):
            save_results_csv(lap_times)
        elif key == ord('a'):
            arm_all_on_whistle = not arm_all_on_whistle
            print(f"[Main] Arm-all-on-whistle: {arm_all_on_whistle}")
        elif key == ord('s'):
            print("[Main] Press lane number (1–9) to arm that lane...")
            key2 = cv2.waitKey(2000) & 0xFF
            if ord('1') <= key2 <= ord('9'):
                lane_n = key2 - ord('0')
                if lane_n in lane_running:
                    lane_running[lane_n] = True
                    print(f"[Main] Lane {lane_n} armed.")
        elif key == ord('r'):
            for i in lane_running:
                lane_running[i] = False
            race_start_time = None
            lap_times.clear()
            print("[Main] All lanes reset.")

    cv2.destroyAllWindows()
    print(f"\n[Main] Session ended.")
    for lap_n, (t, lane_id) in enumerate(lap_times, 1):
        print(f"  Lap {lap_n}  (Lane {lane_id}): {fmt_time(t)}")

    # Hand off to the GoForFit web app for human review (screen3) instead of
    # writing straight to the DB — matches every other AI test's flow.
    stage_dash_50m_results(SESSION_ID, lap_times)


if __name__ == "__main__":
    main()