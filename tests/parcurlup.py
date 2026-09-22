"""
AI-Powered Partial Curl-Up Counter (30 seconds)
================================================
Compatible with mediapipe >= 0.10.30 using the Tasks API

pip install mediapipe==0.10.35 opencv-python numpy pandas
"""

import cv2
import numpy as np
import pandas as pd
import time
import argparse
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TEST_DURATION_SEC       = 30
CURL_DOWN_ANGLE         = 100    # hip angle must drop TO/BELOW → "curled"
CURL_UP_ANGLE           = 120    # hip angle must rise TO/ABOVE → "returned"
# NOTE: tuned for the protocol pose — lying flat with KNEES BENT ~90°. Bent
# knees compress the shoulder-hip-knee angle, so "flat" reads ~130° (not ~180°)
# and a curl bottoms out ~45°. The old 145/160 values were straight-leg numbers
# the returned pose could never reach, so reps never counted. See DIAG capture
# 2026-08-07: flat ≈126-139, curl ≈41-75.
MIN_SHOULDER_RISE_FRAC  = 0.015  # shoulder must travel ≥1.5% of frame height
ANGLE_BUFFER_LEN        = 5      # rolling median smoothing window

# MediaPipe Pose landmark indices (same numbering across all API versions)
IDX = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_HIP": 23,      "R_HIP": 24,
    "L_KNEE": 25,     "R_KNEE": 26,
}

# Colours BGR
C_GREEN  = (50, 220, 80)
C_ORANGE = (0, 165, 255)
C_RED    = (60, 60, 240)
C_WHITE  = (255, 255, 255)
C_DARK   = (20, 20, 20)
C_YELLOW = (0, 215, 255)

# ──────────────────────────────────────────────
# GEOMETRY
# ──────────────────────────────────────────────

def calc_angle(a, b, c):
    """Angle at B formed by A-B-C vectors (2-D, degrees)."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def draw_panel(img, x1, y1, x2, y2, r=12, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), colour, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), colour, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, colour, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def draw_ui(frame, count, phase, angle, elapsed, shoulder_rise, visible, test_started, test_done,
            browser_mode=False):
    h, w = frame.shape[:2]
    remaining = max(0.0, TEST_DURATION_SEC - elapsed)

    # Progress bar
    bar_w = int(w * remaining / TEST_DURATION_SEC)
    cv2.rectangle(frame, (0,0), (w,8), (50,50,50), -1)
    bc = C_GREEN if remaining > 10 else C_ORANGE if remaining > 5 else C_RED
    cv2.rectangle(frame, (0,0), (bar_w,8), bc, -1)

    # Count
    draw_panel(frame, 10, 14, 200, 110)
    cv2.putText(frame, "COUNT", (22,38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{count:02d}", (22,100), cv2.FONT_HERSHEY_SIMPLEX, 2.4, C_GREEN, 4, cv2.LINE_AA)

    # Timer
    draw_panel(frame, w-195, 14, w-10, 110)
    cv2.putText(frame, "TIME LEFT", (w-183,38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
    tc = C_GREEN if remaining > 10 else C_ORANGE if remaining > 5 else C_RED
    cv2.putText(frame, f"{remaining:05.2f}s", (w-185,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, tc, 3, cv2.LINE_AA)

    # Bottom info
    draw_panel(frame, 10, h-75, 450, h-10)
    if visible:
        arrow = "v CURL UP" if phase == "down" else "^ RETURN"
        pc = C_ORANGE if phase == "down" else C_GREEN
        cv2.putText(frame, f"Hip Angle: {angle:.1f} deg", (20,h-48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Shoulder Rise: {shoulder_rise*100:.1f}%  {arrow}",
                    (20,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, pc, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Body not fully visible - adjust camera",
                    (20,h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_YELLOW, 2, cv2.LINE_AA)

    # Waiting for SPACE
    if not test_started and not test_done:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        ready_col = C_GREEN if visible else C_YELLOW
        ready_txt = "Body detected! " if visible else "Position yourself in frame"
        cv2.putText(frame, ready_txt, (w//2 - 260, h//2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, ready_col, 2, cv2.LINE_AA)
        # Browser flow starts the timer via the START button (no keyboard);
        # the standalone OpenCV script uses the SPACE key.
        start_txt = ("Tap START to begin the 30s timer" if browser_mode
                     else "Press SPACE to start 30s timer")
        cv2.putText(frame, start_txt,
                    (w//2 - 300, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_WHITE, 3, cv2.LINE_AA)
        cv2.putText(frame, start_txt,
                    (w//2 - 300, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_GREEN, 2, cv2.LINE_AA)
        return

    # Finished
    if elapsed >= TEST_DURATION_SEC:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "TIME UP!", (w//2-130, h//2-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, C_RED, 5, cv2.LINE_AA)
        cv2.putText(frame, f"Final Count: {count}", (w//2-140, h//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, C_GREEN, 4, cv2.LINE_AA)


# ──────────────────────────────────────────────
# POSE DRAWING (manual — no mp.solutions needed)
# ──────────────────────────────────────────────

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(24,26),(25,27),(26,28),           # legs
]

def draw_pose(frame, lms, w, h):
    for a, b in CONNECTIONS:
        if lms[a].visibility > 0.4 and lms[b].visibility > 0.4:
            p1 = (int(lms[a].x * w), int(lms[a].y * h))
            p2 = (int(lms[b].x * w), int(lms[b].y * h))
            cv2.line(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
    for i in range(33):
        if lms[i].visibility > 0.4:
            cx = int(lms[i].x * w)
            cy = int(lms[i].y * h)
            cv2.circle(frame, (cx, cy), 4, C_GREEN, -1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def _autosave_curl_up(session_id, reps):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/curl_up",
            json={"session_id": session_id, "reps": reps, "duration_sec": 30},
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] curl_up staged for review → reps={reps}  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


def run(source):
    # Download the pose landmarker model if not present
    import urllib.request, os
    MODEL_PATH = "pose_landmarker_full.task"
    MODEL_URL  = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task"
    )
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose model (~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

    # Build landmarker
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    # ── Session ID for DB auto-save ──────────────────────────────────────────
    import sys as _sys, os as _os
    _session_id = None
    for _i, _a in enumerate(_sys.argv[1:], 1):
        if _a == "--session-id" and _i + 1 < len(_sys.argv):
            try: _session_id = int(_sys.argv[_i + 1])
            except ValueError: pass
    if _session_id is None:
        try: _session_id = int(_os.environ.get("SESSION_ID", ""))
        except (ValueError, TypeError): pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # State
    count        = 0
    phase        = "up"
    start_time   = None
    test_started = False
    test_done    = False
    angle_buf    = deque(maxlen=ANGLE_BUFFER_LEN)
    shoulder_buf = deque(maxlen=ANGLE_BUFFER_LEN)
    rep_log      = []
    frame_idx    = 0

    # Create a true fullscreen window
    WIN = "Partial Curl-Up Counter - GoForFit NEP FMS"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get screen resolution via a tiny trick
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    SCREEN_W = root.winfo_screenwidth()
    SCREEN_H = root.winfo_screenheight()
    root.destroy()

    print("Press SPACE to start | R to reset | F to toggle fullscreen | Q to quit.")

    fullscreen = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Stretch frame to fill the screen exactly
        frame = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        frame_idx += 1
        timestamp_ms = int(frame_idx * 1000 / fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        angle         = 0.0
        shoulder_rise = 0.0
        body_visible  = False
        elapsed       = (time.time() - start_time) if test_started else 0.0

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]   # first (only) pose

            req_vis = [lms[i].visibility for i in IDX.values()]
            body_visible = all(v > 0.5 for v in req_vis)

            if body_visible:
                shoulder = (lm_px(lms, IDX["L_SHOULDER"], w, h) +
                            lm_px(lms, IDX["R_SHOULDER"], w, h)) / 2
                hip      = (lm_px(lms, IDX["L_HIP"],      w, h) +
                            lm_px(lms, IDX["R_HIP"],      w, h)) / 2
                knee     = (lm_px(lms, IDX["L_KNEE"],     w, h) +
                            lm_px(lms, IDX["R_KNEE"],     w, h)) / 2

                angle_buf.append(calc_angle(shoulder, hip, knee))
                angle = float(np.median(list(angle_buf)))

                shoulder_buf.append(shoulder[1] / h)
                if len(shoulder_buf) == ANGLE_BUFFER_LEN:
                    shoulder_rise = float(
                        np.max(list(shoulder_buf)) - np.min(list(shoulder_buf))
                    )

                if test_started and not test_done and elapsed < TEST_DURATION_SEC:
                    if phase == "up" and angle <= CURL_DOWN_ANGLE:
                        phase = "down"
                    elif phase == "down" and angle >= CURL_UP_ANGLE:
                        if shoulder_rise >= MIN_SHOULDER_RISE_FRAC:
                            count += 1
                            rep_log.append({
                                "rep"              : count,
                                "min_angle_deg"    : round(angle, 2),
                                "shoulder_rise_pct": round(shoulder_rise * 100, 2),
                                "timestamp_s"      : round(elapsed, 3),
                            })
                        phase = "up"

            draw_pose(frame, lms, w, h)

        if test_started and elapsed >= TEST_DURATION_SEC:
            test_done = True

        draw_ui(frame, count, phase, angle, elapsed, shoulder_rise, body_visible, test_started, test_done)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and not test_started and not test_done:
            start_time   = time.time()
            test_started = True
            print("Timer started!")
        if key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, prop)
        if key == ord('r'):
            count = 0; phase = "up"; start_time = None
            test_started = False; test_done = False
            angle_buf.clear(); shoulder_buf.clear(); rep_log = []

    # ── Auto-save final count ─────────────────────────────────────────────────
    if _session_id is not None and count > 0:
        _autosave_curl_up(_session_id, count)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if rep_log:
        df = pd.DataFrame(rep_log)
        df["grade"] = pd.cut(
            df["min_angle_deg"],
            bins=[0, 130, 140, 150, 160, 180],
            labels=["Excellent", "Good", "Average", "Below Avg", "Minimal"],
        )
        print("\n── Session Report ──────────────────────────────────")
        print(df.to_string(index=False))
        print(f"\nTotal Reps : {count}")
        print(f"Avg Hip Angle : {df['min_angle_deg'].mean():.1f} deg")
        df.to_csv("curl_up_session.csv", index=False)
        print("Log saved -> curl_up_session.csv")
    else:
        print("No reps recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int,  default=0)
    parser.add_argument("--video",  type=str,  default=None)
    args = parser.parse_args()
    run(args.video if args.video else args.camera)