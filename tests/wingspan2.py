"""
╔══════════════════════════════════════════════════════════╗
║         WINGSPAN CALCULATOR  — MediaPipe 0.10+ Tasks API ║
║  Middle-finger-tip to middle-finger-tip in real cm       ║
╚══════════════════════════════════════════════════════════╝

First run — auto-downloads two model files (~10 MB) to the
same folder as this script:
  • hand_landmarker.task
  • pose_landmarker_lite.task

Controls (shown on-screen):
  C — Calibrate   M — Measure   SPACE — 3-sec median snapshot
  S — Save result   Q — Quit

Calibration text entry is done INSIDE the video window —
no terminal typing needed; the video never freezes.
"""

import os, sys, csv, json, time, urllib.request
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from mediapipe.tasks        import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe              import Image, ImageFormat
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
    HandLandmarksConnections,
    PoseLandmarksConnections,
)
import mp_tasks  # cv2-based skeleton drawing — see below

# ══════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════
PERSON_DISTANCE_CM = 180.0      # informational only — not enforced
CALIBRATION_FILE   = "wingspan_calibration.json"
LOG_FILE           = "wingspan_log.csv"
MIDDLE_TIP_IDX     = 12         # MediaPipe hand landmark: middle finger tip
SNAPSHOT_SECS      = 3          # seconds to collect when SPACE is pressed

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# BGR colours
GREEN  = (0,  230,  80)
BLUE   = (80, 160, 255)
RED    = (30,  30, 220)
YELLOW = (0,  220, 240)
CYAN   = (220, 220,  0)
WHITE  = (255, 255, 255)
DARK   = (20,   20,  20)
ORANGE = (0,  140, 255)


# ══════════════════════════════════════════════════════════
# Model download
# ══════════════════════════════════════════════════════════
def ensure_models(folder="."):
    paths = {}
    for fname, url in MODEL_URLS.items():
        dst = os.path.join(folder, fname)
        if not os.path.exists(dst):
            print(f"[DOWNLOAD] {fname} …", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, dst)
                print(f"done  ({os.path.getsize(dst)//1024} KB)")
            except Exception as e:
                print(f"\n[ERROR] {e}\nCheck internet and retry.")
                sys.exit(1)
        paths[fname] = dst
    return paths


# ══════════════════════════════════════════════════════════
# Calibration I/O
# ══════════════════════════════════════════════════════════
def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


def save_calibration(data):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════════
def px_dist(p1, p2):
    return float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))

def pixels_to_cm(px, calib):
    return px * calib["scale_factor"]

def compute_scale(px_span, real_cm):
    return real_cm / px_span


# ══════════════════════════════════════════════════════════
# Arm angle helpers  (NEW)
# ══════════════════════════════════════════════════════════
# MediaPipe Pose landmark indices used:
#   11 = left shoulder   12 = right shoulder
#   23 = left hip        24 = right hip
#   13 = left elbow      14 = right elbow
#   15 = left wrist      16 = right wrist

def _vec(a, b):
    """2-D vector from point a to point b (each a (x,y) tuple)."""
    return np.array([b[0]-a[0], b[1]-a[1]], dtype=float)

def _angle_between(v1, v2):
    """Angle in degrees between two 2-D vectors (0–180)."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def arm_angle(shoulder, hip, wrist):
    """
    Angle (degrees) at the shoulder between:
      • torso vector  : shoulder → hip
      • arm vector    : shoulder → wrist
    Returns None if any point is missing.
    """
    if shoulder is None or hip is None or wrist is None:
        return None
    v_torso = _vec(shoulder, hip)
    v_arm   = _vec(shoulder, wrist)
    return _angle_between(v_torso, v_arm)


ANGLE_MIN = 88.0   # arms must be within this range for a valid measurement
ANGLE_MAX = 92.0

def _angle_ok(a):
    """True if angle is within the valid 88–92° window."""
    return a is not None and ANGLE_MIN <= a <= ANGLE_MAX


def draw_arm_angles(frame, p_result, fw, fh):
    """
    Compute and render the angle each arm makes with the torso.

    Two display layers:
      1. Arc + label drawn AT each shoulder on the body.
      2. A fixed bottom-left panel showing both angles large & always visible.

    Angle definition: at the shoulder, between
      torso vector (shoulder → hip)  and  arm vector (shoulder → wrist).
    Full T-pose ≈ 90°.  Arms down ≈ 0°.

    Returns (l_ang, r_ang) — floats or None — so the caller can gate
    measurement on both angles being within ANGLE_MIN..ANGLE_MAX.
    """
    W, H = frame.shape[1], frame.shape[0]

    # ── extract pose landmarks (low threshold so arms at full stretch still fire)
    l_ang = r_ang = None
    l_shoulder = r_shoulder = None
    l_hip = r_hip = None
    l_wrist = r_wrist = None

    if p_result.pose_landmarks:
        lms = p_result.pose_landmarks[0]

        def lm_px(idx, vis_thresh=0.15):
            if idx >= len(lms):
                return None
            l = lms[idx]
            if l.visibility < vis_thresh:
                return None
            return (l.x * fw, l.y * fh)

        l_shoulder = lm_px(11);  r_shoulder = lm_px(12)
        l_hip      = lm_px(23);  r_hip      = lm_px(24)
        l_wrist    = lm_px(15);  r_wrist    = lm_px(16)

        l_ang = arm_angle(l_shoulder, l_hip, l_wrist)
        r_ang = arm_angle(r_shoulder, r_hip, r_wrist)

        # ── 1. On-body arc + label at each shoulder ──────────────────────
        radius = 40
        sides = [
            ("L", l_shoulder, l_hip, l_wrist, l_ang, BLUE),
            ("R", r_shoulder, r_hip, r_wrist, r_ang, ORANGE),
        ]
        for side_label, shoulder, hip, wrist, angle, col in sides:
            if angle is None or shoulder is None:
                continue
            sx, sy = int(shoulder[0]), int(shoulder[1])

            # arc colour: green if in range, red if out
            arc_col = GREEN if _angle_ok(angle) else RED

            # draw arc spanning the angle at the shoulder
            if hip is not None and wrist is not None:
                v_t   = _vec(shoulder, hip)
                v_a   = _vec(shoulder, wrist)
                ang_t = float(np.degrees(np.arctan2(v_t[1], v_t[0])))
                ang_a = float(np.degrees(np.arctan2(v_a[1], v_a[0])))
                start_a, end_a = min(ang_t, ang_a), max(ang_t, ang_a)
                if end_a - start_a > 180:
                    start_a, end_a = end_a, start_a + 360
                cv2.ellipse(frame, (sx, sy), (radius, radius),
                            0, start_a, end_a, arc_col, 2, cv2.LINE_AA)

            # small floating label next to the shoulder
            angle_lbl = f"{angle:.0f}\u00b0"
            offset_x  = -(55 + 8) if side_label == "L" else radius + 6
            lx = sx + offset_x
            ly = sy - 10
            (tw, th), _ = cv2.getTextSize(
                angle_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame,
                          (lx - 4, ly - th - 4),
                          (lx + tw + 4, ly + 6),
                          (10, 10, 10), -1)
            cv2.putText(frame, angle_lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, arc_col, 2, cv2.LINE_AA)

    # ── 2. Fixed bottom-left panel — always visible ───────────────────────
    both_ok    = _angle_ok(l_ang) and _angle_ok(r_ang)
    border_col = GREEN if both_ok else RED

    PW, PH = 230, 110
    px0    = 8
    py0    = H - PH - 8

    # semi-transparent background
    ov = frame.copy()
    cv2.rectangle(ov, (px0, py0), (px0 + PW, py0 + PH), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px0 + PW, py0 + PH), border_col, 2)

    # header — shows range reminder
    hdr = f"ARM ANGLES  ({ANGLE_MIN:.0f}-{ANGLE_MAX:.0f}\u00b0)"
    cv2.putText(frame, hdr,
                (px0 + 8, py0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1, cv2.LINE_AA)

    # left arm row
    l_col = GREEN if _angle_ok(l_ang) else (RED if l_ang is not None else (120,120,120))
    l_str = f"Left :  {l_ang:.0f}\u00b0" if l_ang is not None else "Left :  --"
    cv2.putText(frame, l_str,
                (px0 + 8, py0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, l_col, 2, cv2.LINE_AA)

    # right arm row
    r_col = GREEN if _angle_ok(r_ang) else (RED if r_ang is not None else (120,120,120))
    r_str = f"Right:  {r_ang:.0f}\u00b0" if r_ang is not None else "Right:  --"
    cv2.putText(frame, r_str,
                (px0 + 8, py0 + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, r_col, 2, cv2.LINE_AA)

    return l_ang, r_ang


# ══════════════════════════════════════════════════════════
# Stable median with outlier filtering
# ══════════════════════════════════════════════════════════
def compute_stable_median(samples):
    """
    Returns a robust median:
    - Filters out samples more than 1.5 std-devs from the raw median
      (removes jitter spikes without losing real readings).
    - Falls back to raw median if filtering would leave too few samples.
    """
    if not samples:
        return 0.0
    arr = np.array(samples, dtype=float)
    raw_med = np.median(arr)
    if len(arr) < 6:
        return float(raw_med)
    std = np.std(arr)
    if std < 0.1:                          # already very stable
        return float(raw_med)
    filtered = arr[np.abs(arr - raw_med) <= 1.5 * std]
    if len(filtered) < max(3, len(arr) // 3):
        return float(raw_med)              # keep raw if too many removed
    return float(np.median(filtered))


# ══════════════════════════════════════════════════════════
# Drawing primitives
# ══════════════════════════════════════════════════════════
def rounded_rect(img, x1, y1, x2, y2, r, col, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), col, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), col, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, col, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def txt(img, text, pos, scale=0.60, col=WHITE, thick=1, shadow=True):
    x, y = int(pos[0]), int(pos[1])
    f = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (x+1, y+1), f, scale, DARK, thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), f, scale, col, thick, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# LIVE measurement bar  (big centred overlay at top)
# ══════════════════════════════════════════════════════════
def draw_live_measurement(img, result_cm, left_tip, right_tip, fw, fh):
    """Always-visible large measurement display + connecting line."""

    # ── line between fingertips ──────────────────────
    if left_tip and right_tip:
        lx = int(left_tip[0]  * fw);  ly = int(left_tip[1]  * fh)
        rx = int(right_tip[0] * fw);  ry = int(right_tip[1] * fh)

        cv2.line(img, (lx, ly), (rx, ry), GREEN, 2, cv2.LINE_AA)

        # endpoint circles
        for pt in [(lx, ly), (rx, ry)]:
            cv2.circle(img, pt, 12, GREEN, -1)
            cv2.circle(img, pt, 15, WHITE, 2)
            cv2.circle(img, pt,  4, WHITE, -1)

    # ── big centred reading at top of frame ──────────
    W = img.shape[1]
    if result_cm is not None:
        label = f"{result_cm:.1f} cm"
        col   = GREEN
    else:
        label = "-- cm  (spread both arms)"
        col   = (120, 120, 120)

    scale = 1.4
    thick = 3
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bx = (W - tw) // 2
    by = 70

    rounded_rect(img, bx-16, by-th-14, bx+tw+16, by+bl+10, 10,
                 (10, 10, 10), alpha=0.70)
    txt(img, label, (bx, by), scale, col, thick)

    # midpoint label on the line
    if left_tip and right_tip and result_cm is not None:
        lx = int(left_tip[0]  * fw);  ly = int(left_tip[1]  * fh)
        rx = int(right_tip[0] * fw);  ry = int(right_tip[1] * fh)
        mx, my = (lx+rx)//2, (ly+ry)//2
        mini = f"{result_cm:.1f} cm"
        (mw, mh), _ = cv2.getTextSize(mini, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        rounded_rect(img, mx-mw//2-6, my-mh-10,
                     mx+mw//2+6, my+4, 5, (20,20,20), 0.75)
        txt(img, mini, (mx-mw//2, my), 0.70, YELLOW, 2)


# ══════════════════════════════════════════════════════════
# Snapshot overlay  (SPACE bar — median)
# ══════════════════════════════════════════════════════════
def draw_snapshot_overlay(img, snap):
    """
    snap dict keys:
      phase      : "COUNTING" | "DONE"
      start_time : time.time() when SPACE was pressed
      samples    : list of cm readings collected so far
      final_cm   : stable median result (set when DONE)
    """
    W, H    = img.shape[1], img.shape[0]
    phase   = snap["phase"]
    elapsed = time.time() - snap["start_time"]
    remain  = max(0.0, SNAPSHOT_SECS - elapsed)

    if phase == "COUNTING":
        # ── countdown ring ───────────────────────────
        cx, cy = W // 2, H // 2
        radius = 90

        cv2.circle(img, (cx, cy), radius + 6, (20, 20, 20), -1)
        cv2.circle(img, (cx, cy), radius + 6, WHITE, 2)

        progress = min(elapsed / SNAPSHOT_SECS, 1.0)
        angle    = int(360 * progress)
        cv2.ellipse(img, (cx, cy), (radius, radius),
                    -90, 0, angle, GREEN, 8, cv2.LINE_AA)

        count_lbl = f"{int(remain) + 1}"
        (cw, ch), _ = cv2.getTextSize(count_lbl,
                                      cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
        txt(img, count_lbl,
            (cx - cw//2, cy + ch//2),
            2.5, WHITE, 4)

        lbl = "Hold still — measuring..."
        (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        txt(img, lbl, (cx - lw//2, cy - radius - 18),
            0.65, YELLOW, 2)

        # live running median preview
        if snap["samples"]:
            live_med = compute_stable_median(snap["samples"])
            med_lbl  = f"~{live_med:.1f} cm"
            (nw, _), _ = cv2.getTextSize(med_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            txt(img, med_lbl, (cx - nw//2, cy + radius + 30),
                0.60, CYAN, 2)

        n_lbl = f"{len(snap['samples'])} samples"
        (nw2, _), _ = cv2.getTextSize(n_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        txt(img, n_lbl, (cx - nw2//2, cy + radius + 58),
            0.48, (160, 160, 160), 1)

    elif phase == "DONE":
        # ── big result card ───────────────────────────
        final = snap["final_cm"]
        n     = len(snap["samples"])
        lbl   = f"{final:.1f} cm"

        scale = 2.2
        thick = 5
        (fw2, fh2), bl = cv2.getTextSize(
            lbl, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)

        cx = W // 2
        bx = cx - fw2 // 2
        by = H // 2 + fh2 // 2

        pad = 28
        rounded_rect(img,
                     bx - pad, by - fh2 - pad - 10,
                     bx + fw2 + pad, by + bl + pad,
                     18, (10, 10, 10), alpha=0.85)

        txt(img, lbl, (bx, by), scale, GREEN, thick)

        sub = f"median of {n} samples over {SNAPSHOT_SECS}s  |  SPACE=redo  S=save"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        txt(img, sub, (cx - sw//2, by + bl + 22), 0.52, YELLOW, 1)


# ══════════════════════════════════════════════════════════
# Side-panel HUD
# ══════════════════════════════════════════════════════════
def draw_hud(img, mode, calibration, result_cm, history, fw, fh):
    W, H = img.shape[1], img.shape[0]
    PW   = 300
    px0  = W - PW - 8

    rounded_rect(img, px0, 8, W-8, H-8, 12, (10,10,10), alpha=0.65)

    px, py, lh = px0 + 10, 38, 28

    txt(img, "WINGSPAN CALC", (px, py), 0.58, YELLOW, 2);  py += lh + 2

    mc = GREEN if mode == "MEASURE" else ORANGE
    txt(img, f"Mode: {mode}", (px, py), 0.58, mc, 2);     py += lh + 4

    # calibration info
    if calibration:
        sf   = calibration["scale_factor"]
        real = calibration["real_wingspan_cm"]
        txt(img, f"Ref span: {real:.1f} cm", (px, py), 0.50, WHITE); py += lh-4
        txt(img, f"Scale: {sf:.4f} cm/px", (px, py), 0.48, CYAN); py += lh
    else:
        txt(img, "No calibration!", (px, py), 0.52, RED);         py += lh-2
        txt(img, "Press [C] first", (px, py), 0.50, RED);         py += lh

    # recent history
    py += 6
    if history:
        txt(img, "Saved measurements:", (px, py), 0.48, YELLOW); py += lh-4
        for v in list(history)[-6:][::-1]:
            txt(img, f"  {v:.1f} cm", (px, py), 0.50, WHITE); py += lh-6

    # controls
    py = H - 140
    txt(img, "Controls:", (px, py), 0.48, YELLOW); py += 22
    for k, d in [("[C]","Calibrate"),("[M]","Measure"),
                 ("[SPC]","Snapshot (median)"),
                 ("[S]","Save reading"),("[Q]","Quit")]:
        txt(img, f"  {k}  {d}", (px, py), 0.44, WHITE); py += 19


# ══════════════════════════════════════════════════════════
# In-window calibration overlay  (no terminal input!)
# ══════════════════════════════════════════════════════════
# calib_state keys:
#   phase      : "SPREAD" | "ENTER_CM" | "CONFIRM"
#   px_buf     : list of pixel spans
#   hold_start : time.time() when stable hold started
#   cm_buf     : str  (typed cm value)
#   avg_px     : float (locked average when moving to ENTER_CM)

def draw_calib_overlay(img, cs, fw, fh,
                       left_tip=None, right_tip=None):
    """Render the full calibration overlay on img in-place."""
    ov = img.copy()
    cv2.rectangle(ov, (0,0), (img.shape[1], img.shape[0]),
                  (8, 8, 50), -1)
    cv2.addWeighted(ov, 0.48, img, 0.52, 0, img)

    W, H  = img.shape[1], img.shape[0]
    cx    = W // 2
    phase = cs["phase"]
    avg   = cs.get("avg_px", 0.0)

    # title
    txt(img, "CALIBRATION MODE", (cx-185, 60), 0.95, YELLOW, 2)

    # live line while spreading
    if phase == "SPREAD" and left_tip and right_tip:
        lx = int(left_tip[0]  * fw);  ly = int(left_tip[1]  * fh)
        rx = int(right_tip[0] * fw);  ry = int(right_tip[1] * fh)
        cv2.line(img, (lx,ly), (rx,ry), ORANGE, 2, cv2.LINE_AA)
        for pt in [(lx,ly),(rx,ry)]:
            cv2.circle(img, pt, 11, ORANGE, -1)
            cv2.circle(img, pt, 14, WHITE,   2)
        if cs["px_buf"]:
            cur = f"Span: {cs['px_buf'][-1]:.0f} px"
            txt(img, cur, (cx-70, ly - 25), 0.60, WHITE, 1)

    steps = [
        ("1", "Stand in front of camera, arms wide apart"),
        ("2", "Spread arms fully — hold still for 2 sec"),
        ("3", "Type your actual wingspan in cm, press ENTER"),
    ]
    phase_idx = {"SPREAD": 1, "ENTER_CM": 2, "CONFIRM": 3}
    done_up_to = phase_idx.get(phase, 0)

    y = 115
    for i, (num, desc) in enumerate(steps):
        step_n = i + 1
        if step_n < done_up_to:
            col = GREEN
        elif step_n == done_up_to:
            col = YELLOW
        else:
            col = (120, 120, 120)
        txt(img, f"  {num}. {desc}", (cx-290, y), 0.56, col, 1)
        y += 32

    y += 12

    if phase == "SPREAD":
        if cs["px_buf"]:
            elapsed = (time.time() - cs["hold_start"]
                       if cs["hold_start"] else 0)
            bar_w   = int(min(elapsed / 2.0, 1.0) * 300)
            cv2.rectangle(img, (cx-150, y), (cx+150, y+16),
                          (60,60,60), -1)
            cv2.rectangle(img, (cx-150, y), (cx-150+bar_w, y+16),
                          GREEN, -1)
            txt(img, "Hold still…", (cx-50, y+13), 0.52, WHITE)
        else:
            txt(img, "Spread both arms wide to begin",
                (cx-195, y), 0.58, ORANGE, 1)
        y += 36
        txt(img, f"Avg pixel span: {avg:.0f} px" if avg else "Avg pixel span: --",
            (cx-150, y), 0.54, CYAN)
        y += 32
        txt(img, "ESC / Q = cancel", (cx-95, y), 0.48, (150,150,150))

    elif phase == "ENTER_CM":
        txt(img, f"Avg span locked: {avg:.0f} px", (cx-160, y),
            0.54, CYAN); y += 34

        prompt = "Your wingspan (cm):"
        txt(img, prompt, (cx-160, y), 0.60, WHITE, 1); y += 30

        # text box
        box_x1, box_y1 = cx-160, y
        box_x2, box_y2 = cx+160, y+38
        cv2.rectangle(img, (box_x1,box_y1), (box_x2,box_y2), WHITE, 2)
        cursor = cs["cm_buf"] + ("|" if int(time.time()*2) % 2 == 0 else "")
        txt(img, cursor, (box_x1+8, box_y2-10), 0.70, GREEN, 2)
        y = box_y2 + 18

        txt(img, "Type on keyboard.  ENTER = confirm  |  BACKSPACE = delete",
            (cx-295, y), 0.46, (180,180,180)); y += 28
        txt(img, "ESC / Q = cancel", (cx-95, y), 0.48, (150,150,150))

    elif phase == "CONFIRM":
        cm_v = cs.get("cm_buf", "?")
        txt(img, f"Wingspan: {cm_v} cm", (cx-160, y), 0.62, GREEN, 2); y+=30
        txt(img, f"Px span:  {avg:.0f} px", (cx-160, y), 0.58, CYAN); y+=40
        txt(img, "Press ENTER to save   |   ESC to cancel",
            (cx-230, y), 0.58, YELLOW)


# ══════════════════════════════════════════════════════════
# Detector builders
# ══════════════════════════════════════════════════════════
def build_hand_detector(model_path):
    opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return HandLandmarker.create_from_options(opts)


def build_pose_detector(model_path):
    opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return PoseLandmarker.create_from_options(opts)


# Detection here was always Tasks API; only the DRAWING went through
# solutions.drawing_utils, which took a protobuf landmark list and was
# removed in mediapipe 0.10.30 along with the rest of solutions. mp_tasks
# draws the same skeletons with plain cv2 calls, so the conversion to
# NormalizedLandmarkList protos that used to sit here is gone entirely.

def draw_hand_skeleton(frame, landmarks_list):
    mp_tasks.draw_hand_skeleton(frame, landmarks_list)


def draw_pose_skeleton(frame, p_result):
    mp_tasks.draw_pose_skeleton(frame, p_result)

# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
def _autosave_wingspan(session_id, wingspan_cm):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:5000/api/stage/wingspan",
            json={"session_id": session_id, "wingspan_cm": wingspan_cm},
            timeout=2,
        )
        print(f"  [DB] wingspan staged for review → {wingspan_cm} cm  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

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

    print("[INIT] Checking model files …")
    models     = ensure_models(script_dir)
    hand_det   = build_hand_detector(models["hand_landmarker.task"])
    pose_det   = build_pose_detector(models["pose_landmarker_lite.task"])
    print("[INIT] Detectors ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {fw}×{fh}  |  Press C=calibrate Q=quit")

    # ── app state ─────────────────────────────────────
    calibration  = load_calibration()
    mode         = "MEASURE"
    result_cm    = None
    history      = []
    snap         = None   # None when idle; dict when SPACE countdown active
    frame_ts_ms  = 0

    # calibration sub-state
    cs = {}

    if calibration:
        print(f"[OK] Calibration loaded: {calibration['real_wingspan_cm']} cm")
    else:
        print("[INFO] No calibration yet — press C.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame       = cv2.flip(frame, 1)
        frame_ts_ms += 33
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img      = Image(image_format=ImageFormat.SRGB, data=rgb)

        h_result = hand_det.detect_for_video(mp_img, frame_ts_ms)
        p_result = pose_det.detect_for_video(mp_img, frame_ts_ms)

        # skeletons
        if h_result.hand_landmarks:
            draw_hand_skeleton(frame, h_result.hand_landmarks)
        draw_pose_skeleton(frame, p_result)

        # ── locate fingertips ─────────────────────────
        left_tip = right_tip = None
        if h_result.hand_landmarks and h_result.handedness:
            for lms, handedness in zip(h_result.hand_landmarks,
                                       h_result.handedness):
                label = handedness[0].category_name
                tip   = lms[MIDDLE_TIP_IDX]
                if label == "Left":
                    left_tip  = (tip.x, tip.y)
                else:
                    right_tip = (tip.x, tip.y)

        # ── pixel span ────────────────────────────────
        span_px = None
        if left_tip and right_tip:
            lx = left_tip[0]  * fw;  ly = left_tip[1]  * fh
            rx = right_tip[0] * fw;  ry = right_tip[1] * fh
            span_px = px_dist((lx, ly), (rx, ry))

        # ══════════════════════════════════════════════
        # MEASURE mode
        # ══════════════════════════════════════════════
        if mode == "MEASURE":
            # ── arm angles (drawn first; returns angles for gating) ───────
            l_ang, r_ang = draw_arm_angles(frame, p_result, fw, fh)
            arms_ok = _angle_ok(l_ang) and _angle_ok(r_ang)

            # measurement is only valid when both arms are at 88-92 deg
            if span_px and calibration and arms_ok:
                result_cm = pixels_to_cm(span_px, calibration)
            else:
                result_cm = None

            draw_live_measurement(frame, result_cm,
                                  left_tip, right_tip, fw, fh)

            # ── "align arms" warning when angles are out of range ─────────
            if not arms_ok:
                W = frame.shape[1]
                warn = "Align arms to 90 deg to measure"
                (ww, wh), _ = cv2.getTextSize(
                    warn, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
                wx = (W - ww) // 2
                wy = 120
                cv2.rectangle(frame,
                              (wx - 10, wy - wh - 8),
                              (wx + ww + 10, wy + 8),
                              (20, 20, 20), -1)
                cv2.rectangle(frame,
                              (wx - 10, wy - wh - 8),
                              (wx + ww + 10, wy + 8),
                              RED, 1)
                cv2.putText(frame, warn, (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, RED, 2, cv2.LINE_AA)

            draw_hud(frame, mode, calibration, result_cm, history, fw, fh)

            # ── snapshot: collect samples & auto-finish ──
            if snap is not None and snap["phase"] == "COUNTING":
                # only add sample when arms are correctly positioned
                if result_cm is not None and arms_ok:
                    snap["samples"].append(result_cm)
                elapsed = time.time() - snap["start_time"]
                if elapsed >= SNAPSHOT_SECS:
                    if snap["samples"]:
                        snap["final_cm"] = compute_stable_median(snap["samples"])
                    else:
                        snap["final_cm"] = 0.0
                    snap["phase"] = "DONE"

            if snap is not None:
                draw_snapshot_overlay(frame, snap)

        # ══════════════════════════════════════════════
        # CALIBRATE mode
        # ══════════════════════════════════════════════
        elif mode == "CALIBRATE":
            phase = cs.get("phase", "SPREAD")

            if phase == "SPREAD":
                if span_px:
                    cs.setdefault("px_buf", []).append(span_px)
                    if cs.get("hold_start") is None:
                        cs["hold_start"] = time.time()
                    elapsed = time.time() - cs["hold_start"]
                    cs["avg_px"] = float(np.mean(cs["px_buf"][-90:]))
                    if elapsed >= 2.0 and len(cs["px_buf"]) >= 10:
                        cs["avg_px"] = float(np.mean(cs["px_buf"][-90:]))
                        cs["phase"]  = "ENTER_CM"
                        cs["cm_buf"] = ""
                else:
                    cs["hold_start"] = None   # reset hold if hands lost

            draw_calib_overlay(frame, cs, fw, fh, left_tip, right_tip)

        cv2.imshow("Wingspan Calculator", frame)

        # ══════════════════════════════════════════════
        # Keyboard handling
        # ══════════════════════════════════════════════
        key = cv2.waitKey(1) & 0xFF

        # ── global quit / cancel ──────────────────────
        if key in (ord('q'), ord('Q'), 27):          # Q or ESC
            if mode == "CALIBRATE":
                mode = "MEASURE"
                cs   = {}
                print("[INFO] Calibration cancelled.")
                continue
            break

        # ── mode switches ─────────────────────────────
        if key in (ord('c'), ord('C')) and mode != "CALIBRATE":
            mode = "CALIBRATE"
            cs   = {"phase": "SPREAD", "px_buf": [],
                    "hold_start": None, "avg_px": 0.0,
                    "cm_buf": ""}
            print("[INFO] Calibration started.")
            continue

        if key in (ord('m'), ord('M')):
            mode = "MEASURE"; cs = {}; snap = None
            print("[INFO] Measurement mode.")
            continue

        # ── save ──────────────────────────────────────
        if key in (ord('s'), ord('S')) and mode == "MEASURE":
            val_to_save = None
            if snap is not None and snap["phase"] == "DONE":
                val_to_save = snap["final_cm"]
            elif result_cm is not None:
                val_to_save = result_cm

            if val_to_save is not None:
                history.append(val_to_save)
                # ── Auto-save to DB ──────────────────────────────────────────
                if _session_id is not None:
                    _autosave_wingspan(_session_id, round(val_to_save, 2))
                ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                newfile = not os.path.exists(LOG_FILE)
                with open(LOG_FILE, "a", newline="") as f:
                    w = csv.writer(f)
                    if newfile:
                        w.writerow(["timestamp", "wingspan_cm",
                                    "distance_cm"])
                    w.writerow([ts, f"{val_to_save:.2f}",
                                PERSON_DISTANCE_CM])
                print(f"[SAVED] {val_to_save:.2f} cm → {LOG_FILE}")
                snap = None
            else:
                print("[WARN] No measurement yet.")
            continue

        # ── spacebar: start / redo snapshot ───────────
        if key == ord(' ') and mode == "MEASURE":
            if calibration:
                snap = {
                    "phase":      "COUNTING",
                    "start_time": time.time(),
                    "samples":    [],
                    "final_cm":   None,
                }
            else:
                print("[WARN] Calibrate first before taking a snapshot.")
            continue

        # ══════════════════════════════════════════════
        # In-window text input for calibration
        # ══════════════════════════════════════════════
        if mode == "CALIBRATE":
            phase = cs.get("phase", "SPREAD")

            if phase == "ENTER_CM":
                if key == 13:           # ENTER → advance to confirm
                    raw = cs["cm_buf"].strip()
                    try:
                        real_cm = float(raw)
                        if real_cm <= 0:
                            raise ValueError
                        cs["phase"] = "CONFIRM"
                    except ValueError:
                        cs["cm_buf"] = ""   # clear bad input

                elif key == 8:          # BACKSPACE
                    cs["cm_buf"] = cs["cm_buf"][:-1]

                elif 32 <= key <= 126:  # printable ASCII
                    ch = chr(key)
                    if ch.isdigit() or (ch == '.' and '.' not in cs["cm_buf"]):
                        cs["cm_buf"] += ch

            elif phase == "CONFIRM":
                if key == 13:           # ENTER → save
                    avg_px  = cs["avg_px"]
                    real_cm = float(cs["cm_buf"])
                    sf      = compute_scale(avg_px, real_cm)
                    calibration = {
                        "real_wingspan_cm": real_cm,
                        "pixel_span":       avg_px,
                        "scale_factor":     sf,
                        "distance_cm":      PERSON_DISTANCE_CM,
                        "timestamp":        datetime.now().isoformat(),
                    }
                    save_calibration(calibration)
                    print(f"[OK] Calibration saved: "
                          f"{real_cm} cm | scale {sf:.5f} cm/px")
                    mode = "MEASURE"
                    cs   = {}

    # ── cleanup ───────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    pose_det.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()