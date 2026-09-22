"""
Sit and Reach Test – Distance Measurement
==========================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Measurement workflow
--------------------
  IDLE        Calibration runs live. Hand position shown but not recorded.
  SPACE #1    Snapshot current hand position as the START point.
  MEASURING   Track hand every frame; record maximum forward displacement.
  SPACE #2    Freeze the max displacement as the final result.
  R           Reset for a new trial.
  Q           Quit and print summary.

Calibration
-----------
  User enters hip-to-ankle length (cm) at startup.
  Every frame: px_per_cm = hip_ankle_pixels / hip_ankle_cm  (smoothed).

Distance
--------
  Both hands (left and right index fingers) are tracked every frame.
  The hand that achieves the greater cumulative displacement from START
  is used as the result.
  Displacement is the true 2D Euclidean distance in the image plane:
      displacement_px = sqrt((x_now - x_start)^2 + (y_now - y_start)^2)
  Signed: positive when moving toward the ankles, negative when pulling back.
  displacement_cm = displacement_px / px_per_cm
  The MAXIMUM displacement recorded during MEASURING is the result.

Usage
-----
  python sit_and_reach.py                    # interactive leg-length prompt
  python sit_and_reach.py --leg-cm 87        # skip prompt
  python sit_and_reach.py --source video.mp4
  python sit_and_reach.py --model pose_landmarker_heavy.task
  python sit_and_reach.py --debug            # print raw finger coords each frame

Model file
----------
  Download a .task model from:
  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
  e.g. pose_landmarker_full.task  (recommended)
       pose_landmarker_lite.task  (faster)
       pose_landmarker_heavy.task (most accurate)
  Place the file in the same directory or pass --model <path>.
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import time
from enum import Enum, auto


# ── MediaPipe (Tasks API) ─────────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions          = mp.tasks.BaseOptions
PoseLandmarker       = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode    = mp_vision.RunningMode

# Landmark indices (PoseLandmark enum values from the 33-point BlazePose topology)
class _LM:
    LEFT_HIP    = 23
    RIGHT_HIP   = 24
    LEFT_ANKLE  = 27
    RIGHT_ANKLE = 28
    LEFT_INDEX  = 19
    RIGHT_INDEX = 20

class L_HIP:    value = _LM.LEFT_HIP
class R_HIP:    value = _LM.RIGHT_HIP
class L_ANKLE:  value = _LM.LEFT_ANKLE
class R_ANKLE:  value = _LM.RIGHT_ANKLE
class L_INDEX:  value = _LM.LEFT_INDEX
class R_INDEX:  value = _LM.RIGHT_INDEX

# Skeleton connections (subset used for drawing – mirrors POSE_CONNECTIONS)
_POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(25,27),(27,29),(27,31),           # left leg
    (24,26),(26,28),(28,30),(28,32),           # right leg
]

# ── Colours (BGR) ────────────────────────────────────────────────────────────────
C_GREEN  = (0,   210,  80)
C_RED    = (30,   50, 220)
C_YELLOW = (0,   210, 220)
C_WHITE  = (240, 240, 240)
C_ORANGE = (0,   165, 255)
C_BLUE   = (255, 140,  40)
C_PURPLE = (200,  60, 255)
C_GRAY   = (130, 130, 130)
C_DARK   = (12,   12,  12)


# ── State machine ────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE      = auto()
    MEASURING = auto()
    LOCKED    = auto()


_STATE_LABEL = {
    State.IDLE:      ("IDLE  –  press SPACE to set start position", C_YELLOW),
    State.MEASURING: ("MEASURING  –  reach forward, then SPACE",    C_GREEN),
    State.LOCKED:    ("LOCKED  –  press R for a new trial",         C_ORANGE),
}


# ── Geometry helpers ─────────────────────────────────────────────────────────────

def lm_xy(lms, lid, w, h):
    """Normalised landmark → (float x, float y) in pixels."""
    p = lms[lid.value]
    return p.x * w, p.y * h

def lm_px(lms, lid, w, h):
    x, y = lm_xy(lms, lid, w, h)
    return int(x), int(y)

def pdist(p1, p2):
    return float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))

def vis(lms, *ids, thr=0.45):
    return all(lms[i.value].visibility >= thr for i in ids)


# ── Smoothing ────────────────────────────────────────────────────────────────────

class RollingMean:
    """One-dimensional rolling mean."""
    def __init__(self, n=8):
        self.n   = n
        self.buf = []

    def push(self, v: float) -> float:
        self.buf.append(float(v))
        if len(self.buf) > self.n:
            self.buf.pop(0)
        return float(np.mean(self.buf))

    def reset(self):
        self.buf.clear()

    @property
    def ready(self):
        return len(self.buf) > 0

    @property
    def value(self):
        return float(np.mean(self.buf)) if self.buf else 0.0


# ── Calibration ──────────────────────────────────────────────────────────────────

def hip_ankle_px(lms, w, h):
    """
    Return average hip→ankle pixel length across visible sides.
    Returns None if neither side is visible.
    """
    segs = []
    for hip_id, ank_id in [(L_HIP, L_ANKLE), (R_HIP, R_ANKLE)]:
        if vis(lms, hip_id, ank_id):
            segs.append(pdist(lm_xy(lms, hip_id, w, h),
                               lm_xy(lms, ank_id, w, h)))
    return float(np.mean(segs)) if segs else None

def best_calib_segment(lms, w, h):
    """Pixel endpoints of the more-visible hip→ankle side (for drawing)."""
    def score(h_id, a_id):
        return lms[h_id.value].visibility + lms[a_id.value].visibility
    pairs = [(L_HIP, L_ANKLE), (R_HIP, R_ANKLE)]
    best  = max(pairs, key=lambda p: score(*p))
    if vis(lms, *best):
        return lm_px(lms, best[0], w, h), lm_px(lms, best[1], w, h)
    return None, None


# ── Finger detection ─────────────────────────────────────────────────────────────

def reach_direction(lms, w, h):
    """
    Return the unit vector pointing from hips toward ankles.
    This is the positive reach direction regardless of camera orientation.
    Falls back to (1, 0) if landmarks unavailable.
    """
    if not vis(lms, L_HIP, R_HIP, L_ANKLE, R_ANKLE):
        return np.array([1.0, 0.0])
    lh = np.array(lm_xy(lms, L_HIP,   w, h))
    rh = np.array(lm_xy(lms, R_HIP,   w, h))
    la = np.array(lm_xy(lms, L_ANKLE, w, h))
    ra = np.array(lm_xy(lms, R_ANKLE, w, h))
    vec = ((la + ra) / 2.0) - ((lh + rh) / 2.0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])


def fingers_raw(lms, w, h):
    """
    Return raw pixel positions of BOTH index fingers.
    Each entry is (x_float, y_float) or None if not visible.
    Also returns the reach unit-vector.
    """
    left  = None
    right = None
    if vis(lms, L_INDEX, thr=0.40):
        left  = lm_xy(lms, L_INDEX, w, h)
    if vis(lms, R_INDEX, thr=0.40):
        right = lm_xy(lms, R_INDEX, w, h)
    rvec = reach_direction(lms, w, h)
    return left, right, rvec


# ── HUD ──────────────────────────────────────────────────────────────────────────

def draw_hud(frame, state,
             leg_cm, px_per_cm, calibrated,
             left_pt, right_pt,       # (int x, int y) or None – current fingers
             start_left, start_right, # locked start points or None
             disp_px, disp_cm,        # current best displacement
             max_disp_cm,
             calib_hip, calib_ank,
             frame_no, debug):

    fh, fw = frame.shape[:2]

    # ── Calibration segment ───────────────────────────────────────────────────
    if calib_hip and calib_ank:
        cv2.line(frame, calib_hip, calib_ank, C_BLUE, 3, cv2.LINE_AA)
        mid = ((calib_hip[0] + calib_ank[0]) // 2,
               (calib_hip[1] + calib_ank[1]) // 2)
        cv2.putText(frame, f"{leg_cm:.0f} cm ref",
                    (mid[0] + 8, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_BLUE, 1, cv2.LINE_AA)

    # ── Both finger dots ──────────────────────────────────────────────────────
    for pt, col, label in [(left_pt,  (80, 200, 255), "L"),
                            (right_pt, (255, 130,  80), "R")]:
        if pt is not None:
            dot_col = col if state != State.LOCKED else C_GRAY
            cv2.circle(frame, pt, 9, dot_col, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 9, C_WHITE,  1, cv2.LINE_AA)
            cv2.putText(frame, label, (pt[0] + 10, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_col, 1, cv2.LINE_AA)

    # ── Start markers + displacement arrows ──────────────────────────────────
    for start_pt, cur_pt in [(start_left, left_pt), (start_right, right_pt)]:
        if start_pt is not None:
            cv2.drawMarker(frame, start_pt, C_YELLOW,
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        if (start_pt is not None and cur_pt is not None
                and state in (State.MEASURING, State.LOCKED)):
            arr_col = C_GREEN if disp_px >= 0 else C_RED
            cv2.arrowedLine(frame, start_pt, cur_pt,
                            arr_col, 2, cv2.LINE_AA, tipLength=0.15)

    # ── Info panel ────────────────────────────────────────────────────────────
    px, py, pw, ph = 12, 12, 340, 225
    ov = frame.copy()
    cv2.rectangle(ov, (px, py), (px + pw, py + ph), C_DARK, -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "SIT & REACH TEST",
                (px + 10, py + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C_ORANGE, 1, cv2.LINE_AA)

    stxt, scol = _STATE_LABEL[state]
    cv2.putText(frame, stxt, (px + 10, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, scol, 1, cv2.LINE_AA)

    ctxt = (f"Cal: {px_per_cm:.1f} px/cm  (ref {leg_cm:.0f} cm)"
            if calibrated else "Calibrating... (show full leg)")
    ccol = (100, 200, 100) if calibrated else C_YELLOW
    cv2.putText(frame, ctxt, (px + 10, py + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, ccol, 1, cv2.LINE_AA)

    if state in (State.MEASURING, State.LOCKED) and (
            start_left is not None or start_right is not None):
        if disp_cm is not None:
            s    = "+" if disp_cm >= 0 else ""
            dtxt = f"Disp : {s}{disp_cm:.1f} cm"
            dcol = C_GREEN if disp_cm >= 0 else C_RED
        else:
            dtxt = f"Disp : {disp_px:.0f} px  (no scale)"
            dcol = C_YELLOW
        cv2.putText(frame, dtxt, (px + 10, py + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, dcol, 2, cv2.LINE_AA)

        mtxt = (f"Max  : +{max_disp_cm:.1f} cm" if max_disp_cm is not None
                else "Max  : --")
        cv2.putText(frame, mtxt, (px + 10, py + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Disp : --", (px + 10, py + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_GRAY, 2, cv2.LINE_AA)

    cv2.putText(frame, "SPACE=start/lock   R=reset   Q=quit",
                (px + 10, py + 207),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1, cv2.LINE_AA)

    # ── Debug overlay (bottom of frame) ───────────────────────────────────────
    if debug:
        lx = f"L({left_pt[0]},{left_pt[1]})"   if left_pt  else "L(--)"
        rx = f"R({right_pt[0]},{right_pt[1]})"  if right_pt else "R(--)"
        dbg = f"f#{frame_no:05d}  {lx}  {rx}  disp={disp_px:.1f}px"
        cv2.putText(frame, dbg, (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_YELLOW, 1, cv2.LINE_AA)


# ── User input ───────────────────────────────────────────────────────────────────

def ask_leg_length():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          SIT & REACH TEST  –  Calibration Setup     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\nMeasure the straight-line distance from your hip joint")
    print("(greater trochanter) to your ankle bone (lateral malleolus)")
    print("while standing upright.\n")
    while True:
        try:
            v = float(input("Enter your hip-to-ankle length in cm: ").strip())
            if 20.0 <= v <= 130.0:
                return v
            print("  ✗  Please enter a value between 20 and 130 cm.")
        except ValueError:
            print("  ✗  Invalid – enter a number, e.g. 87.5")
        except EOFError:
            print("\n[ERROR] No interactive terminal detected and --leg-cm was not provided.")
            print("        Run with: python app.py --leg-cm <value>  (e.g. --leg-cm 87)")
            sys.exit(1)


# ── Displacement helpers ─────────────────────────────────────────────────────────

def signed_disp_2d(start: np.ndarray, current: np.ndarray,
                   rvec: np.ndarray) -> float:
    """
    True 2D displacement of `current` relative to `start`,
    signed by projection onto the reach unit-vector `rvec`.
    Positive = moved toward ankles.
    """
    delta = current - start
    magnitude = float(np.linalg.norm(delta))
    sign = np.sign(float(np.dot(delta, rvec)))
    return sign * magnitude if sign != 0 else magnitude


# ── Main ─────────────────────────────────────────────────────────────────────────

def _stage_sit_reach(session_id, reach_cm, leg_cm, rating=None):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/sit_reach",
            json={
                "session_id": session_id,
                "reach_cm":   reach_cm,
                "leg_cm":     leg_cm,
                "rating":     rating,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] sit_reach staged for review → reach={reach_cm} cm  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sit-and-Reach Test")
    parser.add_argument("--source",  default="0")
    parser.add_argument("--leg-cm",  type=float, default=None,
                        help="Hip-to-ankle length in cm (required when stdin is not a TTY)")
    parser.add_argument("--model",   default="pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file "
                             "(default: pose_landmarker_full.task)")
    parser.add_argument("--smooth",  type=int,   default=6,
                        help="Finger smoothing window in frames (default 6)")
    parser.add_argument("--debug",   action="store_true",
                        help="Print raw finger coords each frame to console "
                             "and show them on the video overlay")
    parser.add_argument("--session-id", type=int, default=None,
                        help="Session ID used to stage the result for review on screen3")
    args = parser.parse_args()

    leg_cm = args.leg_cm if args.leg_cm is not None else ask_leg_length()
    print(f"\n[INFO] Reference leg length : {leg_cm:.1f} cm")
    if args.debug:
        print("[DEBUG] Debug mode ON – raw finger coords will be printed each frame.")

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    print(f"[INFO] Stream  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("[INFO] SPACE → mark start → reach → SPACE → lock   |  R = reset  |  Q = quit\n")

    # ── Per-hand smoothers ────────────────────────────────────────────────────
    # Two independent (x, y) smoothers for left and right index fingers.
    lx_sm = RollingMean(args.smooth)
    ly_sm = RollingMean(args.smooth)
    rx_sm = RollingMean(args.smooth)
    ry_sm = RollingMean(args.smooth)
    cal_sm = RollingMean(20)

    # ── Trial state ───────────────────────────────────────────────────────────
    state = State.IDLE

    # Start positions (float numpy arrays), set at SPACE #1
    start_left  = None   # np.array([x, y])
    start_right = None

    # Reach unit-vector locked at SPACE #1
    rvec_locked = np.array([1.0, 0.0])

    max_disp_px  = None
    max_disp_cm  = None

    # Frozen result (shown in LOCKED state)
    result_disp_px = 0.0
    result_disp_cm = None

    px_per_cm  = None
    frame_no   = 0
    stream_start_time = time.time()
    last_timestamp_ms  = -1

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_segmentation_masks=False,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            frame_no += 1
            fh, fw   = frame.shape[:2]

            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            # Live webcams (esp. on Windows/DirectShow) report CAP_PROP_POS_MSEC
            # unreliably -- it can jitter or go backwards between frames, which
            # violates MediaPipe VIDEO mode's "strictly increasing timestamp"
            # requirement and crashes detect_for_video(). Use a real wall-clock
            # timer instead, with a guard that force-increments it if two frames
            # would otherwise land on (or before) the same millisecond.
            frame_ms = int((time.time() - stream_start_time) * 1000)
            if frame_ms <= last_timestamp_ms:
                frame_ms = last_timestamp_ms + 1
            last_timestamp_ms = frame_ms
            results   = landmarker.detect_for_video(mp_image, frame_ms)

            # Per-frame working variables
            left_pt  = None   # smoothed int pixel tuple for HUD
            right_pt = None
            cur_rvec = np.array([1.0, 0.0])
            calib_h  = calib_a = None
            disp_px  = result_disp_px
            disp_cm  = result_disp_cm

            if results.pose_landmarks:
                lms = results.pose_landmarks[0]

                # ── Calibration ───────────────────────────────────────────────
                seg_px = hip_ankle_px(lms, fw, fh)
                if seg_px is not None and seg_px > 0:
                    px_per_cm = cal_sm.push(seg_px / leg_cm)
                calib_h, calib_a = best_calib_segment(lms, fw, fh)

                # ── Raw finger positions + reach direction ─────────────────────
                raw_left, raw_right, cur_rvec = fingers_raw(lms, fw, fh)

                # Smooth each finger independently
                if raw_left is not None:
                    slx = lx_sm.push(raw_left[0])
                    sly = ly_sm.push(raw_left[1])
                    left_pt = (int(slx), int(sly))
                if raw_right is not None:
                    srx = rx_sm.push(raw_right[0])
                    sry = ry_sm.push(raw_right[1])
                    right_pt = (int(srx), int(sry))

                # ── Debug: print raw coords every frame ────────────────────────
                if args.debug:
                    lstr = (f"L=({raw_left[0]:.1f},{raw_left[1]:.1f})"
                            if raw_left else "L=(--)")
                    rstr = (f"R=({raw_right[0]:.1f},{raw_right[1]:.1f})"
                            if raw_right else "R=(--)")
                    slstr = (f"L_sm=({lx_sm.value:.1f},{ly_sm.value:.1f})"
                             if lx_sm.ready else "L_sm=(--)")
                    srstr = (f"R_sm=({rx_sm.value:.1f},{ry_sm.value:.1f})"
                             if rx_sm.ready else "R_sm=(--)")
                    print(f"[f#{frame_no:05d}] {lstr}  {rstr}  "
                          f"{slstr}  {srstr}  "
                          f"rvec=({cur_rvec[0]:.2f},{cur_rvec[1]:.2f})  "
                          f"ppcm={px_per_cm:.2f}" if px_per_cm else
                          f"[f#{frame_no:05d}] {lstr}  {rstr}  "
                          f"{slstr}  {srstr}  "
                          f"rvec=({cur_rvec[0]:.2f},{cur_rvec[1]:.2f})  "
                          f"ppcm=None")

                # ── Displacement from start ────────────────────────────────────
                if state == State.MEASURING:
                    candidates = []
                    if start_left is not None and lx_sm.ready:
                        cur = np.array([lx_sm.value, ly_sm.value])
                        d   = signed_disp_2d(start_left, cur, rvec_locked)
                        candidates.append(d)
                    if start_right is not None and rx_sm.ready:
                        cur = np.array([rx_sm.value, ry_sm.value])
                        d   = signed_disp_2d(start_right, cur, rvec_locked)
                        candidates.append(d)

                    if candidates:
                        # Use the hand with the greater forward displacement
                        disp_px = max(candidates)
                        disp_cm = (disp_px / px_per_cm) if px_per_cm else None

                        if max_disp_px is None or disp_px > max_disp_px:
                            max_disp_px = disp_px
                            max_disp_cm = disp_cm

                        if args.debug:
                            print(f"           candidates={[f'{c:.1f}' for c in candidates]}  "
                                  f"disp_px={disp_px:.1f}  max={max_disp_px:.1f}")

                # ── Skeleton ──────────────────────────────────────────────────
                for i, j in _POSE_CONNECTIONS:
                    if (lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45):
                        pt1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                        pt2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                        cv2.line(frame, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)
                for idx in range(len(lms)):
                    if lms[idx].visibility >= 0.45:
                        pt = (int(lms[idx].x * fw), int(lms[idx].y * fh))
                        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

            # Build int start points for HUD
            start_left_pt  = (int(start_left[0]),  int(start_left[1]))  if start_left  is not None else None
            start_right_pt = (int(start_right[0]), int(start_right[1])) if start_right is not None else None

            # ── HUD ───────────────────────────────────────────────────────────
            draw_hud(
                frame, state, leg_cm, px_per_cm or 0.0,
                calibrated=(px_per_cm is not None),
                left_pt=left_pt,
                right_pt=right_pt,
                start_left=start_left_pt,
                start_right=start_right_pt,
                disp_px=disp_px,
                disp_cm=disp_cm,
                max_disp_cm=max_disp_cm,
                calib_hip=calib_h,
                calib_ank=calib_a,
                frame_no=frame_no,
                debug=args.debug,
            )

            cv2.imshow("Sit & Reach Test", frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ──────────────────────────────────────────────────
            if key == ord("q"):
                break

            elif key == ord(" "):
                if state == State.IDLE:
                    has_left  = lx_sm.ready
                    has_right = rx_sm.ready
                    if has_left or has_right:
                        start_left  = np.array([lx_sm.value, ly_sm.value]) if has_left  else None
                        start_right = np.array([rx_sm.value, ry_sm.value]) if has_right else None
                        rvec_locked = cur_rvec.copy()
                        max_disp_px = None
                        max_disp_cm = None
                        result_disp_px = 0.0
                        result_disp_cm = None
                        state = State.MEASURING
                        print(f"[INFO] Start locked  "
                              f"L={start_left}  R={start_right}  "
                              f"rvec=({rvec_locked[0]:.2f},{rvec_locked[1]:.2f})")
                    else:
                        print("[WARN] No hands visible yet – move into frame.")

                elif state == State.MEASURING:
                    result_disp_px = max_disp_px if max_disp_px is not None else 0.0
                    result_disp_cm = max_disp_cm
                    state = State.LOCKED
                    if result_disp_cm is not None:
                        s = "+" if result_disp_cm >= 0 else ""
                        print(f"[INFO] Result locked → {s}{result_disp_cm:.1f} cm "
                              f"({result_disp_px:.0f} px)")
                    else:
                        print(f"[INFO] Result locked → {result_disp_px:.0f} px "
                              "(calibration unavailable)")

            elif key == ord("r"):
                state          = State.IDLE
                start_left     = None
                start_right    = None
                max_disp_px    = None
                max_disp_cm    = None
                result_disp_px = 0.0
                result_disp_cm = None
                lx_sm.reset(); ly_sm.reset()
                rx_sm.reset(); ry_sm.reset()
                print("[INFO] Trial reset.")

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    final_cm = result_disp_cm
    final_px = result_disp_px
    print("\n── Sit & Reach Test Result ──────────────────────────────")
    print(f"  Hip-to-ankle reference : {leg_cm:.1f} cm")
    rating = None
    if final_cm is not None:
        s = "+" if final_cm >= 0 else ""
        print(f"  Max displacement       : {s}{final_cm:.1f} cm")
        if   final_cm >= 15: rating = "Excellent"
        elif final_cm >= 5:  rating = "Above average"
        elif final_cm >= 0:  rating = "Average"
        elif final_cm >= -10: rating = "Below average"
        else:                  rating = "Poor – more stretching recommended"
        print(f"  Flexibility rating     : {rating}")
    elif final_px:
        s = "+" if final_px >= 0 else ""
        print(f"  Max displacement       : {s}{final_px:.0f} px  (no calibration)")
    else:
        print("  No measurement was recorded.")
    print("─────────────────────────────────────────────────────────\n")

    if args.session_id is not None and final_cm is not None:
        _stage_sit_reach(args.session_id, round(final_cm, 1), leg_cm, rating)


if __name__ == "__main__":
    main()