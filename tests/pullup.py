"""
AI-Powered Modified Pull-Up Counter
====================================
FMS protocol ("SR 10 Upper Body Modified Pull-ups"): the bar is set at a
height just reachable by the participant lying on their back. They grip it
with an overgrasp and, on "GO", raise the body by flexing the arms until
THE CHIN REACHES THE LEVEL OF THE BAR, then lower back to the start without
the shoulders touching the ground, repeating as many times as possible.
Knees stay straight. The test ends when the participant pauses for two or
more seconds. Score = number of correctly completed pull-ups.

Note this is NOT the Brockport variant, which uses an elastic band 7-8 in
below the bar as the chin target — there is no band here, the bar itself is
the target, which is also what makes the rep checkable from pose alone
(the hands are ON the bar, so the wrists mark its height).

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

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
# A rep is an elbow flex-and-extend cycle in which the chin reached the bar.
# The bar's height doesn't have to be guessed: the participant is gripping it,
# so the wrists ARE the bar line. That makes the protocol's own criterion
# ("chin pulled up to the level of horizontal bar") directly checkable, and it
# holds at any camera distance or framing since both points come from the
# same pose.
# Measured off the FMS demo footage (three reps, pose landmarks): the elbow
# sweeps to 12-29 deg at the top and only back to 126-139 deg at the bottom.
# It does NOT straighten out again between reps — the protocol has them lower
# "with shoulders not touching the ground", so the arm keeps some bend and the
# obvious-looking 150+ "arm is straight" return threshold is never reached.
ELBOW_FLEX_ANGLE    = 75     # elbow angle must drop TO/BELOW -> "pulled up"
ELBOW_EXTEND_ANGLE  = 112    # elbow angle must rise TO/ABOVE -> "returned"
CHIN_MARGIN_FRAC    = 0.01   # chin may finish this far below the bar and still count
                             # (absorbs landmark noise in the chin estimate)
# The chin isn't a pose landmark. It sits roughly this many nose->mouth spans
# below the mouth — a facial proportion, so it scales with the person and with
# how close they are to the camera.
CHIN_BELOW_MOUTH_K  = 1.3

# "The test will stop when the participant pauses for two or more seconds."
PAUSE_SEC           = 2.0
PAUSE_ANGLE_TOL     = 10.0   # degrees of elbow movement within the window
PAUSE_POS_TOL       = 0.012  # chin travel within the window (fraction of frame)

# 3-frame medians, not the 5 curl-ups uses: they still drop single-frame
# landmark outliers, but with half the lag, so a brisk rep still reaches both
# angle thresholds when the processing rate drops to ~10 fps on a weak phone.
ANGLE_BUFFER_LEN    = 3     # rolling median smoothing window (elbow angle)
NOSE_BUFFER_LEN     = 3     # rolling median smoothing window (head position)

# MediaPipe Pose landmark indices (same numbering across all API versions)
IDX = {
    "NOSE": 0,
    "MOUTH_L": 9,     "MOUTH_R": 10,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13,    "R_ELBOW": 14,
    "L_WRIST": 15,    "R_WRIST": 16,
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


def near_arm(lms):
    """(shoulder, elbow, wrist) indices of whichever arm the camera sees best,
    plus that arm's visibility. Filmed from the side the far arm is occluded,
    so requiring both would never pass; filmed head-on either arm works."""
    left = (IDX["L_SHOULDER"], IDX["L_ELBOW"], IDX["L_WRIST"])
    right = (IDX["R_SHOULDER"], IDX["R_ELBOW"], IDX["R_WRIST"])
    l_vis = min(lms[i].visibility for i in left)
    r_vis = min(lms[i].visibility for i in right)
    return (left if l_vis >= r_vis else right), max(l_vis, r_vis)


def chin_y(lms):
    """Estimated chin height, 0..1 of frame height (bigger = lower on screen).

    MediaPipe Pose has no chin landmark, so it's projected down from the face:
    the chin sits about CHIN_BELOW_MOUTH_K nose->mouth spans below the mouth."""
    nose = lms[IDX["NOSE"]].y
    mouth = (lms[IDX["MOUTH_L"]].y + lms[IDX["MOUTH_R"]].y) * 0.5
    return mouth + CHIN_BELOW_MOUTH_K * (mouth - nose)


def bar_y(lms, wrist_idx):
    """Height of the bar, 0..1 of frame height. The participant grips the bar,
    so the wrists mark it — both when they're both visible, otherwise the one
    on the arm the camera sees best."""
    lw, rw = lms[IDX["L_WRIST"]], lms[IDX["R_WRIST"]]
    if lw.visibility > 0.5 and rw.visibility > 0.5:
        return (lw.y + rw.y) * 0.5
    return lms[wrist_idx].y


# ──────────────────────────────────────────────
# REP STATE MACHINE
# ──────────────────────────────────────────────

class PullUpCounter:
    """Counts modified pull-ups from a stream of (elbow angle, chin height,
    bar height) samples. Shared by the desktop script and the browser engine
    so both score a rep identically.

    A rep is one elbow flex-and-extend cycle in which the chin reached the bar
    — the protocol's own criterion. `stopped` goes True once the participant
    holds still for PAUSE_SEC, which is how the test is meant to end.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.phase = "extended"
        self.angle = 0.0
        self.gap = 0.0           # how far the chin still is below the bar (+ = below)
        self.reached = False     # chin has reached the bar during the rep in progress
        self.stopped = False     # ended by the two-second pause rule
        self.idle_sec = 0.0
        self.angle_buf = deque(maxlen=ANGLE_BUFFER_LEN)
        self.chin_buf = deque(maxlen=NOSE_BUFFER_LEN)
        self._ref = None         # last pose the participant actually moved away from
        self._last_move = None   # when that happened
        self._armed = False      # pause rule only applies once they've begun

    def update(self, elbow_angle, chin, bar, now=None, counting=True):
        """Feed one frame. `chin` and `bar` are heights in 0..1 of frame height
        (bigger = lower on screen). Returns True iff this frame completed a
        rep."""
        now = time.time() if now is None else now

        self.angle_buf.append(elbow_angle)
        self.angle = float(np.median(self.angle_buf))
        self.chin_buf.append(chin)
        chin = float(np.median(self.chin_buf))
        self.gap = chin - bar

        if not counting or self.stopped:
            self.reached = False
            return False

        # ── two-second pause rule ──────────────────────────────────────────
        # Movement is measured against the last pose they actually moved away
        # from, not against the previous frame — otherwise a slow-but-steady
        # rep, where each individual frame barely changes, would read as a
        # pause. Armed only after the first pull begins: the participant lies
        # still under the bar waiting for "GO", and that must not end the test.
        if self._ref is None:
            self._ref, self._last_move = (self.angle, chin), now
        if (abs(self.angle - self._ref[0]) > PAUSE_ANGLE_TOL
                or abs(chin - self._ref[1]) > PAUSE_POS_TOL):
            self._ref, self._last_move = (self.angle, chin), now
        self.idle_sec = now - self._last_move
        if self._armed and self.idle_sec >= PAUSE_SEC:
            self.stopped = True
            return False

        # ── rep state machine ──────────────────────────────────────────────
        if self.phase == "extended":
            if self.angle <= ELBOW_FLEX_ANGLE:
                self.phase = "flexed"
                self.reached = False
                self._armed = True
            return False

        # phase == "flexed": did the chin make it up to the bar this rep?
        # `chin` is a median of 3, so the chin has to be at the bar for two
        # frames (~0.1 s) before it counts — a single glitched landmark frame
        # can't award a rep, while a real turnaround at the top dwells there
        # for longer than that.
        if self.gap <= CHIN_MARGIN_FRAC:
            self.reached = True
        if self.angle >= ELBOW_EXTEND_ANGLE:
            self.phase = "extended"
            if self.reached:
                self.count += 1
                return True
        return False


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


def draw_ui(frame, count, phase, angle, elapsed, gap, reached, visible, test_started,
            stopped=False, browser_mode=False):
    h, w = frame.shape[:2]

    # Count
    draw_panel(frame, 10, 14, 200, 110)
    cv2.putText(frame, "COUNT", (22,38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{count:02d}", (22,100), cv2.FONT_HERSHEY_SIMPLEX, 2.4, C_GREEN, 4, cv2.LINE_AA)

    # Elapsed (count-up — this test has no time limit)
    draw_panel(frame, w-195, 14, w-10, 110)
    cv2.putText(frame, "ELAPSED", (w-183,38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
    mm, ss = divmod(int(elapsed), 60)
    cv2.putText(frame, f"{mm:02d}:{ss:02d}", (w-185,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_GREEN, 3, cv2.LINE_AA)

    # Bottom info
    draw_panel(frame, 10, h-75, 470, h-10)
    if visible:
        arrow = "^ PULL UP" if phase == "extended" else "v RETURN"
        pc = C_ORANGE if phase == "extended" else C_GREEN
        cv2.putText(frame, f"Elbow Angle: {angle:.1f} deg", (20,h-48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)
        if reached:
            chin_txt, cc = "CHIN AT BAR", C_GREEN
        else:
            chin_txt, cc = f"Chin {gap*100:.1f}% below bar", C_ORANGE
        cv2.putText(frame, f"{chin_txt}  {arrow}",
                    (20,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cc, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Head, arms and bar must all be in frame",
                    (20,h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_YELLOW, 2, cv2.LINE_AA)

    # Ended by the two-second pause rule
    if stopped:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "TEST ENDED", (w//2-220, h//2-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_RED, 5, cv2.LINE_AA)
        cv2.putText(frame, f"Paused 2s - Final Count: {count}", (w//2-300, h//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_GREEN, 3, cv2.LINE_AA)
        return

    # Waiting for SPACE
    if not test_started:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        ready_col = C_GREEN if visible else C_YELLOW
        ready_txt = "Body detected! " if visible else "Position yourself in frame (side-on)"
        cv2.putText(frame, ready_txt, (w//2 - 300, h//2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, ready_col, 2, cv2.LINE_AA)
        start_txt = ("Tap START to begin" if browser_mode
                     else "Press SPACE to start")
        cv2.putText(frame, start_txt,
                    (w//2 - 300, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_WHITE, 3, cv2.LINE_AA)
        cv2.putText(frame, start_txt,
                    (w//2 - 300, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_GREEN, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# POSE DRAWING (manual — no mp.solutions needed)
# ──────────────────────────────────────────────

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms + shoulders
    (11,23),(12,24),(23,24),                   # torso
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

def _autosave_pull_up(session_id, reps, duration_sec):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly.

    app.py serves on 8080; the older test scripts still post to 5000 and their
    staging has been silently failing. Override with GOFORFIT_URL if the server
    is somewhere else."""
    import os
    base = os.environ.get("GOFORFIT_URL", "http://localhost:8080").rstrip("/")
    try:
        import requests
        requests.post(
            f"{base}/api/stage/pull_up",
            json={"session_id": session_id, "reps": reps, "duration_sec": duration_sec},
            timeout=2,
        )
        print(f"  [DB] pull_up staged for review -> reps={reps}  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


def run(source):
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

    counter      = PullUpCounter()
    start_time   = None
    test_started = False
    rep_log      = []
    frame_idx    = 0

    WIN = "Modified Pull-Up Counter - GoForFit NEP FMS"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        frame = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        frame_idx += 1
        timestamp_ms = int(frame_idx * 1000 / fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        body_visible = False
        elapsed      = (time.time() - start_time) if test_started else 0.0

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]

            (s_i, e_i, wr_i), near_vis = near_arm(lms)
            face_ok = all(lms[IDX[k]].visibility > 0.5
                          for k in ("NOSE", "MOUTH_L", "MOUTH_R"))
            body_visible = near_vis > 0.5 and face_ok

            if body_visible:
                shoulder = lm_px(lms, s_i, w, h)
                elbow    = lm_px(lms, e_i, w, h)
                wrist    = lm_px(lms, wr_i, w, h)

                scored = counter.update(
                    calc_angle(shoulder, elbow, wrist),
                    chin_y(lms),
                    bar_y(lms, wr_i),
                    counting=test_started,
                )
                if scored:
                    rep_log.append({
                        "rep"         : counter.count,
                        "elbow_deg"   : round(counter.angle, 2),
                        "chin_gap_pct": round(counter.gap * 100, 2),
                        "timestamp_s" : round(elapsed, 3),
                    })
                if counter.stopped:
                    print(f"Paused {PAUSE_SEC:.0f}s - test ended. Count: {counter.count}")

            draw_pose(frame, lms, w, h)

        draw_ui(frame, counter.count, counter.phase, counter.angle, elapsed,
                counter.gap, counter.reached, body_visible, test_started,
                stopped=counter.stopped)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and not test_started:
            start_time   = time.time()
            test_started = True
            print("Timer started!")
        if key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, prop)
        if key == ord('r'):
            counter.reset(); start_time = None
            test_started = False; rep_log = []

    final_elapsed = (time.time() - start_time) if (test_started and start_time) else 0.0
    if _session_id is not None and counter.count > 0:
        _autosave_pull_up(_session_id, counter.count, round(final_elapsed, 1))

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if rep_log:
        df = pd.DataFrame(rep_log)
        print("\n-- Session Report ------------------------------------")
        print(df.to_string(index=False))
        print(f"\nTotal Reps : {counter.count}")
        df.to_csv("pull_up_session.csv", index=False)
        print("Log saved -> pull_up_session.csv")
    else:
        print("No reps recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int,  default=0)
    parser.add_argument("--video",  type=str,  default=None)
    args = parser.parse_args()
    run(args.video if args.video else args.camera)
