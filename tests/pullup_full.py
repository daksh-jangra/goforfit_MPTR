"""
AI-Powered Pull-Up Counter
==========================
Brockport protocol ("Pull-Up"): the participant begins in a straight-arm hang
from a bar with an overhand (pronated) grip, pulls the body up until the chin
is above the bar, then lowers back to the full-hang starting position,
repeating as many times as possible. The body must not swing, the knees must
not bend and the legs must not kick. Score = number of correct pull-ups.

A rep needs BOTH the chin above the bar and THE SHOULDERS ABOVE THE ELBOWS,
then the arms straightening again. The chin is the protocol's own criterion.
The shoulders-over-elbows check is the second opinion: the chin has to be
estimated from the face (there is no chin landmark) and a head tipped back at
the top moves that estimate, whereas shoulder and elbow are real landmarks
sitting far apart — measured over a demonstration video's six pulls the chin
cleared the bar by 0.031-0.069 of the frame height while the shoulders cleared
the elbows by 0.049-0.121, so the elbow check has the wider margin and a
drifting chin estimate cannot award a rep on its own.

Shoulders above the HANDS was measured and rejected as the strict rule: it
fails three of those six pulls by a trained adult, and in the manual's own
up-position photo (figure 5.15b) the shoulders sit 0.009 below the hands.

The manual doesn't say how the test ends. Here it ends when the participant
lets go of the bar (auto-detected), or when the coach taps STOP.

Form cues (swinging, bent knees, kicking) are the coach's call and aren't
detected. What IS enforced is what makes a rep valid: the shoulders clear the
elbows at the top, and the arms re-straighten to a full hang before the next
rep.

Geometry helpers (chin estimate, bar line from the wrists, near-arm choice)
are shared with the modified pull-up in pullup.py.

Compatible with mediapipe >= 0.10.30 using the Tasks API

pip install mediapipe==0.10.35 opencv-python numpy pandas
"""

import cv2
import numpy as np
import pandas as pd
import time
import argparse
from collections import deque, namedtuple

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from pullup import (
    IDX, C_GREEN, C_ORANGE, C_RED, C_WHITE, C_YELLOW,
    calc_angle, lm_px, near_arm, chin_y, bar_y, draw_panel, draw_pose,
)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
# A rep is a full-hang -> chin-over-bar-and-shoulders-over-elbows -> full-hang
# cycle. Unlike the modified pull-up (where the arm never re-straightens
# between reps), this protocol explicitly returns to a straight-arm hang.
# 140 rather than a textbook "straight" 170: the manual's own down-position
# photo measures 142 deg on one arm and 152 on the other, so a stricter number
# would reject the hang the protocol itself illustrates.
ELBOW_EXTEND_ANGLE  = 140    # elbow angle must rise TO/ABOVE -> "full hang"

# Letting go of the bar ends the test. The test is wrists-above-shoulders for
# the whole rep, so "let go" needs a clear margin below the shoulders, not a
# bare comparison: at the top of a correct rep the wrists come level with the
# shoulders (0.008 apart in figure 5.15b), and a bare comparison reads that as
# letting go — dropping the frames that score the rep, and ending the test if
# the participant holds there. Arms hanging at the sides put the wrists about
# two upper-arm lengths down, so half an upper arm separates the two cleanly.
# Measuring in upper arms keeps it right at any camera distance.
ON_BAR_MARGIN_ARM   = 0.5
DROP_SEC            = 0.5
# Frames where the pose is lost (spotter crossing the camera, subject drifting
# out of shot, a patch of glare) never reach the counter at all, so a gap
# longer than this between samples means we stopped looking rather than that
# anything happened to the participant. Same limit as the modified pull-up.
SAMPLE_GAP_SEC      = 0.5

# Which way the participant faces, from the shoulders: MediaPipe numbers them
# by the PERSON's left and right, so facing the camera puts their left shoulder
# on the image's right (positive) and facing away swaps them (negative). The
# face landmarks are no help — MediaPipe reports a nose and a mouth with full
# confidence for the back of a head, and the chin estimate built on them looks
# just as confident, which is exactly the silent failure this catches.
# Measured over a demonstration video: -0.23 to -0.32 filmed from behind and
# +0.23 to +0.30 face-on, while a side-on close-up sat near -0.09 and dipped to
# -0.13. Side views must not trip it — side-on is the clearest angle for
# judging chin against bar — so the threshold sits below that dip, and the
# median runs over half a second so a brief swing can't raise it either.
BACK_VIEW_X         = -0.18
FACING_BUFFER_LEN   = 15     # ~0.5 s: the warning shouldn't flicker

# 3-frame medians — see pullup.py: survives landmark jitter while a brisk rep
# still reaches the angle threshold at ~10 fps.
ANGLE_BUFFER_LEN    = 3
CHIN_BUFFER_LEN     = 3
LIFT_BUFFER_LEN     = 3      # shoulder-above-elbow height


# One frame's worth of pose, as the counter wants it. Every height is 0..1 of
# the frame height, bigger = lower on screen; `arm` is the upper arm's length
# in those same units, which is what makes the on-bar margin scale with the
# person and the camera distance. `chin` is None when the face isn't visible.
# `facing` is negative when the camera is behind the participant.
Sample = namedtuple("Sample", "angle chin bar shoulder elbow arm facing")


# ──────────────────────────────────────────────
# REP STATE MACHINE
# ──────────────────────────────────────────────

class FullPullUpCounter:
    """Counts pull-ups from a stream of Samples. Shared by the desktop script
    and the browser engine so both score a rep identically.

    Phases: "start" (waiting for the first straight-arm hang) -> "extended"
    (hanging, pull up) -> "up" (chin over the bar AND shoulders over the
    elbows, lower to a full hang) -> ... Starting in "start" rather than
    "extended" means a participant who jumps up to the bar already pulled up
    doesn't get a free rep — every rep begins from a full hang.

    `stopped` goes True once the participant drops off the bar.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.phase = "start"
        self.angle = 0.0
        self.gap = None          # how far the chin still is below the bar (+ = below),
                                 # None when the face isn't visible; shown to the
                                 # coach, scores nothing
        self.lift = 0.0          # how far the shoulders are above the elbows (+ = above)
        self.chin_ok = False     # chin is over the bar right now
        self.reached = False     # both conditions met in the rep in progress
        self.stopped = False     # ended by dropping off the bar
        self.angle_buf = deque(maxlen=ANGLE_BUFFER_LEN)
        self.chin_buf = deque(maxlen=CHIN_BUFFER_LEN)
        self.lift_buf = deque(maxlen=LIFT_BUFFER_LEN)
        self.facing_buf = deque(maxlen=FACING_BUFFER_LEN)
        self.back_view = False   # camera is behind the participant
        self._hung = False       # seen hanging since counting began
        self._off_since = None   # when the wrists went below the shoulders
        self._last_seen = None   # when a frame last reached the counter

    def _void_rep(self):
        """Abandon the rep in progress and require a fresh, observed full hang
        before the next one can count. The smoothing buffers are dropped too —
        they hold frames from before whatever just happened (a tracking
        dropout, or the participant off the bar), and a stale value surviving
        into the next rep is exactly how an unwatched rep gets credited."""
        self.phase = "start"
        self.reached = False
        self.angle_buf.clear()
        self.chin_buf.clear()
        self.lift_buf.clear()

    def update(self, s, now=None, counting=True):
        """Feed one Sample. Returns True iff this frame completed a rep."""
        now = time.time() if now is None else now

        self.angle_buf.append(s.angle)
        self.angle = float(np.median(self.angle_buf))
        if s.chin is None:
            # Face out of view: nothing to report, and stale frames would
            # report a gap from wherever the head was last seen. No rep can be
            # credited while this lasts — the chin is half the rule — but the
            # count is kept and the test runs on.
            self.chin_buf.clear()
            self.gap = None
        else:
            self.chin_buf.append(s.chin)
            self.gap = float(np.median(self.chin_buf)) - s.bar
        # Positive once the shoulders are higher on screen than the elbows.
        # Medianed like the rest: a rep is awarded off this number, so one
        # glitched landmark frame must not be able to hand out a rep.
        self.lift_buf.append(s.elbow - s.shoulder)
        self.lift = float(np.median(self.lift_buf))
        # Warning only: filmed from behind the chin reading is a guess about a
        # head we can't see, but blocking the count on a view test that misfires
        # would cost a real rep, and the coach can just move the phone.
        self.facing_buf.append(s.facing)
        self.back_view = float(np.median(self.facing_buf)) < BACK_VIEW_X

        if not counting or self.stopped:
            self.reached = False
            return False

        # ── did we stop looking? ───────────────────────────────────────────
        # A gap in the samples is time spent blind, not time observed.
        blind = (self._last_seen is not None
                 and now - self._last_seen > SAMPLE_GAP_SEC)
        self._last_seen = now
        if blind:
            # The drop-off clock only counts time we actually watched the
            # wrists below the shoulders, so an unobserved stretch must not
            # end the test.
            self._off_since = None
            # A rep that was mid-flight is void: we didn't see how it ended,
            # and shoulders over elbows earlier in the cycle is not evidence
            # they lowered again rather than dropping off. Scrapping it can
            # cost a real rep, which the coach can add back on the review
            # screen; crediting an unwatched one inflates the score with
            # nothing to show the coach it happened.
            self._void_rep()

        # ── on the bar? ────────────────────────────────────────────────────
        # Hanging — and at the top of a rep — the wrists (bar line) are at or
        # above the shoulders. Only wrists a clear margin BELOW them mean the
        # arms are down and the bar has been let go.
        if s.bar < s.shoulder + ON_BAR_MARGIN_ARM * s.arm:
            self._hung = True
            self._off_since = None
        else:
            # Arms down — nothing to score, and anything in progress is void.
            # Standing upright puts the shoulders above the elbows all by
            # itself, so these frames must not reach the rep machine or sit in
            # its buffers: a stale one would complete a phantom rep on the
            # first frames back on the bar.
            self._void_rep()
            # Before the first hang they're just standing under the bar waiting
            # to start; after it, they've let go.
            if self._hung:
                if self._off_since is None:
                    self._off_since = now
                if now - self._off_since >= DROP_SEC:
                    self.stopped = True
                    self.reached = False
            return False

        # ── rep state machine ──────────────────────────────────────────────
        if self.phase == "start":
            if self.angle >= ELBOW_EXTEND_ANGLE:
                self.phase = "extended"
            return False

        if self.phase == "extended":
            # The pull registers only when BOTH hold on the same frame: the
            # protocol's chin over the bar, and the shoulders over the elbows
            # agreeing with it. An unreadable chin is not a pass.
            self.chin_ok = self.gap is not None and self.gap <= 0
            if self.chin_ok and self.lift > 0:
                self.phase = "up"
                self.reached = True
            return False

        # phase == "up": both conditions were met, so the rep lands when the
        # arms straighten out again.
        if self.angle >= ELBOW_EXTEND_ANGLE:
            self.phase = "extended"
            self.reached = False
            self.count += 1
            return True
        return False


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def draw_ui(frame, count, phase, angle, elapsed, gap, reached, visible, test_started,
            stopped=False, browser_mode=False, lift=None, back_view=False):
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
        if phase == "start":
            arrow, pc = "HANG STRAIGHT-ARMED", C_YELLOW
        elif phase == "extended":
            arrow, pc = "^ PULL UP", C_ORANGE
        else:
            arrow, pc = "v FULL HANG", C_GREEN
        # Both halves of the rule get a line: which one is missing is the
        # first thing a coach asks when a rep doesn't count.
        chin_note = ("  |  chin not visible" if gap is None else
                     f"  |  chin {abs(gap)*100:.1f}% {'below' if gap > 0 else 'above'} bar")
        cv2.putText(frame, f"Elbow Angle: {angle:.1f} deg{chin_note}", (20,h-48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)
        if reached:
            lift_txt, cc = "CHIN OVER BAR + SHOULDERS OVER ELBOWS", C_GREEN
        elif lift is not None:
            lift_txt = ("Shoulders over elbows - chin still short"
                        if lift > 0 else
                        f"Shoulders {abs(lift)*100:.1f}% below elbows")
            cc = pc
        else:
            lift_txt, cc = "", pc
        cv2.putText(frame, f"{lift_txt}  {arrow}".strip(),
                    (20,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cc, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "One whole arm and the bar must be in frame",
                    (20,h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_YELLOW, 2, cv2.LINE_AA)

    # Filmed from behind: the chin reading is then a guess about a head the
    # camera can't see, and nothing else on screen would say so.
    if back_view and visible and not stopped:
        draw_panel(frame, 10, h-135, 620, h-85, colour=(0, 60, 120), alpha=0.8)
        cv2.putText(frame, "CAMERA IS BEHIND - move it to the front or side",
                    (22, h-102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2, cv2.LINE_AA)

    # Ended by dropping off the bar
    if stopped:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "TEST ENDED", (w//2-220, h//2-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_RED, 5, cv2.LINE_AA)
        cv2.putText(frame, f"Off the bar - Final Count: {count}", (w//2-320, h//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_GREEN, 3, cv2.LINE_AA)
        return

    # Waiting for SPACE
    if not test_started:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        ready_col = C_GREEN if visible else C_YELLOW
        ready_txt = "Body detected! " if visible else "Position yourself in frame"
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


def pose_sample(lms, w, h):
    """(visible, Sample) for one pose, or (False, None) when the near arm isn't
    clear enough to score. Angles and lengths are taken in pixels — normalised
    coords would skew them on a non-square frame — and the arm length is
    converted back to frame-height units, which is what the Sample's heights
    are in.

    A frame without a readable face still arrives, carrying chin=None: the arm
    keeps being tracked and the test runs on, but no rep can be credited while
    the chin — half the rule — can't be seen.

    Mirroring needs no special handling: a mirror doesn't turn anyone around,
    and MediaPipe labels the shoulders by the person it infers, so `facing`
    comes out the same on a flipped frame (measured: -0.226 against -0.225).
    An earlier version negated it for the browser's selfie camera, which
    inverted a correct reading."""
    (s_i, e_i, wr_i), near_vis = near_arm(lms)
    if near_vis <= 0.5:
        return False, None
    face_ok = all(lms[IDX[k]].visibility > 0.5 for k in ("NOSE", "MOUTH_L", "MOUTH_R"))
    shoulder_px, elbow_px = lm_px(lms, s_i, w, h), lm_px(lms, e_i, w, h)
    angle = calc_angle(shoulder_px, elbow_px, lm_px(lms, wr_i, w, h))
    arm = float(np.linalg.norm(shoulder_px - elbow_px)) / h
    facing = lms[IDX["L_SHOULDER"]].x - lms[IDX["R_SHOULDER"]].x
    return True, Sample(angle=angle, chin=chin_y(lms) if face_ok else None,
                        bar=bar_y(lms, wr_i),
                        shoulder=lms[s_i].y, elbow=lms[e_i].y, arm=arm,
                        facing=facing)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def _autosave_full_pull_up(session_id, reps, duration_sec):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/full_pull_up",
            json={"session_id": session_id, "reps": reps, "duration_sec": duration_sec},
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] full_pull_up staged for review -> reps={reps}  session={session_id}")
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

    counter      = FullPullUpCounter()
    start_time   = None
    test_started = False
    rep_log      = []
    frame_idx    = 0
    announced    = False

    WIN = "Pull-Up Counter - GoForFit NEP FMS"
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
            body_visible, sample = pose_sample(lms, w, h)
            if body_visible:
                scored = counter.update(sample, counting=test_started)
                if scored:
                    rep_log.append({
                        "rep"         : counter.count,
                        "elbow_deg"   : round(counter.angle, 2),
                        "lift_pct"    : round(counter.lift * 100, 2),
                        "chin_gap_pct": (round(counter.gap * 100, 2)
                                         if counter.gap is not None else None),
                        "timestamp_s" : round(elapsed, 3),
                    })
                if counter.stopped and not announced:
                    announced = True
                    print(f"Dropped off the bar - test ended. Count: {counter.count}")

            draw_pose(frame, lms, w, h)

        draw_ui(frame, counter.count, counter.phase, counter.angle, elapsed,
                counter.gap, counter.reached, body_visible, test_started,
                stopped=counter.stopped, lift=counter.lift,
                back_view=counter.back_view)
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
            test_started = False; rep_log = []; announced = False

    final_elapsed = (time.time() - start_time) if (test_started and start_time) else 0.0
    if _session_id is not None and counter.count > 0:
        _autosave_full_pull_up(_session_id, counter.count, round(final_elapsed, 1))

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if rep_log:
        df = pd.DataFrame(rep_log)
        print("\n-- Session Report ------------------------------------")
        print(df.to_string(index=False))
        print(f"\nTotal Reps : {counter.count}")
        df.to_csv("full_pull_up_session.csv", index=False)
        print("Log saved -> full_pull_up_session.csv")
    else:
        print("No reps recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int,  default=0)
    parser.add_argument("--video",  type=str,  default=None)
    args = parser.parse_args()
    run(args.video if args.video else args.camera)
