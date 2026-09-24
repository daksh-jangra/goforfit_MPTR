"""
Trunk Lift Test – Chin-to-Floor Height Measurement
===================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.86 / FitnessGram)
--------------------------------------------------------------------
  The participant lies prone on a mat, toes pointed, hands under the thighs,
  eyes on a marker placed on the mat. They lift the upper body SLOWLY to a
  maximum of 12 in (30 cm) and HOLD while the tester measures from the floor
  to the chin. Two trials; the better one is recorded; anything above 12 in
  is recorded as 12 in.

  This script measures ONLY the chin height at each held peak. Everything the
  protocol assigns to the tester stays with the tester: no ballistic bouncing,
  hands under the thighs, toes pointed, head aligned with the spine, and the
  controlled return. The app does not judge those.

Measurement workflow
--------------------
  CALIBRATE   Click the two ends of the measuring stick lying on the mat.
              That single line gives BOTH the scale (px per cm) and the floor
              (the stick rests on the mat), so chin height is measured to the
              floor itself rather than to wherever the chin happens to rest.
  READY       Participant lies flat, chin down.
  RISING      Chin has passed MIN_TRIAL_CM; the peak is tracked.
  HOLD        Height has stayed inside HOLD_BAND_CM for HOLD_S seconds →
              the trial is captured automatically (SPACE also captures).
  CAPTURED    Waits for the chin to return to the mat, then re-arms.
  DONE        Both trials captured. Result = the better one, capped at 12 in.

Chin estimate
-------------
  MediaPipe Pose has no chin landmark. The chin (menton) is extrapolated from
  the eye→mouth axis of the face, which is stable in the side-on view this
  test is filmed from:

      chin = mouth_mid + CHIN_K_EYE * (mouth_mid - eye_mid)

  with a nose→mouth fallback when the eyes are not clearly visible. Both
  constants are facial proportions, so they do not depend on body build — but
  they are approximations, worth ±1–2 cm. The coach corrects the number on
  the review screen before it is saved.

Usage
-----
  python trunk_lift.py                       # 30 cm stick, click both ends
  python trunk_lift.py --stick-cm 45         # any known-length object
  python trunk_lift.py --source video.mp4
  python trunk_lift.py --debug               # per-frame numbers on console

  Mouse : click the two ends of the measuring stick (left-to-right or not,
          either order works).
  Keys  : SPACE = capture the current height as a trial
          C     = clear the calibration and click again
          U     = undo the last trial
          R     = reset the whole test
          Q     = quit and print the summary

Camera placement
----------------
  Phone/webcam on the floor, side-on to the participant, 1.5–2 m away, with
  the whole mat in frame and the measuring stick lying in front of the chin
  (where the protocol already puts it) so the stick and the chin are at the
  same distance from the lens.

Model file
----------
  Download a .task model from:
  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
  e.g. pose_landmarker_full.task  (recommended)
  Place the file in the same directory or pass --model <path>.
"""

import argparse
import sys
import time
from collections import deque
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python           # noqa: F401  (parity with snr.py)
from mediapipe.tasks.python import vision as mp_vision

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode


# ── Protocol constants ───────────────────────────────────────────────────────────
CM_PER_IN   = 2.54
MAX_CM      = 12.0 * CM_PER_IN   # 30.48 cm — scores above this are recorded AS this
DEFAULT_TRIALS   = 2             # "Allow two trials and record the better score"
DEFAULT_STICK_CM = 30.0          # the measuring stick the protocol already requires

# ── Detection tuning ─────────────────────────────────────────────────────────────
REST_CM       = 2.5   # chin at or below this counts as "down on the mat"
MIN_TRIAL_CM  = 3.0   # a lift below this is not a trial, it's the chin twitching
HOLD_S        = 1.0   # the hold that stands in for "held long enough to measure"
HOLD_BAND_CM  = 0.7   # total spread allowed inside the hold window
SMOOTH_FRAMES = 6     # chin position smoothing, same window snr.py uses

# Facial proportions used to extrapolate the chin. See the module docstring.
CHIN_K_EYE  = 0.70    # chin sits ~0.70 eye→mouth spans below the mouth
CHIN_K_NOSE = 1.60    # fallback: ~1.60 nose→mouth spans below the mouth
FACE_MIN_VIS = 0.40

# Landmark indices (33-point BlazePose topology)
IDX_NOSE      = 0
IDX_LEFT_EYE  = 2
IDX_RIGHT_EYE = 5
IDX_MOUTH_L   = 9
IDX_MOUTH_R   = 10

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (27, 29), (27, 31),              # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),              # right leg
]

# ── Colours (BGR) ────────────────────────────────────────────────────────────────
C_GREEN  = (0,   210,  80)
C_RED    = (30,   50, 220)
C_YELLOW = (0,   210, 220)
C_WHITE  = (240, 240, 240)
C_ORANGE = (0,   165, 255)
C_BLUE   = (255, 140,  40)
C_GRAY   = (130, 130, 130)
C_DARK   = (12,   12,  12)


# ── State machine ────────────────────────────────────────────────────────────────
class State(Enum):
    CALIBRATE = auto()
    READY     = auto()
    RISING    = auto()
    HOLD      = auto()
    CAPTURED  = auto()
    DONE      = auto()


_STATE_LABEL = {
    State.CALIBRATE: ("CALIBRATE  -  click both ends of the stick", C_YELLOW),
    State.READY:     ("READY  -  lie flat, chin on the mat",        C_WHITE),
    State.RISING:    ("RISING  -  lift slowly, then hold",          C_GREEN),
    State.HOLD:      ("HOLDING  -  keep still",                     C_GREEN),
    State.CAPTURED:  ("CAPTURED  -  lower down for the next trial", C_ORANGE),
    State.DONE:      ("DONE  -  both trials recorded",              C_ORANGE),
}


# ── Geometry helpers ─────────────────────────────────────────────────────────────

def lm_xy(lms, idx, w, h):
    """Normalised landmark → (x, y) float pixel array."""
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=float)


def vis(lms, *idxs, thr=FACE_MIN_VIS):
    return all(lms[i].visibility >= thr for i in idxs)


def chin_point(lms, w, h):
    """Estimated chin (menton) position in pixels, or None when the face is
    not visible enough to place it. See the module docstring for the two
    proportions used."""
    if not vis(lms, IDX_MOUTH_L, IDX_MOUTH_R):
        return None
    mouth = (lm_xy(lms, IDX_MOUTH_L, w, h) + lm_xy(lms, IDX_MOUTH_R, w, h)) * 0.5

    if vis(lms, IDX_LEFT_EYE, IDX_RIGHT_EYE):
        eye = (lm_xy(lms, IDX_LEFT_EYE, w, h) + lm_xy(lms, IDX_RIGHT_EYE, w, h)) * 0.5
        axis, k = mouth - eye, CHIN_K_EYE
    elif vis(lms, IDX_NOSE):
        axis, k = mouth - lm_xy(lms, IDX_NOSE, w, h), CHIN_K_NOSE
    else:
        return None

    # A near-zero axis means the face is edge-on or badly detected; the
    # extrapolation would point in an arbitrary direction, so refuse it.
    if float(np.linalg.norm(axis)) < 2.0:
        return None
    return mouth + k * axis


class RollingMean:
    """One-dimensional rolling mean (same helper snr.py uses)."""

    def __init__(self, n=8):
        self.n = n
        self.buf = []

    def push(self, v):
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

class FloorCalibration:
    """The measuring stick, clicked end to end, doing two jobs at once.

    Scale  : its known length over its pixel length → px_per_cm.
    Floor  : the stick lies on the mat, so the line through both clicks IS
             the floor line. Chin height is the perpendicular distance from
             the chin to that line, which keeps working when the camera sits
             slightly rotated.
    """

    def __init__(self, p1, p2, stick_cm):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.stick_cm = float(stick_cm)

        span = self.p2 - self.p1
        length_px = float(np.linalg.norm(span))
        if length_px < 10.0:
            raise ValueError("The two calibration points are too close together")
        if self.stick_cm <= 0:
            raise ValueError("Stick length must be positive")

        self.length_px = length_px
        self.px_per_cm = length_px / self.stick_cm
        self.direction = span / length_px

        # Unit normal to the floor line, pointing toward the top of the image
        # (where the lifted chin is). Picking it by image orientation, rather
        # than by click order, means the coach can click the ends either way.
        normal = np.array([-self.direction[1], self.direction[0]])
        self.normal = normal if normal[1] < 0 else -normal

    def height_cm(self, point):
        """Perpendicular height of `point` above the floor line, in cm.
        Negative below the line (chin pressed into the mat, or a sloppy
        calibration click)."""
        return float(np.dot(np.array(point, dtype=float) - self.p1, self.normal)) / self.px_per_cm

    def foot_of(self, point):
        """The point on the floor line directly below `point` — the other end
        of the measurement line drawn on the HUD."""
        point = np.array(point, dtype=float)
        along = float(np.dot(point - self.p1, self.direction))
        return self.p1 + along * self.direction

    def as_dict(self):
        return {
            "p1": [round(float(self.p1[0]), 1), round(float(self.p1[1]), 1)],
            "p2": [round(float(self.p2[0]), 1), round(float(self.p2[1]), 1)],
            "stick_cm": self.stick_cm,
            "px_per_cm": round(self.px_per_cm, 2),
        }


# ── Trial state machine ──────────────────────────────────────────────────────────

class TrunkLiftTracker:
    """Turns a stream of chin heights into trials.

    A trial is captured when the height holds inside HOLD_BAND_CM for HOLD_S
    seconds — the protocol's "held long enough to allow the tester to measure"
    expressed in numbers. The captured value is the MEDIAN of the hold window,
    not the peak, so one bad frame cannot inflate a score. Scores above 12 in
    are recorded as 12 in, exactly as the manual instructs.
    """

    def __init__(self, trials=DEFAULT_TRIALS, hold_s=HOLD_S, band_cm=HOLD_BAND_CM,
                 min_trial_cm=MIN_TRIAL_CM, rest_cm=REST_CM, max_cm=MAX_CM):
        self.trials_required = int(trials)
        self.hold_s = float(hold_s)
        self.band_cm = float(band_cm)
        self.min_trial_cm = float(min_trial_cm)
        self.rest_cm = float(rest_cm)
        self.max_cm = float(max_cm)
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        self.state = State.READY
        self.trials = []          # captured heights in cm, in order
        self.window = deque()     # (timestamp, height_cm) inside the hold window
        self.peak_cm = None       # peak of the current rise (HUD only)
        self.last_height_cm = None
        self.status_msg = ""

    def undo(self):
        """Drop the most recent trial — the coach saw a bad capture."""
        if not self.trials:
            return False
        self.trials.pop()
        self.window.clear()
        self.peak_cm = None
        self.state = State.CAPTURED if self.last_is_up() else State.READY
        self.status_msg = "Last trial removed"
        return True

    def last_is_up(self):
        return self.last_height_cm is not None and self.last_height_cm > self.rest_cm

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, height_cm, now=None):
        """Feed one frame's chin height (or None when the chin or the
        calibration is missing). Returns the trial value if this frame
        completed a capture, else None."""
        now = time.time() if now is None else now
        self.last_height_cm = height_cm

        if height_cm is None:
            # No usable measurement: the hold window would otherwise span the
            # gap and capture a stale value the moment tracking returns.
            self.window.clear()
            return None

        if self.state == State.DONE:
            return None

        if self.state == State.CAPTURED:
            if height_cm <= self.rest_cm:
                self.state = State.READY
                self.peak_cm = None
                self.window.clear()
            return None

        if self.state == State.READY:
            if height_cm < self.min_trial_cm:
                return None
            self.state = State.RISING
            self.peak_cm = height_cm
            self.window.clear()

        # RISING / HOLD ──────────────────────────────────────────────────
        if height_cm <= self.rest_cm:
            # Came back down without ever holding — abandon the attempt.
            self.state = State.READY
            self.peak_cm = None
            self.window.clear()
            return None

        self.peak_cm = height_cm if self.peak_cm is None else max(self.peak_cm, height_cm)
        self.window.append((now, height_cm))
        # Keep one sample from just outside the window, so the span can
        # actually reach hold_s. Trimming everything older than hold_s caps
        # the span just below it, and the capture below never fires.
        while len(self.window) > 2 and (now - self.window[1][0]) >= self.hold_s:
            self.window.popleft()

        if self.is_stable():
            self.state = State.HOLD
            if self.window_span() >= self.hold_s:
                return self._capture(float(np.median([h for _, h in self.window])))
        else:
            self.state = State.RISING
        return None

    def mark(self):
        """Manual capture (SPACE / the MARK button), for when the automatic
        hold does not fire — the coach saw a good hold, so it counts."""
        if self.state == State.DONE:
            return None
        if self.last_height_cm is None:
            self.status_msg = "No chin height yet"
            return None
        if self.last_height_cm < self.min_trial_cm:
            self.status_msg = f"Lift is under {self.min_trial_cm:.0f} cm"
            return None
        value = (float(np.median([h for _, h in self.window]))
                 if self.window else self.last_height_cm)
        return self._capture(value)

    def _capture(self, value_cm):
        value_cm = min(max(value_cm, 0.0), self.max_cm)
        self.trials.append(value_cm)
        self.window.clear()
        self.peak_cm = None
        capped = " (capped at 12 in)" if value_cm >= self.max_cm - 1e-6 else ""
        if len(self.trials) >= self.trials_required:
            self.state = State.DONE
            self.status_msg = f"Trial {len(self.trials)}: {value_cm:.1f} cm{capped} — best {self.best_cm:.1f} cm"
        else:
            self.state = State.CAPTURED
            self.status_msg = f"Trial {len(self.trials)}: {value_cm:.1f} cm{capped}"
        return value_cm

    # ── read-only views ──────────────────────────────────────────────────
    def window_span(self):
        return (self.window[-1][0] - self.window[0][0]) if len(self.window) >= 2 else 0.0

    def is_stable(self):
        if len(self.window) < 2:
            return False
        heights = [h for _, h in self.window]
        return (max(heights) - min(heights)) <= self.band_cm and min(heights) >= self.min_trial_cm

    def hold_progress(self):
        """0..1 — how much of the required hold the current steady stretch
        covers. Drives the HUD bar so the participant can see it fill."""
        if self.state not in (State.RISING, State.HOLD) or not self.is_stable():
            return 0.0
        return min(1.0, self.window_span() / self.hold_s) if self.hold_s > 0 else 1.0

    @property
    def best_cm(self):
        return max(self.trials) if self.trials else None

    @property
    def complete(self):
        return len(self.trials) >= self.trials_required


def to_inches(cm):
    return None if cm is None else cm / CM_PER_IN


# ── HUD ──────────────────────────────────────────────────────────────────────────

def draw_hud(frame, tracker, calib, chin_pt, height_cm,
             calib_points=None, stick_cm=DEFAULT_STICK_CM, debug=False, frame_no=0):
    """Draws the floor line, the chin, the live measurement and the panel.
    Shared by the desktop loop and the browser engine so both show the same
    thing."""
    fh, fw = frame.shape[:2]

    # ── Calibration clicks not yet complete ──────────────────────────────
    for i, pt in enumerate(calib_points or []):
        p = (int(pt[0]), int(pt[1]))
        cv2.drawMarker(frame, p, C_YELLOW, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{i + 1}", (p[0] + 10, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_YELLOW, 1, cv2.LINE_AA)

    # ── Floor line + the stick that defined it ───────────────────────────
    if calib is not None:
        p1 = calib.p1
        far = calib.direction * max(fw, fh) * 2.0
        a = (p1 - far).astype(int)
        b = (p1 + far).astype(int)
        cv2.line(frame, tuple(a), tuple(b), C_GRAY, 1, cv2.LINE_AA)          # floor
        cv2.line(frame, tuple(calib.p1.astype(int)),
                 tuple(calib.p2.astype(int)), C_BLUE, 3, cv2.LINE_AA)        # stick
        mid = ((calib.p1 + calib.p2) * 0.5).astype(int)
        cv2.putText(frame, f"{stick_cm:.0f} cm ref", (mid[0] + 8, mid[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_BLUE, 1, cv2.LINE_AA)

    # ── Chin dot + the measured drop to the floor ────────────────────────
    if chin_pt is not None:
        cp = (int(chin_pt[0]), int(chin_pt[1]))
        dot = C_GREEN if tracker.state in (State.RISING, State.HOLD) else C_WHITE
        cv2.circle(frame, cp, 8, dot, -1, cv2.LINE_AA)
        cv2.circle(frame, cp, 8, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "chin", (cp[0] + 12, cp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot, 1, cv2.LINE_AA)

        if calib is not None and height_cm is not None:
            foot = calib.foot_of(chin_pt).astype(int)
            cv2.line(frame, cp, tuple(foot), dot, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, tuple(foot), dot, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            label = f"{height_cm:.1f} cm / {to_inches(height_cm):.1f} in"
            cv2.putText(frame, label, (cp[0] + 14, (cp[1] + foot[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, dot, 2, cv2.LINE_AA)

    # ── Panel ────────────────────────────────────────────────────────────
    px, py, pw, ph = 12, 12, 360, 232
    ov = frame.copy()
    cv2.rectangle(ov, (px, py), (px + pw, py + ph), C_DARK, -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "TRUNK LIFT TEST", (px + 10, py + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C_ORANGE, 1, cv2.LINE_AA)

    state = State.CALIBRATE if calib is None else tracker.state
    stxt, scol = _STATE_LABEL[state]
    cv2.putText(frame, stxt, (px + 10, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, scol, 1, cv2.LINE_AA)

    ctxt = (f"Cal: {calib.px_per_cm:.1f} px/cm  (stick {stick_cm:.0f} cm)"
            if calib is not None else "Not calibrated - click both stick ends")
    cv2.putText(frame, ctxt, (px + 10, py + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (100, 200, 100) if calib is not None else C_YELLOW, 1, cv2.LINE_AA)

    if height_cm is not None:
        htxt = f"Chin : {height_cm:.1f} cm  ({to_inches(height_cm):.1f} in)"
        hcol = C_GREEN if height_cm >= MIN_TRIAL_CM else C_GRAY
    else:
        htxt, hcol = "Chin : --", C_GRAY
    cv2.putText(frame, htxt, (px + 10, py + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, hcol, 2, cv2.LINE_AA)

    # Hold progress bar — fills over HOLD_S while the height stays steady.
    bx, by, bw, bh = px + 10, py + 122, pw - 20, 8
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_GRAY, 1, cv2.LINE_AA)
    prog = tracker.hold_progress()
    if prog > 0:
        cv2.rectangle(frame, (bx + 1, by + 1),
                      (bx + 1 + int((bw - 2) * prog), by + bh - 1),
                      C_GREEN, -1, cv2.LINE_AA)

    for i in range(tracker.trials_required):
        val = tracker.trials[i] if i < len(tracker.trials) else None
        txt = (f"Trial {i + 1} : {val:.1f} cm  ({to_inches(val):.1f} in)"
               if val is not None else f"Trial {i + 1} : --")
        cv2.putText(frame, txt, (px + 10, py + 152 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    C_WHITE if val is not None else C_GRAY, 1, cv2.LINE_AA)

    best = tracker.best_cm
    btxt = (f"BEST : {best:.1f} cm  ({to_inches(best):.1f} in)"
            if best is not None else "BEST : --")
    cv2.putText(frame, btxt, (px + 10, py + 152 + tracker.trials_required * 22 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                C_ORANGE if best is not None else C_GRAY, 1, cv2.LINE_AA)

    if tracker.status_msg:
        cv2.putText(frame, tracker.status_msg[:52], (px + 10, py + ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_YELLOW, 1, cv2.LINE_AA)

    if debug:
        h = "--" if height_cm is None else f"{height_cm:.2f}"
        dbg = (f"f#{frame_no:05d}  h={h}cm  span={tracker.window_span():.2f}s  "
               f"stable={tracker.is_stable()}  state={tracker.state.name}")
        cv2.putText(frame, dbg, (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_YELLOW, 1, cv2.LINE_AA)


# ── Result staging ───────────────────────────────────────────────────────────────

def _stage_trunk_lift(session_id, lift_cm, trials, stick_cm, rating=None):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/trunk_lift",
            json={
                "session_id": session_id,
                "lift_cm":    lift_cm,
                "trial1_cm":  trials[0] if len(trials) > 0 else None,
                "trial2_cm":  trials[1] if len(trials) > 1 else None,
                "stick_cm":   stick_cm,
                "rating":     rating,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] trunk_lift staged for review → {lift_cm} cm  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trunk Lift Test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--stick-cm", type=float, default=DEFAULT_STICK_CM,
                        help="Length of the measuring stick you click on (default 30)")
    parser.add_argument("--model", default="pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file")
    parser.add_argument("--smooth", type=int, default=SMOOTH_FRAMES,
                        help=f"Chin smoothing window in frames (default {SMOOTH_FRAMES})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="Trials per participant (protocol: 2)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame numbers and show them on the overlay")
    parser.add_argument("--session-id", type=int, default=None,
                        help="Session ID used to stage the result for review on screen3")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    print(f"\n[INFO] Stream  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] Click the two ends of the {args.stick_cm:.0f} cm measuring stick "
          "lying on the mat.")
    print("[INFO] SPACE = capture  |  C = recalibrate  |  U = undo  |  R = reset  |  Q = quit\n")

    clicks = []
    calib = [None]  # boxed so the mouse callback can rebind it

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or calib[0] is not None:
            return
        clicks.append((float(x), float(y)))
        if len(clicks) == 2:
            try:
                calib[0] = FloorCalibration(clicks[0], clicks[1], args.stick_cm)
                print(f"[INFO] Calibrated: {calib[0].px_per_cm:.2f} px/cm")
            except ValueError as exc:
                print(f"[WARN] {exc} — click both ends again.")
                clicks.clear()

    win = "Trunk Lift Test"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    tracker = TrunkLiftTracker(trials=args.trials)
    cx_sm, cy_sm = RollingMean(args.smooth), RollingMean(args.smooth)
    frame_no = 0
    stream_start = time.time()
    last_ts = -1

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
            fh, fw = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Webcams report CAP_PROP_POS_MSEC unreliably, and VIDEO mode needs
            # strictly increasing timestamps — same wall-clock guard snr.py uses.
            ts = int((time.time() - stream_start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            results = landmarker.detect_for_video(mp_image, ts)

            chin_pt = None
            height_cm = None

            if results.pose_landmarks:
                lms = results.pose_landmarks[0]
                raw_chin = chin_point(lms, fw, fh)
                if raw_chin is not None:
                    chin_pt = np.array([cx_sm.push(raw_chin[0]), cy_sm.push(raw_chin[1])])
                    if calib[0] is not None:
                        height_cm = calib[0].height_cm(chin_pt)
                else:
                    cx_sm.reset()
                    cy_sm.reset()

                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        p1 = (int(lms[i].x * fw), int(lms[i].y * fh))
                        p2 = (int(lms[j].x * fw), int(lms[j].y * fh))
                        cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)

            captured = tracker.update(height_cm)
            if captured is not None:
                print(f"[INFO] Trial {len(tracker.trials)} captured → {captured:.1f} cm "
                      f"({to_inches(captured):.1f} in)")

            if args.debug:
                h = "--" if height_cm is None else f"{height_cm:.2f}"
                print(f"[f#{frame_no:05d}] h={h}cm  state={tracker.state.name}  "
                      f"span={tracker.window_span():.2f}s  stable={tracker.is_stable()}")

            draw_hud(frame, tracker, calib[0], chin_pt, height_cm,
                     calib_points=(clicks if calib[0] is None else None),
                     stick_cm=args.stick_cm, debug=args.debug, frame_no=frame_no)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if calib[0] is None:
                    print("[WARN] Calibrate first — click both ends of the stick.")
                elif tracker.complete:
                    # mark() returns None here without touching status_msg,
                    # which still holds the last capture's success line.
                    print(f"[WARN] All {tracker.trials_required} trials recorded "
                          "— press U to redo one, or R to reset.")
                else:
                    v = tracker.mark()
                    if v is not None:
                        print(f"[INFO] Trial {len(tracker.trials)} marked → {v:.1f} cm")
                    else:
                        print(f"[WARN] {tracker.status_msg}")
            elif key == ord("c"):
                if tracker.trials:
                    # Fresh clicks land on slightly different pixels, so the
                    # new scale differs from the one the recorded trials were
                    # measured against. Clear them deliberately first.
                    print("[WARN] Trials already recorded on this scale "
                          "— press R to reset before recalibrating.")
                else:
                    calib[0] = None
                    clicks.clear()
                    print("[INFO] Calibration cleared — click both stick ends again.")
            elif key == ord("u"):
                if tracker.undo():
                    print("[INFO] Last trial removed.")
            elif key == ord("r"):
                tracker.reset()
                cx_sm.reset()
                cy_sm.reset()
                print("[INFO] Test reset.")

    cap.release()
    cv2.destroyAllWindows()

    best = tracker.best_cm
    print("\n── Trunk Lift Result ────────────────────────────────────")
    for i, t in enumerate(tracker.trials, 1):
        print(f"  Trial {i}                 : {t:.1f} cm  ({to_inches(t):.1f} in)")
    if best is not None:
        # The manual records to the nearest inch or centimetre.
        print(f"  Best (recorded)        : {round(best)} cm  ({to_inches(best):.1f} in)")
        if best >= MAX_CM - 1e-6:
            print("  Note                   : capped at 12 in (30.5 cm) per protocol")
    else:
        print("  No trial was recorded.")
    print("─────────────────────────────────────────────────────────\n")

    if args.session_id is not None and best is not None:
        _stage_trunk_lift(args.session_id, round(best),
                          [round(t, 1) for t in tracker.trials], args.stick_cm)


if __name__ == "__main__":
    main()
