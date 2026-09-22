"""
Reverse Curl Test – 45-degree elbow flexion with a 2-second hold
=================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.84)
--------------------------------------------------------------------
  Seated in a chair or wheelchair, the participant picks up a 1 lb (0.5 kg)
  dumbbell with the preferred arm. The weight starts resting on the midpoint
  of the ipsilateral thigh (or a knee-height table). The participant flexes
  the elbow and lifts the weight until the elbow is flexed to at least 45
  degrees, HOLDS that position for 2 seconds, then returns the weight
  eccentrically to the thigh. ONE trial. The item is PASSED if one correct
  reverse curl is performed.

  So this test is pass/fail, not a rep count and not a measurement.

What "flexed to at least 45 degrees" means here
--------------------------------------------------------------------
  Read literally it is ambiguous: 45 degrees OF FLEXION from a straight arm
  (interior elbow angle <= 135) or an INTERIOR elbow angle <= 45. The two
  figures in the manual settle it:

    fig 5.20a (start)  hands resting on the thighs, upper arm hanging,
                       forearm along the thigh -> interior elbow ~90-100 deg
    fig 5.20b (up)     dumbbell up by the chin  -> interior elbow ~45-50 deg

  The start position is already ~90 degrees of flexion, so the "<=135"
  reading would be satisfied before the participant moved at all. This
  script therefore uses the INTERIOR elbow angle and requires it to reach
  UP_ANGLE_DEG (45). Change that one constant if a coach disagrees.

What the app scores vs. what the coach judges
--------------------------------------------------------------------
  SCORED (they define whether the curl happened at all):
    * the elbow reaching UP_ANGLE_DEG
    * the 2-second hold at that position
    * the weight returning to the thigh

  NOT SCORED (the tester's job, per the manual):
    * "the elbow extension on the downward movement must be slower than
      gravitational pull" — the descent rate is measured and shown on the
      HUD in deg/s as information, but it never decides the pass. Judging
      "controlled" against a free-fall model would add a silent failure
      mode for something the coach is already watching from arm's length.
    * forearm pronation, fingers wrapped around the weight, seated posture.

The camera decides — nobody presses PASS or FAIL
--------------------------------------------------------------------
  ONE trial, as the manual says, and the verdict is automatic:

    PASS  a complete correct curl (target, 2 s hold, back to the thigh)
    FAIL  a REAL attempt that falls short: the weight clearly left the thigh
          (the elbow bent at least ATTEMPT_MIN_DEG below its rest angle) and
          came back down without reaching the target or holding 2 s

  A smaller movement — shifting in the chair, adjusting the grip — is not an
  attempt and is ignored, so a fidget can never fail a student. Stopping the
  test before any real attempt leaves it NOT COMPLETED, which is not a fail:
  a camera or setup problem must never be recorded against the participant.
  The manual permits practice curls, so they happen BEFORE the test starts.

Which arm
--------------------------------------------------------------------
  The manual says "preferred arm", which the participant chooses. ARM_AUTO
  picks whichever elbow flexes furthest below its own resting angle, then
  LOCKS that choice for the rest of the attempt so a stray gesture from the
  other arm cannot steal the test mid-curl. --arm left/right forces it.

Usage
-----
  python reverse_curl.py                  # auto-pick the working arm
  python reverse_curl.py --arm right
  python reverse_curl.py --source clip.mp4
  python reverse_curl.py --up-angle 50    # loosen the 45 deg target
  python reverse_curl.py --debug

  Keys : R     = reset the whole test
         Q     = quit and print the summary

Camera placement
----------------
  Phone/webcam at roughly seated-shoulder height, SIDE-ON to the working
  arm, 1.5-2 m away, with the whole torso, the upper arm and the thigh in
  frame. A side-on view is what keeps the elbow angle from foreshortening —
  a straight-on front view reads every angle too wide and good curls fail.

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
UP_ANGLE_DEG = 45.0   # interior elbow angle the lift must reach (see docstring)
HOLD_S       = 2.0    # "held in this position for 2 seconds"

# ── Detection tuning ─────────────────────────────────────────────────────────────
# Rest baseline: the weight sitting on the thigh. Captured only while the elbow
# is genuinely still AND open — never at the moment a threshold is crossed, or
# the baseline lands mid-movement and every attempt under-measures.
REST_MIN_DEG    = 70.0   # elbow must be at least this open to count as "at rest"
REST_BAND_DEG   = 8.0    # spread allowed inside the settle window
REST_SETTLE_S   = 0.5    # how long it must sit still before the baseline is taken

MOVE_DELTA_DEG  = 12.0   # flexion below the baseline that starts WATCHING a lift
ATTEMPT_MIN_DEG = 30.0   # flexion below the baseline that makes it a real try.
                         # A lift that came back down short of this was a
                         # fidget and is ignored; one that got this far and
                         # still fell short is the trial, and it FAILS. From a
                         # ~95 deg rest this is ~65 deg, a third of the way up.
RETURN_TOL_DEG  = 15.0   # back within this of the baseline = weight on the thigh
HOLD_RELEASE_DEG = 6.0   # angle may drift this far above UP_ANGLE without
                         # breaking the hold — one jittery frame at 46 deg must
                         # not reset a good 2-second hold
DROPOUT_GRACE_S = 0.35   # tracking may vanish this long mid-hold before the
                         # hold is voided (glare, motion blur, the arm crossing
                         # the torso). Longer than one dropped frame, shorter
                         # than anything a participant could hold through.

SMOOTH_FRAMES   = 3      # median window. 3 survives landmark jitter while still
                         # letting a brisk curl reach the threshold at ~10 fps;
                         # 5 starts missing the turnaround.
MIN_VIS         = 0.50   # landmark visibility floor for the arm joints

ARM_AUTO, ARM_LEFT, ARM_RIGHT = "auto", "left", "right"

# Landmark indices (33-point BlazePose topology)
IDX_L_SHOULDER, IDX_R_SHOULDER = 11, 12
IDX_L_ELBOW,    IDX_R_ELBOW    = 13, 14
IDX_L_WRIST,    IDX_R_WRIST    = 15, 16
IDX_L_HIP,      IDX_R_HIP      = 23, 24
IDX_L_KNEE,     IDX_R_KNEE     = 25, 26

ARM_IDX = {
    ARM_LEFT:  (IDX_L_SHOULDER, IDX_L_ELBOW, IDX_L_WRIST),
    ARM_RIGHT: (IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST),
}

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (24, 26),                                 # thighs
]

# HUD palette (matches parcurlup.py / trunk_lift.py)
C_DARK  = (28, 28, 28)
C_OK    = (80, 220, 120)
C_WARN  = (60, 190, 250)
C_BAD   = (80, 80, 240)
C_TEXT  = (245, 245, 245)


class State(Enum):
    READY   = auto()   # waiting for a still, open elbow to set the baseline
    SET     = auto()   # baseline captured, weight on the thigh, ready to lift
    LIFTING = auto()   # elbow flexing, has not reached UP_ANGLE_DEG yet
    HOLD    = auto()   # at or past UP_ANGLE_DEG, the 2 s timer is running
    LOWERING = auto()  # hold satisfied, waiting for the weight to reach the thigh
    PASSED  = auto()   # a complete correct reverse curl
    FAILED  = auto()   # a real attempt fell short — the one trial is used


# ── small helpers ────────────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    """Angle at B formed by A-B-C vectors (2-D, degrees)."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def vis(lms, *idxs, thr=MIN_VIS):
    return all(lms[i].visibility >= thr for i in idxs)


def elbow_angle(lms, arm, w, h):
    """Interior elbow angle for one arm, or None if the joints aren't clear."""
    idxs = ARM_IDX[arm]
    if not vis(lms, *idxs):
        return None
    s, e, wr = (lm_px(lms, i, w, h) for i in idxs)
    return calc_angle(s, e, wr)


class MedianSmoother:
    """Short median window — kills single-frame landmark spikes without the
    lag a longer mean would add to the turnaround at the top of the curl."""

    def __init__(self, n=SMOOTH_FRAMES):
        self.n = int(n)
        self.buf = deque(maxlen=self.n)

    def push(self, v):
        self.buf.append(float(v))
        return float(np.median(self.buf))

    def reset(self):
        self.buf.clear()

    @property
    def value(self):
        return float(np.median(self.buf)) if self.buf else None


class ReverseCurlTracker:
    """Turns a stream of elbow angles into a pass/fail verdict.

    One trial, as the manual specifies. A PASS needs all three parts of a
    correct reverse curl, in order:

        1. the elbow flexes from its resting angle to UP_ANGLE_DEG or less
        2. it stays there for HOLD_S seconds
        3. the weight comes back down to the thigh

    One trial, decided by the camera. A real attempt that falls short FAILS
    it; a movement too small to be an attempt (see ATTEMPT_MIN_DEG) is
    ignored and the participant simply stays ready. Once PASSED or FAILED
    the verdict is final until reset() — later movement changes nothing.

    Everything is driven through update(), so the whole state machine can be
    exercised with synthetic angle sequences — no camera required.
    """

    def __init__(self, up_angle=UP_ANGLE_DEG, hold_s=HOLD_S,
                 rest_min=REST_MIN_DEG, move_delta=MOVE_DELTA_DEG,
                 return_tol=RETURN_TOL_DEG, settle_s=REST_SETTLE_S,
                 rest_band=REST_BAND_DEG):
        self.up_angle = float(up_angle)
        self.hold_s = float(hold_s)
        self.rest_min = float(rest_min)
        self.move_delta = float(move_delta)
        self.return_tol = float(return_tol)
        self.settle_s = float(settle_s)
        self.rest_band = float(rest_band)
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        self.state = State.READY
        self.baseline_deg = None      # resting elbow angle, weight on the thigh
        self.angle_deg = None         # latest smoothed angle
        self.peak_deg = None          # deepest flexion of the current attempt
        self.hold_elapsed = 0.0       # seconds held at/under the target so far
        self.hold_best_s = 0.0        # longest hold seen in this trial
        self.attempts = []            # one dict per completed attempt
        self.descent_deg_s = None     # measured on the way down, never scored
        self.passed = False
        self.verdict_by = None        # "auto" once decided — no manual verdicts
        self.status_msg = "Rest the weight on the thigh and hold still"
        self._settle = deque()        # (t, angle) while finding the baseline
        self._hold_start = None
        self._lower_start = None
        self._dropout_start = None
        self._now = None

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, angle_deg, now=None):
        """Feed one frame's interior elbow angle, or None when the arm isn't
        tracked. Returns the result dict on the frame the trial passes, else
        None."""
        now = time.time() if now is None else now
        self._now = now

        if self.state in (State.PASSED, State.FAILED):
            return None

        if angle_deg is None:
            return self._on_dropout(now)

        self._dropout_start = None
        self.angle_deg = float(angle_deg)
        angle = self.angle_deg

        if self.state == State.READY:
            self._seek_baseline(angle, now)
            return None

        if self.state == State.SET:
            if angle <= self.baseline_deg - self.move_delta:
                self.state = State.LIFTING
                self.peak_deg = angle
                self.status_msg = "Lifting"
            else:
                # Still at rest: keep the baseline current, so a participant
                # who shifts in the chair between attempts is measured from
                # where the weight actually rests now.
                self._seek_baseline(angle, now, refresh=True)
            return None

        if self.state == State.LIFTING:
            self.peak_deg = min(self.peak_deg, angle)
            if angle <= self.up_angle:
                self.state = State.HOLD
                self._hold_start = now
                self.hold_elapsed = 0.0
                self.status_msg = f"Hold for {self.hold_s:.0f} s"
            elif angle >= self.baseline_deg - self.return_tol:
                return self._end_attempt(f"Came back down at {self.peak_deg:.0f} deg "
                                         f"— target is {self.up_angle:.0f} deg")
            return None

        if self.state == State.HOLD:
            self.peak_deg = min(self.peak_deg, angle)
            if angle > self.up_angle + HOLD_RELEASE_DEG:
                # Hold broken before the 2 s were up.
                self._note_hold(now)
                self._hold_start = None
                self.hold_elapsed = 0.0
                if angle >= self.baseline_deg - self.return_tol:
                    return self._end_attempt(f"Held {self.hold_best_s:.1f} s of "
                                             f"{self.hold_s:.0f} s before lowering")
                else:
                    self.state = State.LIFTING
                    self.status_msg = "Lost the hold — back to the target"
                return None

            self.hold_elapsed = now - self._hold_start
            self.hold_best_s = max(self.hold_best_s, self.hold_elapsed)
            if self.hold_elapsed >= self.hold_s:
                self.state = State.LOWERING
                self._lower_start = now
                self.status_msg = "Hold complete — lower the weight to the thigh"
            return None

        if self.state == State.LOWERING:
            if angle >= self.baseline_deg - self.return_tol:
                return self._pass(now, angle)
            return None

        return None

    # ── internals ────────────────────────────────────────────────────────
    def _on_dropout(self, now):
        """The arm stopped being tracked this frame."""
        if self._dropout_start is None:
            self._dropout_start = now
        gap = now - self._dropout_start

        # The baseline must never be built across a gap — the window would
        # span the dropout and read as "perfectly still".
        self._settle.clear()

        if self.state == State.HOLD and gap > DROPOUT_GRACE_S:
            # Void the hold rather than credit time nobody could see. Same
            # rule the pull-up counter uses for a rep that was mid-flight
            # when tracking dropped.
            self._note_hold(self._dropout_start)
            self._hold_start = None
            self.hold_elapsed = 0.0
            self.state = State.LIFTING
            self.status_msg = "Tracking lost mid-hold — hold again"
        return None

    def _seek_baseline(self, angle, now, refresh=False):
        """Capture (or refresh) the resting elbow angle. Only ever taken from
        a stretch where the elbow is both OPEN and STILL, so the baseline is
        the real rest position and not a frame from mid-lift."""
        if angle < self.rest_min:
            self._settle.clear()
            if not refresh:
                self.status_msg = (f"Open the elbow past {self.rest_min:.0f} deg "
                                   f"— rest the weight on the thigh")
            return
        self._settle.append((now, angle))
        # Keep one sample from just outside the window so the span can
        # actually reach settle_s (trimming to exactly settle_s caps it just
        # below, and the capture never fires).
        while len(self._settle) > 2 and (now - self._settle[1][0]) >= self.settle_s:
            self._settle.popleft()
        if len(self._settle) < 2:
            return
        vals = [a for _, a in self._settle]
        span = self._settle[-1][0] - self._settle[0][0]
        if span >= self.settle_s and (max(vals) - min(vals)) <= self.rest_band:
            self.baseline_deg = float(np.median(vals))
            if not refresh:
                self.state = State.SET
                self.status_msg = (f"Rest angle {self.baseline_deg:.0f} deg "
                                   f"— lift to {self.up_angle:.0f} deg")

    def _note_hold(self, when):
        if self._hold_start is not None:
            self.hold_best_s = max(self.hold_best_s, when - self._hold_start)

    def _end_attempt(self, msg):
        """The weight came back to the thigh without a correct curl. If the
        lift was big enough to be a real try, that was the one trial and it
        FAILS — returns the result. Otherwise it was a fidget: forget it and
        stay ready — returns None."""
        # Reaching the target zone is always a real try, even for someone who
        # rests at a low angle and so has less than ATTEMPT_MIN_DEG to give.
        real_try = (self.peak_deg is not None
                    and (self.peak_deg <= self.baseline_deg - ATTEMPT_MIN_DEG
                         or self.peak_deg <= self.up_angle))
        if not real_try:
            self.peak_deg = None
            self._hold_start = None
            self.hold_elapsed = 0.0
            self.hold_best_s = 0.0
            self.state = State.SET
            self.status_msg = f"Ready — lift to {self.up_angle:.0f} deg"
            return None
        self.attempts.append({
            "peak_angle_deg": round(self.peak_deg, 1),
            "hold_s": round(self.hold_best_s, 2),
            "reached_target": bool(self.peak_deg <= self.up_angle),
            "outcome": "fail",
        })
        self._hold_start = None
        self.hold_elapsed = 0.0
        self.passed = False
        self.verdict_by = "auto"
        self.state = State.FAILED
        self.status_msg = f"FAIL — {msg}"
        return self.result()

    def _pass(self, now, angle):
        if self._lower_start is not None and now > self._lower_start and self.peak_deg is not None:
            # Information only — the manual gives "slower than gravitational
            # pull" to the tester, so this number is shown, never enforced.
            self.descent_deg_s = round((angle - self.peak_deg) / (now - self._lower_start), 1)
        self.attempts.append({
            "peak_angle_deg": round(self.peak_deg, 1) if self.peak_deg is not None else None,
            "hold_s": round(self.hold_best_s, 2),
            "reached_target": True,
            "outcome": "pass",
        })
        self.passed = True
        self.verdict_by = "auto"
        self.state = State.PASSED
        self.status_msg = "PASS — correct reverse curl"
        return self.result()

    # ── read-only views ──────────────────────────────────────────────────
    def hold_progress(self):
        """0..1 — how much of the 2 s hold is banked. Drives the HUD bar so
        the participant can watch it fill."""
        if self.state != State.HOLD or self.hold_s <= 0:
            return 1.0 if self.state in (State.LOWERING, State.PASSED) else 0.0
        return min(1.0, self.hold_elapsed / self.hold_s)

    @property
    def best_peak_deg(self):
        peaks = [a["peak_angle_deg"] for a in self.attempts
                 if a["peak_angle_deg"] is not None]
        if self.peak_deg is not None:
            peaks.append(round(self.peak_deg, 1))
        return min(peaks) if peaks else None

    @property
    def resolved(self):
        return self.state in (State.PASSED, State.FAILED)

    def result(self):
        return {
            "passed": bool(self.passed),
            "peak_angle_deg": self.best_peak_deg,
            "hold_s": round(self.hold_best_s, 2),
            "target_angle_deg": self.up_angle,
            "required_hold_s": self.hold_s,
            "descent_deg_s": self.descent_deg_s,
            "attempts": len(self.attempts),
            "verdict_by": self.verdict_by,
        }


class ArmSelector:
    """Decides which arm is doing the test.

    The manual lets the participant use their "preferred arm", so the app has
    to work it out. Each arm gets a running estimate of its own resting angle;
    the working arm is whichever has flexed furthest BELOW its own rest. A
    hysteresis margin stops the choice flickering while both arms sit still
    (they often rest at different angles, and a selector that flipped between
    them would feed the tracker a jumping baseline that never settles).

    Once a real lift starts the choice LOCKS, so the other hand moving — to
    steady the chair, say — cannot steal the test mid-curl.
    """

    SWITCH_MARGIN_DEG = 8.0
    REST_WINDOW = 15          # frames kept for each arm's rest estimate

    def __init__(self, forced=ARM_AUTO, lock_delta=MOVE_DELTA_DEG):
        self.forced = forced if forced in (ARM_LEFT, ARM_RIGHT) else None
        self.lock_delta = float(lock_delta)
        self.reset()

    def reset(self):
        self.arm = self.forced
        self.locked = self.forced is not None
        self._rest = {ARM_LEFT: deque(maxlen=self.REST_WINDOW),
                      ARM_RIGHT: deque(maxlen=self.REST_WINDOW)}

    def _rest_est(self, arm):
        buf = self._rest[arm]
        return float(np.median(buf)) if buf else None

    def update(self, angles):
        """angles: {ARM_LEFT: deg|None, ARM_RIGHT: deg|None} -> chosen arm|None"""
        if self.locked:
            return self.arm

        for a in (ARM_LEFT, ARM_RIGHT):
            v = angles.get(a)
            if v is not None and v >= REST_MIN_DEG:
                self._rest[a].append(v)   # only open elbows describe "rest"

        deltas = {}
        for a in (ARM_LEFT, ARM_RIGHT):
            v, r = angles.get(a), self._rest_est(a)
            if v is not None and r is not None:
                deltas[a] = r - v         # positive = flexed below its rest

        if not deltas:
            # Nothing measurable yet: fall back to whichever arm is visible.
            if self.arm is None:
                for a in (ARM_LEFT, ARM_RIGHT):
                    if angles.get(a) is not None:
                        self.arm = a
                        break
            return self.arm

        best = max(deltas, key=deltas.get)
        if self.arm is None:
            self.arm = best
        elif best != self.arm:
            other = deltas.get(self.arm, 0.0)
            if deltas[best] - other >= self.SWITCH_MARGIN_DEG:
                self.arm = best

        if self.arm in deltas and deltas[self.arm] >= self.lock_delta:
            self.locked = True
        return self.arm


# ── HUD ──────────────────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _hud_text(txt):
    """cv2.putText only draws ASCII: anything else is rendered as "???".
    Status messages keep their proper punctuation for screen4's HTML status
    line and are flattened here, at the one place that draws them."""
    txt = txt.replace("\u2014", "-").replace("\u2013", "-").replace("\u00b0", " deg")
    return txt.encode("ascii", "replace").decode("ascii")


def draw_hud(frame, tracker, arm=None, angle=None, arm_pts=None):
    """Draw the live state onto the frame. Called with the engine's lock held,
    so it only reads the tracker."""
    h, w = frame.shape[:2]
    st = tracker.state

    colour = {
        State.READY:    C_WARN,
        State.SET:      C_WARN,
        State.LIFTING:  C_WARN,
        State.HOLD:     C_OK,
        State.LOWERING: C_OK,
        State.PASSED:   C_OK,
        State.FAILED:   C_BAD,
    }.get(st, C_WARN)

    # Elbow marker + the two arm segments, so the coach can see what is
    # being measured rather than trusting a bare number.
    if arm_pts is not None:
        s, e, wr = arm_pts
        cv2.line(frame, tuple(np.int32(s)), tuple(np.int32(e)), colour, 3, cv2.LINE_AA)
        cv2.line(frame, tuple(np.int32(e)), tuple(np.int32(wr)), colour, 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.int32(e)), 9, colour, -1, cv2.LINE_AA)

    # Top bar: the hold progress, which is the thing the participant is
    # actually trying to fill.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = tracker.hold_progress()
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 10), C_OK, -1)

    draw_panel(frame, 12, 22, 430, 196)
    y = 48
    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    line(f"REVERSE CURL  [{st.name}]", colour, 0.68, 30)
    line(f"Arm: {arm or '--'}    Target: <= {tracker.up_angle:.0f} deg")
    if angle is not None:
        base = f"  (rest {tracker.baseline_deg:.0f})" if tracker.baseline_deg is not None else ""
        line(f"Elbow: {angle:.0f} deg{base}",
             C_OK if angle <= tracker.up_angle else C_TEXT)
    else:
        line("Elbow: not tracked", C_BAD)
    line(f"Hold: {tracker.hold_elapsed:.1f} / {tracker.hold_s:.0f} s"
         f"   (best {tracker.hold_best_s:.1f} s)")
    if tracker.descent_deg_s is not None:
        # Information only — see the module docstring.
        line(f"Descent: {tracker.descent_deg_s:.0f} deg/s (coach judges)", C_WARN, 0.52, 22)

    if tracker.status_msg:
        msg = _hud_text(tracker.status_msg)
        draw_panel(frame, 12, h - 54, min(w - 12, 12 + 13 * len(msg) + 24), h - 12)
        cv2.putText(frame, msg, (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 2, cv2.LINE_AA)

    if tracker.resolved:
        verdict = "PASS" if tracker.passed else "FAIL"
        vcol = C_OK if tracker.passed else C_BAD
        size = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)[0]
        org = ((w - size[0]) // 2, (h + size[1]) // 2)
        draw_panel(frame, org[0] - 30, org[1] - size[1] - 24,
                   org[0] + size[0] + 30, org[1] + 24, C_DARK, 0.72)
        cv2.putText(frame, verdict, org, cv2.FONT_HERSHEY_SIMPLEX, 2.2, vcol, 5, cv2.LINE_AA)
    return frame


def _stage_reverse_curl(session_id, result, arm, rating=None):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/reverse_curl",
            json={
                "session_id":     session_id,
                "passed":         bool(result.get("passed")),
                "arm":            arm,
                "peak_angle_deg": result.get("peak_angle_deg"),
                "hold_s":         result.get("hold_s"),
                "rating":         rating,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] reverse_curl staged for review → "
              f"{'PASS' if result.get('passed') else 'FAIL'}  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Reverse Curl Test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--arm", choices=[ARM_AUTO, ARM_LEFT, ARM_RIGHT], default=ARM_AUTO,
                        help="Which arm performs the test (default: auto-detect)")
    parser.add_argument("--up-angle", type=float, default=UP_ANGLE_DEG,
                        help=f"Interior elbow angle that counts as the up position "
                             f"(default {UP_ANGLE_DEG:.0f})")
    parser.add_argument("--hold-s", type=float, default=HOLD_S,
                        help=f"Seconds the up position must be held (protocol: {HOLD_S:.0f})")
    parser.add_argument("--model", default="pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file")
    parser.add_argument("--smooth", type=int, default=SMOOTH_FRAMES,
                        help=f"Angle smoothing window in frames (default {SMOOTH_FRAMES})")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame numbers to the console")
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
    print(f"[INFO] Seated, weight on the mid-thigh. Lift to <= {args.up_angle:.0f} deg, "
          f"hold {args.hold_s:.0f} s, lower to the thigh.")
    print("[INFO] The camera decides PASS / FAIL.  R = reset  |  Q = quit\n")

    tracker = ReverseCurlTracker(up_angle=args.up_angle, hold_s=args.hold_s)
    selector = ArmSelector(forced=args.arm)
    smooth = MedianSmoother(args.smooth)
    staged = False
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

    win = "Reverse Curl Test"
    cv2.namedWindow(win)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            fh, fw = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int((time.time() - stream_start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            results = landmarker.detect_for_video(mp_image, ts)

            angle = None
            arm = selector.arm
            arm_pts = None

            if results.pose_landmarks:
                lms = results.pose_landmarks[0]
                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(lms[i].x * fw), int(lms[i].y * fh)),
                                 (int(lms[j].x * fw), int(lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)

                raw = {a: elbow_angle(lms, a, fw, fh) for a in (ARM_LEFT, ARM_RIGHT)}
                arm = selector.update(raw)
                if arm is not None and raw.get(arm) is not None:
                    angle = smooth.push(raw[arm])
                    arm_pts = [lm_px(lms, i, fw, fh) for i in ARM_IDX[arm]]
                else:
                    # Arm lost: drop the history so the angle doesn't creep
                    # back from stale frames when it is re-detected.
                    smooth.reset()
            else:
                smooth.reset()

            outcome = tracker.update(angle)
            if outcome is not None and not staged and args.session_id:
                _stage_reverse_curl(args.session_id, outcome, arm)
                staged = True

            if args.debug:
                print(f"arm={arm} angle={angle if angle is None else round(angle,1)} "
                      f"state={tracker.state.name} hold={tracker.hold_elapsed:.2f}")

            draw_hud(frame, tracker, arm=arm, angle=angle, arm_pts=arm_pts)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                tracker.reset()
                selector.reset()
                smooth.reset()
                staged = False

    cap.release()
    cv2.destroyAllWindows()

    res = tracker.result()
    print("\n── Reverse Curl ─────────────────────────────")
    verdict = ("PASS" if res["passed"] else "FAIL") if tracker.resolved else "NOT COMPLETED"
    print(f"  Result       : {verdict}")
    print(f"  Arm          : {selector.arm or '--'}")
    print(f"  Best flexion : {res['peak_angle_deg']} deg (target <= {res['target_angle_deg']:.0f})")
    print(f"  Longest hold : {res['hold_s']} s (required {res['required_hold_s']:.0f})")
    if res["descent_deg_s"] is not None:
        print(f"  Descent      : {res['descent_deg_s']} deg/s  (coach judges control)")
    print(f"  Attempts     : {res['attempts']}")


if __name__ == "__main__":
    main()
