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
      gravitational pull". Not measured or shown at all: judging
      "controlled" against a free-fall model would add a silent failure
      mode, and a speed read-out misled more than it helped (it timed from
      the end of the 2 s hold, so a longer hold made a fast drop look slow).
      The coach is watching from arm's length.
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
REST_MAX_DEG    = 130.0  # ...and no more open than this. The start position is the
                         # weight on the mid-thigh (~90-100 deg in the manual's
                         # figure); an arm hanging at the side reads 150-180. If a
                         # hanging arm were accepted as "rest", moving the hand onto
                         # the thigh would look like a lift, and hanging it again
                         # like a failed attempt — a FAIL with no curl at all.
REST_BAND_DEG   = 8.0    # spread allowed inside the settle window
REST_SETTLE_S   = 0.5    # how long it must sit still before the baseline is taken
# The two above only pick the FIRST resting reading, while the participant is
# just sitting there. Every later "it's resting now" call — updating a rest
# reading, or deciding the weight was set down somewhere else — needs the arm
# SETTLED: under 4 deg for a full second. 8 deg in 0.5 s let anything slower
# than 16 deg/s pass as still, so a slow lift dragged the rest angle down with
# it and a slow, controlled lowering could count as "back down" too early. A
# hand resting on the thigh easily meets 4 deg/1 s; a lift or a lowering,
# however slow, doesn't.
SETTLED_BAND_DEG = 4.0
SETTLED_S        = 1.0

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
HOLD_WOBBLE_S   = 0.30   # the angle may sit above UP_ANGLE (inside the release
                         # band) this long before the hold breaks. Absorbs a shaky
                         # frame or two; without a time limit, touching 45 once and
                         # then "holding" at 50 passed.
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


def in_frame(lms, *idxs):
    """MediaPipe still returns a "confident" guess for joints outside the
    picture (normalised coordinates below 0 or above 1). An arm below the
    bottom edge then produces a made-up angle, so only joints actually in the
    picture count."""
    return all(0.0 <= lms[i].x <= 1.0 and 0.0 <= lms[i].y <= 1.0 for i in idxs)


def elbow_angle(lms, arm, w, h):
    """Interior elbow angle for one arm, or None if the joints aren't clear
    or aren't in the picture."""
    idxs = ARM_IDX[arm]
    if not vis(lms, *idxs) or not in_frame(lms, *idxs):
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
                 rest_min=REST_MIN_DEG, rest_max=REST_MAX_DEG,
                 move_delta=MOVE_DELTA_DEG, return_tol=RETURN_TOL_DEG,
                 settle_s=REST_SETTLE_S, rest_band=REST_BAND_DEG):
        self.up_angle = float(up_angle)
        self.hold_s = float(hold_s)
        self.rest_min = float(rest_min)
        self.rest_max = float(rest_max)
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
        self.passed = False
        self.verdict_by = None        # "auto" once decided — no manual verdicts
        self.status_msg = "Rest the weight on the thigh and hold still"
        self._settle = deque()        # (t, angle) while finding the baseline
        self._hold_start = None
        self._above_since = None      # when the hold last drifted above the target
        self._return = deque()        # (t, angle) while waiting for the weight to settle
        self._dropout_start = None
        self._last_frame_t = None
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

        # A gap between frames (the phone or network stalled) is time nobody
        # saw. Mid-hold it must not count toward the 2 s, and a rest reading
        # must never be built across it. None frames are handled below; this
        # catches frames that simply stopped arriving.
        prev_t, self._last_frame_t = self._last_frame_t, now
        if prev_t is not None and now - prev_t > DROPOUT_GRACE_S:
            self._settle.clear()
            self._return.clear()
            if self.state == State.HOLD:
                self._note_hold(prev_t)
                self._hold_start = None
                self._above_since = None
                self.hold_elapsed = 0.0
                self.state = State.LIFTING
                self.status_msg = "Video paused mid-hold - hold again"

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
                self._above_since = None
                self.hold_elapsed = 0.0
                self.status_msg = f"Hold for {self.hold_s:.0f} s"
            elif angle >= self.baseline_deg - self.return_tol:
                return self._end_attempt(f"Came back down at {self.peak_deg:.0f} deg "
                                         f"- target is {self.up_angle:.0f} deg")
            return None

        if self.state == State.HOLD:
            self.peak_deg = min(self.peak_deg, angle)
            if angle > self.up_angle:
                if self._above_since is None:
                    self._above_since = now
            else:
                self._above_since = None
            wobbled_too_long = (self._above_since is not None
                                and now - self._above_since > HOLD_WOBBLE_S)
            if angle > self.up_angle + HOLD_RELEASE_DEG or wobbled_too_long:
                # Hold broken before the 2 s were up: too far above the
                # target, or above it for longer than a wobble.
                self._note_hold(now)
                self._hold_start = None
                self._above_since = None
                self.hold_elapsed = 0.0
                if angle >= self.baseline_deg - self.return_tol:
                    return self._end_attempt(f"Held {self.hold_best_s:.1f} s of "
                                             f"{self.hold_s:.0f} s before lowering")
                else:
                    self.state = State.LIFTING
                    self.status_msg = "Lost the hold - back to the target"
                return None

            self.hold_elapsed = now - self._hold_start
            self.hold_best_s = max(self.hold_best_s, self.hold_elapsed)
            if self.hold_elapsed >= self.hold_s:
                self.state = State.LOWERING
                self._return.clear()
                self.status_msg = "Hold complete - lower the weight to the thigh"
            return None

        if self.state == State.LOWERING:
            # The hold is done; only the return is left. Back near where it
            # started counts at once. The weight may also be set down on a
            # different spot of the thigh — that counts once the elbow is in
            # the resting range and SETTLED (see SETTLED_S), so passing through
            # the range on the way down — even slowly — doesn't end it early. (Only the
            # PASS side is relaxed: a struggling lift on the way UP is judged
            # against the start as before, so it is never failed early.)
            if angle >= self.baseline_deg - self.return_tol:
                return self._pass(now, angle)
            if self._still_in_rest_range(self._return, angle, now,
                                         SETTLED_BAND_DEG, SETTLED_S):
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

        if gap > DROPOUT_GRACE_S:
            # Same reasoning for the window that proves the weight was set
            # down: it means "the elbow sat still for SETTLED_S", and a window
            # whose first sample predates a blackout proves nothing about the
            # seconds nobody saw. Without this, tracking lost for seconds while
            # lowering and regained anywhere in the resting range passed on the
            # first frame back. One dropped frame (inside the grace) still
            # keeps the window, like the hold.
            self._return.clear()

        if self.state == State.HOLD and gap > DROPOUT_GRACE_S:
            # Void the hold rather than credit time nobody could see. Same
            # rule the pull-up counter uses for a rep that was mid-flight
            # when tracking dropped.
            self._note_hold(self._dropout_start)
            self._hold_start = None
            self.hold_elapsed = 0.0
            self.state = State.LIFTING
            self.status_msg = "Tracking lost mid-hold - hold again"
        return None

    def _seek_baseline(self, angle, now, refresh=False):
        """Capture (or refresh) the resting elbow angle. Only ever taken from
        a stretch where the elbow is both OPEN and STILL, so the baseline is
        the real rest position and not a frame from mid-lift."""
        if angle < self.rest_min:
            self._settle.clear()
            if not refresh:
                self.status_msg = (f"Open the elbow past {self.rest_min:.0f} deg "
                                   f"- rest the weight on the thigh")
            return
        if angle > self.rest_max:
            # Arm hanging at the side — or, filmed from the FRONT, a forearm
            # resting along the thigh points at the lens and reads nearly
            # straight (the manual's own front-view photo reads 168 deg).
            # Either way this isn't a usable resting angle; say which to fix.
            self._settle.clear()
            if not refresh:
                self.status_msg = "Arm looks straight - rest the weight on the thigh, film side-on"
            return
        band, settle_s = ((SETTLED_BAND_DEG, SETTLED_S) if refresh
                          else (self.rest_band, self.settle_s))
        if self._still_in_rest_range(self._settle, angle, now, band, settle_s):
            vals = [a for _, a in self._settle]
            self.baseline_deg = float(np.median(vals))
            if not refresh:
                self.state = State.SET
                self.status_msg = (f"Rest angle {self.baseline_deg:.0f} deg "
                                   f"- lift to {self.up_angle:.0f} deg")

    def _still_in_rest_range(self, buf, angle, now, band, settle_s):
        """Feed one angle into a stillness window. True once the elbow has sat
        inside the resting range (hand on the thigh), within `band`, for
        `settle_s`. Leaving the range empties the window."""
        if not (self.rest_min <= angle <= self.rest_max):
            buf.clear()
            return False
        buf.append((now, angle))
        # Keep one sample from just outside the window so the span can
        # actually reach settle_s (trimming to exactly settle_s caps it just
        # below, and it never fires).
        while len(buf) > 2 and (now - buf[1][0]) >= settle_s:
            buf.popleft()
        if len(buf) < 2:
            return False
        vals = [a for _, a in buf]
        return (buf[-1][0] - buf[0][0] >= settle_s
                and max(vals) - min(vals) <= band)

    def adopt_baseline(self, deg):
        """The working arm changed before the lift began — take that arm's
        own resting angle. A rest reading belongs to ONE arm: measuring the
        other arm against it misreads every movement (a curl from the thigh
        can look like it never came back down). No usable reading for the new
        arm: start the rest search over."""
        if self.state not in (State.READY, State.SET):
            return
        self._settle.clear()
        self.peak_deg = None
        if deg is None or not (self.rest_min <= deg <= self.rest_max):
            self.baseline_deg = None
            self.state = State.READY
            self.status_msg = "Rest the weight on the thigh and hold still"
            return
        self.baseline_deg = float(deg)
        self.state = State.SET
        self.status_msg = (f"Rest angle {self.baseline_deg:.0f} deg "
                           f"- lift to {self.up_angle:.0f} deg")

    def _note_hold(self, when):
        if self._hold_start is not None:
            self.hold_best_s = max(self.hold_best_s, when - self._hold_start)

    def _end_attempt(self, msg):
        """The weight came back to the thigh without a correct curl. If the
        lift was big enough to be a real try, that was the one trial and it
        FAILS — returns the result. Otherwise it was a fidget: forget it and
        stay ready — returns None."""
        if not self._is_real_try():
            self.peak_deg = None
            self._hold_start = None
            self.hold_elapsed = 0.0
            self.hold_best_s = 0.0
            self.state = State.SET
            self.status_msg = f"Ready - lift to {self.up_angle:.0f} deg"
            return None
        return self._fail(msg)

    def _is_real_try(self):
        """Did the current lift go far enough to be THE trial? Reaching the
        target zone always counts, even for someone who rests at a low angle
        and so has less than ATTEMPT_MIN_DEG to give."""
        return (self.peak_deg is not None and self.baseline_deg is not None
                and (self.peak_deg <= self.baseline_deg - ATTEMPT_MIN_DEG
                     or self.peak_deg <= self.up_angle))

    def _fail(self, msg):
        self.attempts.append({
            "peak_angle_deg": round(self.peak_deg, 1),
            "hold_s": round(self.hold_best_s, 2),
            "reached_target": bool(self.peak_deg <= self.up_angle),
            "outcome": "fail",
        })
        self._hold_start = None
        self._above_since = None
        self.hold_elapsed = 0.0
        self.passed = False
        self.verdict_by = "auto"
        self.state = State.FAILED
        self.status_msg = f"FAIL - {msg}"
        return self.result()

    def finish(self):
        """The test was stopped before the camera reached a verdict. The
        manual allows ONE trial and passes only a correct curl, so:

          * a real attempt was under way             -> FAIL (the trial happened)
          * the hold was done and the weight is back
            in the resting range, just not settled
            long enough to be confirmed yet          -> PASS (the curl happened)
          * no real attempt was seen                 -> None: NOT COMPLETED —
            usually the camera couldn't see the arm; not the student's fail

        Returns the result dict, or None for not completed. Already decided:
        returns that verdict unchanged."""
        if self.resolved:
            return self.result()
        now = self._now if self._now is not None else time.time()
        if (self.state == State.LOWERING and self.angle_deg is not None
                and self.baseline_deg is not None
                and self.rest_min <= self.angle_deg <= self.rest_max
                and self.angle_deg >= self.baseline_deg - ATTEMPT_MIN_DEG):
            # "Back down" is measured against THIS participant's own rest
            # angle, not the bottom of the resting range. Taking the range's
            # floor instead meant someone who rests at 125 deg could be
            # stopped at 72 — elbow still bent, weight halfway up — and be
            # credited with a PASS, while the same arm one degree lower
            # FAILED. Still flexed by ATTEMPT_MIN_DEG or more from rest is as
            # much bend as a real attempt, so the weight is not on the thigh.
            return self._pass(now, self.angle_deg)
        if self.state in (State.LIFTING, State.HOLD, State.LOWERING) and self._is_real_try():
            if self.state == State.HOLD:
                self._note_hold(now)
            if self.state == State.LOWERING:
                return self._fail("stopped with the weight still up")
            return self._fail("stopped mid-attempt (one trial only)")
        return None

    def _pass(self, now, angle):
        self.attempts.append({
            "peak_angle_deg": round(self.peak_deg, 1) if self.peak_deg is not None else None,
            "hold_s": round(self.hold_best_s, 2),
            "reached_target": True,
            "outcome": "pass",
        })
        self.passed = True
        self.verdict_by = "auto"
        self.state = State.PASSED
        self.status_msg = "PASS - correct reverse curl"
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
            "attempts": len(self.attempts),
            "verdict_by": self.verdict_by,
        }


class ArmSelector:
    """Decides which arm is doing the test.

    The manual lets the participant use their "preferred arm", so the app has
    to work it out. Each arm gets its own resting angle; the working arm is
    whichever has flexed furthest BELOW its own rest. A hysteresis margin
    stops the choice flickering while both arms sit still (they often rest at
    different angles, and a selector that flipped between them would feed the
    tracker a jumping baseline that never settles).

    An arm's resting angle is recorded ONLY while that arm is still, inside
    the resting range — the same rule the tracker uses. Taking every in-range
    frame instead let the "rest" drift down with a slow lift, so a slow curl
    never looked like movement: the camera kept watching the idle arm, and
    the rest angle it handed the tracker came out ~15 deg too low.

    Once a real lift starts the choice LOCKS, so the other hand moving — to
    steady the chair, say — cannot steal the test mid-curl. The lock needs
    flexion worth a real attempt (ATTEMPT_MIN_DEG): locking at the smaller
    "watching" threshold meant a 12-15 deg shift of the IDLE hand — about what
    adjusting the grip looks like — pinned the camera to the wrong arm. It is
    released again by note_idle() once the tracker is back at rest.
    """

    SWITCH_MARGIN_DEG = 8.0

    def __init__(self, forced=ARM_AUTO, lock_delta=ATTEMPT_MIN_DEG):
        self.forced = forced if forced in (ARM_LEFT, ARM_RIGHT) else None
        self.lock_delta = float(lock_delta)
        self.reset()

    def reset(self):
        self.arm = self.forced
        self.locked = self.forced is not None
        self._still = {ARM_LEFT: deque(), ARM_RIGHT: deque()}   # (t, deg)
        self._rest = {ARM_LEFT: None, ARM_RIGHT: None}          # last still reading
        self._idle_since = None      # when the tracker last went back to rest

    def _track_rest(self, arm, v, now):
        """Update this arm's resting angle from a still, in-range stretch."""
        buf = self._still[arm]
        if v is None or not (REST_MIN_DEG <= v <= REST_MAX_DEG):
            buf.clear()
            return
        # First reading: the easy rule. After that the arm must be SETTLED to
        # move its reading, so a slow lift can't drag "rest" down with it.
        band, settle_s = ((REST_BAND_DEG, REST_SETTLE_S) if self._rest[arm] is None
                          else (SETTLED_BAND_DEG, SETTLED_S))
        buf.append((now, v))
        while len(buf) > 2 and (now - buf[1][0]) >= settle_s:
            buf.popleft()
        vals = [x for _, x in buf]
        if (len(buf) >= 2 and buf[-1][0] - buf[0][0] >= settle_s
                and max(vals) - min(vals) <= band):
            self._rest[arm] = float(np.median(vals))

    def rest_of(self, arm):
        """This arm's own resting angle, or None if it hasn't rested yet."""
        return self._rest.get(arm)

    def lock(self):
        """Pin the current choice — the lift has begun on this arm."""
        self._idle_since = None
        if self.arm is not None:
            self.locked = True

    def note_idle(self, now):
        """No attempt is under way — the weight is on the thigh. Once that has
        been true for SETTLED_S the lock is released, so a movement the tracker
        decided was NOT an attempt cannot own the rest of the trial: a lock
        taken on a fidget used to leave the camera watching the idle hand while
        a perfect curl happened on the other arm, ending the trial NOT
        COMPLETED. The delay is what makes it safe — a slow lift flaps between
        "watching" and "at rest" frame by frame, and releasing on the first
        such frame let the other hand take over a real lift in progress. A
        pinned arm (--arm left/right) stays pinned."""
        if self.forced is not None:
            return
        if self._idle_since is None:
            self._idle_since = now
        elif now - self._idle_since >= SETTLED_S:
            self.locked = False

    def update(self, angles, now=None):
        """angles: {ARM_LEFT: deg|None, ARM_RIGHT: deg|None} -> chosen arm|None"""
        if self.locked:
            return self.arm
        now = time.time() if now is None else now

        for a in (ARM_LEFT, ARM_RIGHT):
            self._track_rest(a, angles.get(a), now)

        deltas = {}
        for a in (ARM_LEFT, ARM_RIGHT):
            v, r = angles.get(a), self._rest[a]
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
        if self.arm is None or self._rest[self.arm] is None:
            # The current pick has never rested on the thigh (e.g. it hangs at
            # the side) — prefer an arm that has.
            self.arm = best
        elif best != self.arm:
            other = deltas.get(self.arm, 0.0)
            if deltas[best] - other >= self.SWITCH_MARGIN_DEG:
                self.arm = best

        if self.arm in deltas and deltas[self.arm] >= self.lock_delta:
            self.locked = True
        return self.arm


def step_arm(selector, tracker, smooth, raw, prev_arm, now=None):
    """One frame of the pipeline: pick the working arm, smooth its angle and
    feed the verdict. Shared by the web engine and the desktop loop so the
    two can't drift apart. `raw` is {ARM_LEFT: deg|None, ARM_RIGHT: deg|None}
    (both None when nobody is detected). Returns (arm, angle, outcome).

    Two rules live here because they need the selector and the tracker
    together: when the choice of arm changes before the lift, the tracker
    takes the NEW arm's own rest angle; and once the tracker sees a lift, the
    choice locks, so the arm can't change mid-attempt."""
    now = time.time() if now is None else now
    arm = selector.update(raw, now=now)
    if arm != prev_arm and arm is not None:
        smooth.reset()
        if prev_arm is not None:
            tracker.adopt_baseline(selector.rest_of(arm))
    value = raw.get(arm) if arm is not None else None
    if value is None:
        smooth.reset()       # don't let stale frames creep back on re-detect
        angle = None
    else:
        angle = smooth.push(value)
    outcome = tracker.update(angle, now=now)
    if tracker.state in (State.READY, State.SET):
        # Nothing under way — the weight is on the thigh. A lock taken on a
        # movement that turned out not to be an attempt is released once that
        # has held for a moment (see note_idle), so the arm that really curls
        # can still take over.
        selector.note_idle(now)
    else:
        selector.lock()
    return arm, angle, outcome


# ── HUD ──────────────────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _hud_text(txt):
    """cv2.putText only draws ASCII: anything else is rendered as "???".
    The status lines are written in plain ASCII, so this is a safety net for
    anything added later (a dash, a degree sign) reaching the overlay."""
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

    # A recording has a timeline of its own. Timing the 2 s hold by the wall
    # clock while frames are decoded faster or slower than real time measures
    # the computer instead of the participant: 2 s of video that takes 4 s to
    # decode credits a 4 s hold, and a clip decoded quickly fails a good one.
    # So a file is clocked by its own frame rate (the same way pullup_full.py
    # times clips) while a camera keeps the wall clock, which for a camera IS
    # the truth.
    live = isinstance(source, int)
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not live and not (1.0 <= clip_fps <= 240.0):
        clip_fps = 30.0
        print(f"[WARN] {args.source} reports no usable frame rate - "
              f"timing it as {clip_fps:.0f} fps")

    print(f"\n[INFO] Stream  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] Seated, weight on the mid-thigh. Lift to <= {args.up_angle:.0f} deg, "
          f"hold {args.hold_s:.0f} s, lower to the thigh.")
    print("[INFO] The camera decides PASS / FAIL.  R = reset  |  Q = quit\n")

    tracker = ReverseCurlTracker(up_angle=args.up_angle, hold_s=args.hold_s)
    selector = ArmSelector(forced=args.arm)
    smooth = MedianSmoother(args.smooth)
    staged = False
    prev_arm = None
    stream_start = time.time()
    frame_no = 0
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

            frame_no += 1
            now = time.time() if live else stream_start + (frame_no - 1) / clip_fps

            fh, fw = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int((now - stream_start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            results = landmarker.detect_for_video(mp_image, ts)

            arm_pts = None
            raw = {ARM_LEFT: None, ARM_RIGHT: None}
            lms = None

            if results.pose_landmarks:
                lms = results.pose_landmarks[0]
                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(lms[i].x * fw), int(lms[i].y * fh)),
                                 (int(lms[j].x * fw), int(lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)

                raw = {a: elbow_angle(lms, a, fw, fh) for a in (ARM_LEFT, ARM_RIGHT)}

            arm, angle, outcome = step_arm(selector, tracker, smooth, raw, prev_arm, now=now)
            prev_arm = arm
            if angle is not None and lms is not None:
                arm_pts = [lm_px(lms, i, fw, fh) for i in ARM_IDX[arm]]
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
                prev_arm = None
                staged = False

    cap.release()
    cv2.destroyAllWindows()

    outcome = tracker.finish()          # same STOP rule as the web engine
    if outcome is not None and not staged and args.session_id:
        _stage_reverse_curl(args.session_id, outcome, selector.arm)
    res = tracker.result()
    print("\n── Reverse Curl ─────────────────────────────")
    verdict = ("PASS" if res["passed"] else "FAIL") if tracker.resolved else "NOT COMPLETED"
    print(f"  Result       : {verdict}")
    print(f"  Arm          : {selector.arm or '--'}")
    print(f"  Best flexion : {res['peak_angle_deg']} deg (target <= {res['target_angle_deg']:.0f})")
    print(f"  Longest hold : {res['hold_s']} s (required {res['required_hold_s']:.0f})")
    print(f"  Attempts     : {res['attempts']}")


if __name__ == "__main__":
    main()
