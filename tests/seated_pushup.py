"""
Seated Push-Up Test – lift the body off the seat on straight arms, up to 20 s
============================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.85, figure 5.21)
--------------------------------------------------------------------
  The participant sits with the hands on push-up blocks, on the armrests of
  a wheelchair or armchair, or on the wheels of a sport wheelchair, and
  pushes down until the buttocks come off the seat with the elbows
  extended. That position is held for as long as possible, to a maximum of
  20 seconds. ONE trial; the score is the time held, to the nearest second.

  Timing begins when the body is raised with the elbows extended and ends
  when the position is no longer held, or at 20 s. A participant who cannot
  fully extend the elbows is timed from their own maximal extension - the
  owner chose to allow that: the clock starts at whatever elbow angle the
  student has once they are up and steady. Feet on the floor must not help;
  that is for the coach to watch.

What the camera measures
--------------------------------------------------------------------
  FRONT-ON, facing the participant - the owner's choice. Nothing here
  depends on which way the camera looks at the body, though: the lift is
  measured against the hands and each arm is read on its own, so a side-on
  camera works too. The manual's wheelchair photo (fig 5.21b) IS side-on and reads fine;
  the blocks photo (fig 5.21a) is roughly front-on.

    lift    how much further above the HANDS the hips are than when they
            sat, in torso lengths (mid-shoulder to mid-hip, measured while
            seated). On blocks the arms are already straight while sitting,
            so straight arms alone cannot tell sitting from lifted - the hips
            can. The hands stay planted on the supports, so measuring against
            them rather than the screen means a bumped phone or a wheelchair
            that rolls moves hips and hands together and the lift is unchanged.
            Each hand is its own reference, so losing one from view does not
            shift the reading when the two sit at different heights.
    elbow   shoulder-elbow-wrist angle, each arm separately

  The manual's photos, cropped to the participant, read the near elbow at
  178 (blocks) and 176 (wheelchair); the far arm is hidden in both and
  drops out on its own. The whole wheelchair photo locks the
  pose model onto the kneeling tester instead - only the student should be
  in frame.

  From the front, an elbow bending straight BACKWARD barely changes the
  angle the camera sees. The body sinking is what that looks like here, so
  the hips coming back down is the rule that catches it.

The clock
--------------------------------------------------------------------
  SEAT   first the participant sits still, hands on the supports, for
         SEAT_STILL_S. How far the hips then sit above each hand is the seat.
         A later still sit that reads LOWER replaces it, so a student who was
         already part-way up when the camera first saw them is not held to
         that. The seat also keeps where the hips are on SCREEN, and a later
         still sit corrects it:
           * hips level on screen but the hands moved on screen by more than
             HAND_MOVE_TORSO (from the lap down onto the supports, or just
             re-gripping a little lower) -> the hands moved, not the body:
             the seat is learned again. Even a small shift matters - left
             in, it counts as lift, and a student on blocks who sits back
             down with straight arms would still read as "up". A real lift
             moves the hips, not the hands, so it never triggers this.
           * hips level against the hands but moved on screen (a bumped
             phone, or sitting down after standing still in front of the
             camera) -> the on-screen seat is moved to match
  START  once the hips are LIFT_MIN_TORSO above the seat against the hands
         AND have risen at least SCREEN_RISE_MIN_TORSO on screen, at least one
         whole arm (shoulder, elbow, wrist) is in view, and the body has been
         steady
         for SETTLE_S. A real lift moves the hips on screen and away from the
         hands; hands shifting on the supports only does the second, and a
         bumped phone only the first, so neither alone can start the clock.
         That second is credited - they were holding it. The
         elbow angles and the lift over that second become the participant's
         own starting position. With no arm in view there is no elbow to
         check, so the clock waits rather than run on the hips alone.
  STOP   when either of these lasts longer than BREAK_GRACE_S:
           * the hips come back down most of the way to the seat
             (below SINK_FRAC of the starting lift, and never less than
             SEAT_TOUCH_TORSO)
           * an elbow bends ELBOW_BEND_DEG below the straightest it has been
             during the hold. Each arm's mark starts at its angle when the
             clock started and rises as the arm straightens further - only a
             steady reading (a full smoothing window) raises it, never one
             odd frame - so straightening and then bending back is caught.
         The time stops at the last frame the position was still correct,
         not after the grace - the grace only stops one shaky frame from
         ending a good hold.
  CAP    20 s is the maximum; the clock stops itself there.
  LOST   tracking (or the phone's frames) gone longer than DROPOUT_GRACE_S
         ends the hold at the last frame the position was seen correct. A
         shorter gap - school Wi-Fi stalls for a second at a time - does not:
         if the student is still up when the picture comes back, the clock
         carries on; if they sat down meanwhile, it stops at the last frame
         they were seen up. Smoothing is cleared after any gap, so the first
         frame back is judged on what it shows, not on readings from before.
         If a break had begun before the picture was lost, its reason is
         the one given. "Seen"
         means the hips, a hand that has a seat reading, and an arm that has
         an elbow mark - without those the hold cannot be checked. An arm
         that was hidden when the clock started gets its own mark from its
         first steady reading, so the arms swapping in and out of view
         mid-hold does not count as losing the student.

  Hips more than MAX_LIFT_TORSO above the seat is standing up, not a seated
  push-up: it never starts the clock and it ends a hold. This one is judged
  on SCREEN height, not against the hands - a student who stands takes the
  hands off the supports and they rise too, so against the hands standing
  reads as an ordinary 0.25-0.6 lift. A bumped phone moves the picture a
  few centimetres; standing moves the hips a whole torso, so screen height
  is safe for this rule alone.

  Stopping before the clock ever started is NOT COMPLETED, not a zero - a
  camera or setup problem must never be recorded against the participant.

Usage
-----
  python seated_pushup.py
  python seated_pushup.py --source clip.mp4
  python seated_pushup.py --debug

  Keys : R = reset the trial     Q = quit and print the summary

Camera placement
----------------
  Phone at seat height, FACING the participant, 1.5-2 m away, with the head,
  both hands and the hips in frame.
"""

import argparse
import math
import sys
import time
from collections import deque
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision as mp_vision

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode


# ── Protocol constants ───────────────────────────────────────────────────────────
MAX_HOLD_S = 20.0     # "timing ends ... after a maximum of 20 seconds"

# ── Sitting: where the seat is ───────────────────────────────────────────────────
SEAT_STILL_S     = 1.0    # sitting still this long -> the seat is known
SEAT_BAND_TORSO  = 0.10   # hip-above-hand spread allowed while sitting still
                          # (~4 cm). The gap carries the hand's wobble as well
                          # as the hip's - about twice a hip alone - and at
                          # 0.04 a normally jittery student never read as still.

# ── Lifted: the clock can start ──────────────────────────────────────────────────
LIFT_MIN_TORSO   = 0.08   # hips this far above the seat = off it (~3.4 cm on
                          # a child's ~42 cm torso). Set below the 5 cm aimed
                          # for because of landmark wobble: at 0.12 a real 5 cm
                          # lift started the clock ~1 in 10 wobbly tries, at
                          # 0.08 every time - and a minute of sitting still
                          # with the same wobble never started it.
HAND_MOVE_TORSO  = 0.04   # hands moved this far on screen while the hips
                          # stayed put -> re-learn the seat (~1.7 cm). A still
                          # stretch's median hand height wobbles ~0.01.
SCREEN_RISE_MIN_TORSO = 0.06  # ... and at least this far up on screen (~2.5 cm)
                              # - the check that hands moving lower on the
                              # supports is not mistaken for a lift.
MAX_LIFT_TORSO   = 0.80   # hips this far up, on screen or against the
                          # hands -> stood up. A seated push-up from bent
                          # arms lifts ~0.3-0.5; standing ~1.0.
SETTLE_S         = 1.0    # lifted and steady this long -> clock starts
SETTLE_LIFT_BAND = 0.10   # lift spread allowed inside the settle window - wide
                          # for the same reason; the lift itself is >= 0.12
SETTLE_ELBOW_BAND = 8.0   # elbow spread allowed inside the settle window

# ── Breaking the hold (measured against the participant's own start) ────────────
SINK_FRAC        = 0.35   # lift falls below this share of the starting lift
                          # -> "came back down onto the seat"
SEAT_TOUCH_TORSO = 0.07   # ... but never a line closer to the seat than this
                          # (~2.9 cm). It must sit above the largest hand
                          # re-grip the seat re-learn can miss under wobble
                          # (~0.06): at 0.05, a small re-grip plus a ~3.4 cm
                          # push kept the clock running after the student sat
                          # back down. Costs a little on ~4 cm holds, which
                          # wobble now cuts short a bit more often; 5 cm and
                          # up are unaffected.
ELBOW_BEND_DEG   = 20.0   # an elbow closes this far -> "elbows bent"
BREAK_GRACE_S    = 0.30   # a break must last this long to end the hold
DROPOUT_GRACE_S  = 2.00   # mid-hold, tracking or the phone's frames may
                          # vanish this long (0.6 ended good Isometric holds
                          # on a single school Wi-Fi stall). The time is
                          # still only credited if the student is seen up
                          # again afterwards.
SETTLE_GAP_S     = 0.60   # any gap in the frames longer than this empties
                          # the seat / settle windows and the smoothing -
                          # stillness can't be proven across time nobody saw

SMOOTH_FRAMES    = 3      # median window per measurement
MIN_VIS          = 0.50   # landmark visibility floor for every joint used

SIDE_LEFT, SIDE_RIGHT = "left", "right"
ARMS = (SIDE_LEFT, SIDE_RIGHT)

# Landmark indices (33-point BlazePose topology)
SHOULDER = {SIDE_LEFT: 11, SIDE_RIGHT: 12}
ELBOW    = {SIDE_LEFT: 13, SIDE_RIGHT: 14}
WRIST    = {SIDE_LEFT: 15, SIDE_RIGHT: 16}
HIP      = {SIDE_LEFT: 23, SIDE_RIGHT: 24}

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (24, 26), (25, 27), (26, 28),             # legs
]

# HUD palette (matches reverse_curl.py)
C_DARK  = (28, 28, 28)
C_OK    = (80, 220, 120)
C_WARN  = (60, 190, 250)
C_BAD   = (80, 80, 240)
C_TEXT  = (245, 245, 245)
C_SEAT  = (200, 160, 60)

# Plain-words reason for each break, shown on screen and staged for screen3.
BREAK_WORDS = {
    "sink":  "came back down towards the seat",
    "elbow": "elbows bent",
    "stood": "stood up off the seat",
}


class State(Enum):
    READY    = auto()   # finding the seat / waiting for the lift
    SETTLING = auto()   # lifted - confirming it is held steady
    HOLDING  = auto()   # the clock is running
    DONE     = auto()   # the one trial is over; the score is final


# ── geometry ─────────────────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    """Angle at B formed by A-B-C (2-D, degrees)."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _clear(lms, idx):
    """Visible enough, and inside the picture (MediaPipe still returns a
    confident-looking guess for joints off the edge)."""
    lm = lms[idx]
    return lm.visibility >= MIN_VIS and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0


def body_metrics(lms, w, h):
    """What the test compares, for one frame - or None when the shoulders or
    hips are unclear. A hand or an arm that is unclear reads None on its own;
    the other side and the hips still count.

      hip_y    mid-hip height on screen, pixels (down = bigger)
      torso    mid-shoulder to mid-hip, pixels
      wrist_*  wrist height on screen per hand, pixels, or None
      elbow_*  shoulder-elbow-wrist angle per arm, degrees, or None
    """
    if lms is None:
        return None
    for s in ARMS:
        if not (_clear(lms, SHOULDER[s]) and _clear(lms, HIP[s])):
            return None
    sh = (lm_px(lms, SHOULDER[SIDE_LEFT], w, h) + lm_px(lms, SHOULDER[SIDE_RIGHT], w, h)) / 2
    hp = (lm_px(lms, HIP[SIDE_LEFT], w, h) + lm_px(lms, HIP[SIDE_RIGHT], w, h)) / 2
    torso = float(np.linalg.norm(sh - hp))
    if torso < 1.0:
        return None

    out = {"hip_y": float(hp[1]), "torso": torso}
    for s in ARMS:
        wrist_ok = _clear(lms, WRIST[s])
        out["wrist_" + s] = float(lm_px(lms, WRIST[s], w, h)[1]) if wrist_ok else None
        if wrist_ok and _clear(lms, ELBOW[s]):
            out["elbow_" + s] = calc_angle(lm_px(lms, SHOULDER[s], w, h),
                                           lm_px(lms, ELBOW[s], w, h),
                                           lm_px(lms, WRIST[s], w, h))
        else:
            out["elbow_" + s] = None
    return out


class MedianSmoother:
    """Short median per measurement - kills single-frame landmark spikes. A
    hand or arm that drops out starts its window afresh when it comes back."""

    KEYS = ("hip_y", "torso", "wrist_left", "wrist_right", "elbow_left", "elbow_right")

    def __init__(self, n=SMOOTH_FRAMES):
        self.n = int(n)
        self.bufs = {k: deque(maxlen=self.n) for k in self.KEYS}

    def push(self, m):
        out = dict(m)
        for k, buf in self.bufs.items():
            if m.get(k) is None:
                buf.clear()
                out[k] = None
                continue
            buf.append(float(m[k]))
            out[k] = float(np.median(buf))
        return out

    def full(self, k):
        """True once k has a whole window behind it - a steady reading, not
        the first frame or two after it came back into view."""
        return len(self.bufs[k]) == self.n

    def reset(self):
        for buf in self.bufs.values():
            buf.clear()


def _band(vals):
    return max(vals) - min(vals) if vals else 0.0


def hand_gaps(m):
    """{side: how far the hips sit above that hand, pixels} for each hand in
    view. Grows as the body lifts; a camera shift moves both ends alike."""
    return {s: m["wrist_" + s] - m["hip_y"] for s in ARMS if m.get("wrist_" + s) is not None}


class SeatedPushUpTracker:
    """Turns per-frame body readings into the hold time.

    update() takes one body_metrics() dict (or None) and an optional injected
    clock, so the whole state machine can be driven with synthetic readings
    - no camera required.
    """

    def __init__(self, max_hold_s=MAX_HOLD_S, settle_s=SETTLE_S, seat_still_s=SEAT_STILL_S):
        self.max_hold_s = float(max_hold_s)
        self.settle_s = float(settle_s)
        self.seat_still_s = float(seat_still_s)
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        self.state = State.READY
        self.current = None           # latest smoothed readings
        self.seat = None              # {side: hip-above-hand gap while sitting, px}
        self.seat_torso = None        # torso length while sitting, pixels
        self.seat_hip_y = None        # mid-hip SCREEN height while sitting - only
                                      # for spotting a stand-up (see the notes)
        self.lift = None              # current lift above the seat, torso lengths
        self.baseline = None          # starting lift, and each arm's elbow mark
        self.hold_start = None
        self.hold_end = None
        self.end_reason = None        # plain words: why the clock stopped
        self.capped = False
        self.status_msg = "Sit facing the camera, hands on the supports"
        self._seat_win = deque()      # (t, hand gaps, torso, hip_y) while sitting
        self._settle = deque()        # (t, metrics with 'lift')
        self._smooth = MedianSmoother()
        self._bad_since = None
        self._bad_reason = None
        self._last_seen = None        # last frame the hold could be checked
        self._last_good = None        # last frame the position was still correct -
                                      # where the clock stops, whatever ends it
        self._last_frame_t = None
        self._now = None

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, body, now=None):
        """Feed one frame. Returns the result dict on the frame the hold ends,
        else None."""
        now = time.time() if now is None else now
        self._now = now
        if self.state == State.DONE:
            return None

        # Frames that simply stopped arriving (phone or network stalled) are
        # time nobody saw - the same as a tracking dropout.
        prev_t, self._last_frame_t = self._last_frame_t, now
        gap = (now - prev_t) if prev_t is not None else 0.0
        if gap > SETTLE_GAP_S:
            # Readings from before the gap say nothing about now: a median
            # still holding them would let a student who sat down during the
            # gap read as still up on the first frame back.
            self._seat_win.clear()
            self._settle.clear()
            self._smooth.reset()
            if self.state == State.HOLDING and gap > DROPOUT_GRACE_S:
                return self._end_lost(
                    "the video paused - the camera lost sight of the student")

        if body is None:
            self._smooth.reset()
            self.current = None
        else:
            self.current = self._smooth.push(body)
        m = self.current
        self.lift = self._lift_of(hand_gaps(m)) if m is not None else None

        if self.state == State.HOLDING:
            if m is not None:
                self._mark_new_arms(m)
            # A stand-up shows from the hips alone - the hands may well have
            # left the supports and the picture - so it is judged regardless.
            if not (self._stood(m) or self._checkable(m)):
                if now - self._last_seen > DROPOUT_GRACE_S:
                    return self._end_lost("the camera lost sight of the student")
                # Unseen time is never capped or credited on its own: the
                # clock only moves on if the position is seen again.
                return None
            self._last_seen = now
            return self._check_hold(m, now)

        # READY / SETTLING
        if m is None:
            self._seat_win.clear()
            self._settle.clear()
            self.state = State.READY
            self.status_msg = "Face the camera - head, both hands and hips in frame"
            return None

        if not hand_gaps(m):
            self._seat_win.clear()
            self._settle.clear()
            self.state = State.READY
            self.status_msg = "Keep both hands in view"
            return None

        self._track_seat(m, now)
        if self.seat is None:
            self.state = State.READY
            self.status_msg = "Sit still, hands on the supports"
            return None
        if self._stood(m):
            self._settle.clear()
            self.state = State.READY
            self.status_msg = "Stay seated - push up with the arms"
            return None
        if self.lift is None:           # only a hand the seat was not read on
            self._settle.clear()
            self.state = State.READY
            self.status_msg = "Keep both hands in view"
            return None

        if self.lift < LIFT_MIN_TORSO or self._screen_rise(m) < SCREEN_RISE_MIN_TORSO:
            self._settle.clear()
            self.state = State.READY
            # A student already up when the seat was taken reads as sitting
            # here, so the message covers both cases.
            self.status_msg = "Push up off the seat (already up? sit down first)"
            return None
        if all(m["elbow_" + s] is None for s in ARMS):
            # No elbow to check means no way to see the arms give way, so the
            # clock does not start on the hips alone.
            self._settle.clear()
            self.state = State.READY
            self.status_msg = "Keep both arms in view"
            return None

        self.state = State.SETTLING
        self.status_msg = "Off the seat - hold still"
        self._settle.append((now, dict(m, lift=self.lift)))
        # Keep only the steady stretch: drop samples from the front until the
        # window agrees with itself, so a wobble restarts the settle.
        while len(self._settle) > 1 and not self._window_still():
            self._settle.popleft()
        if self._settle[-1][0] - self._settle[0][0] >= self.settle_s:
            self._start_hold(now)
        return None

    # ── internals ────────────────────────────────────────────────────────
    def _lift_of(self, gaps, seat=None):
        """Lift in torso lengths, averaged over the hands that have a seat
        reading - or None when no such hand is in view."""
        seat = self.seat if seat is None else seat
        if seat is None:
            return None
        common = [s for s in gaps if s in seat]
        if not common:
            return None
        return float(np.mean([gaps[s] - seat[s] for s in common])) / self.seat_torso

    def _checkable(self, m):
        """Mid-hold: can this frame be judged? Needs the hips, a hand with a
        seat reading, and an arm with an elbow mark."""
        if m is None or self.lift is None:
            return False
        return any(self.baseline["elbow_" + s] is not None and m["elbow_" + s] is not None
                   for s in ARMS)

    def _track_seat(self, m, now):
        """Learn where the hips sit, relative to each hand. A still stretch of
        SEAT_STILL_S becomes the seat if there is none yet, or if it reads
        LOWER than the one held - the lowest still sit is the seat, whatever
        was seen first. A hand first seen during a still sit at the seat's
        level is added to it."""
        win = self._seat_win
        win.append((now, hand_gaps(m), m["torso"], m["hip_y"]))

        def still():
            # Still against the hands AND on screen - standing then sitting
            # can leave the hands-to-hips gap unchanged.
            band = SEAT_BAND_TORSO * np.median([t for _, _, t, _ in win])
            if _band([y for _, _, _, y in win]) > band:
                return False
            return all(_band([g[s] for _, g, _, _ in win if s in g]) <= band for s in ARMS)

        while len(win) > 1 and not still():
            win.popleft()
        if win[-1][0] - win[0][0] < self.seat_still_s:
            return
        # A hand counts if it was in view for at least half the stretch.
        gaps = {}
        for s in ARMS:
            vals = [g[s] for _, g, _, _ in win if s in g]
            if 2 * len(vals) >= len(win):
                gaps[s] = float(np.median(vals))
        if not gaps:
            return
        torso = float(np.median([t for _, _, t, _ in win]))
        hip_y = float(np.median([y for _, _, _, y in win]))

        if self.seat is None:
            self._set_seat(gaps, torso, hip_y, m)
            return
        lift = self._lift_of(gaps)
        screen = (self.seat_hip_y - hip_y) / self.seat_torso
        if lift is None or lift < -SEAT_BAND_TORSO:
            # Sitting lower than the seat held (or only on hands it has no
            # reading for): this is the seat now.
            self._set_seat(gaps, torso, hip_y, m)
        elif abs(lift - screen) > HAND_MOVE_TORSO and screen <= SEAT_BAND_TORSO:
            # lift - screen is how far the hands moved on screen. They moved,
            # but the hips did not: the HANDS moved (lap -> supports, or a
            # re-grip), not the body. Learn it again.
            self._set_seat(gaps, torso, hip_y, m)
        elif abs(lift) <= SEAT_BAND_TORSO:
            for s, g in gaps.items():
                if s not in self.seat:
                    self.seat[s] = g - lift * self.seat_torso
            # Sitting, by the hands - so wherever the hips are on screen now
            # is the seat on screen (after a bumped phone, or after sitting
            # down from standing). Taken at zero lift, so the two references
            # stay in step.
            self.seat_hip_y = hip_y + lift * self.seat_torso

    def _set_seat(self, gaps, torso, hip_y, m):
        self.seat, self.seat_torso, self.seat_hip_y = dict(gaps), torso, hip_y
        self.lift = self._lift_of(hand_gaps(m))
        self._settle.clear()

    def _window_still(self):
        samples = [m for _, m in self._settle]
        if _band([m["lift"] for m in samples]) > SETTLE_LIFT_BAND:
            return False
        for s in ARMS:
            vals = [m["elbow_" + s] for m in samples if m["elbow_" + s] is not None]
            if _band(vals) > SETTLE_ELBOW_BAND:
                return False
        return True

    def _start_hold(self, now):
        samples = [m for _, m in self._settle]
        self.baseline = {"lift": float(np.median([m["lift"] for m in samples]))}
        for s in ARMS:
            vals = [m["elbow_" + s] for m in samples if m["elbow_" + s] is not None]
            self.baseline["elbow_" + s] = float(np.median(vals)) if vals else None
        self.hold_start = self._settle[0][0]
        self._last_seen = now
        self._last_good = now
        self._bad_since = None
        self._settle.clear()
        self.state = State.HOLDING
        self.status_msg = "Clock running - hold it"

    @property
    def sink_line(self):
        """Lift below which the hips count as back on the seat."""
        if self.baseline is None:
            return None
        return max(SEAT_TOUCH_TORSO, SINK_FRAC * self.baseline["lift"])

    @property
    def seat_found(self):
        return self.seat is not None

    @property
    def seat_y(self):
        """Where the seat is on screen right now (pixels), from the hands in
        view - for drawing. None if it cannot be placed this frame."""
        m = self.current
        if m is None or self.seat is None:
            return None
        ys = [m["wrist_" + s] - self.seat[s] for s in ARMS
              if s in self.seat and m.get("wrist_" + s) is not None]
        return float(np.mean(ys)) if ys else None

    def _screen_rise(self, m):
        """How far the hips are above the seat on SCREEN, torso lengths."""
        return (self.seat_hip_y - m["hip_y"]) / self.seat_torso

    def _stood(self, m):
        """Hips risen past MAX_LIFT_TORSO - on screen, or against the hands."""
        if m is None or self.seat_hip_y is None:
            return False
        if self._screen_rise(m) > MAX_LIFT_TORSO:
            return True
        return self.lift is not None and self.lift > MAX_LIFT_TORSO

    def _mark_new_arms(self, m):
        """Mid-hold, an arm that was hidden when the clock started gets its
        mark from its first steady reading (a full smoothing window), so it
        can carry the hold when the other arm goes out of view."""
        for s in ARMS:
            k = "elbow_" + s
            if self.baseline[k] is None and m[k] is not None and self._smooth.full(k):
                self.baseline[k] = m[k]

    def _raise_marks(self, m):
        """An arm that straightens further than its mark raises the mark, so
        a later bend is measured from the straightest it has been. Only a
        steady reading (a full smoothing window) counts."""
        for s in ARMS:
            k = "elbow_" + s
            base = self.baseline[k]
            if (base is not None and m[k] is not None and m[k] > base
                    and self._smooth.full(k)):
                self.baseline[k] = m[k]

    def _broken(self, m):
        """The first rule this frame breaks, or None. A stand-up comes first:
        it is the one rule judged without the hands, so lift may be None."""
        if self._stood(m):
            return "stood"
        if self.lift < self.sink_line:
            return "sink"
        for s in ARMS:
            base, now_deg = self.baseline["elbow_" + s], m["elbow_" + s]
            if base is not None and now_deg is not None and now_deg < base - ELBOW_BEND_DEG:
                return "elbow"
        return None

    def _check_hold(self, m, now):
        self._raise_marks(m)
        why = self._broken(m)
        if why is None:
            self._bad_since = None
            self._bad_reason = None
            self._last_good = now
            self.status_msg = "Clock running - hold it"
            return self._check_cap(now)
        if self._bad_since is None:
            self._bad_since = now
            self._bad_reason = why
        if now - self._bad_since > BREAK_GRACE_S:
            # The hold ended at the last correct frame, not when the break
            # was confirmed.
            return self._end(self._last_good, BREAK_WORDS[self._bad_reason])
        # No cap on a frame that may be a break: if it is one, the clock
        # stops at the last good frame anyway; if not, the next good frame caps.
        return None

    def _end_lost(self, reason):
        """The picture is gone for good. Stop at the last frame the position
        was correct - and if a break had already begun, that is the reason."""
        if self._bad_since is not None:
            reason = BREAK_WORDS[self._bad_reason]
        return self._end(self._last_good, reason)

    def _check_cap(self, now):
        if self.hold_start is not None and now - self.hold_start >= self.max_hold_s:
            self.capped = True
            return self._end(self.hold_start + self.max_hold_s,
                             f"reached {self.max_hold_s:.0f} s - the maximum")
        return None

    def _end(self, when, reason):
        self.hold_end = max(self.hold_start, min(when, self.hold_start + self.max_hold_s))
        self.end_reason = reason
        self.state = State.DONE
        self.status_msg = f"Done - {self.score_s} s ({reason})"
        return self.result()

    def finish(self):
        """The coach pressed STOP. A hold under way ends at the last frame the
        position was still correct. No hold started -> None: NOT COMPLETED
        (not a zero). Already done -> that result unchanged."""
        if self.state == State.DONE:
            return self.result()
        if self.state != State.HOLDING:
            return None
        return self._end(self._last_good, "stopped by the coach")

    # ── read-only views ──────────────────────────────────────────────────
    @property
    def elapsed_s(self):
        """Seconds on the clock right now (exact)."""
        if self.hold_start is None:
            return 0.0
        end = self.hold_end if self.hold_end is not None else self._now
        if end is None:
            return 0.0
        return max(0.0, min(self.max_hold_s, end - self.hold_start))

    @property
    def score_s(self):
        """The protocol score: time held, to the nearest second."""
        return int(min(self.max_hold_s, math.floor(self.elapsed_s + 0.5)))

    @property
    def resolved(self):
        return self.state == State.DONE

    def result(self):
        return {
            "hold_sec": self.score_s,
            "held_exact_s": round(self.elapsed_s, 2),
            "end_reason": self.end_reason,
            "capped": bool(self.capped),
            "max_hold_s": self.max_hold_s,
        }


# ── HUD ──────────────────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _hud_text(txt):
    """cv2.putText only draws ASCII."""
    txt = txt.replace("—", "-").replace("–", "-").replace("°", " deg")
    return txt.encode("ascii", "replace").decode("ascii")


def _fmt_elbows(m, b=None):
    parts = []
    for s, tag in ((SIDE_LEFT, "L"), (SIDE_RIGHT, "R")):
        v = m.get("elbow_" + s)
        txt = f"{tag} {v:.0f}" if v is not None else f"{tag} --"
        if b is not None and b.get("elbow_" + s) is not None:
            txt += f" (best {b['elbow_' + s]:.0f})"
        parts.append(txt)
    return "  ".join(parts)


def draw_hud(frame, tracker, body_pts=None):
    """Draw the live state onto the frame. body_pts: {'shoulder'|'elbow'|
    'wrist'|'hip': {side: pixel point}} for the joints being read, or None."""
    h, w = frame.shape[:2]
    st = tracker.state
    colour = {State.READY: C_WARN, State.SETTLING: C_WARN,
              State.HOLDING: C_OK, State.DONE: C_OK}.get(st, C_WARN)
    if tracker._bad_since is not None and st == State.HOLDING:
        colour = C_BAD

    # The seat line and the sink line, so the coach can see what the camera
    # is comparing the hips with.
    seat_y = tracker.seat_y
    if seat_y is not None:
        sy = int(seat_y)
        cv2.line(frame, (0, sy), (w, sy), C_SEAT, 2, cv2.LINE_AA)
        cv2.putText(frame, "seat", (w - 70, sy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, C_SEAT, 2, cv2.LINE_AA)
        # Dashed: the line to clear before the clock, the line not to sink
        # below while it runs.
        stop = tracker.sink_line
        line_at = stop if stop is not None else LIFT_MIN_TORSO
        ny = int(seat_y - line_at * tracker.seat_torso)
        for x in range(0, w, 24):
            cv2.line(frame, (x, ny), (min(w, x + 12), ny), C_SEAT, 1, cv2.LINE_AA)
        # Left edge: the two lines sit only a few pixels apart, and "seat"
        # already labels the right.
        cv2.putText(frame, "stop" if stop is not None else "need", (10, ny - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_SEAT, 2, cv2.LINE_AA)

    if body_pts is not None:
        for s in ARMS:
            sh, el, wr = (body_pts[k].get(s) for k in ("shoulder", "elbow", "wrist"))
            if sh is not None and el is not None and wr is not None:
                sh, el, wr = (tuple(np.int32(p)) for p in (sh, el, wr))
                cv2.line(frame, sh, el, colour, 3, cv2.LINE_AA)
                cv2.line(frame, el, wr, colour, 3, cv2.LINE_AA)
                cv2.circle(frame, el, 8, colour, -1, cv2.LINE_AA)
        hips = [body_pts["hip"].get(s) for s in ARMS]
        if all(p is not None for p in hips):
            mid = tuple(np.int32((hips[0] + hips[1]) / 2))
            cv2.circle(frame, mid, 10, colour, -1, cv2.LINE_AA)

    # Top bar: time held out of the 20 s maximum.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = tracker.elapsed_s / tracker.max_hold_s if tracker.max_hold_s else 0.0
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * min(1.0, prog)), 10), C_OK, -1)

    draw_panel(frame, 12, 22, 430, 170)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    line(f"SEATED PUSH-UP  [{st.name}]", colour, 0.68, 30)
    line(f"Time: {tracker.elapsed_s:.1f} / {tracker.max_hold_s:.0f} s")
    m = tracker.current
    if m is not None:
        if tracker.lift is not None:
            if tracker.sink_line is not None:
                line(f"Hips up: {tracker.lift:+.2f} torso  (stops below {tracker.sink_line:.2f})")
            else:
                line(f"Hips up: {tracker.lift:+.2f} torso  (need {LIFT_MIN_TORSO:.2f})")
        else:
            line("Hips up: finding the seat")
        line(f"Elbows: {_fmt_elbows(m, tracker.baseline)}")
    else:
        line("Body not fully tracked", C_BAD)

    if tracker.status_msg:
        msg = _hud_text(tracker.status_msg)
        draw_panel(frame, 12, h - 54, min(w - 12, 12 + 13 * len(msg) + 24), h - 12)
        cv2.putText(frame, msg, (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 2, cv2.LINE_AA)

    if tracker.resolved:
        big = f"{tracker.score_s} s"
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)[0]
        org = ((w - size[0]) // 2, (h + size[1]) // 2)
        draw_panel(frame, org[0] - 30, org[1] - size[1] - 24,
                   org[0] + size[0] + 30, org[1] + 24, C_DARK, 0.72)
        cv2.putText(frame, big, org, cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_OK, 5, cv2.LINE_AA)
    return frame


def tracked_points(tracker, lms, w, h):
    """Pixel joints the tracker is reading, for the HUD."""
    if lms is None or tracker.current is None:
        return None
    pts = {}
    for name, idx in (("shoulder", SHOULDER), ("elbow", ELBOW),
                      ("wrist", WRIST), ("hip", HIP)):
        pts[name] = {s: (lm_px(lms, idx[s], w, h) if _clear(lms, idx[s]) else None)
                     for s in ARMS}
    return pts


def _stage_seated_pushup(session_id, result, rating=None):
    """Stages the result on the server for review/save on screen3 - does NOT
    write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/seated_push_up",
            json={
                "session_id":   session_id,
                "hold_sec":     result.get("hold_sec"),
                "held_exact_s": result.get("held_exact_s"),
                "end_reason":   result.get("end_reason"),
                "rating":       rating,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] seated_push_up staged for review -> "
              f"{result.get('hold_sec')} s  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Seated Push-Up Test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file")
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

    # A recording is clocked by its own frame rate, a camera by the wall
    # clock - otherwise a clip decoded slowly would be credited extra seconds.
    live = isinstance(source, int)
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not live and not (1.0 <= clip_fps <= 240.0):
        clip_fps = 30.0
        print(f"[WARN] {args.source} reports no usable frame rate - "
              f"timing it as {clip_fps:.0f} fps")

    print(f"[INFO] Seated push-up: lift off the seat and hold, max {MAX_HOLD_S:.0f} s. "
          f"R = reset  |  Q = quit\n")

    tracker = SeatedPushUpTracker()
    staged = False
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

    win = "Seated Push-Up Test"
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

            lms = results.pose_landmarks[0] if results.pose_landmarks else None
            if lms is not None:
                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(lms[i].x * fw), int(lms[i].y * fh)),
                                 (int(lms[j].x * fw), int(lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)

            outcome = tracker.update(body_metrics(lms, fw, fh), now=now)
            if outcome is not None and not staged and args.session_id:
                _stage_seated_pushup(args.session_id, outcome)
                staged = True

            if args.debug:
                m = tracker.current
                lift = f"{tracker.lift:+.3f}" if tracker.lift is not None else "  -  "
                print(f"t={now - stream_start:6.2f} state={tracker.state.name:8s} "
                      f"clock={tracker.elapsed_s:5.2f} lift={lift} "
                      + (_fmt_elbows(m) if m else "no body"))

            draw_hud(frame, tracker, tracked_points(tracker, lms, fw, fh))
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                tracker.reset()
                staged = False

    cap.release()
    cv2.destroyAllWindows()

    outcome = tracker.finish()          # same STOP rule as the web engine
    if outcome is not None and not staged and args.session_id:
        _stage_seated_pushup(args.session_id, outcome)
    print("\n-- Seated Push-Up ---------------------------")
    if tracker.resolved:
        res = tracker.result()
        print(f"  Score  : {res['hold_sec']} s  (exact {res['held_exact_s']} s)")
        print(f"  Ended  : {res['end_reason']}")
    else:
        print("  Result : NOT COMPLETED (the clock never started)")


if __name__ == "__main__":
    main()
