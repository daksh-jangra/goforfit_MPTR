"""
Bench Press Test - as many presses as possible, max 50 boys / 30 girls
======================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.69-70)
--------------------------------------------------------------------
  Lying on the back on a bench, knees bent, feet on the floor (or on mats).
  The participant holds a 35 lb (15.9 kg) barbell above the shoulders with
  the elbows bent, the bar at the chest: the ready position (fig 5.6b). On
  command they press it up to straight arms at 90 degrees to the body
  (fig 5.6c) and bring it back to the chest. They keep going without rest,
  one rep every 3-4 s, until they can't raise the bar or reach 50 reps
  (males) / 30 (females). ONE trial.

  A rep counts when the bar has touched the chest and both arms end up
  straight. The score is the number of correct reps. Spotters stand beside
  the rib cage, not behind the head.

  Grip: the manual says about shoulder width. The owner chose the wider
  1.5x shoulder-width grip from the reference video (Jeff Nippard, "How To
  Bench Press With Perfect Technique", 2026-10-22). That only changes the
  instructions; the count is the same.

What counts as a rep - the ELBOW ANGLE, side-on (owner, 2026-10-22)
--------------------------------------------------------------------
  On the arm nearest the camera:

    elbow   shoulder-elbow-wrist angle (straight ~ 160-180)
    elev    how far the upper arm is raised above the body line
            (shoulder -> hip), in degrees. 0 = lying along the body,
            90 = pointing at the ceiling.

    hand    how high the wrist is above the shoulder, across the body line
            (screen height without the hip), in this frame's forearm lengths

    CHEST   elbow <= DOWN_DEG (90) AND elev <= ELEV_DOWN_DEG (30): the bar is
            down at the chest. (Without the hip in view, the elbow alone.)
    TOP     elbow >= UP_DEG (155) AND hand >= HAND_UP_FA (1.75): arms
            straight and the bar really up.

  A rep is chest -> top, counted the moment the arms are straight. It must
  come back to the chest before the next one counts.

  Measured on the side-on frames of the reference video (pose model, pixel
  coordinates): bar on the chest elbow 73-81 with the upper arm 0-6 deg
  off the body line; halfway down 99-131; locked out at the top 155-179
  (155 when the straight arm tips toward the head). Side-on the elbow is in
  the picture plane, so it reads cleanly. Hand height was tried and is NOT
  used: with the camera a little high and angled, the bar on the chest
  still read 1.2-1.3 upper arms above the shoulder.

  Why the hand height too (review 2026-10-22, owner chose the fix): a
  side-on camera can't see how far the elbow sticks out TOWARD it. With the
  elbows flared out sideways (common with a wide grip) a bent arm looks
  straight: in a 3-D arm model the camera read 155+ while the real elbow
  was only ~118 with the elbows 75 deg out, and only ~66 (40% of the way
  up) at 90 deg out. The hand height doesn't have that blind spot: a true
  lockout puts the wrist ~2 forearms above the shoulder (the video's side-on
  lockouts read 1.83-2.11), the flared fakes 1.0-1.7. Each frame's own
  forearm is the yardstick: it stays upright under the bar at the chest and
  at the top. When the arm leans toward the lens it looks short, which only
  makes the reading higher (the video's high camera did that: 2.3-2.6) -
  measured against the longest forearm seen instead, those real lockouts
  read ~1.6 and were missed.

  Not counted (shown on screen, the count is unchanged):
    * the bar was lowered partway and pressed back up without reaching the
      chest - but only when the elbow dipped below SHALLOW_DIP_DEG (140),
      so a straight arm wobbling near 155 isn't logged as an attempt;
    * the bar was pressed partway (elbow past REBOUND_DEG, 130) and came
      back down to the chest without the arms straightening.

  The body must be lying down (shoulder-hip line within BODY_TILT_MAX_DEG
  of level) for anything to count. A student sitting up on the bench is
  "not lying down" and the rep under way is dropped.

What the coach judges (not scored here)
--------------------------------------------------------------------
  The FAR arm (one side-on camera can't see it - the manual's "both arms
  end up straight" is the coach's call for that side), the bar weight, the
  3-4 s pace, the feet and spotting.

When the test ends (Dumbbell Press rules, owner 2026-10-22)
--------------------------------------------------------------------
  * MAX_REPS: 50 for a boy, 30 for a girl. The coach picks Boy / Girl
    before START (START waits for it).
  * No rep counted within FIRST_REP_S (15 s) of START: the score is 0.
  * REST_S (4 s) without working, counted from the first rep on. Working
    means a new rep, getting back to the chest after a rep, a press still
    climbing (the elbow opening another PROGRESS_DEG beyond the most it has
    been since the bar was last at the chest), or the bar still coming down
    (the elbow closing another PROGRESS_DEG below the least since the last
    rep). The HAND height counts the same way (PROGRESS_FA, TRAVEL_FA): with
    the elbows flared the camera reads the arm as straight from the top
    down to ~40% of the way, so a slow lowering (~7 s) showed no elbow
    movement and ended the test mid-set. Getting back to the chest after ONE press that didn't lock out
    restarts the clock too (a near miss is still working); a second miss in
    a row doesn't. Each step also needs real TRAVEL (TRAVEL_DEG over TRAVEL_S,
    each end averaged over TRAVEL_AVG_S), so tracking wobble can't keep
    the test alive. A slow press or lowering that keeps going never times
    out; lying still - at the chest, holding the bar up, or stopped partway
    - does, and so do repeated presses that never get any higher ("can no
    longer raise the barbell").
  * The coach presses STOP.
  The rest clock only runs on frames where the arm is seen, and each frame
  adds at most MAX_FRAME_DT: a Wi-Fi gap can't end the test by itself.

Tracking dropouts and glitches (as push_up.py)
--------------------------------------------------------------------
  When the arm is lost the phase is kept, and the 3-frame smoothing is kept
  through a gap of up to SHORT_GAP_S. An elbow reading that JUMPS more than
  JUMP_DEG from the last trusted one (or JUMP_SPEED per second after a gap)
  is only trusted once it has held JUMP_HOLD_FRAMES readings: two wild
  frames in a row get past the median and would fake a trip to the chest
  or the top.

Camera placement
----------------
  SIDE-ON to the bench, at about bench height, 2-3 m away, with the head,
  the arm nearest the camera, the bar overhead and the hips in frame. The
  spotter stands on the FAR side of the bench, not between the camera and
  the student.

Usage
-----
  python bench_press.py --sex boy
  python bench_press.py --source clip.mp4 --sex girl --debug

  Keys : B / G = boy (50) / girl (30)
         SPACE = start counting ("GO")
         R     = reset
         Q     = quit and print the summary
"""

import argparse
import math
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision as mp_vision

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode


# ── Protocol constants ───────────────────────────────────────────────────────────
SEX_BOY, SEX_GIRL = "boy", "girl"
SEXES = (SEX_BOY, SEX_GIRL)
MAX_REPS_BY_SEX = {SEX_BOY: 50, SEX_GIRL: 30}   # "50 for males, 30 for females"
REST_S      = 4.0    # Dumbbell Press's rest rule (owner's choice)
FIRST_REP_S = 15.0   # no counted rep this long after START = score 0 (as Dumbbell Press)

# ── Detection tuning ─────────────────────────────────────────────────────────────
UP_DEG          = 155.0   # elbow at least this = arms straight (the top)...
HAND_UP_FA      = 1.75    # ...with the wrist this many forearms above the shoulder
                          # (true lockouts 1.83-2.11; flared-elbow fakes 1.0-1.7)
DOWN_DEG        = 90.0    # elbow at most this = bar at the chest...
ELEV_DOWN_DEG   = 30.0    # ...with the upper arm no more than this above the body line
SHALLOW_DIP_DEG = 140.0   # a dip from the top must go below this to be a
                          # "not down to the chest" attempt
REBOUND_DEG     = 130.0   # from the chest up past this, then back to the chest
                          # = the arms were not straightened
BODY_TILT_MAX_DEG = 50.0  # shoulder-hip line flatter than this = lying down
PROGRESS_DEG    = 15.0    # elbow opening / closing this much = still working
TRAVEL_S        = 1.2     # a "still working" step also needs the elbow to have
TRAVEL_DEG      = 10.0    # moved this far that way over this long
TRAVEL_AVG_S    = 0.5     # (each end averaged over this long)
PROGRESS_FA     = 0.20    # the same for the hand height, in forearms (chest
TRAVEL_FA       = 0.15    # ~0.9 to top ~2.0). Stress test, 3000 runs: with a
                          # shaky hand reading (+-0.15) 0.18/0.12 ended 28 late
                          # (up to 9 s), 0.20/0.15 6 (up to 7.8 s); 0.25/0.15
                          # only 2 but cut off an 8 s flared lowering, which
                          # 0.20/0.15 still allows (twice the 3-4 s pace).
MAX_FRAME_DT    = 0.5     # most one frame may add to any clock (Wi-Fi gaps)
SHORT_GAP_S     = 0.3     # arm lost no longer than this: keep the smoothing
JUMP_DEG        = 40.0    # elbow reading this far from the trusted one = a jump
JUMP_SPEED      = 300.0   # ...or further than this many deg/s since it, after a gap
JUMP_HOLD_FRAMES = 3      # ...trusted only once it has held this many readings
SIDE_SWITCH_VIS = 0.10    # the other side must be this much clearer to switch

SMOOTH_FRAMES   = 3       # median window, as in the other rep counters
MIN_VIS         = 0.50    # landmark visibility floor

SIDE_LEFT, SIDE_RIGHT = "left", "right"
SIDES = (SIDE_LEFT, SIDE_RIGHT)

# Landmark indices (33-point BlazePose topology)
SIDE_IDX = {
    #            shoulder elbow wrist hip
    SIDE_LEFT:  (11, 13, 15, 23),
    SIDE_RIGHT: (12, 14, 16, 24),
}

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (24, 26), (25, 27), (26, 28),             # legs
]

# Phases of the press
PH_UNKNOWN  = "unknown"    # not yet seen at the chest or the top
PH_UP       = "up"         # arms straight
PH_LOWERING = "lowering"   # left the top, not at the chest yet
PH_DOWN     = "down"       # bar at the chest, ready to press
PH_PRESSING = "pressing"   # left the chest, arms not straight yet

# Why a rep was not counted (key -> plain words)
NOT_COUNTED = {
    "not_chest":    "bar not down to the chest",
    "not_straight": "arms not straightened at the top",
}

END_MAX     = "max reps"
END_REST    = "rested 4 s"
END_NO_REP  = f"no rep in {FIRST_REP_S:.0f} s"
END_STOPPED = "stopped by coach"

# HUD palette (matches reverse_curl.py)
C_DARK  = (28, 28, 28)
C_OK    = (80, 220, 120)
C_WARN  = (60, 190, 250)
C_BAD   = (80, 80, 240)
C_TEXT  = (245, 245, 245)


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


def _clear(lm):
    return lm.visibility >= MIN_VIS and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0


def side_metrics(lms, side, w, h):
    """The readings for one side of the body, or None when the shoulder,
    elbow or wrist is unclear or outside the picture (MediaPipe still
    returns a confident-looking guess for joints off the edge). The hip is
    optional: without it elev and body_tilt are None, and the hand height
    is measured straight up the picture."""
    idxs = SIDE_IDX[side]
    if not all(_clear(lms[i]) for i in idxs[:3]):
        return None
    s, e, wr = (lm_px(lms, i, w, h) for i in idxs[:3])
    upper = e - s
    forearm = float(np.linalg.norm(wr - e))
    if np.linalg.norm(upper) < 1.0 or forearm < 1.0:
        return None

    elev = tilt = None
    up = np.array([0.0, -1.0])     # without the hip: straight up the picture
    if _clear(lms[idxs[3]]):
        hp = lm_px(lms, idxs[3], w, h)
        body = hp - s
        blen = float(np.linalg.norm(body))
        if blen >= 1.0:
            d = body / blen
            tilt = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
            # The side of the body line facing up the picture (the ceiling,
            # with the phone level). Upper arm raised that way = positive.
            n = np.array([d[1], -d[0]])
            if n[1] > 0:
                n = -n
            elev = math.degrees(math.asin(float(np.clip(
                np.dot(upper, n) / np.linalg.norm(upper), -1.0, 1.0))))
            up = n

    return {
        "elbow": calc_angle(s, e, wr),
        "hand": float(np.dot(wr - s, up)) / forearm,
        "elev": elev,
        "body_tilt": tilt,
        "vis": float(np.mean([lms[i].visibility for i in idxs[:3]])),
    }


def read_sides(lms, w, h):
    return {sd: side_metrics(lms, sd, w, h) for sd in SIDES}


class MedianSmoother:
    """Short median per measurement - kills single-frame landmark spikes.
    elev / body_tilt may be missing (hip out of view): their median is over
    the frames that had them, and None while this frame doesn't."""

    ALWAYS = ("elbow", "hand")
    OPTIONAL = ("elev", "body_tilt")

    def __init__(self, n=SMOOTH_FRAMES):
        self.n = int(n)
        self.bufs = {k: deque(maxlen=self.n) for k in self.ALWAYS + self.OPTIONAL}

    def push(self, m):
        out = dict(m)
        for k in self.ALWAYS:
            self.bufs[k].append(float(m[k]))
            out[k] = float(np.median(self.bufs[k]))
        for k in self.OPTIONAL:
            self.bufs[k].append(m[k])
            vals = [v for v in self.bufs[k] if v is not None]
            out[k] = float(np.median(vals)) if (m[k] is not None and vals) else None
        return out

    def reset(self):
        for buf in self.bufs.values():
            buf.clear()

    @property
    def full(self):
        return len(self.bufs["elbow"]) == self.n


class BenchPressTest:
    """The whole trial: phases, the count, the rest clock and the end rule.
    Driven entirely through update(sides, now), so it can be exercised with
    made-up readings - no camera needed."""

    def __init__(self, rest_s=REST_S, first_rep_s=FIRST_REP_S):
        self.rest_s = float(rest_s)
        self.first_rep_s = float(first_rep_s)
        self.sex = None            # SEX_BOY / SEX_GIRL, picked by the coach
        self.reset()

    def reset(self):
        # self.sex is deliberately NOT reset: it is set from outside and
        # survives RESET (same student, second attempt).
        self.phase = PH_UNKNOWN
        self.count = 0
        self.not_counted = {k: 0 for k in NOT_COUNTED}
        self.started = False
        self.start_t = None
        self.end_t = None
        self.stopped = False
        self.end_reason = None
        self.rest_elapsed = 0.0
        self._last_t = None
        self._lost_since = None    # when the arm was last lost (None = seen)
        self._e_ok = None          # last trusted elbow reading
        self._e_pend = []          # readings since a jump, not trusted yet
        self._e_ok_t = None        # when the trusted reading was taken
        self.side = None           # the side being read (MediaPipe naming)
        self._sm = MedianSmoother()
        self.m = None              # smoothed readings this frame (None = not seen)
        self.in_position = True    # lying down (unknown without the hip: assume so)
        self.problem = None        # what is wrong right now, for the screen
        self._dip = None           # lowest elbow since leaving the top
        self._peak = None          # highest elbow since leaving the chest
        self._best = None          # highest elbow since back at the chest (rest clock)
        self._low = None           # lowest elbow since the last rep (rest clock)
        self._climbed = False      # credited a climb since back at the chest
        self._hbest = None         # the same two ratchets for the hand height
        self._hlow = None
        self._misses = 0           # presses not locked out since the last rep
        self._trail = deque()      # (t, elbow, hand), last TRAVEL_S + TRAVEL_AVG_S
        self.status_msg = ("Press START when ready" if self.sex
                           else "Pick Boy or Girl, then press START")

    # ── read-outs ───────────────────────────────────────────────────────
    @property
    def tracked(self):
        return self.m is not None

    @property
    def max_reps(self):
        return MAX_REPS_BY_SEX.get(self.sex)

    @property
    def total_not_counted(self):
        return sum(self.not_counted.values())

    def elapsed(self, now=None):
        if not self.started:
            return 0.0
        end = self.end_t if self.stopped else (time.time() if now is None else now)
        return max(0.0, end - self.start_t)

    def rest_progress(self):
        if not self.started or self.stopped or self.count == 0:
            return 0.0
        return min(1.0, self.rest_elapsed / self.rest_s)

    def start_problem(self):
        """Why START can't go yet (plain words), or None."""
        if self.started:
            return "Test already started"
        if self.sex not in SEXES:
            return "Pick Boy or Girl first"
        return None

    # ── control ─────────────────────────────────────────────────────────
    def set_sex(self, sex):
        """Boy (max 50) or girl (max 30). Only before START: changing it
        mid-trial would move the finish line. Returns False if refused."""
        if self.started or sex not in SEXES:
            return False
        self.sex = sex
        self.status_msg = "Press START when ready"
        return True

    def start(self, now=None):
        if self.start_problem():
            return False
        self.started = True
        self.start_t = time.time() if now is None else now
        self.rest_elapsed = 0.0
        self.status_msg = "Go - press the bar up to straight arms"
        return True

    def stop(self, reason=END_STOPPED, now=None):
        if self.stopped or not self.started:
            return
        self.stopped = True
        self.end_reason = reason
        self.end_t = time.time() if now is None else now

    # ── per frame ───────────────────────────────────────────────────────
    def _pick(self, sides):
        """The nearer (clearer) side. Stays on the current side unless it is
        lost or the other is clearly better, so it doesn't flicker."""
        cands = {sd: m for sd, m in sides.items() if m is not None}
        if not cands:
            return None
        best = max(cands, key=lambda sd: cands[sd]["vis"])
        cur = cands.get(self.side)
        if cur is not None and cands[best]["vis"] < cur["vis"] + SIDE_SWITCH_VIS:
            best = self.side
        if best != self.side:
            self.side = best
            self._sm.reset()    # the other side's numbers must not blend in
            self._e_ok, self._e_pend = None, []
        return cands[best]

    def update(self, sides, now=None):
        """sides: {SIDE_LEFT: side_metrics|None, SIDE_RIGHT: ...}.
        Returns the end reason on the frame the test ends, else None."""
        now = time.time() if now is None else now
        dt = 0.0 if self._last_t is None else min(MAX_FRAME_DT, max(0.0, now - self._last_t))
        self._last_t = now
        if self.stopped:
            return None
        counting = self.started

        # No rep at all: after FIRST_REP_S the score is 0. Checked on every
        # frame, seen or not, and timed from START.
        if counting and self.count == 0 and now - self.start_t >= self.first_rep_s:
            self.stop(END_NO_REP, self.start_t + self.first_rep_s)
            self.status_msg = f"No rep in {self.first_rep_s:.0f} s - score 0"
            return END_NO_REP

        raw = self._pick(sides)
        if raw is None:
            # Not seen. The phase is kept - the arm is still wherever it
            # was. The smoothing is kept for a short gap, dropped after.
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since > SHORT_GAP_S:
                self._sm.reset()
                self._e_ok, self._e_pend = None, []
            self.m = None
            return None
        self._lost_since = None
        m = self._sm.push(raw)
        if not self._sm.full:
            self.m = None          # just (re)found: one wild frame would decide
            return None
        if not self._trust(m["elbow"], now):
            self.m = None          # a jump not yet confirmed
            return None
        self.m = m

        e, h = m["elbow"], m["hand"]
        self._trail.append((now, e, h))
        while self._trail and now - self._trail[0][0] > TRAVEL_S + TRAVEL_AVG_S:
            self._trail.popleft()

        ev = self._step(m, counting)
        if not counting:
            return None
        if ev == "rep":
            self.rest_elapsed = 0.0
            self._misses = 0
            self._best = self._hbest = None   # nothing to climb until it's back down
            self._low, self._hlow = e, h      # the lowering is measured from here
            self.status_msg = f"Rep {self.count}"
            if self.count >= self.max_reps:
                self.stop(END_MAX, now)
                self.status_msg = f"{self.max_reps} reps - test complete"
                return END_MAX
            return None
        if ev == "armed" or (ev == "not_counted" and self.phase == PH_DOWN
                             and self._misses == 1):
            # Back at the chest. After ONE press that didn't lock out, too:
            # a press that just misses the line is still working, and its
            # lowering plus the next press ran past 4 s and ended the test
            # mid-set (stress test: 1 session in 60). A second miss in a row
            # gets no such restart - "can no longer raise the barbell".
            self.rest_elapsed = 0.0
            self._best, self._hbest = e, h
            self._climbed = False
            return None
        if self.count == 0:
            return None             # nothing to rest from before the first rep
        if self._working(e, h, now):
            self.rest_elapsed = 0.0
            return None
        self.rest_elapsed += dt
        if self.rest_elapsed >= self.rest_s:
            self.stop(END_REST, now)
            self.status_msg = f"Rested {self.rest_s:.0f} s - test over"
            return END_REST
        return None

    def _working(self, e, h, now):
        """A slow press still climbing, or the bar still coming down (see
        the docstring), judged on the elbow and on the hand height.
        Ratchets: the same height again isn't progress."""
        climbing = self.phase in (PH_DOWN, PH_PRESSING)
        lowering = (self.phase in (PH_UP, PH_LOWERING)
                    or (self.phase == PH_DOWN and not self._climbed))
        working = False
        if climbing and self._best is not None:
            if not self._climbed and (e < self._best or h < self._hbest):
                # Still settling onto the chest: the climb is measured from
                # the real bottom, not from where the elbow crossed 90.
                self._best = min(self._best, e)
                self._hbest = min(self._hbest, h)
            else:
                if e >= self._best + PROGRESS_DEG and self._travelled(now, 1) >= TRAVEL_DEG:
                    self._best = e
                    working = True
                if h >= self._hbest + PROGRESS_FA and self._travelled(now, 2) >= TRAVEL_FA:
                    self._hbest = h
                    working = True
                if working:
                    self._climbed = True
                    return True
        if lowering and self._low is not None:
            if e <= self._low - PROGRESS_DEG and self._travelled(now, 1) <= -TRAVEL_DEG:
                self._low = e
                working = True
            if h <= self._hlow - PROGRESS_FA and self._travelled(now, 2) <= -TRAVEL_FA:
                self._hlow = h
                working = True
        return working

    def _travelled(self, now, i):
        """How far the elbow (i=1) or the hand (i=2) has really moved over
        the last TRAVEL_S: the mean of the last TRAVEL_AVG_S minus the mean
        TRAVEL_S before that. Up / opening is positive. Jitter nets out near
        0. 0 until there is enough history."""
        if not self._trail or now - self._trail[0][0] < TRAVEL_S:
            return 0.0
        recent = [x[i] for x in self._trail if x[0] >= now - TRAVEL_AVG_S]
        old = [x[i] for x in self._trail if x[0] <= now - TRAVEL_S]
        if not recent or not old:
            return 0.0
        return sum(recent) / len(recent) - sum(old) / len(old)

    def _trust(self, e, now):
        """The elbow the phases run on (as push_up.py). A reading that jumps
        further than a real press moves is only trusted once it has held
        JUMP_HOLD_FRAMES readings; if it comes back first it was a glitch.
        Returns False for a reading not trusted (yet)."""
        if self._e_ok is None:
            self._e_ok, self._e_ok_t = e, now
            return True
        limit = max(JUMP_DEG, JUMP_SPEED * (now - self._e_ok_t))
        if not self._e_pend:
            if abs(e - self._e_ok) <= limit:
                self._e_ok, self._e_ok_t = e, now
                return True
            self._e_pend = [e]
            return False
        if abs(e - self._e_ok) <= JUMP_DEG:
            self._e_pend = []           # came back: it was a glitch
            self._e_ok, self._e_ok_t = e, now
            return True
        if abs(e - self._e_pend[-1]) > JUMP_DEG:
            self._e_pend = [e]          # jumped again: start over
            return False
        self._e_pend.append(e)
        if len(self._e_pend) >= JUMP_HOLD_FRAMES:
            self._e_ok, self._e_ok_t, self._e_pend = e, now, []
            return True
        return False

    def _step(self, m, counting):
        """The phase machine. Returns "rep", "armed", "not_counted" or None."""
        if m["body_tilt"] is not None:
            self.in_position = m["body_tilt"] <= BODY_TILT_MAX_DEG
        if not self.in_position:
            # Sitting up or standing: the rep under way is dropped.
            self.phase = PH_UNKNOWN
            self.problem = "Lie down on the bench"
            return None
        self.problem = None

        e = m["elbow"]
        at_top = e >= UP_DEG and m["hand"] >= HAND_UP_FA
        at_chest = e <= DOWN_DEG and (m["elev"] is None or m["elev"] <= ELEV_DOWN_DEG)

        if self.phase == PH_UNKNOWN:
            if at_chest:
                self.phase = PH_DOWN
            elif at_top:
                self.phase = PH_UP
            return None

        if self.phase == PH_UP:
            if at_top:
                return None
            self._dip = e
            self.phase = PH_LOWERING

        if self.phase == PH_LOWERING:
            self._dip = min(self._dip, e)
            if at_chest:
                self.phase = PH_DOWN
                return "armed"
            if at_top:
                self.phase = PH_UP
                if self._dip > SHALLOW_DIP_DEG:
                    return None    # a wobble at the top, not an attempt
                return self._reject("not_chest", counting)
            return None

        if self.phase == PH_DOWN:
            if e <= DOWN_DEG:
                return None
            self._peak = e
            self.phase = PH_PRESSING

        # PH_PRESSING
        if at_top:
            self.phase = PH_UP
            if not counting:
                return None
            self.count += 1
            return "rep"
        self._peak = max(self._peak, e)
        if at_chest:
            self.phase = PH_DOWN
            if self._peak >= REBOUND_DEG:
                return self._reject("not_straight", counting)
        return None

    def _reject(self, why, counting):
        if not counting:
            return None
        self.not_counted[why] += 1
        if why == "not_straight":
            self._misses += 1
        self.status_msg = f"Not counted - {NOT_COUNTED[why]}"
        return "not_counted"

    def result(self, now=None):
        return {
            "reps": self.count,
            "sex": self.sex,
            "duration_sec": round(self.elapsed(now), 1),
            "end_reason": self.end_reason,
            "not_counted": dict(self.not_counted),
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


def body_points(lms, side, w, h):
    """Shoulder, elbow, wrist, hip of the side being read (hip None when
    it isn't clearly seen)."""
    idxs = SIDE_IDX[side]
    hip = lm_px(lms, idxs[3], w, h) if _clear(lms[idxs[3]]) else None
    return [lm_px(lms, i, w, h) for i in idxs[:3]] + [hip]


def draw_hud(frame, test, pts=None, now=None):
    h, w = frame.shape[:2]
    colour = C_BAD if test.problem else (C_OK if test.phase == PH_UP else C_WARN)

    if pts is not None:
        s, e, wr = (tuple(np.int32(p)) for p in pts[:3])
        for p1, p2 in ((s, e), (e, wr)):
            cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)
        if pts[3] is not None:
            cv2.line(frame, s, tuple(np.int32(pts[3])), colour, 3, cv2.LINE_AA)
        cv2.circle(frame, e, 9, colour, -1, cv2.LINE_AA)

    # Top bar: the rest clock filling toward the 4 s end.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = test.rest_progress()
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 10), C_BAD if prog > 0.6 else C_WARN, -1)

    draw_panel(frame, 12, 22, 430, 200)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _hud_text(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    state = "DONE" if test.stopped else "COUNTING" if test.started else "READY"
    line(f"BENCH PRESS  [{state}]", colour, 0.68, 30)
    line(f"Reps: {test.count} / {test.max_reps or '--'}", C_OK, 0.9, 34)
    if test.m is not None:
        line(f"Elbow {test.m['elbow']:.0f}/{UP_DEG:.0f}  Hand {test.m['hand']:.1f}/{HAND_UP_FA}")
        line(f"Phase: {test.phase}   Not counted: {test.total_not_counted}")
    else:
        line("Arm not tracked - side-on, head to hips in frame", C_BAD)
    if test.started and not test.stopped and test.count > 0:
        line(f"Rest: {test.rest_elapsed:.1f} / {test.rest_s:.0f} s")
    elif test.started and not test.stopped:
        line(f"First rep: {test.elapsed(now):.0f} / {test.first_rep_s:.0f} s")

    if test.problem and test.m is not None:
        warn = _hud_text(test.problem)
        size = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        x0 = max(12, (w - size[0]) // 2 - 14)
        draw_panel(frame, x0, 214, min(w - 12, x0 + size[0] + 28), 254, C_BAD, 0.8)
        cv2.putText(frame, warn, (x0 + 14, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    C_TEXT, 2, cv2.LINE_AA)

    if test.status_msg:
        msg = _hud_text(test.status_msg)
        draw_panel(frame, 12, h - 54, min(w - 12, 12 + 13 * len(msg) + 24), h - 12)
        cv2.putText(frame, msg, (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 2, cv2.LINE_AA)
    return frame


def _stage_bench_press(session_id, res):
    """Stages the result on the server for human review/save on screen3 -
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/bench_press",
            json={
                "session_id":   session_id,
                "reps":         res["reps"],
                "sex":          res["sex"],
                "duration_sec": res["duration_sec"],
                "end_reason":   res["end_reason"],
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] bench_press staged for review → {res['reps']} reps  "
              f"session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bench Press Test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file")
    parser.add_argument("--sex", choices=SEXES, default=None,
                        help="boy (max 50) or girl (max 30); or press B / G")
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
    # clock: the 4 s rest must be the participant's, not the decoder's.
    live = isinstance(source, int)
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not live and not (1.0 <= clip_fps <= 240.0):
        clip_fps = 30.0

    print("[INFO] Side-on camera at bench height, spotter on the far side. "
          "B / G = boy / girl, SPACE = start, R = reset, Q = quit\n")

    test = BenchPressTest()
    if args.sex:
        test.set_sex(args.sex)
    staged = False
    stream_start = time.time()
    last_now = stream_start
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

    win = "Bench Press Test"
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

            sides = {SIDE_LEFT: None, SIDE_RIGHT: None}
            lms = None
            if results.pose_landmarks:
                lms = results.pose_landmarks[0]
                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(lms[i].x * fw), int(lms[i].y * fh)),
                                 (int(lms[j].x * fw), int(lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)
                sides = read_sides(lms, fw, fh)

            ended = test.update(sides, now=now)
            last_now = now
            if ended and not staged and args.session_id:
                _stage_bench_press(args.session_id, test.result(now))
                staged = True

            pts = None
            if test.tracked and lms is not None and test.side:
                pts = body_points(lms, test.side, fw, fh)
            if args.debug and test.m is not None:
                m = test.m
                elev = "-" if m["elev"] is None else f"{m['elev']:.0f}"
                print(f"side={test.side} phase={test.phase} elbow={m['elbow']:.0f} "
                      f"hand={m['hand']:.2f} elev={elev} reps={test.count} rest={test.rest_elapsed:.1f}")
            draw_hud(frame, test, pts, now=now)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                test.set_sex(SEX_BOY)
            elif key == ord("g"):
                test.set_sex(SEX_GIRL)
            elif key == ord(" "):
                why = test.start_problem()
                if why:
                    print(f"[START] {why}")
                else:
                    test.start(now)
            elif key == ord("r"):
                test.reset()
                staged = False

    cap.release()
    cv2.destroyAllWindows()

    # The last frame's time, not the wall clock (see dumbbell_press.py).
    test.stop(END_STOPPED, now=last_now)
    res = test.result(now=last_now)
    if test.started and not staged and args.session_id:
        _stage_bench_press(args.session_id, res)
    print("\n── Bench Press ──────────────────────────────")
    print(f"  Reps          : {res['reps']} / {test.max_reps or '--'}")
    print(f"  Boy / girl    : {res['sex'] or '--'}")
    print(f"  Time          : {res['duration_sec']} s")
    print(f"  Ended         : {res['end_reason'] or '--'}")
    for k, n in res["not_counted"].items():
        if n:
            print(f"  Not counted   : {n} x {NOT_COUNTED[k]}")


if __name__ == "__main__":
    main()
