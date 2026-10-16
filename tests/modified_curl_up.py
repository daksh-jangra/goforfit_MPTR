"""
Modified Curl-Up Test - as many curl-ups as possible at 1 every 3 s, up to 75
=============================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.70-72)
--------------------------------------------------------------------
  The Curl-Up (p.70-71) with the hands on the thighs. Supine on a mat,
  knees bent about 140 degrees, feet flat on the floor, legs slightly
  apart, HANDS ON THE FRONT OF THE THIGHS. Curl up slowly, sliding the
  hands along the thighs until the fingertips touch the kneecaps (about
  10 cm, or beyond if necessary - fig 5.8a), then return to the start.
  The tester calls the cadence, about one curl every 3 seconds. ONE trial.

  The score is the number of correct curl-ups, up to 75. One is counted
  for every return to the supine position. Not counted if the feet
  completely leave the floor at any time, if the fingertips don't reach
  the required distance, if the participant doesn't return to the start
  position, or any other incorrect curl-up. The test ends when the
  participant can't keep the pace or completes 75.

What the camera checks (owner's choices, 2026-10-15)
--------------------------------------------------------------------
  Camera SIDE-ON at floor level, whole body in frame INCLUDING THE FEET.
  On the side nearest the camera:

    trunk   hip->shoulder line against the floor, degrees. The floor is
            the line through the hip and the heel - both lie on the mat -
            so a tilted phone doesn't move it. Lying reads about -5 to
            -14 on the manual's photo (fig 5.7a; the hip joint sits higher
            than the heel), curled up +16 to +33 (fig 5.7b, 5.8b).
    reach   the fingertip (MediaPipe's index-finger point) along the
            thigh, hip = 0, knee = 1. Touching the kneecap read 0.93-1.04
            on fig 5.8b's crops.
    feet    heel and toe height above where they rested, against the hip,
            in thigh lengths.

  Lengths are in thigh lengths, so the student's distance doesn't matter.

  A REP is one cycle from lying and back:
    LYING   the trunk within RETURN_DEG of the student's own lying level
            (learned while they lie still; the first time, within
            LYING_MAX_DEG of the floor)
    UP      the trunk rises RISE_DEG above lying - an attempt has started
    REACHED the fingertip at REACH_MIN of the way to the knee AND slid at
            least SLIDE_MIN forward from where it rested while lying
  counted when the trunk is back down to lying, unless during that cycle
  the FEET left the floor: heel AND toe of one foot both more than
  FOOT_LIFT above their resting place for FAULT_MIN_S. Those show
  "Not counted - <why>".
  Coming back down without reaching is "fingertips didn't reach the knees";
  reaching, sliding the hands at least halfway back (REBOUND_FRAC) and
  reaching again - clearly forward again, REBOUND_HYST from their lowest -
  without lying down is "didn't lie back down".

  While a curl-up is under way the floor line is the one learned lying
  down, frozen: a lifting heel would tilt the live hip-heel line, hiding
  its own lift and inflating the trunk angle.

  The rest position is measured while the student is genuinely lying -
  not where the hands are when the curl starts (by then they've moved).
  Only the trunk rising starts an attempt: straightening bent arms while
  lying slides the hands but is not a curl-up and counts nothing.

  The feet are judged on the foot nearest the camera (the far one hides
  behind it side-on). If no foot is clearly in view the count still runs
  and the screen says the feet can't be checked - the coach watches them.

  Not checked here (the coach's call): the head touching the mat, jerky or
  "any other incorrect" curl-ups.

  Before START the phases are tracked too, so the camera already knows
  the student is lying down. A student not lying when START is pressed
  must lie down first: the first curl-up starts from lying.

When the test ends
--------------------------------------------------------------------
  * 75 curl-ups (the maximum).
  * No curl-up for GAP_S (6 s = two 3 s beats), from START or from the
    last one: "can't keep the pace". A curl-up that was NOT counted still
    resets it: the student is still going, and one bad one in a good set
    must not end the test. A student who keeps going with every one wrong is the
    coach's STOP.
  * The coach presses STOP.
  The gap clock only runs on frames where the body is seen, and each frame
  adds at most MAX_FRAME_DT, so a Wi-Fi gap can't end the test by itself.
  The 3 s beat itself is played by the browser (screen4); this file only
  ends the test.

Usage
-----
  python modified_curl_up.py
  python modified_curl_up.py --source clip.mp4 --debug

  Keys : SPACE = start counting ("GO")
         R     = reset
         Q     = quit and print the summary

Camera placement
----------------
  Phone on the floor (or just above it), SIDE-ON to the participant,
  1.5-2.5 m away, with the whole body from the head to the feet in frame.
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
CADENCE_S = 3.0    # "one curl every 3 seconds" (the browser beeps it)
GAP_S     = 6.0    # no curl-up for two beats ends the test
MAX_REPS  = 75     # the test maximum

# ── Detection tuning ─────────────────────────────────────────────────────────────
LYING_MAX_DEG   = 10.0   # trunk at most this = lying, before the student's own
                         # lying level is known (photo: lying -5..-14, up +16..+33)
RISE_DEG        = 10.0   # trunk this far above lying = a curl-up has started
RETURN_DEG      = 6.0    # trunk back within this of lying = returned to the mat
REACH_MIN       = 0.85   # fingertip this far along the thigh = at the kneecap
                         # (fig 5.8b read 0.93-1.04 touching it)
SLIDE_MIN       = 0.12   # ...and slid at least this far from where it rested
                         # (the protocol's ~10 cm is ~0.25-0.3 of a child's thigh)
REBOUND_FRAC    = 0.5    # hands slid back this share of the way to rest, then
                         # reach again without lying down = "didn't lie back down"
REBOUND_HYST    = 0.10   # ...and came forward again at least this far from
                         # their lowest (one shaky fingertip is not a re-reach)
REST_FALLBACK   = 0.65   # fingertip rest before one is learned (~10 cm short)
FOOT_LIFT       = 0.15   # heel AND toe this far above rest (thigh lengths,
                         # ~5 cm) = the foot left the floor
FAULT_MIN_S     = 0.2    # a fault must be seen this long within one curl-up
FLOOR_TILT_MAX_DEG = 35.0  # hip-heel line flatter than this = lying on the mat
REST_WINDOW_S   = 1.0    # rest position = the last second spent lying...
REST_MIN_S      = 0.25   # ...once at least this much of it has been seen
REF_STILL_DEG   = 6.0    # a lying second this steady may set the lying level
MAX_FRAME_DT    = 0.5    # most one frame may add to any clock (Wi-Fi gaps)
SIDE_SWITCH_VIS = 0.10   # the other side must be this much clearer to switch

SMOOTH_FRAMES   = 3      # median window, as in the other rep counters
MIN_VIS         = 0.50   # landmark visibility floor

SIDE_LEFT, SIDE_RIGHT = "left", "right"
SIDES = (SIDE_LEFT, SIDE_RIGHT)
OTHER = {SIDE_LEFT: SIDE_RIGHT, SIDE_RIGHT: SIDE_LEFT}

# Landmark indices (33-point BlazePose topology)
SIDE_IDX = {
    SIDE_LEFT:  {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27,
                 "heel": 29, "toe": 31, "wrist": 15, "index": 19},
    SIDE_RIGHT: {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28,
                 "heel": 30, "toe": 32, "wrist": 16, "index": 20},
}
CORE = ("shoulder", "hip", "knee", "index")   # needed to read a side at all

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (24, 26), (25, 27), (26, 28),             # legs
    (27, 29), (29, 31), (27, 31),                       # left foot
    (28, 30), (30, 32), (28, 32),                       # right foot
]

# Phases of the curl-up
PH_UNKNOWN = "unknown"    # not yet seen lying
PH_DOWN    = "down"       # lying on the mat
PH_RISING  = "rising"     # curling up, fingertips not at the knees yet
PH_REACHED = "reached"    # fingertips at the knees; counts once back down

# Why a curl-up was not counted (key -> plain words)
NOT_COUNTED = {
    "short":    "fingertips didn't reach the knees",
    "not_down": "didn't lie back down",
    "feet":     "feet left the floor",
}

END_GAP     = "no curl-up for 6 s"
END_MAX     = "reached 75"
END_STOPPED = "stopped by coach"

# HUD palette (matches reverse_curl.py)
C_DARK  = (28, 28, 28)
C_OK    = (80, 220, 120)
C_WARN  = (60, 190, 250)
C_BAD   = (80, 80, 240)
C_TEXT  = (245, 245, 245)


# ── geometry ─────────────────────────────────────────────────────────────────────
def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _clear(lm):
    return lm.visibility >= MIN_VIS and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0


def read_points(lms, w, h):
    """{side: {name: pixel point or None}} for every point the test uses;
    a point is None when unclear or outside the picture (MediaPipe still
    returns a confident-looking guess for joints off the edge). "vis" is the
    side's mean visibility over its CORE points."""
    out = {}
    for sd in SIDES:
        idx = SIDE_IDX[sd]
        pts = {name: (lm_px(lms, i, w, h) if _clear(lms[i]) else None)
               for name, i in idx.items()}
        pts["vis"] = float(np.mean([lms[idx[n]].visibility for n in CORE]))
        out[sd] = pts
    return out


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


def _up_normal(d):
    """The floor's perpendicular that points up the picture."""
    n = np.array([d[1], -d[0]])
    return n if n[1] < 0 else -n


def measure(pts, side, floor_d=None):
    """What the test compares, from one side's points (with the other side's
    heel/ankle as a fallback floor anchor). None when the side can't be read.
    floor_d: the floor direction (feet -> head, unit, pixels) to measure
    against instead of this frame's hip-heel line - the test freezes it
    during a curl-up, because a lifting heel tilts that line and would hide
    its own lift (and inflate the trunk angle).
    Foot heights (heel_left, toe_left, heel_right, toe_right) are
    hip-relative and raw - the test subtracts each foot's own rest; None
    when that foot's heel or toe is unclear."""
    p = pts.get(side)
    if p is None or any(p[n] is None for n in CORE):
        return None
    o = pts.get(OTHER[side]) or {}
    # Heels first (both lie at the same height on the mat), ankles only
    # when no heel is clear - a switch between them tilts the line ~4 deg.
    anchor = next((q for q in (p["heel"], o.get("heel"), p["ankle"], o.get("ankle"))
                   if q is not None), None)
    if anchor is None:
        return None
    sh, hp, kn, ix = p["shoulder"], p["hip"], p["knee"], p["index"]
    thigh_v = kn - hp
    thigh = float(np.linalg.norm(thigh_v))
    cur_d = _unit(hp - anchor)        # along the floor, feet -> head
    if thigh < 1.0 or cur_d is None:
        return None
    d = cur_d if floor_d is None else floor_d
    n = _up_normal(d)
    v = sh - hp

    out = {
        "trunk": math.degrees(math.atan2(float(v @ n), float(v @ d))),
        "reach": float((ix - hp) @ thigh_v / (thigh * thigh)),
        "floor_tilt": math.degrees(math.atan2(abs(cur_d[1]), abs(cur_d[0]))),
        "floor_d": cur_d,
        "vis": p["vis"],
    }
    # Each foot on its own: heel and toe height, or None when unclear.
    for sd in SIDES:
        q = pts.get(sd) or {}
        clear = q.get("heel") is not None and q.get("toe") is not None
        for part in ("heel", "toe"):
            out[f"{part}_{sd}"] = float((q[part] - hp) @ n / thigh) if clear else None
    return out


class MedianSmoother:
    """Short median per measurement - kills single-frame landmark spikes.
    The foot heights may be missing (foot out of view): their median is over
    the frames that had them, and None while this frame doesn't."""

    KEYS = ("trunk", "reach", "floor_tilt")
    OPT = tuple(f"{part}_{sd}" for sd in SIDES for part in ("heel", "toe"))

    def __init__(self, n=SMOOTH_FRAMES):
        self.n = int(n)
        self.bufs = {k: deque(maxlen=self.n) for k in self.KEYS + self.OPT}

    def push(self, m):
        out = dict(m)
        for k in self.KEYS:
            self.bufs[k].append(float(m[k]))
            out[k] = float(np.median(self.bufs[k]))
        for k in self.OPT:
            self.bufs[k].append(m[k])
            vals = [v for v in self.bufs[k] if v is not None]
            out[k] = float(np.median(vals)) if (m[k] is not None and vals) else None
        return out

    def reset(self):
        for buf in self.bufs.values():
            buf.clear()

    @property
    def full(self):
        return len(self.bufs["trunk"]) == self.n


class ModifiedCurlUpTest:
    """The whole trial: phases, the feet check, the count, the gap clock and
    the end. Driven entirely through update(pts, now), so it can be
    exercised with made-up body points - no camera needed."""

    def __init__(self, gap_s=GAP_S):
        self.gap_s = float(gap_s)
        self.reset()

    def reset(self):
        self.phase = PH_UNKNOWN
        self.count = 0
        self.not_counted = {k: 0 for k in NOT_COUNTED}
        self.started = False
        self.start_t = None
        self.end_t = None
        self.stopped = False
        self.end_reason = None
        self.gap_elapsed = 0.0
        self._last_t = None
        self.side = None            # the side being read (MediaPipe naming)
        self._sm = MedianSmoother()
        self.m = None               # smoothed readings this frame (None = not seen)
        self.in_position = False    # lying on the mat, side-on
        self.feet_seen = False      # a foot clear enough to check this frame
        self.feet_up = False        # ...and off the floor right now
        self.problem = None         # what is wrong right now, for the screen
        # Learned while lying still: the lying trunk level, where the
        # fingertip rests, where the foot rests.
        self.lying_ref = None
        self.rest_reach = None
        self.rest_feet = {}         # "heel_left" etc. -> rest height
        self.floor_d = None         # the floor direction while lying
        self._rest = deque()        # (t, smoothed readings) while lying
        self._feet_t = 0.0          # feet-off-the-floor time this curl-up
        self._slid_back = False     # hands slid back since reaching
        self._low = None            # ...and the lowest they went since
        self.peak_reach = None      # furthest fingertip this curl-up
        self.status_msg = "Press START when ready"

    # ── read-outs ───────────────────────────────────────────────────────
    @property
    def tracked(self):
        return self.m is not None

    @property
    def total_not_counted(self):
        return sum(self.not_counted.values())

    def elapsed(self, now=None):
        if not self.started:
            return 0.0
        end = self.end_t if self.stopped else (time.time() if now is None else now)
        return max(0.0, end - self.start_t)

    def gap_progress(self):
        if not self.started or self.stopped:
            return 0.0
        return min(1.0, self.gap_elapsed / self.gap_s)

    def rise(self):
        """Trunk above the lying level, degrees (None until known)."""
        if self.m is None or self.lying_ref is None:
            return None
        return self.m["trunk"] - self.lying_ref

    # ── control ─────────────────────────────────────────────────────────
    def start(self, now=None):
        if self.started:
            return False
        self.started = True
        self.start_t = time.time() if now is None else now
        self.gap_elapsed = 0.0
        # The first curl-up starts from lying. Caught partway, the student
        # has to lie down first.
        if self.phase != PH_DOWN:
            self.phase = PH_UNKNOWN
        self._clear_rep()
        self.status_msg = "Go - one curl-up every beep"
        return True

    def stop(self, reason=END_STOPPED, now=None):
        if self.stopped or not self.started:
            return
        self.stopped = True
        self.end_reason = reason
        self.end_t = time.time() if now is None else now

    def _clear_rep(self):
        self._feet_t = 0.0
        self._slid_back = False
        self._low = None
        self.peak_reach = None

    # ── per frame ───────────────────────────────────────────────────────
    def _pick(self, pts):
        """The nearer (clearer) side. Stays on the current side unless it is
        lost or the other is clearly better, so it doesn't flicker."""
        frozen = self.floor_d if self.phase in (PH_RISING, PH_REACHED) else None
        cands = {sd: measure(pts, sd, frozen) for sd in SIDES}
        cands = {sd: m for sd, m in cands.items() if m is not None}
        if not cands:
            return None
        best = max(cands, key=lambda sd: cands[sd]["vis"])
        cur = cands.get(self.side)
        if cur is not None and cands[best]["vis"] < cur["vis"] + SIDE_SWITCH_VIS:
            best = self.side
        if best != self.side:
            self.side = best
            self._sm.reset()    # the other side's numbers must not blend in
        return cands[best]

    def update(self, pts, now=None):
        """pts: read_points() output (or None when no pose).
        Returns the end reason on the frame the test ends, else None."""
        now = time.time() if now is None else now
        dt = 0.0 if self._last_t is None else min(MAX_FRAME_DT, max(0.0, now - self._last_t))
        self._last_t = now
        if self.stopped:
            return None

        raw = self._pick(pts) if pts else None
        if raw is None:
            # Not seen. The phase is kept - the body is still wherever it
            # was - but stale frames must not blend into the next reading.
            self._sm.reset()
            self.m = None
            return None
        m = self._sm.push(raw)
        if not self._sm.full:
            # Just (re)found: one wild frame would decide on its own.
            self.m = None
            return None
        self.m = m

        ev = self._step(m, dt, now, self.started)
        if not self.started:
            return None
        if ev == "rep" and self.count >= MAX_REPS:
            self.stop(END_MAX, now)
            self.status_msg = f"{MAX_REPS} curl-ups - test complete"
            return END_MAX
        if ev in ("rep", "not_counted"):
            # A curl-up attempt, counted or not: still going.
            self.gap_elapsed = 0.0
            return None
        self.gap_elapsed += dt
        if self.gap_elapsed >= self.gap_s:
            self.stop(END_GAP, now)
            self.status_msg = f"No curl-up for {self.gap_s:.0f} s - test over"
            return END_GAP
        return None

    def _learn_rest(self, m, now):
        """While lying: the lying level, the fingertip's and the foot's rest."""
        self._rest.append((now, m))
        while self._rest and now - self._rest[0][0] > REST_WINDOW_S:
            self._rest.popleft()
        if now - self._rest[0][0] < REST_MIN_S - 1e-9:
            return
        trunk = [r[1]["trunk"] for r in self._rest]
        # The lowest steady lying level seen. Never raised by a lie-down
        # that stopped short, so "back on the mat" can't creep upward rep
        # by rep.
        if max(trunk) - min(trunk) <= REF_STILL_DEG:
            level = float(np.median(trunk))
            if self.lying_ref is None or level < self.lying_ref:
                self.lying_ref = level
        # Where the hands rest: the low end (most lying-down) of the window.
        self.rest_reach = float(np.percentile([r[1]["reach"] for r in self._rest], 25))
        self.floor_d = _unit(np.median([r[1]["floor_d"] for r in self._rest], axis=0))
        for k in MedianSmoother.OPT:
            vals = [r[1][k] for r in self._rest if r[1][k] is not None]
            if vals:
                self.rest_feet[k] = float(np.median(vals))

    def _foot_lift(self, m):
        """(seen, up) for the foot nearest the camera, else the far one -
        whichever is in view with a rest position to compare against. Up =
        heel AND toe both above their rest: the whole foot off the floor.
        Seen = some foot is clearly in view, rest learned or not."""
        seen = False
        for sd in (self.side, OTHER[self.side]):
            heel, toe = m[f"heel_{sd}"], m[f"toe_{sd}"]
            if heel is None or toe is None:
                continue
            seen = True
            rh, rt = self.rest_feet.get(f"heel_{sd}"), self.rest_feet.get(f"toe_{sd}")
            if rh is None or rt is None:
                continue
            return True, (heel - rh > FOOT_LIFT and toe - rt > FOOT_LIFT)
        return seen, False

    def _is_down(self, m):
        if self.lying_ref is None:
            return m["trunk"] <= LYING_MAX_DEG
        return m["trunk"] <= self.lying_ref + RETURN_DEG

    def _reached(self, m):
        return (m["reach"] >= REACH_MIN
                and (self.rest_reach is None or m["reach"] - self.rest_reach >= SLIDE_MIN))

    def _step(self, m, dt, now, counting):
        """The phase machine. Returns "rep", "not_counted" or None."""
        self.in_position = m["floor_tilt"] <= FLOOR_TILT_MAX_DEG
        self.feet_seen, self.feet_up = self._foot_lift(m)
        if not self.in_position:
            # Standing, sitting on a chair, camera not side-on: the curl-up
            # under way is dropped.
            self.phase = PH_UNKNOWN
            self._rest.clear()
            self._clear_rep()
            self.problem = "Lie on your back on the mat, side-on to the camera"
            return None
        self.problem = ("Feet off the floor - keep them flat" if self.feet_up
                        else None if self.feet_seen
                        else "Feet not in view - the coach watches them")

        if self.phase == PH_UNKNOWN:
            if self._is_down(m):
                self.phase = PH_DOWN
                self._rest.clear()
                self._learn_rest(m, now)
            return None

        if self.phase == PH_DOWN:
            rise = None if self.lying_ref is None else m["trunk"] - self.lying_ref
            if rise is None or rise < RISE_DEG:
                self._learn_rest(m, now)
                return None
            # The trunk is coming up: a new curl-up. Its faults start here,
            # measured against the rest just learned.
            self._rest.clear()
            self._clear_rep()
            self.phase = PH_RISING

        # A curl-up under way.
        if self.feet_up:
            self._feet_t += dt
        self.peak_reach = m["reach"] if self.peak_reach is None else max(self.peak_reach, m["reach"])

        if self.phase == PH_RISING:
            if self._reached(m):
                self.phase = PH_REACHED
                self._slid_back = False
                self._low = None
            elif self._is_down(m):
                self._enter_down(m, now)
                return self._reject("short", counting)
            return None

        # PH_REACHED
        if self._is_down(m):
            self._enter_down(m, now)
            if not counting:
                return None
            if self._feet_t >= FAULT_MIN_S - 1e-9:
                return self._reject("feet", counting)
            self.count += 1
            self.status_msg = f"Curl-up {self.count}"
            return "rep"
        rest = REST_FALLBACK if self.rest_reach is None else self.rest_reach
        if m["reach"] <= rest + REBOUND_FRAC * (self.peak_reach - rest):
            self._slid_back = True
        if self._slid_back:
            self._low = m["reach"] if self._low is None else min(self._low, m["reach"])
        if (self._slid_back and self._reached(m)
                and m["reach"] >= self._low + REBOUND_HYST):
            # Slid the hands back and reached again without lying down: that
            # one never returned. The next one starts at this reach. The
            # hands must come clearly forward again from their lowest - a
            # shaky fingertip on the way down bobbing back over the knee
            # line is not a second reach (it used to read as one, and the
            # reject wiped that curl-up's feet record, so a feet-off
            # curl-up could be counted).
            self._clear_rep()
            self.peak_reach = m["reach"]
            return self._reject("not_down", counting)
        return None

    def _enter_down(self, m, now):
        self.phase = PH_DOWN
        self._rest.clear()
        self._learn_rest(m, now)

    def _reject(self, why, counting):
        if not counting:
            return None
        self.not_counted[why] += 1
        self.status_msg = f"Not counted - {NOT_COUNTED[why]}"
        return "not_counted"

    def result(self, now=None):
        return {
            "reps": self.count,
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


def body_points(pts, side):
    """The side being read, for the overlay: shoulder, hip, knee, fingertip,
    and the knee-target point at REACH_MIN along the thigh."""
    p = pts[side]
    if any(p[n] is None for n in CORE):
        return None
    target = p["hip"] + REACH_MIN * (p["knee"] - p["hip"])
    return {"shoulder": p["shoulder"], "hip": p["hip"], "knee": p["knee"],
            "index": p["index"], "target": target,
            "heel": p["heel"], "toe": p["toe"]}


def draw_hud(frame, test, bp=None):
    h, w = frame.shape[:2]
    colour = (C_BAD if test.feet_up
              else C_OK if test.phase == PH_REACHED
              else C_WARN if test.phase == PH_RISING
              else C_TEXT)

    if bp is not None:
        P = lambda k: tuple(np.int32(bp[k]))
        cv2.line(frame, P("shoulder"), P("hip"), colour, 3, cv2.LINE_AA)
        cv2.line(frame, P("hip"), P("knee"), colour, 3, cv2.LINE_AA)
        if bp["heel"] is not None and bp["toe"] is not None:
            cv2.line(frame, P("heel"), P("toe"), C_BAD if test.feet_up else C_OK, 3, cv2.LINE_AA)
        # The kneecap target, and the fingertip heading for it.
        cv2.circle(frame, P("target"), 11, C_OK if test.phase == PH_REACHED else C_WARN, 2, cv2.LINE_AA)
        cv2.circle(frame, P("index"), 7, colour, -1, cv2.LINE_AA)

    # Top bar: the gap clock filling toward the end.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = test.gap_progress()
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 10), C_BAD if prog > 0.6 else C_WARN, -1)

    draw_panel(frame, 12, 22, 450, 200)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _hud_text(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    state = "DONE" if test.stopped else "COUNTING" if test.started else "READY"
    line(f"MODIFIED CURL-UP  [{state}]", colour, 0.68, 30)
    line(f"Curl-ups: {test.count}", C_OK, 0.9, 34)
    if test.m is not None:
        rise = test.rise()
        line(f"Reach: {test.m['reach']:.2f}  (knee {REACH_MIN:.2f})   "
             f"Up: {'-' if rise is None else f'{rise:.0f}'} deg")
        line(f"Phase: {test.phase}   Not counted: {test.total_not_counted}")
    else:
        line("Body not tracked - side-on, head to feet in frame", C_BAD)
    if test.started and not test.stopped:
        line(f"Since last curl-up: {test.gap_elapsed:.1f} / {test.gap_s:.0f} s")

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


def _stage_modified_curl_up(session_id, res):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/modified_curl_up",
            json={
                "session_id":   session_id,
                "reps":         res["reps"],
                "duration_sec": res["duration_sec"],
                "end_reason":   res["end_reason"],
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] modified_curl_up staged for review → {res['reps']} reps  "
              f"session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Modified Curl-Up Test")
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
    # clock: the 6 s gap must be the participant's, not the decoder's.
    live = isinstance(source, int)
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not live and not (1.0 <= clip_fps <= 240.0):
        clip_fps = 30.0

    print("[INFO] Side-on camera at floor level, head to feet in frame. "
          "SPACE = start, R = reset, Q = quit (no cadence beep here)\n")

    test = ModifiedCurlUpTest()
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

    win = "Modified Curl-Up Test"
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

            pts = None
            if results.pose_landmarks:
                lms = results.pose_landmarks[0]
                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(lms[i].x * fw), int(lms[i].y * fh)),
                                 (int(lms[j].x * fw), int(lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)
                pts = read_points(lms, fw, fh)

            ended = test.update(pts, now=now)
            last_now = now
            if ended and not staged and args.session_id:
                _stage_modified_curl_up(args.session_id, test.result(now))
                staged = True

            bp = body_points(pts, test.side) if (test.tracked and pts and test.side) else None
            if args.debug and test.m is not None:
                m = test.m
                rise = test.rise()
                print(f"side={test.side} phase={test.phase} trunk={m['trunk']:.0f} "
                      f"rise={'-' if rise is None else round(rise)} reach={m['reach']:.2f} "
                      f"rest={'-' if test.rest_reach is None else round(test.rest_reach, 2)} "
                      f"feet_up={test.feet_up} tilt={m['floor_tilt']:.0f} "
                      f"reps={test.count} gap={test.gap_elapsed:.1f}")
            draw_hud(frame, test, bp)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
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
        _stage_modified_curl_up(args.session_id, res)
    print("\n── Modified Curl-Up ─────────────────────────")
    print(f"  Curl-ups      : {res['reps']}")
    print(f"  Time          : {res['duration_sec']} s")
    print(f"  Ended         : {res['end_reason'] or '--'}")
    for k, n in res["not_counted"].items():
        if n:
            print(f"  Not counted   : {n} x {NOT_COUNTED[k]}")


if __name__ == "__main__":
    main()
