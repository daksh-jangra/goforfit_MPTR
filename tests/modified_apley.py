"""
Modified Apley Test - upper-body flexibility, left arm then right arm
=================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual, pp. 88-89, fig. 5.24)
--------------------------------------------------------------------
  The participant reaches back with one hand and tries to touch the
  superior medial angle (the top inner corner) of the OPPOSITE shoulder
  blade. No equipment. One trial per arm. Each arm is scored:

      3  touches the top inner corner of the opposite shoulder blade and
         holds it 1 to 2 seconds
      2  touches the top of the head
      1  touches the mouth
      0  can't touch the mouth

  "Ballistic or reflexive touches are not acceptable": the position must be
  held briefly. Here EVERY level needs a 1-second hold, so a hand flicking
  past the head on its way to the back can't score a 2.

  The student tries the shoulder blade first; if they can't, the top of the
  head; then the mouth. The app keeps the best level held for each arm, so
  it doesn't matter what order they try them in.

How it's read from the camera
--------------------------------------------------------------------
  The phone goes BEHIND the student (the back camera, never mirrored, as
  for the Shoulder Stretch): the hand on the shoulder blade can only be
  seen from behind. From behind, the student's LEFT arm is on the LEFT of
  the picture, so the LEFT hand reaching to the RIGHT shoulder blade ends
  up on the RIGHT of the head.

  Measured from the head, in the hand's own PALM lengths (wrist to the
  middle knuckle), never from the pose's shoulders: from behind, with an
  arm raised over the head, MediaPipe's shoulder points jump around by 2-3x
  within one steady hold (Brockport Test Demo video, C0xamv81CCE), while its
  head points and the hand tracker's hand stay put. The palm length is a
  per-student yardstick, so a child and an adult use the same lines.

  3  a hand pointing DOWN (it came over the head, not up the back) whose
     fingertips are past the middle of the head toward the opposite side
     by REACH_CROSS_MIN, below the ears by REACH_DEPTH_MIN, with the wrist
     no lower than REACH_WRIST_MAX below the ears. Measured on the demo
     video's held reaches (both arms): fingertips 0.75-2.0 across,
     1.45-2.1 below, wrist -0.3..+0.3. Two things are refused:
       * the regular Apley reach down the SAME side of the back (the
         FITNESSGRAM-style reach in the "Fitnessgram and Brockport Fitness
         Tips" video, BjO9ZtPC7Rw): its fingertips stay on the hand's own
         side, 0 to -1.0 across;
       * a hand on the FRONT of the opposite shoulder, facing the camera:
         the wrist is down at the chest, far below REACH_WRIST_MAX.
  2  a hand resting on top of the head: the palm's middle between
     HEAD_TOP_MIN and HEAD_TOP_MAX palms above the ears and within
     HEAD_DX_MAX of the middle of the head. When the hand tracker misses
     the hand, the pose's own wrist and index-finger points are used
     instead (fig. 5.24b, front-on: 0.95 and 1.06 head units above the
     ears).
  1  a fingertip on the mouth. The pose's mouth points are used; on the
     manual's photo (fig. 5.24c) the pose's index finger sat 0.11 head units
     from them.

  The camera can't tell the student's front from their back (MediaPipe
  puts a "face" on the back of the head, and the face detector misses a
  face with a hand over the mouth). So a hand on the BACK of the neck,
  from behind, can read as the mouth. That can only ever give a 1 to a
  student who reached somewhere harder than the mouth, so it is left.

What the app scores vs. what the coach judges
--------------------------------------------------------------------
  SCORED: the best level held for 1 second, per arm. The coach can't change
  it (as for the other camera-scored tests) - only end an arm early (NEXT)
  or RESET the run.

  NOT SCORED (the tester's job): warm-up, practice, which arm the student
  uses for the head and the mouth, and "no physical help during the test".
  The manual's tip of holding the tester's fingertips on the shoulder blade
  as a target is NOT used: a second person's hand in the shot confuses the
  hand tracker. The app's own line is the target.

Run
--------------------------------------------------------------------
  LEFT arm first, then RIGHT. An arm ends by itself once it holds a 3 (it
  can't do better); otherwise the coach presses NEXT when the student has
  tried. An arm the camera never saw the student on is "not tested".

Usage
-----
  python modified_apley.py
  python modified_apley.py --source clip.mp4

  Keys : N = end this arm now     R = reset     Q = quit
"""

import argparse
import os
import sys
import time
from collections import deque, namedtuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision as mp_vision

# Re-exported so web_engines.py builds both detectors through this module
# (`import modified_apley as apl_mod`), as it does for shoulder_stretch.py.
# The legacy mediapipe.solutions module is deliberately not imported.
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode


# ── Protocol constants ───────────────────────────────────────────────────────────
HOLD_S = 1.0             # "hold that position for 1 to 2 seconds"
MAX_SCORE = 3

SIDE_LEFT, SIDE_RIGHT = "left", "right"
SIDES = (SIDE_LEFT, SIDE_RIGHT)          # the order they are tested in

LEVEL_TEXT = {
    3: "opposite shoulder blade",
    2: "top of the head",
    1: "mouth",
    0: "can't reach the mouth",
}

# ── Detection tuning ─────────────────────────────────────────────────────────────
HOLD_GAP_S      = 0.5    # no reading for longer than this restarts the 1 s hold
HOLD_DIP_S      = 0.3    # a lower reading for longer than this breaks it
MIN_HOLD_READS  = 4      # ... and it needs this many readings at the level

# Score 3, in palm lengths from the middle of the head (the ears' midpoint).
# Demo video, held reaches: across 0.75-2.0, below 1.45-2.1, wrist -0.3..+0.3.
# Same-side reach (refused): across 0 to -1.0. A hand only behind the neck
# reaches ~1 palm down and barely past the middle.
DOWN_MIN        = 0.5    # wrist -> middle knuckle pointing down (+1 = straight down)
REACH_CROSS_MIN = 0.35   # fingertips past the middle of the head, toward the far side
REACH_DEPTH_MIN = 1.1    # fingertips below the ears
REACH_WRIST_MAX = 0.8    # wrist no lower than this below the ears (it came over the head)

# Score 2, hand tracker: the palm's middle above the ears. A hand on the
# forehead sits ~0.5 palm up, a hand stretched straight overhead 3.5+.
HEAD_TOP_MIN    = 0.8
HEAD_TOP_MAX    = 2.8
HEAD_DX_MAX     = 1.5
# Score 2, pose fallback, in HEAD units (shoulder line to ears, ~13-15 cm).
# Fig. 5.24b: wrist 0.95, index finger 1.06 above the ears.
POSE_TOP_MIN    = 0.6
POSE_TOP_MAX    = 1.8
POSE_TOP_DX_MAX = 0.9

# Score 1: a fingertip this close to the mouth. Chin and nose are ~0.3 head
# units / ~0.5 palm away. Fig. 5.24c: pose index finger 0.11 head units.
MOUTH_NEAR_PALM = 0.6
MOUTH_NEAR_HEAD = 0.30

# The hand's yardstick: its palm length, but never less than its usual
# length over the last few seconds. A hand lying flat on top of the head is
# seen edge-on and looks short, which would stretch every distance.
PALM_WINDOW_S   = 15.0
PALM_PCTL       = 0.75

# The head's last seen place is kept while the hand stays tracked: a frame
# with neither the pose nor a hand for longer than this forgets it.
HEAD_KEEP_GAP_S = 1.0
# A real head's ears sit ~1.3-1.5 palms above the shoulder line (demo
# video). One the pose made up for a head cut off the picture sits level
# with the shoulders or below them (kid video close-ups: -0.6 to +0.3).
HEAD_RISE_MIN   = 0.6

MIN_VIS = 0.50           # pose landmark visibility floor

# Pose landmark indices (33-point BlazePose)
EAR_L, EAR_R, MOUTH_L, MOUTH_R = 7, 8, 9, 10
SH_L, SH_R = 11, 12
POSE_HAND_POINTS = (15, 16, 19, 20)      # wrists, index fingers
POSE_FINGER_POINTS = (19, 20, 21, 22)    # index fingers, thumbs
# Hand landmark indices (21-point MediaPipe hand)
H_WRIST, H_MID_MCP = 0, 9
H_PALM = (0, 5, 9, 13, 17)
H_REACH_TIPS = (8, 12, 16)               # index, middle, ring
H_TIPS = (4, 8, 12, 16, 20)

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# HUD palette (matches the other tests)
C_DARK = (28, 28, 28)
C_OK   = (80, 220, 120)
C_WARN = (60, 190, 250)
C_BAD  = (80, 80, 240)
C_TEXT = (245, 245, 245)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POSE_MODEL = os.path.join(_HERE, "pose_landmarker_full.task")
DEFAULT_HAND_MODEL = os.path.join(_HERE, "hand_landmarker.task")
MODEL_URLS = {
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


# ── Geometry (pure functions on pixel points; the logic checks call these) ───────
def _v(p):
    return np.asarray(p, dtype=np.float64)


def reach_dir(side):
    """Which way the hand must cross the head, seen from BEHIND and not
    mirrored: +1 = toward picture-right. The LEFT hand goes to the RIGHT
    shoulder blade, which is on the right of the picture from behind."""
    return 1.0 if side == SIDE_LEFT else -1.0


def opposite(side):
    return SIDE_RIGHT if side == SIDE_LEFT else SIDE_LEFT


def palm_length(hand):
    """Wrist to middle knuckle, pixels. hand: (21, 2) pixel points."""
    return float(np.hypot(*(hand[H_MID_MCP] - hand[H_WRIST])))


def down_ness(hand):
    """+1 if the hand points straight down (wrist above the knuckles), -1 if
    straight up, 0 if sideways."""
    palm = palm_length(hand)
    return 0.0 if palm < 1e-6 else float((hand[H_MID_MCP][1] - hand[H_WRIST][1]) / palm)


def shoulder_blade_touch(hand, ears, unit, side):
    """Score 3: (True, None) when this hand is on the opposite shoulder
    blade, else (False, why) - why is a hint for the student, or None when
    the hand isn't trying it at all (not pointing down)."""
    if down_ness(hand) < DOWN_MIN:
        return False, None
    ears = _v(ears)
    tips = hand[list(H_REACH_TIPS)].mean(axis=0)
    across = (tips[0] - ears[0]) * reach_dir(side) / unit
    below = (tips[1] - ears[1]) / unit
    wrist_below = (hand[H_WRIST][1] - ears[1]) / unit
    if wrist_below > REACH_WRIST_MAX:
        return False, None              # a hand on the chest or down the side
    far = opposite(side).upper()
    if across < REACH_CROSS_MIN:
        return False, f"Reach ACROSS behind the head to the {far} shoulder blade"
    if below < REACH_DEPTH_MIN:
        return False, f"Lower - down to the top of the {far} shoulder blade"
    return True, None


def head_top_touch(hand, ears, unit):
    """Score 2 from the hand tracker: the palm rests on top of the head."""
    ears = _v(ears)
    palm_mid = hand[list(H_PALM)].mean(axis=0)
    up = (ears[1] - palm_mid[1]) / unit
    dx = abs(palm_mid[0] - ears[0]) / unit
    return HEAD_TOP_MIN <= up <= HEAD_TOP_MAX and dx <= HEAD_DX_MAX


def mouth_touch(hand, mouth, unit):
    """Score 1 from the hand tracker: a fingertip on the mouth."""
    mouth = _v(mouth)
    d = min(float(np.hypot(*(hand[i] - mouth))) for i in H_TIPS)
    return d / unit <= MOUTH_NEAR_PALM


def pose_head_top(points, ears, head_unit):
    """Score 2 from the pose's own hand points: a wrist AND an index finger
    of the same arm sitting on top of the head. points: {index: (x, y)} of
    the visible ones."""
    ears = _v(ears)
    for wr, ix in ((15, 19), (16, 20)):
        if wr not in points or ix not in points:
            continue
        ok = True
        for i in (wr, ix):
            p = _v(points[i])
            up = (ears[1] - p[1]) / head_unit
            if not (POSE_TOP_MIN <= up <= POSE_TOP_MAX
                    and abs(p[0] - ears[0]) / head_unit <= POSE_TOP_DX_MAX):
                ok = False
        if ok:
            return True
    return False


def pose_mouth(points, mouth, head_unit):
    """Score 1 from the pose's own finger points."""
    mouth = _v(mouth)
    return any(float(np.hypot(*(_v(points[i]) - mouth))) / head_unit <= MOUTH_NEAR_HEAD
               for i in POSE_FINGER_POINTS if i in points)


# ── Landmarks -> one reading ─────────────────────────────────────────────────────
Reading = namedtuple("Reading", "level problem marks")


def _px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _vis(lm, thr=MIN_VIS):
    return getattr(lm, "visibility", 1.0) >= thr


class PalmYardstick:
    """The palm length used as the unit: this frame's, or the usual one over
    the last PALM_WINDOW_S if that is longer (an edge-on hand looks short)."""

    def __init__(self, window_s=PALM_WINDOW_S, pctl=PALM_PCTL):
        self.window_s, self.pctl = float(window_s), float(pctl)
        self.buf = deque()       # (t, palm px)

    def reset(self):
        self.buf.clear()

    def push(self, t, palm):
        self.buf.append((t, float(palm)))
        while self.buf and self.buf[0][0] < t - self.window_s:
            self.buf.popleft()

    def unit(self, palm):
        if not self.buf:
            return palm
        vals = sorted(v for _, v in self.buf)
        return max(palm, vals[int(self.pctl * (len(vals) - 1))])


class ApleyMeter:
    """Turns one frame's landmarks into one LEVEL (0-3) for the arm under
    test, or None when the frame can't say (nobody there, or no hand found
    and nothing from the pose either - hands drop out a lot from behind, and
    a dropout must not read as 'can't reach').

    The head is remembered: from behind, once the arm goes up over the head
    the POSE model loses the whole body - on the demo video for the entire
    10-second hold at 15 and 30 fps - while the HAND tracker keeps the hand
    in every frame. So the head's last seen place is used for as long as a
    hand stays tracked without a break of more than HEAD_KEEP_GAP_S."""

    def __init__(self):
        self.palms = PalmYardstick()
        self.reset()

    def reset(self):
        self.palms.reset()
        self._head = None         # (ears, mouth) last seen by the pose
        self._track_t = None      # last frame the head was seen or carried on

    def _head_ok(self, lms, ears, w, h):
        """Whether the pose's head is a real head. With the head cut off the
        picture (a phone too close or tilted down), MediaPipe still reports
        one - in the middle of the back, level with the shoulders - and a
        hand reaching up the back then reads as on top of it (a 2 that
        wasn't there, "Fitnessgram and Brockport Fitness Tips" 269-272 s)."""
        if not (0.0 < ears[0] < w and 0.0 < ears[1] < h):
            return False
        if not (_vis(lms[SH_L]) and _vis(lms[SH_R])):
            return True
        rise = 0.5 * (_px(lms[SH_L], w, h)[1] + _px(lms[SH_R], w, h)[1]) - ears[1]
        return rise > HEAD_RISE_MIN * self.palms.unit(0.0)

    def measure(self, lms, hands, w, h, side, now=None):
        """lms: one body's 33 pose landmarks (normalised) or None. hands: the
        hand tracker's landmark lists (may be empty). Returns a Reading;
        marks = {'ears', 'unit', 'hands'} for the HUD."""
        now = time.time() if now is None else now
        if lms is not None and (_vis(lms[EAR_L]) or _vis(lms[EAR_R])):
            ears = 0.5 * (_px(lms[EAR_L], w, h) + _px(lms[EAR_R], w, h))
            mouth = 0.5 * (_px(lms[MOUTH_L], w, h) + _px(lms[MOUTH_R], w, h))
            if not self._head_ok(lms, ears, w, h):
                return Reading(None, "Head not in view - the whole head and both shoulders "
                                     "must be in the picture", {})
            self._head = (ears, mouth)
            self._track_t = now
        elif (self._head is not None and hands
              and now - self._track_t <= HEAD_KEEP_GAP_S):
            ears, mouth = self._head
            self._track_t = now
            lms = None            # the pose's own hand points are gone too
        else:
            return Reading(None, "No student detected - head and shoulders in view", {})

        level, problem = None, None
        hand_pts = [np.array([[p.x * w, p.y * h] for p in hl]) for hl in hands or []]
        marks = {"ears": ears, "unit": None, "hands": hand_pts}
        for hp in hand_pts:
            palm = palm_length(hp)
            if palm < 4:
                continue
            self.palms.push(now, palm)
            unit = self.palms.unit(palm)
            marks["unit"] = unit
            lv = 0
            ok, why = shoulder_blade_touch(hp, ears, unit, side)
            if ok:
                lv = 3
            elif head_top_touch(hp, ears, unit):
                lv = 2
            elif mouth_touch(hp, mouth, unit):
                lv = 1
            if why and lv < 3:
                problem = why
            level = lv if level is None else max(level, lv)

        # The pose's own hand points, for the head and the mouth. Front-on
        # they are good (the manual's photos); they back up a missed hand.
        if lms is not None and (level is None or level < 2):
            sh_ok = _vis(lms[SH_L]) and _vis(lms[SH_R])
            head_unit = (0.5 * (_px(lms[SH_L], w, h)[1] + _px(lms[SH_R], w, h)[1]) - ears[1]
                         if sh_ok else 0.0)
            if head_unit > 5:
                pts = {i: _px(lms[i], w, h) for i in POSE_HAND_POINTS + POSE_FINGER_POINTS
                       if _vis(lms[i])}
                lv = 2 if pose_head_top(pts, ears, head_unit) else \
                    1 if pose_mouth(pts, mouth, head_unit) else None
                if lv is not None:
                    level = lv if level is None else max(level, lv)
        return Reading(level, problem, marks)


# ── The 1-second hold ────────────────────────────────────────────────────────────
class LevelHold:
    """The highest level HELD for HOLD_S: read at that level (or higher) from
    the first reading to the last, with no break. A reading below it for
    more than HOLD_DIP_S (one or two jittery frames are let through), or no
    reading at all for more than HOLD_GAP_S, breaks it. The hold is timed
    from the readings themselves, so a dropout can't stretch half a second
    of touching into a full one."""

    def __init__(self, hold_s=HOLD_S, gap_s=HOLD_GAP_S, dip_s=HOLD_DIP_S,
                 min_reads=MIN_HOLD_READS):
        self.hold_s, self.gap_s, self.dip_s = float(hold_s), float(gap_s), float(dip_s)
        self.min_reads = int(min_reads)
        self.reset()

    def reset(self):
        self.last_t = None
        # Per level: [first reading at it, last reading at it, readings at it]
        self.runs = {lv: None for lv in range(1, MAX_SCORE + 1)}

    def push(self, t, level):
        if self.last_t is not None and t - self.last_t > self.gap_s:
            self.reset()
        self.last_t = t
        held = None
        for lv in self.runs:
            run = self.runs[lv]
            if level >= lv:
                if run is None:
                    run = self.runs[lv] = [t, t, 0]
                run[1] = t
                run[2] += 1
                if run[1] - run[0] >= self.hold_s * 0.95 and run[2] >= self.min_reads:
                    held = lv
            elif run is not None and t - run[1] > self.dip_s:
                self.runs[lv] = None
        return held


# ── The run: left arm, then right arm ────────────────────────────────────────────
class ApleyTracker:
    """Left arm then right arm. Fed one level per frame (or None when the
    frame couldn't say) with an injectable clock, so the whole run can be
    checked with made-up sequences - no camera.

    Each arm keeps the best level HELD for 1 s. It ends by itself on a held
    3; otherwise the coach presses NEXT. An arm on which the student was
    seen but held nothing scores 0; one never seen stays not tested."""

    def __init__(self, hold_s=HOLD_S):
        self.hold_s = float(hold_s)
        self.reset()

    def reset(self):
        self.side_idx = 0
        self.results = {s: None for s in SIDES}       # 0-3, or None = not tested
        self.end_reasons = {s: None for s in SIDES}
        self.ended_by = None
        self._hold = LevelHold(hold_s=self.hold_s)
        self._new_side()

    def _new_side(self):
        self.current = None       # this frame's level
        self.held = None          # level held over the last second
        self.best = None          # best held level on this arm
        self.seen = False         # the student was measured on this arm
        self.problem = None
        self._hold.reset()
        self.status_msg = self._position_msg()

    # ── read-only views ──────────────────────────────────────────────────
    @property
    def side(self):
        return SIDES[self.side_idx] if self.side_idx < len(SIDES) else None

    @property
    def resolved(self):
        return self.side_idx >= len(SIDES)

    @property
    def measured_any(self):
        return any(r is not None for r in self.results.values())

    def _position_msg(self):
        if self.side is None:
            return ""
        return (f"{self.side.upper()} arm: back to the camera, reach behind the head to the "
                f"{opposite(self.side).upper()} shoulder blade and hold")

    def score(self):
        """The one number the website gets: the lower arm."""
        vals = [r for r in self.results.values() if r is not None]
        return min(vals) if vals else None

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, level, now=None, problem=None):
        """Feed this frame's level (None = the frame couldn't say). Returns
        the result dict on the frame the run finishes, else None."""
        now = time.time() if now is None else now
        if self.resolved:
            return None
        self.problem = problem
        if level is None:
            self.current = None
            self.status_msg = problem or (
                "Hold the position in view" if self.seen else self._position_msg())
            return None

        self.seen = True
        self.current = int(level)
        self.held = self._hold.push(now, level)
        if self.held is not None and (self.best is None or self.held > self.best):
            self.best = self.held
        if self.best is not None and self.best >= MAX_SCORE:
            return self._end_side("reached 3")

        if problem:
            self.status_msg = problem
        elif self.best:
            self.status_msg = (f"{self.side.upper()}: held {LEVEL_TEXT[self.best]} = "
                               f"{self.best} - try higher, or press NEXT")
        else:
            self.status_msg = (f"{self.side.upper()} arm: shoulder blade, else top of the "
                               "head, else the mouth - hold 1 s")
        return None

    # ── ending arms ──────────────────────────────────────────────────────
    def next_side(self):
        """Coach: end the arm under way now, on what it has held so far."""
        if self.resolved:
            return None
        return self._end_side("coach")

    def finish(self):
        """STOP pressed: close the arm under way on what it has, leave a later
        arm not tested. Returns the result if anything was measured."""
        while not self.resolved:
            self._end_side("stopped")
        return self.result() if self.measured_any else None

    def _end_side(self, why):
        side = self.side
        self.results[side] = (self.best or 0) if self.seen else None
        self.end_reasons[side] = why
        self.side_idx += 1
        done = (f"{side.upper()}: {self.results[side]}" if self.results[side] is not None
                else f"{side.upper()}: not tested")
        self.current = self.held = self.best = None
        if self.resolved:
            self.ended_by = why
            self.status_msg = "DONE - " + "   ".join(self._summary())
            return self.result()
        self._new_side()
        self.status_msg = f"{done}.  Now the {self.side.upper()} - {self._position_msg()}"
        return None

    def _summary(self):
        return [f"{s.upper()} {self.results[s]}" if self.results[s] is not None
                else f"{s.upper()} --" for s in SIDES]

    def result(self):
        return {
            "left_score": self.results[SIDE_LEFT],
            "right_score": self.results[SIDE_RIGHT],
            "score": self.score(),
            "ended_by": self.ended_by,
        }


# ── Drawing ──────────────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _ascii(txt):
    """cv2's Hershey fonts draw '?' for anything outside ASCII."""
    return txt.encode("ascii", "replace").decode("ascii")


def draw_pose(frame, lms):
    if lms is None:
        return
    h, w = frame.shape[:2]
    for i, j in _POSE_CONNECTIONS:
        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
            cv2.line(frame, (int(lms[i].x * w), int(lms[i].y * h)),
                     (int(lms[j].x * w), int(lms[j].y * h)), (200, 200, 200), 2, cv2.LINE_AA)


def draw_hud(frame, tracker, marks=None):
    """Draw the live state onto the frame, with the shoulder-blade target
    (where the fingertips need to get to for a 3) when a hand gives the size."""
    h, w = frame.shape[:2]
    tr = tracker
    marks = marks or {}
    colour = (C_OK if tr.held else C_WARN if tr.current is not None else C_BAD)

    for hp in marks.get("hands") or []:
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, tuple(np.int32(hp[a])), tuple(np.int32(hp[b])), colour, 2, cv2.LINE_AA)

    ears, unit = marks.get("ears"), marks.get("unit")
    if ears is not None and unit and not tr.resolved:
        d = reach_dir(tr.side)
        x1 = int(ears[0] + d * REACH_CROSS_MIN * unit)
        x2 = int(ears[0] + d * (REACH_CROSS_MIN + 1.6) * unit)
        y1 = int(ears[1] + REACH_DEPTH_MIN * unit)
        y2 = int(ears[1] + (REACH_DEPTH_MIN + 1.4) * unit)
        cv2.rectangle(frame, (min(x1, x2), y1), (max(x1, x2), y2), C_WARN, 2, cv2.LINE_AA)
        cv2.putText(frame, "3", (min(x1, x2) + 6, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    C_WARN, 2, cv2.LINE_AA)

    draw_panel(frame, 12, 22, 560, 170)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _ascii(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2,
                    cv2.LINE_AA)
        y += dy

    line("MODIFIED APLEY TEST", colour, 0.64, 30)
    if not tr.resolved:
        line(f"Arm: {tr.side.upper()} ({tr.side_idx + 1} of 2)")
        line("Now: " + ("--" if tr.current is None else str(tr.current)) +
             "   Held 1 s: " + ("--" if tr.held is None else str(tr.held)))
        if tr.best:
            line(f"Best held: {tr.best} ({LEVEL_TEXT[tr.best]})", C_OK)
    for s in SIDES:
        if tr.end_reasons[s] is not None:
            r = tr.results[s]
            line(f"{s.upper()}: {r}" if r is not None else f"{s.upper()}: not tested",
                 C_OK if r is not None else C_BAD)

    if tr.status_msg:
        scale, lines, cur = 0.55, [], ""
        for word in _ascii(tr.status_msg).split():
            trial = f"{cur} {word}".strip()
            if cur and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 48:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines = (lines + [cur])[-3:]
        top = h - 20 - 26 * len(lines) - 8
        draw_panel(frame, 12, top, w - 12, h - 12)
        for k, txt in enumerate(lines):
            cv2.putText(frame, txt, (24, top + 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        C_TEXT, 2, cv2.LINE_AA)

    if tr.resolved and tr.measured_any:
        big = "   ".join(f"{s[0].upper()} {tr.results[s]}" if tr.results[s] is not None
                         else f"{s[0].upper()} -" for s in SIDES)
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
        org = ((w - size[0]) // 2, (h + size[1]) // 2)
        draw_panel(frame, org[0] - 30, org[1] - size[1] - 24,
                   org[0] + size[0] + 30, org[1] + 24, C_DARK, 0.72)
        cv2.putText(frame, big, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, C_OK, 5, cv2.LINE_AA)
    return frame


# ── Desktop run (webcam or clip) ─────────────────────────────────────────────────
def _ensure_model(path):
    if os.path.exists(path):
        return
    import ssl
    import urllib.request
    url = MODEL_URLS[os.path.basename(path)]
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    print(f"Downloading {os.path.basename(path)} ...")
    with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as fh:
        fh.write(resp.read())


def main():
    ap = argparse.ArgumentParser(description="Brockport Modified Apley Test")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--model", default=DEFAULT_POSE_MODEL)
    ap.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    ap.add_argument("--no-window", action="store_true",
                    help="don't show the video (print the result only)")
    args = ap.parse_args()

    _ensure_model(args.model)
    _ensure_model(args.hand_model)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source!r}")
    clip = not isinstance(src, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.45, min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.hand_model),
        running_mode=VisionRunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45))

    # Filmed from behind and never mirrored: the student's left arm is on
    # the left of the picture.
    tracker = ApleyTracker()
    meter = ApleyMeter()
    t0 = time.time()
    n = 0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        fh, fw = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # A clip runs on its own clock, so it scores the same at any speed.
        now = n / fps if clip else time.time() - t0
        ts = max(last_ts + 1, int(now * 1000))
        last_ts = ts
        res = pose.detect_for_video(image, ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        hands = hand_det.detect_for_video(image, ts).hand_landmarks or []
        draw_pose(frame, lms)
        marks = {}
        if not tracker.resolved:
            rd = meter.measure(lms, hands, fw, fh, tracker.side, now=now)
            marks = rd.marks
            if tracker.update(rd.level, now=now, problem=rd.problem) is not None:
                print(f"{now:6.1f}s", tracker.result())
        if args.no_window:
            continue
        draw_hud(frame, tracker, marks)
        cv2.imshow("Modified Apley Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n") and tracker.next_side() is not None:
            print(tracker.result())
        if key == ord("r"):
            tracker.reset()
            meter.reset()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hand_det.close()
    if not tracker.resolved:
        print(tracker.finish())


if __name__ == "__main__":
    main()
