"""
Target Stretch Test - range of motion at one joint, left and right
=================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual, pp. 94-100, form 5.1)
--------------------------------------------------------------------
  The target stretch test (TST) screens how far a joint moves. For each
  joint the participant moves as far as they can, HOLDS the end position
  for at least 1 second, and the tester reads the angle off a clock face to
  the nearest "half hour" (15 degrees). The angle is then turned into a
  score of 0, 1 or 2 with table 5.3:

      joint                         score 2    score 1
      wrist extension                 60         30
      elbow extension                  0        -15     (0 = fully straight)
      shoulder extension              60         30
      shoulder abduction             165        120
      shoulder external rotation      75         30
      forearm supination              90         45
      forearm pronation               90         45
      knee extension                   0        -15

  Anything below the score-1 line scores 0. The coach picks which joint is
  tested; every run here measures that ONE joint on the LEFT side and then
  the RIGHT side (owner's choice: both sides, every time).

  Knee extension: the manual lies the participant on their side and the
  tester looks down from above. That needs the phone overhead, so this app
  tests it SEATED with a side-on camera (owner's choice) - the knee angle is
  the same angle, the manual allows a changed position as long as it is
  noted.

What the app scores vs. what the coach judges
--------------------------------------------------------------------
  SCORED: the angle the joint reaches and holds for 1 second, rounded to
  15 degrees, and the 0/1/2 score that angle earns. The coach can't change
  it (owner's choice) - they can only RESET the run.

  NOT SCORED (the tester's job, as in the manual): the trunk staying erect,
  the elbow staying at the side for external rotation, the elbow at 90
  degrees, palm direction, "help the participant find their best position".

How each joint is read from the camera
--------------------------------------------------------------------
  SIDE-ON camera (the tested side toward the lens):
    wrist extension     hand line (wrist -> middle knuckle) against the
                        forearm; bending the hand UP (back) counts
    elbow extension     how straight the elbow gets; 0 = straight, minus =
                        degrees short, plus = over-straight
    shoulder extension  upper arm against the trunk line, BEHIND the body
    external rotation   the forearm starts pointing forward and shortens on
                        screen as it turns outward toward the lens - the
                        shortening gives the angle (cos). The full length is
                        learned from the start position itself
    knee extension      seated; how straight the knee gets, like the elbow
  FRONT-ON camera (the student faces the lens):
    shoulder abduction  upper arm against the trunk line, out to the side
    supination /        elbow bent, fist pointing at the lens, thumb up.
    pronation           The app can't see the manual's pencil, so it
                        measures the line across the knuckles instead
                        (little-finger knuckle -> index knuckle) - it turns
                        by the same angle the pencil would

  Which side is which, without trusting MediaPipe's left/right labels
  (they swap when the frame is mirrored and are unreliable side-on):
    side-on   a student showing their LEFT side to an un-mirrored camera
              faces the LEFT of the picture. The face direction is read
              from the nose against the ears, so turning around between
              the two sides is checked, not assumed.
    front-on  the student's LEFT arm is on the RIGHT of an un-mirrored
              picture.
  Mirrored frames (the selfie camera) flip both rules.

Hold rule
--------------------------------------------------------------------
  A reading counts as HELD when the last 1 second of readings has its 20th
  percentile at that angle - i.e. the joint stayed at or past it for about
  four fifths of a second, which rides out landmark jitter without letting
  a one-frame flick through the end range score. The best held angle in
  the side's 10-second window is that side's result. A gap in readings of
  more than half a second (the student out of view, a Wi-Fi stall) restarts
  the hold.

  Each side's 10 seconds start once the joint has been measured steadily
  for a second (the first held reading, counted from that second's start),
  so getting into position does not eat the time and one stray frame
  can't start the clock. The coach can end a side
  early (NEXT). A side with no reading for 30 seconds is left unmeasured.

Usage
-----
  python target_stretch.py --joint shoulder_abduction
  python target_stretch.py --joint knee_extension --source clip.mp4

  Keys : N = end this side now     R = reset     Q = quit
"""

import argparse
import math
import os
import sys
import time
from collections import deque, namedtuple
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision as mp_vision

# Re-exported so web_engines.py builds both detectors through this module
# (`import target_stretch as tst_mod`), as it does for shoulder_stretch.py.
# The legacy mediapipe.solutions module is deliberately not imported.
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode


# ── Protocol constants ───────────────────────────────────────────────────────────
CLOCK_STEP_DEG = 15      # "to the nearest half hour" on the clock face
HOLD_S         = 1.0     # "hold their final position for at least 1 second"

SIDE_LEFT, SIDE_RIGHT = "left", "right"
SIDES = (SIDE_LEFT, SIDE_RIGHT)          # the order they are tested in

VIEW_SIDE, VIEW_FRONT = "side", "front"

Joint = namedtuple("Joint", "key title view score2 score1 hand straight instruction")

# Table 5.3. `straight` joints (elbow, knee) are measured as degrees short of
# straight (0 best, minus = short), so their readings are not clamped at 0.
# `hand` joints need the hand tracker.
JOINTS = {
    "wrist_extension": Joint(
        "wrist_extension", "Wrist extension", VIEW_SIDE, 60, 30, True, False,
        "elbow bent, forearm forward palm down - bend the hand back (up) as far as it goes"),
    "elbow_extension": Joint(
        "elbow_extension", "Elbow extension", VIEW_SIDE, 0, -15, False, True,
        "upper arm at the side, palm forward - straighten the elbow as far as it goes"),
    "shoulder_extension": Joint(
        "shoulder_extension", "Shoulder extension", VIEW_SIDE, 60, 30, False, False,
        "arm at the side - swing the arm BACK as far as it goes, trunk upright"),
    "shoulder_abduction": Joint(
        "shoulder_abduction", "Shoulder abduction", VIEW_FRONT, 165, 120, False, False,
        "arm at the side - raise it out to the side and up as far as it goes"),
    "shoulder_external_rotation": Joint(
        "shoulder_external_rotation", "Shoulder external rotation", VIEW_SIDE, 75, 30,
        False, False,
        "seated, elbow at the side bent 90, forearm forward - turn the hand OUT as far as it goes"),
    "forearm_supination": Joint(
        "forearm_supination", "Forearm supination", VIEW_FRONT, 90, 45, True, False,
        "elbow bent, fist pointing at the camera thumb up - turn the palm UP as far as it goes"),
    "forearm_pronation": Joint(
        "forearm_pronation", "Forearm pronation", VIEW_FRONT, 90, 45, True, False,
        "elbow bent, fist pointing at the camera thumb up - turn the palm DOWN as far as it goes"),
    "knee_extension": Joint(
        "knee_extension", "Knee extension", VIEW_SIDE, 0, -15, False, True,
        "seated - straighten the knee as far as it goes"),
}
DEFAULT_JOINT = "shoulder_abduction"

# ── Detection tuning ─────────────────────────────────────────────────────────────
SIDE_WINDOW_S   = 10.0   # each side's time to reach and hold, from its first reading
NO_READING_S    = 30.0   # a side never measured in this long is left unmeasured
HOLD_GAP_S      = 0.5    # a longer gap between readings restarts the 1 s hold
HOLD_PCTL       = 0.20   # the held angle = this percentile of the last HOLD_S
MIN_HOLD_READS  = 4      # ... and only once the window has this many readings

MIN_VIS = 0.50           # pose landmark visibility floor for measured joints

# Side-on check: (nose x - mean ear x) / |nose - shoulder midpoint|. Side-on
# the nose sits well ahead of the ears (~0.4); facing the lens it sits
# between them (fig 5.29, student facing the lens with the head a little
# turned: -0.09..-0.11).
FACING_MIN = 0.20
# Shoulder spread: shoulder separation / |nose - shoulder midpoint|. Facing
# the lens it is wide (fig 5.29: 2.8); side-on the shoulders overlap. Front
# joints need at least FRONT_MIN; side joints at most SIDE_MAX_SPREAD, so a
# turned head alone can't pass a front-on student as side-on.
FRONT_MIN = 0.80
SIDE_MAX_SPREAD = 1.40

# External rotation reads the forearm's on-screen length against its length
# at the start (pointing forward). Lengths are in UPPER-ARM lengths so the
# student stepping nearer the camera doesn't change them. The start length
# is the running max of a 3-reading median, so one long jittery frame can't
# inflate it.
ROT_SMOOTH = 3
# Turning the hand IN (toward the belly) shortens the forearm on screen the
# same way turning it OUT does. Turning out brings the wrist nearer the lens
# than the elbow; MediaPipe's depth (z, smaller = nearer) tells them apart.
# A wrist this much FARTHER than the elbow is a turn inward.
ROT_INWARD_Z = 0.03
ROT_INWARD_MIN_DEG = 20.0   # only judge direction once the turn is this big

# A hand from the hand tracker belongs to the tested arm when its wrist is
# within this many upper-arm lengths of the pose wrist.
HAND_MATCH_UPPERARM = 0.8

# Pose landmark indices (33-point BlazePose)
NOSE, EAR_L, EAR_R = 0, 7, 8
ARM = {   # shoulder, elbow, wrist, hip, pinky, index
    "L": (11, 13, 15, 23, 17, 19),
    "R": (12, 14, 16, 24, 18, 20),
}
LEG = {   # hip, knee, ankle
    "L": (23, 25, 27),
    "R": (24, 26, 28),
}
# Hand landmark indices (21-point MediaPipe hand)
H_WRIST, H_INDEX_MCP, H_MID_MCP, H_PINKY_MCP = 0, 5, 9, 17

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
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


# ── Scoring ──────────────────────────────────────────────────────────────────────
def to_clock(deg):
    """Round to the nearest half hour (15 deg), halves rounding up."""
    return int(math.floor(deg / CLOCK_STEP_DEG + 0.5)) * CLOCK_STEP_DEG


def score_for(joint, clock_deg):
    """0, 1 or 2 from table 5.3 for an angle already rounded to the clock."""
    if clock_deg >= joint.score2:
        return 2
    if clock_deg >= joint.score1:
        return 1
    return 0


SideResult = namedtuple("SideResult", "deg score exact")


def side_result(joint, exact):
    deg = to_clock(exact)
    return SideResult(deg=deg, score=score_for(joint, deg), exact=round(float(exact), 1))


# ── Geometry (pure functions on pixel points; the logic checks call these) ───────
def _v(p):
    return np.asarray(p, dtype=np.float64)


def angle_between(a, b):
    """Unsigned angle between two vectors, degrees 0..180."""
    a, b = _v(a), _v(b)
    na, nb = np.hypot(*a), np.hypot(*b)
    if na < 1e-6 or nb < 1e-6:
        return None
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def signed_angle(a, b):
    """Angle from a to b in IMAGE coordinates (y down), degrees -180..180.
    Positive = b is turned clockwise on screen from a."""
    a, b = _v(a), _v(b)
    if np.hypot(*a) < 1e-6 or np.hypot(*b) < 1e-6:
        return None
    cross = a[0] * b[1] - a[1] * b[0]
    return math.degrees(math.atan2(cross, float(np.dot(a, b))))


def facing_from_head(nose, ear_l, ear_r, shoulder_mid, min_ratio=FACING_MIN):
    """+1 if the face points to the RIGHT of the picture, -1 to the LEFT,
    None if the student is not clearly side-on."""
    nose, sm = _v(nose), _v(shoulder_mid)
    scale = float(np.hypot(*(nose - sm)))
    if scale < 1e-6:
        return None
    ratio = (nose[0] - 0.5 * (_v(ear_l)[0] + _v(ear_r)[0])) / scale
    if ratio >= min_ratio:
        return 1
    if ratio <= -min_ratio:
        return -1
    return None


def shoulder_spread(nose, sh_a, sh_b):
    """Shoulder separation over the nose-to-shoulders distance: wide facing
    the lens, small side-on. None if degenerate."""
    sm = 0.5 * (_v(sh_a) + _v(sh_b))
    scale = float(np.hypot(*(_v(nose) - sm)))
    if scale < 1e-6:
        return None
    return float(np.hypot(*(_v(sh_a) - _v(sh_b)))) / scale


def front_on(nose, sh_a, sh_b, min_ratio=FRONT_MIN):
    """True when the shoulders are spread across the picture like a student
    facing (or with their back to) the lens."""
    r = shoulder_spread(nose, sh_a, sh_b)
    return r is not None and r >= min_ratio


def expected_facing(side, mirrored):
    """Which way a student showing `side` to the lens faces in the picture.
    Un-mirrored, the LEFT side toward the camera faces picture-left (-1)."""
    f = -1 if side == SIDE_LEFT else 1
    return -f if mirrored else f


def expected_image_side(side, mirrored):
    """Where the tested arm of a student FACING the lens sits: +1 = the
    right half of the picture. Un-mirrored, their LEFT arm is on the right."""
    s = 1 if side == SIDE_LEFT else -1
    return -s if mirrored else s


def abduction_deg(shoulder, elbow, hip):
    """Upper arm against the trunk line: 0 at the side, 90 out level, 180 up."""
    return angle_between(_v(elbow) - _v(shoulder), _v(hip) - _v(shoulder))


def shoulder_extension_deg(shoulder, elbow, hip, facing):
    """How far the upper arm swings BEHIND the trunk line. Forward (flexion)
    reads negative. Measured against the trunk, so leaning doesn't add."""
    a = signed_angle(_v(hip) - _v(shoulder), _v(elbow) - _v(shoulder))
    return None if a is None else a * facing


def elbow_extension_deg(shoulder, elbow, wrist, facing):
    """0 = straight, minus = degrees short of straight, plus = over-straight.
    The elbow bends FORWARD, so the face direction tells a bent elbow from
    an over-straight one (both look like 'not 180' as a plain angle)."""
    a = signed_angle(_v(elbow) - _v(shoulder), _v(wrist) - _v(elbow))
    return None if a is None else a * facing


def knee_extension_deg(hip, knee, ankle, facing):
    """0 = straight, minus = short of straight. The knee bends BACKWARD,
    the opposite way to the elbow, hence the opposite sign."""
    a = signed_angle(_v(knee) - _v(hip), _v(ankle) - _v(knee))
    return None if a is None else -a * facing


def wrist_extension_deg(elbow, wrist, hand_wrist, knuckle, facing):
    """Hand line (wrist -> middle knuckle) against the forearm. With the
    palm down and the forearm pointing forward, bending the hand back
    (extension) lifts it; bending it down reads negative."""
    a = signed_angle(_v(wrist) - _v(elbow), _v(knuckle) - _v(hand_wrist))
    return None if a is None else -a * facing


def rotation_from_length(length, start_length):
    """External rotation from the forearm's on-screen length: pointing
    forward (side-on) it shows its full length, turned by t it shows cos t."""
    if start_length is None or start_length <= 1e-6:
        return None
    r = max(0.0, min(1.0, length / start_length))
    return math.degrees(math.acos(r))


def forearm_rotation_deg(pinky_knuckle, index_knuckle, lateral_sign):
    """The knuckle line's turn from upright, seen end-on (fist at the lens).
    Plus = the thumb side turned OUTWARD (away from the body) = supination,
    minus = turned inward = pronation."""
    k = _v(index_knuckle) - _v(pinky_knuckle)
    if np.hypot(*k) < 1e-6:
        return None
    return math.degrees(math.atan2(k[0] * lateral_sign, -k[1]))


# ── Landmarks -> one reading ─────────────────────────────────────────────────────
Reading = namedtuple("Reading", "extent problem segments")


def _px(lms, i, w, h):
    return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float64)


def _vis_ok(lms, idxs, thr=MIN_VIS):
    return all(getattr(lms[i], "visibility", 1.0) >= thr for i in idxs)


def _near_label(lms, chains):
    """The side-on limb nearest the lens: the better-seen chain, and the
    nearer one (MediaPipe z) when they are seen equally well. Read from the
    landmarks, not from their left/right names."""
    def vis(lab):
        return float(np.mean([getattr(lms[i], "visibility", 1.0) for i in chains[lab]]))

    def depth(lab):
        return float(np.mean([lms[i].z for i in chains[lab]]))

    vl, vr = vis("L"), vis("R")
    if abs(vl - vr) > 0.05:
        return "L" if vl > vr else "R"
    return "L" if depth("L") <= depth("R") else "R"


def _front_label(lms, side, mirrored):
    """The tested arm of a student facing the lens, by where it is in the
    picture rather than by its left/right name."""
    want = expected_image_side(side, mirrored)
    xl, xr = lms[ARM["L"][0]].x, lms[ARM["R"][0]].x
    if want > 0:
        return "L" if xl >= xr else "R"
    return "L" if xl < xr else "R"


def _match_hand(hands, wrist_px, upper_arm_px, w, h):
    """The detected hand whose wrist is nearest the arm's pose wrist, if it
    is close enough to be that arm's hand."""
    best, best_d = None, None
    for hl in hands or []:
        d = float(np.hypot(*(_px(hl, H_WRIST, w, h) - wrist_px)))
        if best_d is None or d < best_d:
            best, best_d = hl, d
    if best is None or best_d > HAND_MATCH_UPPERARM * upper_arm_px:
        return None
    return best


class JointMeter:
    """Turns one frame's landmarks into one angle (the reading) for the joint
    and side under test. Holds the only per-side memory a measurement needs:
    the external-rotation start length."""

    def __init__(self, joint):
        self.joint = JOINTS[joint] if isinstance(joint, str) else joint
        self.reset()

    def reset(self):
        self.start_len = {s: None for s in SIDES}
        self._len_buf = {s: deque(maxlen=ROT_SMOOTH) for s in SIDES}

    # The check every side-on joint shares: the right side faces the lens.
    def _side_view(self, lms, w, h, side, mirrored):
        nose, sl, sr = _px(lms, NOSE, w, h), _px(lms, 11, w, h), _px(lms, 12, w, h)
        spread = shoulder_spread(nose, sl, sr)
        facing = None
        if spread is not None and spread <= SIDE_MAX_SPREAD:
            facing = facing_from_head(nose, _px(lms, EAR_L, w, h), _px(lms, EAR_R, w, h),
                                      0.5 * (sl + sr))
        if facing is None:
            return None, (f"Turn SIDE-ON - the {side.upper()} side toward the camera, "
                          "face looking ahead")
        if facing != expected_facing(side, mirrored):
            return None, f"Turn around - the {side.upper()} side must face the camera"
        return facing, None

    def measure(self, lms, hands, w, h, side, mirrored):
        """lms: one body's 33 pose landmarks (normalised) or None. hands: the
        hand tracker's landmark lists (may be empty). Returns a Reading."""
        j = self.joint
        if lms is None:
            return Reading(None, "No student detected", [])

        if j.view == VIEW_FRONT:
            if not front_on(_px(lms, NOSE, w, h), _px(lms, 11, w, h), _px(lms, 12, w, h)):
                return Reading(None, "Face the camera", [])
            lab = _front_label(lms, side, mirrored)
            facing = None
        else:
            facing, problem = self._side_view(lms, w, h, side, mirrored)
            if problem:
                return Reading(None, problem, [])
            lab = _near_label(lms, LEG if j.key == "knee_extension" else
                              {k: v[:3] for k, v in ARM.items()})

        sh_i, el_i, wr_i, hip_i, pk_i, ix_i = ARM[lab]
        sh, el, wr = _px(lms, sh_i, w, h), _px(lms, el_i, w, h), _px(lms, wr_i, w, h)
        hip = _px(lms, hip_i, w, h)
        upper = float(np.hypot(*(el - sh)))

        if j.key == "shoulder_abduction":
            if not _vis_ok(lms, (sh_i, el_i, hip_i)):
                return Reading(None, "Arm or hip not visible", [])
            a = abduction_deg(sh, el, hip)
            return Reading(a, None, [(sh, el), (el, wr), (sh, hip)])

        if j.key == "shoulder_extension":
            if not _vis_ok(lms, (sh_i, el_i, hip_i)):
                return Reading(None, "Arm or hip not visible", [])
            a = shoulder_extension_deg(sh, el, hip, facing)
            return Reading(None if a is None else max(0.0, a), None,
                           [(sh, el), (el, wr), (sh, hip)])

        if j.key == "elbow_extension":
            if not _vis_ok(lms, (sh_i, el_i, wr_i)):
                return Reading(None, "Arm not fully visible", [])
            return Reading(elbow_extension_deg(sh, el, wr, facing), None,
                           [(sh, el), (el, wr)])

        if j.key == "knee_extension":
            hp_i, kn_i, an_i = LEG[lab]
            if not _vis_ok(lms, (hp_i, kn_i, an_i)):
                return Reading(None, "Hip, knee or ankle not visible", [])
            hp, kn, an = _px(lms, hp_i, w, h), _px(lms, kn_i, w, h), _px(lms, an_i, w, h)
            return Reading(knee_extension_deg(hp, kn, an, facing), None,
                           [(hp, kn), (kn, an)])

        if j.key == "wrist_extension":
            if not _vis_ok(lms, (sh_i, el_i, wr_i)):
                return Reading(None, "Arm not fully visible", [])
            hand = _match_hand(hands, wr, upper, w, h)
            if hand is not None:
                hw, kn = _px(hand, H_WRIST, w, h), _px(hand, H_MID_MCP, w, h)
            elif _vis_ok(lms, (pk_i, ix_i), 0.3):
                # The pose's own finger points: rougher, but better than nothing
                # when the hand tracker misses a side-on hand.
                hw, kn = wr, 0.5 * (_px(lms, pk_i, w, h) + _px(lms, ix_i, w, h))
            else:
                return Reading(None, "Hand not visible", [(el, wr)])
            a = wrist_extension_deg(el, wr, hw, kn, facing)
            return Reading(None if a is None else max(0.0, a), None,
                           [(el, wr), (hw, kn)])

        if j.key == "shoulder_external_rotation":
            if not _vis_ok(lms, (sh_i, el_i, wr_i)) or upper < 1.0:
                return Reading(None, "Arm not fully visible", [])
            length = float(np.hypot(*(wr - el))) / upper
            buf = self._len_buf[side]
            buf.append(length)
            med = float(sorted(buf)[len(buf) // 2])
            if self.start_len[side] is None or med > self.start_len[side]:
                self.start_len[side] = med
            a = rotation_from_length(length, self.start_len[side])
            if a is not None and a >= ROT_INWARD_MIN_DEG \
                    and lms[wr_i].z > lms[el_i].z + ROT_INWARD_Z:
                return Reading(0.0, "That is turning IN - turn the hand OUT",
                               [(sh, el), (el, wr)])
            return Reading(a, None, [(sh, el), (el, wr)])

        if j.key in ("forearm_supination", "forearm_pronation"):
            if not _vis_ok(lms, (sh_i, el_i)):
                return Reading(None, "Arm not visible", [])
            other = _px(lms, ARM["R" if lab == "L" else "L"][0], w, h)
            lateral = 1.0 if sh[0] >= other[0] else -1.0
            hand = _match_hand(hands, wr, upper, w, h)
            if hand is not None:
                pk, ix = _px(hand, H_PINKY_MCP, w, h), _px(hand, H_INDEX_MCP, w, h)
            elif _vis_ok(lms, (pk_i, ix_i), 0.3):
                pk, ix = _px(lms, pk_i, w, h), _px(lms, ix_i, w, h)
            else:
                return Reading(None, "Fist not visible - point it at the camera", [(sh, el)])
            r = forearm_rotation_deg(pk, ix, lateral)
            if r is None:
                return Reading(None, "Fist not visible - point it at the camera", [(sh, el)])
            if j.key == "forearm_pronation":
                r = -r
            return Reading(max(0.0, r), None, [(sh, el), (pk, ix)])

        return Reading(None, f"Unknown joint {j.key}", [])


# ── The 1-second hold ────────────────────────────────────────────────────────────
class HoldWindow:
    """The angle the joint has HELD over the last second: the HOLD_PCTL
    percentile of the readings in the window, once the window spans HOLD_S
    with enough readings. A gap longer than HOLD_GAP_S restarts it."""

    def __init__(self, hold_s=HOLD_S, gap_s=HOLD_GAP_S, pctl=HOLD_PCTL,
                 min_reads=MIN_HOLD_READS):
        self.hold_s, self.gap_s = float(hold_s), float(gap_s)
        self.pctl, self.min_reads = float(pctl), int(min_reads)
        self.buf = deque()       # (t, value)

    def reset(self):
        self.buf.clear()

    def push(self, t, v):
        if self.buf and t - self.buf[-1][0] > self.gap_s:
            self.buf.clear()
        self.buf.append((t, float(v)))
        # Keep exactly one reading at or before the window's start, so the
        # span test below can tell a full second from "nearly".
        while len(self.buf) >= 2 and self.buf[1][0] <= t - self.hold_s:
            self.buf.popleft()
        if len(self.buf) < self.min_reads or t - self.buf[0][0] < self.hold_s * 0.95:
            return None
        vals = sorted(x for _, x in self.buf)
        return vals[int(self.pctl * (len(vals) - 1))]


# ── The run: left side, then right side ──────────────────────────────────────────
class Phase(Enum):
    POSITION  = auto()   # this side not measured yet - getting into view
    MEASURING = auto()   # this side's 10 s are running
    DONE      = auto()   # both sides finished


class TargetStretchTracker:
    """One joint, left side then right side. Fed one angle per frame (or None
    when the frame couldn't be measured) with an injectable clock, so the
    whole run can be checked with made-up sequences - no camera.

    Each side keeps the best angle HELD for 1 s inside its 10 s window and
    turns it into clock degrees and a 0/1/2 score. The engine decides the
    score; the coach can only end a side early (NEXT) or RESET the run."""

    def __init__(self, joint, side_window_s=SIDE_WINDOW_S, no_reading_s=NO_READING_S,
                 hold_s=HOLD_S):
        self.joint = JOINTS[joint] if isinstance(joint, str) else joint
        self.side_window_s = float(side_window_s)
        self.no_reading_s = float(no_reading_s)
        self.hold_s = float(hold_s)
        self.reset()

    def reset(self):
        self.side_idx = 0
        self.phase = Phase.POSITION
        self.results = {s: None for s in SIDES}       # SideResult, or None = not measured
        self.end_reasons = {s: None for s in SIDES}
        self.ended_by = None
        self._hold = HoldWindow(hold_s=self.hold_s)
        self._new_side(None)

    def _new_side(self, now):
        self.current = None       # this frame's angle
        self.held = None          # angle held over the last second
        self.best = None          # best held angle on this side
        self.problem = None
        self._hold.reset()
        self._first_t = None      # start of the first steady second (starts the 10 s)
        self._side_t0 = now       # when this side began (no-reading backstop)
        self._now = now
        self.status_msg = self._position_msg()

    # ── read-only views ──────────────────────────────────────────────────
    @property
    def side(self):
        return SIDES[self.side_idx] if self.side_idx < len(SIDES) else None

    @property
    def resolved(self):
        return self.phase == Phase.DONE

    def time_left(self, now=None):
        if self._first_t is None or self.resolved:
            return None
        now = self._now if now is None else now
        return max(0.0, self.side_window_s - (now - self._first_t))

    @property
    def measured_any(self):
        return any(r is not None for r in self.results.values())

    def _position_msg(self):
        if self.side is None:
            return ""
        where = ("side toward the camera" if self.joint.view == VIEW_SIDE
                 else "arm - face the camera")
        return f"{self.side.upper()} {where}: {self.joint.instruction}"

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, extent, now=None, problem=None):
        """Feed this frame's angle (None = not measurable this frame, with
        `problem` saying why). Returns the result dict on the frame the run
        finishes, else None."""
        now = time.time() if now is None else now
        self._now = now
        if self.resolved:
            return None
        if self._side_t0 is None:
            self._side_t0 = now
        self.problem = problem

        if extent is None:
            self.current = None
            if problem:
                self.status_msg = problem
            elif self.phase == Phase.MEASURING:
                self.status_msg = "Hold the position in view"
            else:
                self.status_msg = self._position_msg()
            return self._check_time(now)

        self.current = float(extent)
        self.held = self._hold.push(now, extent)
        if self._first_t is None and self.held is not None:
            self._first_t = self._hold.buf[0][0]
            self.phase = Phase.MEASURING
        if self.held is not None and (self.best is None or self.held > self.best):
            self.best = self.held
        if problem:
            self.status_msg = problem
        elif self.best is None:
            self.status_msg = f"{self.side.upper()}: {self.joint.instruction} - and hold 1 s"
        else:
            r = side_result(self.joint, self.best)
            self.status_msg = (f"{self.side.upper()}: best held {r.deg} deg (score {r.score})"
                               " - go further and hold if you can")
        return self._check_time(now)

    def _check_time(self, now):
        if self._first_t is not None and now - self._first_t >= self.side_window_s:
            return self._end_side(now, "time")
        if self._first_t is None and now - self._side_t0 >= self.no_reading_s:
            return self._end_side(now, "not seen")
        return None

    # ── ending sides ─────────────────────────────────────────────────────
    def next_side(self, now=None):
        """Coach: end the side under way now, on what it has held so far."""
        if self.resolved:
            return None
        now = time.time() if now is None else now
        return self._end_side(now, "coach")

    def finish(self, now=None):
        """STOP pressed: close the side under way on what it has, leave any
        later side unmeasured. Returns the result if anything was measured."""
        now = time.time() if now is None else now
        if not self.resolved:
            self._end_side(now, "stopped")
            while not self.resolved:
                self._end_side(now, "stopped")
        return self.result() if self.measured_any else None

    def _end_side(self, now, why):
        side = self.side
        self.results[side] = None if self.best is None else side_result(self.joint, self.best)
        self.end_reasons[side] = why
        self.side_idx += 1
        done_msg = (f"{side.upper()}: {self.results[side].deg} deg, score "
                    f"{self.results[side].score}" if self.results[side]
                    else f"{side.upper()}: not measured")
        if self.side_idx >= len(SIDES):
            self.phase = Phase.DONE
            self.ended_by = why
            self.current = self.held = self.best = None
            self._first_t = None
            self.status_msg = "DONE - " + "   ".join(self._summary())
            return self.result()
        self.phase = Phase.POSITION
        self._new_side(now)
        self.status_msg = f"{done_msg}.  Now the {self.side.upper()} - {self._position_msg()}"
        return None

    def _summary(self):
        out = []
        for s in SIDES:
            r = self.results[s]
            out.append(f"{s.upper()} {r.deg} deg = {r.score}" if r else f"{s.upper()} --")
        return out

    def result(self):
        res = {"joint": self.joint.key, "joint_title": self.joint.title}
        for s in SIDES:
            r = self.results[s]
            res[f"{s}_deg"] = r.deg if r else None
            res[f"{s}_score"] = r.score if r else None
            res[f"{s}_exact"] = r.exact if r else None
        res["ended_by"] = self.ended_by
        res["rating"] = self.joint.title
        return res


# ── Drawing ──────────────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _ascii(txt):
    """cv2's Hershey fonts draw '?' for anything outside ASCII."""
    return txt.encode("ascii", "replace").decode("ascii")


def draw_hand_points(frame, hands, colour=(200, 200, 200)):
    h, w = frame.shape[:2]
    for hl in hands or []:
        for lm in hl:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, colour, -1, cv2.LINE_AA)


def draw_hud(frame, tracker, segments=None):
    """Draw the live state onto the frame. segments: the measured limb lines
    in pixels, so the coach sees exactly what the angle is taken from."""
    h, w = frame.shape[:2]
    tr = tracker
    colour = C_OK if tr.held is not None else (C_WARN if tr.current is not None else C_BAD)
    if tr.resolved:
        colour = C_OK

    for p1, p2 in segments or []:
        a, b = tuple(np.int32(p1)), tuple(np.int32(p2))
        cv2.line(frame, a, b, colour, 4, cv2.LINE_AA)
        cv2.circle(frame, a, 7, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, b, 7, colour, -1, cv2.LINE_AA)

    # Top bar: this side's 10 s.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    left = tr.time_left()
    if left is not None and tr.side_window_s > 0:
        cv2.rectangle(frame, (0, 0), (int(w * left / tr.side_window_s), 10), C_WARN, -1)

    draw_panel(frame, 12, 22, 560, 196)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _ascii(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2,
                    cv2.LINE_AA)
        y += dy

    line(f"TARGET STRETCH - {tr.joint.title.upper()}", colour, 0.64, 30)
    if not tr.resolved:
        line(f"Side: {tr.side.upper()} ({tr.side_idx + 1} of 2)" +
             (f"   {left:.0f} s left" if left is not None else ""))
        now_txt = "--" if tr.current is None else f"{tr.current:+.0f}" if tr.joint.straight \
            else f"{tr.current:.0f}"
        line(f"Now: {now_txt} deg   Held 1 s: " +
             ("--" if tr.held is None else f"{tr.held:.0f} deg"))
        if tr.best is not None:
            r = side_result(tr.joint, tr.best)
            line(f"Best held: {r.deg} deg -> score {r.score}", C_OK)
        else:
            line(f"Score 2 at {tr.joint.score2} deg, 1 at {tr.joint.score1} deg")
    for s in SIDES:
        r = tr.results[s]
        if r is not None:
            line(f"{s.upper()}: {r.deg} deg -> score {r.score}", C_OK)
        elif tr.end_reasons[s] is not None:
            line(f"{s.upper()}: not measured", C_BAD)

    if tr.status_msg:
        msg = _ascii(tr.status_msg)
        scale = 0.55 if len(msg) < 80 else 0.45
        draw_panel(frame, 12, h - 54, w - 12, h - 12)
        cv2.putText(frame, msg[:120], (24, h - 26), cv2.FONT_HERSHEY_SIMPLEX, scale, C_TEXT,
                    2, cv2.LINE_AA)

    if tr.resolved and tr.measured_any:
        big = "   ".join(f"{s[0].upper()} {tr.results[s].score}" if tr.results[s]
                         else f"{s[0].upper()} -" for s in SIDES)
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
        org = ((w - size[0]) // 2, (h + size[1]) // 2)
        draw_panel(frame, org[0] - 30, org[1] - size[1] - 24,
                   org[0] + size[0] + 30, org[1] + 24, C_DARK, 0.72)
        cv2.putText(frame, big, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, C_OK, 5, cv2.LINE_AA)
    return frame


def draw_pose(frame, lms):
    if lms is None:
        return
    h, w = frame.shape[:2]
    for i, j in _POSE_CONNECTIONS:
        if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
            cv2.line(frame, (int(lms[i].x * w), int(lms[i].y * h)),
                     (int(lms[j].x * w), int(lms[j].y * h)), (200, 200, 200), 2, cv2.LINE_AA)


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
    ap = argparse.ArgumentParser(description="Brockport Target Stretch Test")
    ap.add_argument("--joint", default=DEFAULT_JOINT, choices=list(JOINTS))
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--model", default=DEFAULT_POSE_MODEL)
    ap.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't selfie-flip the frame (back camera / clip)")
    args = ap.parse_args()

    _ensure_model(args.model)
    _ensure_model(args.hand_model)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source!r}")
    mirrored = not args.no_mirror

    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.hand_model),
        running_mode=VisionRunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45))

    tracker = TargetStretchTracker(args.joint)
    meter = JointMeter(args.joint)
    t0 = time.time()
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirrored:
            frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        ts = max(last_ts + 1, int((now - t0) * 1000))
        last_ts = ts
        res = pose.detect_for_video(image, ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        hands = hand_det.detect_for_video(image, ts).hand_landmarks if tracker.joint.hand else []
        draw_pose(frame, lms)
        draw_hand_points(frame, hands)
        segs = []
        if not tracker.resolved:
            rd = meter.measure(lms, hands, fw, fh, tracker.side, mirrored)
            segs = rd.segments
            if tracker.update(rd.extent, now=now, problem=rd.problem) is not None:
                print(tracker.result())
        draw_hud(frame, tracker, segs)
        cv2.imshow("Target Stretch", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n") and tracker.next_side(now) is not None:
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
