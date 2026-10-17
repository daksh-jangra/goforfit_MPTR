"""
Shoulder Stretch Test – fingertips touching behind the back
=================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual, figure 5.27)
--------------------------------------------------------------------
  The participant reaches over one shoulder and down the back with one arm,
  and across and up the back with the other, and tries to touch the
  fingertips of the two hands together between the shoulder blades.

  The trial is designated RIGHT or LEFT by the arm reaching OVER THE
  SHOULDER — the right arm over the right shoulder is a RIGHT-arm stretch,
  whichever way the other hand comes up.

  No equipment. ONE trial. Scored PASS/FAIL: the item is PASSED if the
  fingers touch.

  So this test is pass/fail, not a rep count and not a measurement.

How "the fingers touch" is measured here
--------------------------------------------------------------------
  "Touching" has no number in the manual, and a camera cannot see skin
  contact — it sees two clusters of fingertips that are either on top of
  each other or not. The metric is therefore the smallest distance between
  ANY fingertip of one hand and ANY fingertip of the other (5 x 5 pairs),
  divided by the mean PALM LENGTH of the two hands.

  Normalising by the hand's own size is what makes one threshold work for
  a small child and a tall adolescent, at whatever distance from the camera
  they happen to stand. It is deliberately NOT normalised by shoulder width
  — shoulder width is a body scale, not a hand scale, and swapping them
  silently changes what PASS_RATIO means (see the constant below).

What the app scores vs. what the coach judges
--------------------------------------------------------------------
  SCORED (it is the whole item):
    * the fingertips of the two hands meeting, held long enough to be a
      real touch rather than one lucky frame

  NOT SCORED (the tester's job):
    * which hand goes over and which comes up — the app only REPORTS the
      side it saw, it never corrects the participant
    * trunk posture, head position, whether the elbow points up
    * "close enough" judgements. The engine decides this trial from the
      measurement; the coach ends the trial, they do not overrule it.

Which side
--------------------------------------------------------------------
  The over-the-shoulder arm is the one whose elbow is ABOVE its own
  shoulder (image y decreases upward), so the side is decided by comparing
  (shoulder.y - elbow.y) for the left arm (11/13) against the right arm
  (12/14). On a twisted torso BOTH elbows can read above their shoulders,
  so a clear margin between the two sides is required before the reading is
  trusted at all — an ambiguous frame votes for nothing. SIDE_LEFT /
  SIDE_RIGHT pin it by hand, exactly as reverse_curl.py lets the coach pin
  the working arm.

  MediaPipe's hand "handedness" labels are NOT used: they assume a front /
  mirror view, and this test is shot from behind, where they are wrong.

Usage
-----
  python shoulder_stretch.py
  python shoulder_stretch.py --source clip.mp4
  python shoulder_stretch.py --pass-ratio 0.25  # loosen the touch threshold
  python shoulder_stretch.py --debug

  Keys : E = end the trial now (records FAIL unless BOTH sides were done)
         R = reset the whole test
         Q = quit and print the summary

  There is no side pin: BOTH sides must be touched inside one 15 s trial, so
  the arm is named per frame from the hands and follows the student when
  they switch. Pinning a side would make the other side impossible to bank.

Camera placement
----------------
  Phone/webcam BEHIND the participant at about shoulder height, 1.5-2 m
  away, with both shoulders, both elbows and the whole back in frame. The
  hands meet BETWEEN THE SHOULDER BLADES, so a front view cannot see the
  contact at all — it sees two elbows and nothing else. Use the back lens
  and do NOT mirror the frame: mirroring swaps the pose's left and right
  landmarks, which would invert the RIGHT/LEFT trial label.

  Only the STUDENT should be in frame when the camera starts: the app locks
  onto the one body it sees then and follows it, and it will not lock while
  two people are in frame. A coach who steps in afterwards is ignored.

  Hand detection is intermittent in this pose — the hands are partly hidden
  by the torso and are seen from the back. That is expected; the tracker
  survives dropouts instead of assuming continuous landmarks.

Model files
-----------
  Download the .task models from:
  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
  e.g. pose_landmarker_full.task + hand_landmarker.task  (recommended)
  Both are looked for next to this file; pass --model / --hand-model to
  override, and a missing file is downloaded once on first run.
"""

import argparse
import os
import sys
import time
from collections import deque, namedtuple
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python           # noqa: F401  (parity with snr.py)
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# Re-exported so web_engines.py can build both detectors through this module
# (`import shoulder_stretch as ss_mod`) without importing mediapipe itself,
# the way it already does for reverse_curl.py.
# NOTE: the legacy `mediapipe.solutions` module is intentionally NOT imported
# anywhere here. Some MediaPipe wheels ship without it and the ImportError
# takes every live engine down with it; skeletons are drawn with OpenCV.
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode


# ── Protocol constants ───────────────────────────────────────────────────────────
# There is no number in the manual: one trial, and it passes if the fingers
# touch. Everything below is detection tuning for "touch", not protocol.
TRIALS = 1

# ── Detection tuning ─────────────────────────────────────────────────────────────
# PASS threshold for the contact metric (min fingertip gap / mean palm length).
# Calibrated from real footage rather than guessed:
#     genuine fingertip touches measured 0.09 - 0.15
#     a clear ~3.5-inch (9 cm) miss measured 0.96
#     interpolating between those puts a 1-inch (2.5 cm) miss at ~0.26
# 0.20 sits between a real touch and a 1-inch miss — roughly 2 cm of
# tolerance, which is about the landmark error of a fingertip seen from
# behind. Raising it past ~0.26 starts passing visible gaps; dropping it
# below ~0.15 starts failing touches the coach can see.
PASS_RATIO = 0.20

# Sustained-touch rule: K qualifying readings (ratio <= PASS_RATIO) inside a
# ROLLING window of T seconds. Old readings expire; a dropout never resets the
# window (expiry handles staleness); a reading above TOUCH_RELEASE_RATIO is
# the hands really coming apart and clears it. One frame under the threshold
# is landmark jitter, and a MINIMUM over the trial would bias every trial
# toward PASS, so neither decides the verdict.
#
# The numbers come from a Monte-Carlo of this exact rule (scratchpad mc/sim.c,
# 20 000 trials per cell, 12 fps clock, readings = truth + 0.05 sd noise, half
# OU (tau 0.5 s) half white; a 1-inch miss is truth 0.26, a touch is 0.12):
#   * the previous rule (4 detections, no time window, reset only above 0.28)
#     false-PASSed a 1-inch miss 9 / 18 / 32 / 54 / 80 % at 5/10/20/40/80 s —
#     it was a ratchet that converged on PASS. And at ~1 reading/s its 0.35 s
#     dropout grace reset the count between almost every pair of readings,
#     so a genuine 10 s touch FAILED 71 % of the time.
#   * K=5 in T=10 s with release 0.24, at 1 reading/s: false-PASS 2.3 % over
#     80 s, false-FAIL 4.8 % on a 10 s touch. No (K, T) meets both <=2 % and
#     <=5 % without a cap, because false-PASS keeps climbing with trial length
#     under ANY windowed rule (0.2 / 0.4 / 1.1 / 2.3 % at 10/20/40/80 s).
#   * so the trial is CAPPED at 20 s from the first near-touch: 0.4 % false
#     PASS / 4.8 % false FAIL at 1 reading/s; 2.6 % false PASS at 2/s. At
#     0.5 reading/s a 10 s touch fails ~50 %: the fix for that is yield, not K.
#   * release 0.24 (was 0.28) was the single biggest lever on false-PASS.
#   * the rule wants the touch HELD until the bar fills — about 8-10 s at
#     1 reading/s (a 5 s hold fails ~50 % there). At the 4+ readings/s
#     measured on the single-person clip at 12 fps it fills in ~1-2 s.
TOUCH_READINGS      = 5      # K
TOUCH_WINDOW_S      = 6.0    # T (was 10 s; the whole trial is now 15 s)
TOUCH_RELEASE_RATIO = 0.24   # a reading between PASS_RATIO and this adds
                             # nothing and clears nothing; above it clears.
# BOTH sides must be touched inside ONE 15 s trial, and the clock starts at
# the first frame that shows the stretch (a real attempt), not at camera start,
# so getting into position does not eat the trial.
TRIAL_CAP_S         = 15.0   # from the first reading of the trial
# A separate, longer backstop: the student is on camera but NOTHING is ever
# measured (bad angle, poor light, hands never found). Without this the trial
# stays open forever, because the 15 s clock only starts at the first reading.
# It resolves as "not completed" rather than a fail — nothing was scored.
NO_ATTEMPT_CAP_S    = 30.0   # from when the student is locked on
TOUCH_FRAMES = TOUCH_READINGS   # old name, kept: web_engines and the CLI pass
                                # touch_frames= and it now means K.

MIN_PALM_PX = 12.0       # below this the hand is too small/far in frame for a
                         # normalised gap to mean anything — the ratio is
                         # reported as unmeasurable rather than as a huge
                         # number that would read as a confident FAIL.

# ── Whose hands: the participant lock and the behind-the-back gate ───────────
# Every allowance is in TORSO-DIAGONAL units (max of shoulder->opposite hip),
# not shoulder separation: seen from behind with one arm up, the shoulder
# separation collapses to 0.10-0.62 of the torso diagonal, and on f_011 it
# made the participant's own hand look 0.82 "shoulders" from her wrist.
#
# No per-hand wrist test: behind the back the pose wrist is the landmark that
# goes wrong (f_011 vis 0.36, 90 px off), the elbow is a forearm away by
# construction, and on a_088 a bystander's hands match HIS OWN pose wrists at
# 0.02 td — no per-hand threshold can say "this pose is not the participant".
# Identity is the ParticipantLock's job; the gate below is plausibility.
#
# Contact box: the midpoint of the closest fingertip pair, in the locked
# body's frame (origin shoulder midpoint, u along 11->12, v perpendicular and
# pointing down the image), both in td. Measured on every real touch
# (raw <= 0.28) across the clips: 196 of 197 inside; |u| p95 0.15, v from
# -0.09 (p5) to +0.28 (max). Bystander self-touches against the participant:
# 0 of 31 inside (a_088 u = -0.78, a_109 v = +0.75).
CONTACT_U_MAX_TD = 0.35
CONTACT_V_MIN_TD = -0.30
CONTACT_V_MAX_TD = 0.45
# Both hand wrists (hand lm 0) within this of the locked body's shoulder
# midpoint. Genuine max 0.79; bystander min 1.12, a_109 1.09, a_088 1.15.
HAND_R_MAX_TD = 1.00

# Participant lock. Seeded only when EXACTLY ONE body is in frame (the field
# condition: the student alone at the start), then held by centroid
# continuity. Size, centrality and facing were all measured to pick the
# coach, so none of them seeds. At 12 fps the participant's shoulder midpoint
# moves p99 0.11 td frame to frame and 0.13-0.27 td across a camera cut; the
# coach stands >= 0.66 td away (p1 0.87).
LOCK_JUMP_TD = 0.35
BODY_HOLD_S  = 1.0       # reuse the last locked body this long when the pose
                         # detector returns nobody (6 % of rear-view frames
                         # have hands but no pose, e.g. a_019).
NUM_POSES    = 2         # the pose detector must SEE a second body for the
                         # lock to refuse it: at num_poses=1 MediaPipe locked
                         # onto the coach on 98-100 % of two-person frames.

# Side detection: the over-the-shoulder elbow must beat the other side by
# this much, as a fraction of the shoulder separation. On a twisted torso
# both elbows can read above their shoulders with only ~10 px between them,
# which is noise — at a typical 150 px shoulder width this asks for ~18 px.
SIDE_MARGIN_FRAC = 0.12

SMOOTH_FRAMES   = 3      # median window for the HUD / gap_ratio DISPLAY only.
                         # The tracker gets the raw per-frame ratio: feeding
                         # it smoothed values raised false-PASS from 2.3 % to
                         # 4.4 % (overlapping medians are correlated, and the
                         # K-in-T window already does the de-jittering).
SMOOTH_MAX_AGE_S = 2.0   # samples older than this leave the window; this
                         # replaces the old reset-on-every-dropout, which kept
                         # the window at one sample under normal dropouts.
MIN_VIS         = 0.50   # landmark visibility floor for the pose joints.
                         # Pose landmarks only — HandLandmarker landmarks have
                         # no .visibility attribute at all.

SIDE_AUTO, SIDE_LEFT, SIDE_RIGHT = "auto", "left", "right"

# Landmark indices (33-point BlazePose topology)
IDX_L_SHOULDER, IDX_R_SHOULDER = 11, 12
IDX_L_ELBOW,    IDX_R_ELBOW    = 13, 14
IDX_L_WRIST,    IDX_R_WRIST    = 15, 16
IDX_L_HIP,      IDX_R_HIP      = 23, 24

SIDE_IDX = {
    SIDE_LEFT:  (IDX_L_SHOULDER, IDX_L_ELBOW, IDX_L_WRIST),
    SIDE_RIGHT: (IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST),
}
POSE_WRIST_IDX = (IDX_L_WRIST, IDX_R_WRIST)

# Hand landmark indices (21-point MediaPipe hand topology)
HAND_WRIST_IDX   = 0
HAND_MID_MCP_IDX = 9                    # middle-finger knuckle
FINGERTIP_IDX    = (4, 8, 12, 16, 20)   # thumb, index, middle, ring, little

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
]

# HUD palette (matches parcurlup.py / trunk_lift.py / reverse_curl.py)
C_DARK  = (28, 28, 28)
C_OK    = (80, 220, 120)
C_WARN  = (60, 190, 250)
C_BAD   = (80, 80, 240)
C_TEXT  = (245, 245, 245)

# Model files live next to this script, like every other test in tests/.
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


class State(Enum):
    READY    = auto()   # no usable two-hand reading yet
    REACHING = auto()   # both hands seen and owned, fingertips still apart
    TOUCHING = auto()   # under the threshold, banking the sustained detections
    PASSED   = auto()   # the fingers touched
    FAILED   = auto()   # the single trial ended without a touch


# ── small helpers ────────────────────────────────────────────────────────────────
def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def vis(lms, *idxs, thr=MIN_VIS):
    """POSE landmarks only — hand landmarks carry no .visibility."""
    return all(lms[i].visibility >= thr for i in idxs)


def px_dist(p1, p2):
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def pose_body_scale(pose_lms, w, h):
    """Shoulder separation in pixels, or None if the shoulders aren't clear.

    From behind, this is the only stable body scale in frame — the hips are
    often cropped and the wrists are exactly what we are trying to validate.
    It scales the ownership gate and the side margin, never the contact
    metric (that one is scaled by palm length, deliberately)."""
    if not vis(pose_lms, IDX_L_SHOULDER, IDX_R_SHOULDER):
        return None
    ls = lm_px(pose_lms, IDX_L_SHOULDER, w, h)
    rs = lm_px(pose_lms, IDX_R_SHOULDER, w, h)
    d = px_dist(ls, rs)
    return d if d > 1.0 else None


def palm_length_px(hand_lms, w, h):
    """Wrist (0) to middle-finger MCP (9) — the hand's own size in pixels."""
    wrist = np.array([hand_lms[HAND_WRIST_IDX].x * w,
                      hand_lms[HAND_WRIST_IDX].y * h], dtype=np.float32)
    mcp = np.array([hand_lms[HAND_MID_MCP_IDX].x * w,
                    hand_lms[HAND_MID_MCP_IDX].y * h], dtype=np.float32)
    return px_dist(wrist, mcp)


def fingertip_px(hand_lms, w, h):
    """The five fingertips of one hand as pixel points."""
    return [np.array([hand_lms[i].x * w, hand_lms[i].y * h], dtype=np.float32)
            for i in FINGERTIP_IDX]


def torso_diagonal_px(body_lms, w, h):
    """The body scale every ownership allowance is expressed in: the longer of
    the two torso diagonals, left shoulder -> right hip (11 -> 24) and right
    shoulder -> left hip (12 -> 23), in pixels. None if degenerate.

    No visibility gate, on purpose: from behind with the arms folded up the
    back MediaPipe under-scores these joints on perfectly good frames, and the
    diagonal is only a SCALE — a few px of error moves no decision. Unlike the
    shoulder separation it does not collapse when the torso turns."""
    ls = lm_px(body_lms, IDX_L_SHOULDER, w, h)
    rs = lm_px(body_lms, IDX_R_SHOULDER, w, h)
    lh = lm_px(body_lms, IDX_L_HIP, w, h)
    rh = lm_px(body_lms, IDX_R_HIP, w, h)
    td = max(px_dist(ls, rh), px_dist(rs, lh))
    return td if td > 1.0 else None


def shoulder_mid_px(body_lms, w, h):
    """Midpoint of the two shoulders in pixels — the body's anchor point for
    the participant lock and the origin of its back frame."""
    return 0.5 * (lm_px(body_lms, IDX_L_SHOULDER, w, h)
                  + lm_px(body_lms, IDX_R_SHOULDER, w, h))


def back_frame_uv(point, body_lms, w, h):
    """(u, v) of an image point in the body's own back frame, in torso-diagonal
    units, or None if the body has no usable scale.

    Origin at the shoulder midpoint; u along left shoulder -> right shoulder
    (11 -> 12); v perpendicular to that, oriented DOWN the image, so a point
    between the shoulder blades has small |u| and small positive v."""
    td = torso_diagonal_px(body_lms, w, h)
    if td is None:
        return None
    ls = lm_px(body_lms, IDX_L_SHOULDER, w, h)
    rs = lm_px(body_lms, IDX_R_SHOULDER, w, h)
    axis = rs - ls
    n = float(np.hypot(axis[0], axis[1]))
    if n < 1e-6:
        return None
    ax = axis / n
    perp = np.array([-ax[1], ax[0]], dtype=np.float32)
    if perp[1] < 0:
        perp = -perp
    rel = np.asarray(point, dtype=np.float32) - 0.5 * (ls + rs)
    return float(np.dot(rel, ax) / td), float(np.dot(rel, perp) / td)


def nearest_pose_wrist(hand_lms, pose_lms, w, h):
    """ADVISORY ONLY — not part of the gate. (pose landmark index, distance
    px, torso diagonal px) for this hand's wrist against the nearer of the
    body's two arm anchors, or None if the body has no scale.

    Each arm's anchor is its pose WRIST or ELBOW, whichever the model scores
    as more visible: behind the back the wrist is exactly the landmark that
    goes unreliable (f_011: vis 0.36, 90 px off) while the elbow stays put.
    Kept for debugging / the HUD; frame_gap_ratio does not use it, because a
    per-hand distance cannot tell whose POSE it is (a_088: a bystander's
    hands match his own wrists at 0.02 td)."""
    td = torso_diagonal_px(pose_lms, w, h)
    if td is None:
        return None
    hand_wrist = np.array([hand_lms[HAND_WRIST_IDX].x * w,
                           hand_lms[HAND_WRIST_IDX].y * h], dtype=np.float32)
    best_idx, best_d = None, None
    for i_el, i_wr in ((IDX_L_ELBOW, IDX_L_WRIST), (IDX_R_ELBOW, IDX_R_WRIST)):
        idx = (i_wr if pose_lms[i_wr].visibility >= pose_lms[i_el].visibility
               else i_el)
        d = px_dist(hand_wrist, lm_px(pose_lms, idx, w, h))
        if best_d is None or d < best_d:
            best_idx, best_d = idx, d
    return best_idx, best_d, td


def hand_belongs_to_pose(hand_lms, pose_lms, w, h, max_td=HAND_R_MAX_TD):
    """Is this hand within reach of this body: its wrist (hand lm 0) no further
    than max_td torso diagonals from the body's shoulder midpoint? This is the
    radius half of the frame gate. It does NOT establish identity — that is
    ParticipantLock's job — it only rejects hands a body away."""
    td = torso_diagonal_px(pose_lms, w, h)
    if td is None:
        return False
    hand_wrist = np.array([hand_lms[HAND_WRIST_IDX].x * w,
                           hand_lms[HAND_WRIST_IDX].y * h], dtype=np.float32)
    return px_dist(hand_wrist, shoulder_mid_px(pose_lms, w, h)) <= max_td * td


class ParticipantLock:
    """WHOSE body the trial is measured against.

    The field norm is one student in front of the camera; this is the guard
    for a coach stepping into frame, not a multi-person tracker.

      * SEEDS only on a frame with EXACTLY ONE body (the student alone at the
        start). With two or more in frame it stays unlocked — nothing in the
        pose data picks the participant out of two (size, centrality and
        facing were all measured to pick the coach).
      * HOLDS by centroid continuity: each frame, the body whose shoulder
        midpoint is nearest the locked one, accepted only within
        LOCK_JUMP_TD x td. Losing the body is NOT a re-seed; only reset()
        (or an explicit seed()) unlocks it.
      * body_for(now) is the matched body, or the last one for up to
        hold_s when the pose detector returns nobody — hands are often found
        on frames where the pose is not.

    Geometry is in pixels, so w/h are passed with the bodies (normalised
    landmarks on a non-square frame are not isotropic)."""

    def __init__(self, jump_td=LOCK_JUMP_TD, hold_s=BODY_HOLD_S):
        self.jump_td = float(jump_td)
        self.hold_s = float(hold_s)
        self.reset()

    def reset(self):
        self.centroid = None       # shoulder midpoint of the locked body, px
        self.td = None             # its torso diagonal, px
        self.last_body = None      # its last matched landmarks
        self.last_seen_t = None
        self.fresh = None          # body matched on the latest update(), or None

    @property
    def locked(self):
        return self.centroid is not None

    def seed(self, body_lms, w, h, now=None):
        """Lock onto this body explicitly (a harness, or a future coach
        'this is the student' tap). Returns True if it had a usable scale."""
        td = torso_diagonal_px(body_lms, w, h)
        if td is None:
            return False
        now = time.time() if now is None else now
        self.centroid = shoulder_mid_px(body_lms, w, h)
        self.td = td
        self.last_body = body_lms
        self.last_seen_t = now
        self.fresh = body_lms
        return True

    def update(self, bodies, w, h, now=None):
        """bodies: this frame's pose_landmarks list (may be empty). Returns the
        participant's body if it was matched on THIS frame, else None."""
        now = time.time() if now is None else now
        self.fresh = None
        bodies = [b for b in (bodies or []) if torso_diagonal_px(b, w, h) is not None]
        if not bodies:
            return None
        if not self.locked:
            if len(bodies) == 1:
                self.seed(bodies[0], w, h, now)
                return self.fresh
            return None
        best, best_d = None, None
        for b in bodies:
            d = px_dist(shoulder_mid_px(b, w, h), self.centroid)
            if best_d is None or d < best_d:
                best, best_d = b, d
        if best_d > self.jump_td * self.td:
            return None
        self.centroid = shoulder_mid_px(best, w, h)
        self.td = torso_diagonal_px(best, w, h)
        self.last_body = best
        self.last_seen_t = now
        self.fresh = best
        return best

    def body_for(self, now=None):
        """The body to measure this frame against: the one matched on the
        latest update(), else the last one if seen within hold_s, else None."""
        if self.fresh is not None:
            return self.fresh
        if self.last_body is None:
            return None
        now = time.time() if now is None else now
        return self.last_body if now - self.last_seen_t <= self.hold_s else None


def _r3(v):
    return round(v, 3) if v is not None else None


GapReading = namedtuple("GapReading",
                        "ratio gap_px palm_px point_a point_b")


def fingertip_gap_ratio(hand_a, hand_b, w, h):
    """Contact metric for two hands: the smallest distance over all 5 x 5
    fingertip pairs, divided by the mean palm length of the two hands.

    Returns a GapReading (ratio, gap in px, mean palm length in px, and the
    two fingertip points that were closest — the HUD draws that pair) or None
    when the hands are too small in frame for the ratio to mean anything.

    No cv2, no HUD, no state: the web engine and the logic checks call this
    directly with landmark lists."""
    palm_a = palm_length_px(hand_a, w, h)
    palm_b = palm_length_px(hand_b, w, h)
    palm = 0.5 * (palm_a + palm_b)
    if palm < MIN_PALM_PX:
        return None

    tips_a = fingertip_px(hand_a, w, h)
    tips_b = fingertip_px(hand_b, w, h)
    best_d, best_pair = None, None
    for pa in tips_a:
        for pb in tips_b:
            d = px_dist(pa, pb)
            if best_d is None or d < best_d:
                best_d, best_pair = d, (pa, pb)

    # The + 1e-6 guard is the same idiom the angle helpers use — palm is
    # already floored by MIN_PALM_PX, this only rules out a division by an
    # exact zero if that floor is ever tuned down.
    ratio = float(best_d / (palm + 1e-6))
    return GapReading(ratio=ratio, gap_px=float(best_d), palm_px=float(palm),
                      point_a=best_pair[0], point_b=best_pair[1])


def contact_on_back(reading, hand_lms_list, body_lms, w, h):
    """The behind-the-back plausibility gate for one GapReading against the
    locked body. Returns (ok, why, u, v, r_max) so probes can see which half
    rejected a frame; frame_gap_ratio only uses ok.

      a) the contact point (midpoint of the closest fingertip pair) must lie on
         the back: |u| <= CONTACT_U_MAX_TD and
         CONTACT_V_MIN_TD <= v <= CONTACT_V_MAX_TD in the body's back frame;
      b) both hand wrists within HAND_R_MAX_TD of the shoulder midpoint.

    A chest-height gesture by someone standing beside the participant fails
    (a) even when a pose wrist is hallucinated onto their hand (a_109)."""
    td = torso_diagonal_px(body_lms, w, h)
    if td is None:
        return False, "no body scale", None, None, None
    contact = 0.5 * (np.asarray(reading.point_a, dtype=np.float32)
                     + np.asarray(reading.point_b, dtype=np.float32))
    uv = back_frame_uv(contact, body_lms, w, h)
    if uv is None:
        return False, "no body scale", None, None, None
    u, v = uv
    mid = shoulder_mid_px(body_lms, w, h)
    r_max = max(px_dist(np.array([hl[HAND_WRIST_IDX].x * w,
                                  hl[HAND_WRIST_IDX].y * h], dtype=np.float32), mid)
                for hl in hand_lms_list[:2]) / td
    if abs(u) > CONTACT_U_MAX_TD or not (CONTACT_V_MIN_TD <= v <= CONTACT_V_MAX_TD):
        return False, "contact not on the back", u, v, r_max
    if r_max > HAND_R_MAX_TD:
        return False, "hand too far from the body", u, v, r_max
    return True, "ok", u, v, r_max


HAND_DIR_MIN = 0.5   # |down-ness| each hand needs. Measured (palm-normalised
                     # wrist->knuckle, +1 = straight down): real touches, top
                     # hand +0.74..+0.97 and bottom hand -0.95..-1.00; a
                     # coach's hands -0.28..-0.04; facing the camera both
                     # hands point the same way. 0.5 sits between them.


def hand_down_ness(hand_lms, w, h):
    """+1 if the hand points straight down (wrist above the knuckles), -1 if
    straight up, 0 if sideways — the wrist->middle-knuckle direction."""
    wr = np.array([hand_lms[HAND_WRIST_IDX].x * w, hand_lms[HAND_WRIST_IDX].y * h])
    mc = np.array([hand_lms[HAND_MID_MCP_IDX].x * w, hand_lms[HAND_MID_MCP_IDX].y * h])
    palm = float(np.hypot(*(mc - wr)))
    return 0.0 if palm < 1e-6 else float((mc[1] - wr[1]) / palm)


def stretch_hands(hand_pair, w, h, min_dir=HAND_DIR_MIN):
    """The shape of the stretch, read from the hands alone: one points DOWN
    (over the shoulder) and the other points UP (up the back). This is what
    rejects hands brought together in front of the face or chest — the false
    PASS seen on a laptop — and a bystander's gesturing hands. The pose's
    elbows can't do this job: from behind the model often reports both
    elbows down while one is plainly raised."""
    if len(hand_pair) < 2:
        return False
    d = sorted(hand_down_ness(hl, w, h) for hl in hand_pair[:2])
    return d[0] <= -min_dir and d[1] >= min_dir


SIDE_HAND_MARGIN_TD = 0.03   # both cues must clear this much of a torso
                             # diagonal. Measured on real touches: the
                             # nearest-shoulder margin ran 0.06-0.18 td and
                             # the midline offset 0.06-0.08 td.


def side_from_hands(hand_pair, body_lms, w, h, margin_td=SIDE_HAND_MARGIN_TD):
    """Which arm reaches OVER the shoulder, read from the HANDS: the hand
    pointing DOWN is by definition the one coming over the shoulder, so the
    trial is named by which shoulder it came from.

    Two independent cues must AGREE, or this says nothing: the down hand's
    wrist is nearer that shoulder, and it sits on that side of the spine.
    Both are read from the picture (un-mirrored, filmed from behind), never
    from MediaPipe's left/right labels.

    This replaces the elbows for naming: from behind the pose often reports
    both elbows down while one is plainly raised, and on real touch frames
    the elbow rule named 1 of 3 while this named 3 of 3."""
    if body_lms is None or len(hand_pair) < 2:
        return None
    td = torso_diagonal_px(body_lms, w, h)
    if td is None:
        return None
    down = max(hand_pair[:2], key=lambda hl: hand_down_ness(hl, w, h))
    if hand_down_ness(down, w, h) < HAND_DIR_MIN:
        return None
    wrist = np.array([down[HAND_WRIST_IDX].x * w, down[HAND_WRIST_IDX].y * h],
                     dtype=np.float32)
    # Image-left / image-right shoulder: from behind these ARE the
    # participant's left and right (fig 5.27, "right shoulder").
    sh_l, sh_r = sorted((lm_px(body_lms, IDX_L_SHOULDER, w, h),
                         lm_px(body_lms, IDX_R_SHOULDER, w, h)),
                        key=lambda p: p[0])
    d_l, d_r = px_dist(wrist, sh_l) / td, px_dist(wrist, sh_r) / td
    if abs(d_l - d_r) < margin_td:
        return None
    by_shoulder = SIDE_RIGHT if d_r < d_l else SIDE_LEFT

    offset = (wrist[0] - shoulder_mid_px(body_lms, w, h)[0]) / td
    if abs(offset) < margin_td:
        return None
    by_spine = SIDE_RIGHT if offset > 0 else SIDE_LEFT

    return by_shoulder if by_shoulder == by_spine else None


def frame_gap_ratio(hand_lms_list, body_lms, w, h):
    """The whole per-frame measurement: contact metric, then the
    behind-the-back gate against the locked participant.

    `hand_lms_list` is h_result.hand_landmarks (0, 1 or 2 hands).
    `body_lms` is the LOCKED participant's 33 pose landmarks — normally
    ParticipantLock.body_for(now), which may be the body reused from a recent
    frame when this frame has no pose. It is NOT "whatever pose MediaPipe
    returned first": with a second person in frame that is often the coach.

    Returns a GapReading, or None when this frame cannot be measured (no
    locked body, fewer than two hands, hands too small, or a contact that is
    not on the participant's back / hands a body away). A None is fed to the
    tracker as a dropout."""
    if body_lms is None or not hand_lms_list or len(hand_lms_list) < 2:
        return None
    # Only the stretch counts: one hand coming DOWN over the shoulder, the
    # other coming UP the back. Hands together anywhere else are not a try.
    if not stretch_hands(hand_lms_list[:2], w, h):
        return None
    hand_a, hand_b = hand_lms_list[0], hand_lms_list[1]
    reading = fingertip_gap_ratio(hand_a, hand_b, w, h)
    if reading is None:
        return None
    ok = contact_on_back(reading, hand_lms_list, body_lms, w, h)[0]
    return reading if ok else None


def detect_side(pose_lms, w, h, margin_frac=SIDE_MARGIN_FRAC):
    """Which arm is reaching OVER the shoulder: SIDE_LEFT, SIDE_RIGHT or None.
    It doubles as the POSTURE check: None means "this is not the stretch".

    The stretch posture (Brockport fig 5.27) is one elbow clearly ABOVE its
    shoulder (the over-the-shoulder arm) and the other elbow BELOW its
    shoulder (the arm reaching up the back). Hands held together in front of
    the face or chest have both elbows down, so they are not the stretch —
    that is the false PASS a coach saw on a laptop before anyone moved.

    The side is decided by WHERE the raised elbow is in the picture, never by
    MediaPipe's left/right labels: the labels assume the person faces the
    camera, and from behind they come out swapped. The frame must be
    UN-mirrored and shot from behind, so the participant's right arm is on
    the right of the picture (as in fig 5.27, "right shoulder").

    None when either arm's shoulder/elbow isn't visible, when no elbow is up
    by margin_frac of the shoulder separation, or when the other elbow is not
    lower by that margin too (a twisted torso reads both elbows up by nearly
    the same amount — too close to name a side)."""
    scale = pose_body_scale(pose_lms, w, h)
    if scale is None:
        return None

    arms = []
    for i_sh, i_el in ((IDX_L_SHOULDER, IDX_L_ELBOW), (IDX_R_SHOULDER, IDX_R_ELBOW)):
        if not vis(pose_lms, i_sh, i_el):
            return None          # one arm can't make a comparison
        sh = lm_px(pose_lms, i_sh, w, h)
        el = lm_px(pose_lms, i_el, w, h)
        arms.append((float(sh[1] - el[1]), el))   # + = elbow above the shoulder

    (lift_up, elbow_up), (lift_down, _) = sorted(arms, key=lambda a: a[0], reverse=True)
    if lift_up < margin_frac * scale:
        return None                              # no elbow clearly up
    if lift_up - lift_down < margin_frac * scale:
        return None                              # the other elbow isn't lower
    mid_x = shoulder_mid_px(pose_lms, w, h)[0]
    return SIDE_RIGHT if elbow_up[0] > mid_x else SIDE_LEFT


class MedianSmoother:
    """Short, age-limited median window for the DISPLAYED gap (HUD and the
    engine's gap_ratio). It is NOT in the decision path — the tracker takes
    the raw per-frame ratio.

    Samples expire by age (SMOOTH_MAX_AGE_S) rather than being thrown away on
    every unmeasurable frame: under the normal intermittent detection of this
    pose the old reset-on-dropout kept the window at one sample, i.e. no
    smoothing at all. push() returns sorted(values)[len // 2] — the true
    median for 1 or 3 samples and the LARGER of two (conservative against a
    touch), never the mean np.median gives at an even length."""

    def __init__(self, n=SMOOTH_FRAMES, max_age_s=SMOOTH_MAX_AGE_S):
        self.n = int(n)
        self.max_age_s = float(max_age_s)
        self.buf = deque(maxlen=self.n)      # (t, value)

    def _expire(self, now):
        while self.buf and now - self.buf[0][0] > self.max_age_s:
            self.buf.popleft()

    def push(self, v, now=None):
        now = time.time() if now is None else now
        self._expire(now)
        self.buf.append((now, float(v)))
        return self.value

    def reset(self):
        """Only for RESET, a side pin or a new trial — not for a dropout."""
        self.buf.clear()

    @property
    def value(self):
        if not self.buf:
            return None
        vals = sorted(v for _, v in self.buf)
        return float(vals[len(vals) // 2])


class ShoulderStretchTracker:
    """Turns a stream of fingertip-gap ratios into a pass/fail verdict.

    ONE trial, and BOTH sides must be touched inside it: the student reaches
    right-arm-over and left-arm-over, and passes only if the fingertips touch
    on BOTH (FITNESSGRAM's rule, chosen over the Brockport PDF's single side).

    Each side needs K (touch_frames) qualifying readings (ratio <= pass_ratio)
    inside a rolling window of window_s seconds:
      * readings older than window_s expire — touches a minute apart are not
        a sustained touch (the old counter had no window and ratcheted);
      * a reading above release_ratio is the hands coming apart and clears
        that side's window; one between pass_ratio and release_ratio adds
        nothing and clears nothing (jitter on a hand seen from behind);
      * a dropout (None) never clears anything — hands vanishing behind the
        back is normal here, and expiry already handles staleness;
      * a qualifying reading whose SIDE is unknown is banked for neither
        side: it cannot be credited to an arm that hasn't been named.

    The whole trial is capped at cap_s from the FIRST reading — the first
    frame showing the stretch, so getting into position does not eat the
    clock. Reaching the cap resolves the trial on what was achieved: a PASS
    only if both sides were done.

    The minimum gap is RECORDED per side (best_gap_ratio, and left/right) so
    the threshold can be retuned later from real data, but it never decides
    the verdict.

    The engine decides this test. There is no coach PASS/FAIL override —
    end_trial() closes the trial on whatever it measured, so verdict_by is
    always "auto".

    Everything is driven through update(), which takes a plain float or None,
    a side or None, and an injectable clock, so the whole state machine can
    be exercised with synthetic sequences — no camera, no MediaPipe objects.
    """

    SIDES = (SIDE_LEFT, SIDE_RIGHT)

    def __init__(self, pass_ratio=PASS_RATIO, touch_frames=TOUCH_READINGS,
                 release_ratio=TOUCH_RELEASE_RATIO, window_s=TOUCH_WINDOW_S,
                 cap_s=TRIAL_CAP_S, no_attempt_s=NO_ATTEMPT_CAP_S,
                 grace_s=None):
        # grace_s is accepted and ignored: a dropout no longer resets
        # anything, the kwarg only keeps old callers from breaking.
        self.pass_ratio = float(pass_ratio)
        self.touch_frames = int(touch_frames)          # K, per side
        self.release_ratio = float(release_ratio)
        self.window_s = float(window_s)                # T
        self.cap_s = None if cap_s is None else float(cap_s)
        self.no_attempt_s = None if no_attempt_s is None else float(no_attempt_s)
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        self.state = State.READY
        self.ratio = None             # latest raw ratio fed in
        self.side = None              # side of the latest reading
        self.best_gap_ratio = None    # best of either side — a RECORD
        self.best_by_side = {s: None for s in self.SIDES}
        self.done = set()             # sides whose touch is banked
        self.frames_seen = 0
        self.qualifying_total = 0
        self.passed = False
        self.verdict_by = "auto"      # the engine decides this test, always
        self.ended_by = None          # "pass" | "cap" | "coach"
        self.status_msg = ("Camera behind the participant - reach over one "
                           "shoulder and up the back with the other hand")
        self._qual = {s: deque() for s in self.SIDES}
        self._cap_start = None
        self._visible_since = None    # student locked on, nothing measured yet
        self._now = None

    @property
    def touch_count(self):
        """Qualifying readings in the window for the side being worked on."""
        return max(len(q) for q in self._qual.values())

    def side_done(self, side):
        return side in self.done

    @property
    def no_attempt(self):
        """Resolved without a single measurable frame — the camera never saw
        the stretch, so there is nothing to score."""
        return self.resolved and self.frames_seen == 0

    def mark_visible(self, now=None):
        """The student is locked on and the engine is watching. Starts the
        no-attempt backstop; the 15 s trial clock still waits for a real
        attempt, so getting into position does not eat it."""
        if self._visible_since is None and not self.resolved:
            self._visible_since = time.time() if now is None else now

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, ratio, now=None, side=None):
        """Feed one frame's RAW fingertip-gap ratio (or None when the frame
        couldn't be measured) and the side it belongs to (or None when the
        arm couldn't be named). Returns the result dict on the frame the
        trial resolves, else None."""
        now = time.time() if now is None else now
        self._now = now

        if self.state in (State.PASSED, State.FAILED):
            return None
        if side is not None:
            self.side = side

        self._expire(now)

        if ratio is None:
            if not any(self._qual.values()) and self.state == State.TOUCHING:
                self.state = State.READY
                self.status_msg = "Hands not visible - hold the position in view"
            return self._check_cap(now)

        self.ratio = float(ratio)
        self.frames_seen += 1
        if self.best_gap_ratio is None or self.ratio < self.best_gap_ratio:
            self.best_gap_ratio = self.ratio
        if self._cap_start is None:
            # The clock starts at the first frame showing the stretch.
            self._cap_start = now
        if side in self.SIDES:
            prev = self.best_by_side[side]
            if prev is None or self.ratio < prev:
                self.best_by_side[side] = self.ratio

        if self.ratio <= self.pass_ratio:
            self.qualifying_total += 1
            if side not in self.SIDES:
                self.state = State.TOUCHING
                self.status_msg = "Touching - hold still so the arm can be named"
                return self._check_cap(now)
            if side not in self.done:
                self._qual[side].append(now)
                if len(self._qual[side]) >= self.touch_frames:
                    self.done.add(side)
                    self._qual[side].clear()
                    if len(self.done) == len(self.SIDES):
                        return self._pass(now)
                    other = SIDE_RIGHT if side == SIDE_LEFT else SIDE_LEFT
                    self.state = State.REACHING
                    self.status_msg = (f"{side.capitalize()} side done - "
                                       f"now the {other} arm over the shoulder")
                    return self._check_cap(now)
            self.state = State.TOUCHING
            self.status_msg = (f"Fingers touching ({side}) - hold it "
                               f"({len(self._qual[side])}/{self.touch_frames})")
        elif self.ratio <= self.release_ratio:
            if self.state != State.TOUCHING:
                self.state = State.REACHING
                self.status_msg = "Almost - bring the fingertips together"
        else:
            if side in self.SIDES:
                self._qual[side].clear()
            else:
                for q in self._qual.values():
                    q.clear()
            self.state = State.REACHING
            self.status_msg = self._todo_msg()
        return self._check_cap(now)

    # ── internals ────────────────────────────────────────────────────────
    def _todo_msg(self):
        left = [s for s in self.SIDES if s not in self.done]
        if len(left) == len(self.SIDES):
            return "Reach further - the fingertips are still apart"
        return f"Now the {left[0]} arm over the shoulder"

    def _expire(self, now):
        for q in self._qual.values():
            while q and now - q[0] > self.window_s:
                q.popleft()

    def _check_cap(self, now):
        if (self.cap_s is not None and self._cap_start is not None
                and now - self._cap_start >= self.cap_s):
            return self.end_trial(now, by="cap")
        # Nothing has EVER been measured and the student has been in view a
        # long time: close it rather than leave the trial open forever.
        if (self._cap_start is None and self.no_attempt_s is not None
                and self._visible_since is not None
                and now - self._visible_since >= self.no_attempt_s):
            return self.end_trial(now, by="no_attempt")
        return None

    def _pass(self, now):
        self.passed = True
        self.verdict_by = "auto"
        self.ended_by = "pass"
        self.state = State.PASSED
        self.status_msg = "PASS - fingertips touched on both sides"
        return self.result()

    def end_trial(self, now=None, by="coach"):
        """Close the trial. This is not a coach verdict: the result is what
        the measurement earned — a PASS only if BOTH sides were banked."""
        if self.resolved:
            return None
        self._now = time.time() if now is None else now
        self.passed = len(self.done) == len(self.SIDES)
        self.verdict_by = "auto"
        self.ended_by = "pass" if self.passed else by
        self.state = State.PASSED if self.passed else State.FAILED
        if self.passed:
            self.status_msg = "PASS - fingertips touched on both sides"
        elif by == "no_attempt":
            self.status_msg = ("NOT COMPLETED - the camera never saw the "
                               "stretch. Check the view and run it again")
        elif not self.done:
            self.status_msg = ("FAIL - time limit" if by == "cap"
                               else "FAIL - the fingertips did not touch")
        else:
            got = ", ".join(sorted(self.done))
            self.status_msg = f"FAIL - only the {got} side was completed"
        return self.result()

    # ── read-only views ──────────────────────────────────────────────────
    def touch_progress(self):
        """0..1 across the WHOLE test: each banked side is half, and the side
        being worked on fills the rest of its half."""
        if self.state == State.PASSED:
            return 1.0
        if self.touch_frames <= 0:
            return 0.0
        per = 1.0 / len(self.SIDES)
        got = len(self.done) * per
        working = max((len(q) for s, q in self._qual.items()
                       if s not in self.done), default=0)
        return min(1.0, got + per * working / self.touch_frames)

    def cap_remaining(self, now=None):
        """Seconds left in the 15 s trial, or None before it starts."""
        if self.cap_s is None or self._cap_start is None or self.resolved:
            return None
        now = (self._now if now is None else now)
        if now is None:
            return None
        return max(0.0, self.cap_s - (now - self._cap_start))

    @property
    def resolved(self):
        return self.state in (State.PASSED, State.FAILED)

    def result(self):
        return {
            "passed": bool(self.passed),          # both sides touched
            "left_passed": SIDE_LEFT in self.done,
            "right_passed": SIDE_RIGHT in self.done,
            # Trial minimum per side, recorded so PASS_RATIO can be retuned
            # later from real student data. NOT what decided the verdict.
            "best_gap_left": _r3(self.best_by_side[SIDE_LEFT]),
            "best_gap_right": _r3(self.best_by_side[SIDE_RIGHT]),
            "best_gap_ratio": _r3(self.best_gap_ratio),
            "threshold_ratio": self.pass_ratio,
            "required_frames": self.touch_frames,     # K, per side
            "window_s": self.window_s,                # T
            "trial_cap_s": self.cap_s,
            "qualifying_readings_total": self.qualifying_total,
            "ended_by": self.ended_by,
            "verdict_by": self.verdict_by,
        }


class SideVoter:
    """Which arm is over the shoulder RIGHT NOW.

    Both sides must be done inside one trial, so this deliberately does NOT
    lock: the answer has to follow the student when they switch arms. It is
    a short rolling majority over the frames that actually named a side,
    which rides out the odd mis-read without freezing the label.

    There is no coach pin any more: pinning one side would make the other
    side impossible to bank."""

    VOTE_WINDOW = 5

    def __init__(self):
        self.reset()

    def reset(self):
        self.side = None
        self._votes = deque(maxlen=self.VOTE_WINDOW)

    def update(self, side):
        """side: SIDE_LEFT | SIDE_RIGHT | None (this frame names nothing)."""
        if side is not None:
            self._votes.append(side)
        if not self._votes:
            return self.side
        counts = {sd: sum(1 for v in self._votes if v == sd)
                  for sd in (SIDE_LEFT, SIDE_RIGHT)}
        best = max(counts, key=counts.get)
        other = SIDE_RIGHT if best == SIDE_LEFT else SIDE_LEFT
        if counts[best] > counts[other]:
            self.side = best
        return self.side


def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _ascii(txt):
    """cv2.putText draws ASCII only; anything else shows as '???'."""
    return (str(txt).replace("—", "-").replace("–", "-")
            .replace("≤", "<=").encode("ascii", "replace").decode("ascii"))


def draw_hand_skeleton(frame, hand_lms_list, colour=(200, 200, 200)):
    """Hand skeletons straight onto the BGR frame with OpenCV — the legacy
    mediapipe.solutions drawing helpers are deliberately never imported."""
    h, w = frame.shape[:2]
    for lms in hand_lms_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        n = len(pts)
        for c in HandLandmarksConnections.HAND_CONNECTIONS:
            if 0 <= c.start < n and 0 <= c.end < n:
                cv2.line(frame, pts[c.start], pts[c.end], colour, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 2, colour, -1, cv2.LINE_AA)


def draw_hud(frame, tracker, side=None, ratio=None, tip_pts=None):
    """Draw the live state onto the frame. Called with the engine's lock held,
    so it only reads the tracker."""
    h, w = frame.shape[:2]
    st = tracker.state

    colour = {
        State.READY:    C_WARN,
        State.REACHING: C_WARN,
        State.TOUCHING: C_OK,
        State.PASSED:   C_OK,
        State.FAILED:   C_BAD,
    }.get(st, C_WARN)

    # The winning fingertip pair, so the coach can see exactly which gap is
    # being measured rather than trusting a bare number.
    if tip_pts is not None:
        pa, pb = tip_pts
        cv2.line(frame, tuple(np.int32(pa)), tuple(np.int32(pb)), colour, 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.int32(pa)), 9, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.int32(pb)), 9, colour, -1, cv2.LINE_AA)

    # Top bar: the sustained-touch progress, which is the thing the
    # participant is actually trying to fill.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = tracker.touch_progress()
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 10), C_OK, -1)

    draw_panel(frame, 12, 22, 540, 196)
    y = 48
    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _ascii(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    line(f"SHOULDER STRETCH  [{st.name}]", colour, 0.68, 30)
    # Each side carries its own verdict, so the words are drawn one at a time:
    # line() paints a whole string in a single colour, and R and L need
    # different ones. "...." only while the trial is still running — once it
    # is over, a side that was never banked is a FAIL.
    def side_word(sd):
        if tracker.side_done(sd):
            return "PASS", C_OK
        if tracker.resolved:
            return "FAIL", C_BAD
        return "....", C_TEXT

    r_word, r_col = side_word(SIDE_RIGHT)
    l_word, l_col = side_word(SIDE_LEFT)
    x = 24
    for seg, seg_col in (("Both sides needed:  R ", C_TEXT), (r_word, r_col),
                         ("   L ", C_TEXT), (l_word, l_col),
                         (f"    now: {side or '--'}", C_TEXT)):
        seg = _ascii(seg)
        cv2.putText(frame, seg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    seg_col, 2, cv2.LINE_AA)
        x += cv2.getTextSize(seg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    y += 26
    if ratio is not None:
        line(f"Gap: {ratio:.2f} palm",
             C_OK if ratio <= tracker.pass_ratio else C_TEXT)
    elif tracker.resolved:
        line("Gap: --")
    else:
        line("Gap: hands not tracked", C_BAD)
    best = ("--" if tracker.best_gap_ratio is None
            else f"{tracker.best_gap_ratio:.2f}")
    left = tracker.cap_remaining()
    line(f"Touch: {tracker.touch_count} / {tracker.touch_frames}"
         f"   (closest {best})"
         + (f"   {left:.0f}s left of {tracker.cap_s:.0f}" if left is not None
            else f"   {tracker.cap_s:.0f}s once you start"))

    if tracker.status_msg:
        draw_panel(frame, 12, h - 54, min(w - 12, 12 + 13 * len(tracker.status_msg) + 24), h - 12)
        cv2.putText(frame, _ascii(tracker.status_msg), (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 2, cv2.LINE_AA)

    return frame


def _stage_shoulder_stretch(session_id, result, rating=None):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/shoulder_stretch",
            json={
                "session_id":     session_id,
                "passed":         bool(result.get("passed")),
                "left_passed":    bool(result.get("left_passed")),
                "right_passed":   bool(result.get("right_passed")),
                "best_gap_left":  result.get("best_gap_left"),
                "best_gap_right": result.get("best_gap_right"),
                "best_gap_ratio": result.get("best_gap_ratio"),
                "rating":         rating,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] shoulder_stretch staged for review → "
              f"{'PASS' if result.get('passed') else 'FAIL'}  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


def ensure_model(path):
    """Download a .task model next to this script if it isn't there yet, so a
    fresh checkout works without a manual step. Both files ship with the repo,
    so on this machine nothing is fetched."""
    if os.path.exists(path):
        return path
    url = MODEL_URLS.get(os.path.basename(path))
    if url is None:
        return path          # a --model the caller pointed somewhere else
    import ssl
    import urllib.request
    # Python on macOS ships without the system CA bundle wired into OpenSSL,
    # so a plain urlretrieve raises CERTIFICATE_VERIFY_FAILED. Verify against
    # certifi's bundle (a hard dependency of mediapipe) instead.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    print(f"[DOWNLOAD] {os.path.basename(path)} …", end=" ", flush=True)
    with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as fh:
        fh.write(resp.read())
    print(f"done  ({os.path.getsize(path) // 1024} KB)")
    return path


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Shoulder Stretch Test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--pass-ratio", type=float, default=PASS_RATIO,
                        help=f"Fingertip gap / palm length that counts as a touch "
                             f"(default {PASS_RATIO:.2f})")
    parser.add_argument("--touch-frames", type=int, default=TOUCH_FRAMES,
                        help=f"Qualifying readings within {TOUCH_WINDOW_S:.0f} s "
                             f"needed for a PASS (default {TOUCH_FRAMES})")
    parser.add_argument("--model", default=DEFAULT_POSE_MODEL,
                        help="Path to MediaPipe PoseLandmarker .task model file")
    parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL,
                        help="Path to MediaPipe HandLandmarker .task model file")
    parser.add_argument("--smooth", type=int, default=SMOOTH_FRAMES,
                        help=f"Ratio smoothing window in frames (default {SMOOTH_FRAMES})")
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
    print("[INFO] Camera BEHIND the participant. One arm over the shoulder, the "
          "other up the back, fingertips together between the shoulder blades.")
    print("[INFO] E = end trial  |  R = reset  |  Q = quit\n")

    tracker = ShoulderStretchTracker(pass_ratio=args.pass_ratio,
                                     touch_frames=args.touch_frames)
    selector = SideVoter()
    smooth = MedianSmoother(args.smooth)
    lock = ParticipantLock()
    staged = False
    stream_start = time.time()
    last_ts = -1

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(args.model)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=NUM_POSES,     # must SEE a second person to refuse them
        # 0.45, following wingspan2.build_pose_detector's production values —
        # NOT reverse_curl.py's 0.55. Seen from behind with the arms folded up
        # the back, 0.55 drops the pose on perfectly good frames. Do not "fix"
        # this back to 0.55 by analogy with the other pass/fail test.
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_segmentation_masks=False,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(args.hand_model)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        # wingspan2.build_hand_detector's production values.
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    win = "Shoulder Stretch Test"
    cv2.namedWindow(win)

    # Neither landmarker is used as a context manager: the hand detector is
    # not one in the house code, so both are created up front and closed in
    # the finally block. An exception mid-loop would otherwise leak them.
    landmarker = PoseLandmarker.create_from_options(pose_options)
    hand_det = HandLandmarker.create_from_options(hand_options)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            # No cv2.flip here, deliberately: mirroring swaps the pose's left
            # and right landmarks, which would invert the RIGHT/LEFT label of
            # the trial. reverse_curl.py does not flip either.
            fh, fw = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int((time.time() - stream_start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            # Both graphs get the SAME timestamp for the same frame; each
            # keeps its own strictly-increasing check in VIDEO mode.
            results = landmarker.detect_for_video(mp_image, ts)
            h_result = hand_det.detect_for_video(mp_image, ts)

            ratio = None           # raw ratio -> the tracker
            shown = None           # smoothed ratio -> the HUD only
            tip_pts = None
            now = time.time()

            # Whose body: the lock seeds on a lone body and then follows it by
            # continuity, so a coach stepping in cannot take the trial over.
            fresh = lock.update(results.pose_landmarks, fw, fh, now)
            body_lms = lock.body_for(now)
            if body_lms is not None:
                tracker.mark_visible(now)
            if body_lms is not None:
                for i, j in _POSE_CONNECTIONS:
                    if body_lms[i].visibility >= 0.45 and body_lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(body_lms[i].x * fw), int(body_lms[i].y * fh)),
                                 (int(body_lms[j].x * fw), int(body_lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)
            if h_result.hand_landmarks:
                draw_hand_skeleton(frame, h_result.hand_landmarks)

            reading = frame_gap_ratio(h_result.hand_landmarks, body_lms, fw, fh)

            # Which arm is over the shoulder, only while the stretch is in
            # view. Read from the HANDS (the down-pointing hand came over the
            # shoulder); the elbows are a fallback for a frame the hands
            # can't name.
            if reading is not None and not tracker.resolved:
                named = side_from_hands(h_result.hand_landmarks, body_lms, fw, fh)
                if named is None and fresh is not None:
                    named = detect_side(fresh, fw, fh)
                selector.update(named)
            side = selector.side
            if reading is not None:
                ratio = reading.ratio
                shown = smooth.push(reading.ratio, now)
                tip_pts = (reading.point_a, reading.point_b)
            # No smooth.reset() on a dropout: its samples expire by age.

            outcome = tracker.update(ratio, now=now, side=side)
            if outcome is not None and not staged and args.session_id:
                _stage_shoulder_stretch(args.session_id, outcome)
                staged = True

            if args.debug:
                print(f"side={side} locked={lock.locked} "
                      f"ratio={ratio if ratio is None else round(ratio,3)} "
                      f"state={tracker.state.name} touch={tracker.touch_count}")

            if not lock.locked:
                tracker.status_msg = "Only the student in frame to start"
            draw_hud(frame, tracker, side=side, ratio=shown, tip_pts=tip_pts)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                outcome = tracker.end_trial(by="coach")
                if outcome is not None and not staged and args.session_id:
                    _stage_shoulder_stretch(args.session_id, outcome)
                    staged = True
            elif key == ord("r"):
                tracker.reset()
                selector.reset()
                smooth.reset()
                lock.reset()
                staged = False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for det in (landmarker, hand_det):
            try:
                det.close()
            except Exception:
                pass

    res = tracker.result()
    print("\n── Shoulder Stretch ─────────────────────────")
    print(f"  Result       : {'PASS' if res['passed'] else 'FAIL / not completed'}")
    print(f"  Right side   : {'touched' if res['right_passed'] else 'not done'}"
          f"   (closest {res['best_gap_right']})")
    print(f"  Left side    : {'touched' if res['left_passed'] else 'not done'}"
          f"   (closest {res['best_gap_left']})")
    print(f"  Closest gap  : {res['best_gap_ratio']} palm "
          f"(pass <= {res['threshold_ratio']:.2f}, both sides needed)")
    print(f"  Trial        : {res['required_frames']} readings per side within "
          f"{res['window_s']:.0f} s, {res['trial_cap_s']:.0f} s for the whole "
          f"test (qualifying readings: {res['qualifying_readings_total']}, "
          f"ended by {res['ended_by']})")


if __name__ == "__main__":
    main()
