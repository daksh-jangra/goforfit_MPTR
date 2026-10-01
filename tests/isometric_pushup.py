"""
Isometric Push-Up Test – hold the up position of a push-up, up to 40 seconds
============================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.77-78, figure 5.14)
--------------------------------------------------------------------
  The participant holds the UP position of a push-up for as long as they can,
  to a maximum of 40 seconds: front-leaning rest, hands directly below the
  shoulders, arms extended, the whole body in a straight line, toes on the
  mat. The test is terminated when any movement - bending, sagging or
  swaying - occurs at the elbows, shoulders, trunk or knees. ONE trial; the
  score is the time held, to the nearest second.

  A demo video of a BOTTOM hold (elbows at 90 degrees) was also looked at;
  the owner chose the PDF: arms straight, top of the push-up.

What the camera measures
--------------------------------------------------------------------
  Side-on, floor level. On the side of the body nearest the camera:

    elbow   shoulder-elbow-wrist angle         (straight arm ~ 170-180)
    bend    how far the hip sits off the shoulder-ankle line, in degrees:
            + = hips piked up, - = hips sagging    (straight body ~ 0)
    knee    hip-knee-ankle angle               (straight leg ~ 170-180)
    sway    shoulder position over the hand, in torso lengths - the hands
            are planted, so this moves only when the body rocks

  The manual's own photo (fig 5.14, cropped to the child) reads elbow
  168-171, bend 1-2, knee 167-172, body 11 degrees off flat, hands under the
  shoulders. The entry gates below leave room either side of that.

The clock
--------------------------------------------------------------------
  START  once the participant has been in the correct position, and still,
         for SETTLE_S. That second is credited - they were holding it. The
         readings over it become the participant's own starting shape, and
         every later frame is compared with THAT, so a child whose arms lock
         at 160 is not held to someone else's 175.
  STOP   when any of these lasts longer than BREAK_GRACE_S:
           * the elbows bend        (ELBOW_BEND_DEG below the start,
                                     or below HOLD_ELBOW_MIN_DEG at all)
           * the hips sag or pike   (BEND_MOVE_DEG away from the start,
                                     or more than HOLD_BEND_MAX_DEG off the line)
           * the knees bend         (KNEE_BEND_DEG below the start,
                                     or below HOLD_KNEE_MIN_DEG at all)
           * the body sways         (SWAY_MOVE shoulder travel over the hand)
         The fixed limits matter for a student who only just qualified: arms
         that start at 150 could otherwise bend to 131 before "20 below the
         start" caught them.
         The time stops at the last frame the position was still correct,
         not after the grace - the grace only stops one shaky frame from
         ending a good hold.
  CAP    40 s is the maximum; the clock stops itself there.
  LOST   tracking (or the phone's frames) gone longer than DROPOUT_GRACE_S
         ends the hold at the last frame the position was seen. A shorter gap
         - school Wi-Fi stalls for a second at a time - does not: if the
         student is still in position when the picture comes back, the clock
         carries on.

  Stopping before the clock ever started is NOT COMPLETED, not a zero - a
  camera or setup problem must never be recorded against the participant.

  Form cues the manual leaves to the tester (hand placement taught before the
  test, tactual help getting into position) are not scored.

Usage
-----
  python isometric_pushup.py
  python isometric_pushup.py --source clip.mp4
  python isometric_pushup.py --debug

  Keys : R = reset the trial     Q = quit and print the summary

Camera placement
----------------
  Phone on the floor (or just above it), SIDE-ON to the participant,
  1.5-2 m away, with the whole body from hands to toes in frame.
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
MAX_HOLD_S = 40.0     # "hold a raised push-up position for as long as 40 seconds"

# ── "In position" gates (checked before the clock starts) ────────────────────────
ELBOW_MIN_DEG     = 150.0   # arms extended          (figure: 168-171)
BEND_MAX_DEG      = 20.0    # body in a straight line (figure: 1-2)
KNEE_MIN_DEG      = 150.0   # legs straight, knees off the mat (figure: 167-172)
BODY_TILT_MAX_DEG = 40.0    # shoulder-to-ankle line near horizontal (figure: 11).
                            # This is what stops a standing person qualifying.
ARM_TILT_MAX_DEG  = 35.0    # shoulder-to-wrist line near vertical: hands under
                            # the shoulders, not out in front

# ── Settling: how the start is confirmed ─────────────────────────────────────────
SETTLE_S          = 1.0     # in position and still this long -> clock starts
SETTLE_ELBOW_BAND = 8.0     # spread allowed inside the settle window
SETTLE_BEND_BAND  = 8.0
SETTLE_KNEE_BAND  = 10.0
SETTLE_SWAY_BAND  = 0.10

# ── Breaking the hold (all measured against the participant's own start) ────────
ELBOW_BEND_DEG  = 20.0   # elbow closes this far -> "elbows bent"
BEND_MOVE_DEG   = 15.0   # hip leaves its start line this far -> sag / pike
KNEE_BEND_DEG   = 20.0   # knee closes this far -> "knees bent"
SWAY_MOVE       = 0.25   # shoulder travels this many torso lengths forward or
                         # back over the planted hand -> "swayed". ~11 cm on a
                         # child: clearly visible, well above landmark jitter
                         # (~0.03). Deliberately tolerant - sway is the rule
                         # most likely to cut a good hold short.
# Fixed limits while the clock runs, whatever the start was. A little looser
# than the entry gates so a student who only just qualified isn't stopped by
# landmark jitter.
HOLD_ELBOW_MIN_DEG = 140.0
HOLD_BEND_MAX_DEG  = 28.0
HOLD_KNEE_MIN_DEG  = 140.0
BREAK_GRACE_S   = 0.30   # a break must last this long to end the hold
DROPOUT_GRACE_S = 2.00   # mid-hold, tracking or the phone's frames may vanish
                         # this long. 0.6 s ended good holds on a single Wi-Fi
                         # stall; the time is still only credited if the
                         # position is seen again afterwards.
SETTLE_GAP_S    = 0.60   # before the clock starts, any gap longer than this
                         # empties the settle window - stillness can't be
                         # proven across time nobody saw

SMOOTH_FRAMES   = 3      # median window per measurement
MIN_VIS         = 0.50   # landmark visibility floor for every joint used

SIDE_LEFT, SIDE_RIGHT = "left", "right"

# Landmark indices (33-point BlazePose topology)
SIDE_IDX = {
    #            shoulder elbow wrist hip knee ankle
    SIDE_LEFT:  (11, 13, 15, 23, 25, 27),
    SIDE_RIGHT: (12, 14, 16, 24, 26, 28),
}

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

METRICS = ("elbow", "bend", "knee", "sway")

# Plain-words reason for each break, shown on screen and staged for screen3.
BREAK_WORDS = {
    "elbow": "elbows bent",
    "bend":  "hips sagged or piked",
    "knee":  "knees bent",
    "sway":  "body swayed",
}


class State(Enum):
    READY    = auto()   # waiting for the correct up position
    SETTLING = auto()   # in position - confirming it is held and still
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


def side_metrics(lms, side, w, h):
    """Everything the test compares, for one side of the body - or None when
    any of its six joints is unclear or outside the picture (MediaPipe still
    returns a confident-looking guess for joints off the edge)."""
    idxs = SIDE_IDX[side]
    for i in idxs:
        lm = lms[i]
        if lm.visibility < MIN_VIS or not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return None
    s, e, wr, hp, k, a = (lm_px(lms, i, w, h) for i in idxs)

    torso = float(np.linalg.norm(s - hp))
    if torso < 1.0 or abs(a[0] - s[0]) < 1.0:
        return None

    # Hip off the shoulder-ankle line. The magnitude is the bend at the hip;
    # the sign comes from screen height (above the line = piked), so it means
    # the same thing whichever way the participant faces or the feed mirrors.
    line_y = s[1] + (a[1] - s[1]) * (hp[0] - s[0]) / (a[0] - s[0])
    bend = 180.0 - calc_angle(s, hp, a)
    if hp[1] > line_y:
        bend = -bend

    dx_arm, dy_arm = wr[0] - s[0], wr[1] - s[1]
    arm_tilt = (math.degrees(math.atan2(abs(dx_arm), dy_arm)) if dy_arm > 0
                else 180.0)          # wrist level with or above the shoulder

    return {
        "elbow": calc_angle(s, e, wr),
        "bend": bend,
        "knee": calc_angle(hp, k, a),
        "sway": float((s[0] - wr[0]) / torso),
        "body_tilt": math.degrees(math.atan2(abs(a[1] - s[1]), abs(a[0] - s[0]))),
        "arm_tilt": arm_tilt,
        "vis": float(np.mean([lms[i].visibility for i in idxs])),
    }


def position_problem(m):
    """None if these readings are the correct up position, else what to fix
    (in plain words, for the coach's status line)."""
    if m["body_tilt"] > BODY_TILT_MAX_DEG:
        return "Get down into the push-up position"
    if m["elbow"] < ELBOW_MIN_DEG:
        return "Straighten the arms"
    if m["arm_tilt"] > ARM_TILT_MAX_DEG:
        return "Hands directly below the shoulders"
    if m["knee"] < KNEE_MIN_DEG:
        return "Straighten the legs - knees off the mat"
    if m["bend"] > BEND_MAX_DEG:
        return "Hips too high - body in a straight line"
    if m["bend"] < -BEND_MAX_DEG:
        return "Hips sagging - body in a straight line"
    return None


class MedianSmoother:
    """Short median per measurement - kills single-frame landmark spikes."""

    def __init__(self, n=SMOOTH_FRAMES):
        self.bufs = {k: deque(maxlen=int(n)) for k in METRICS + ("body_tilt", "arm_tilt")}

    def push(self, m):
        out = dict(m)
        for k, buf in self.bufs.items():
            buf.append(float(m[k]))
            out[k] = float(np.median(buf))
        return out

    def reset(self):
        for buf in self.bufs.values():
            buf.clear()


class IsometricPushUpTracker:
    """Turns per-frame body readings into the hold time.

    update() takes {SIDE_LEFT: metrics|None, SIDE_RIGHT: metrics|None} (from
    side_metrics) and an optional injected clock, so the whole state machine
    can be driven with synthetic readings - no camera required.

    The side nearest the camera is picked by landmark visibility and LOCKED
    when the settle window opens: the near and far arm read ~20 degrees apart
    side-on, so switching sides mid-hold would look like a bend.
    """

    def __init__(self, max_hold_s=MAX_HOLD_S, settle_s=SETTLE_S):
        self.max_hold_s = float(max_hold_s)
        self.settle_s = float(settle_s)
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        self.state = State.READY
        self.side = None              # side being read; locked once settling
        self.current = None           # latest smoothed readings (tracked side)
        self.baseline = None          # the participant's own starting shape
        self.hold_start = None
        self.hold_end = None
        self.end_reason = None        # plain words: why the clock stopped
        self.capped = False
        self.status_msg = "Get into the push-up position, side-on to the camera"
        self._settle = deque()        # (t, metrics)
        self._smooth = {SIDE_LEFT: MedianSmoother(), SIDE_RIGHT: MedianSmoother()}
        self._bad_since = None        # first frame of a possible break (grace timer)
        self._bad_reason = None
        self._last_seen = None        # last frame the locked side was tracked
        self._last_good = None        # last frame the position was still correct -
                                      # where the clock stops, whatever ends it
        self._last_frame_t = None
        self._now = None

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, sides, now=None):
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
            # still holding them would let a student who collapsed during the
            # gap read as "still in position" on the first frame back.
            for sm in self._smooth.values():
                sm.reset()
            self._settle.clear()
            if self.state == State.HOLDING and gap > DROPOUT_GRACE_S:
                return self._end(self._last_good,
                                 "the video paused - the camera lost sight of the student")

        m = self._pick(sides)

        if self.state == State.HOLDING:
            if m is None:
                # Unseen time is never capped or credited on its own: the
                # clock only moves on if the position is seen again.
                if now - self._last_seen > DROPOUT_GRACE_S:
                    return self._end(self._last_good,
                                     "the camera lost sight of the student")
                return None
            self._last_seen = now
            return self._check_hold(m, now)

        # READY / SETTLING
        if m is None:
            self._settle.clear()
            self.state = State.READY
            self.status_msg = "Whole body side-on to the camera, hands to toes in frame"
            return None

        problem = position_problem(m)
        if problem is not None:
            self._settle.clear()
            self.state = State.READY
            self.status_msg = problem
            return None

        self.state = State.SETTLING       # locks self.side (see _pick)
        self.status_msg = "Good position - hold still"
        self._settle.append((now, m))
        # Keep only the still stretch: drop samples from the front until the
        # window agrees with itself, so a wobble restarts the settle.
        while len(self._settle) > 1 and not self._window_still():
            self._settle.popleft()
        if self._settle[-1][0] - self._settle[0][0] >= self.settle_s:
            self._start_hold(now)
        return None

    # ── internals ────────────────────────────────────────────────────────
    def _pick(self, sides):
        """Smoothed readings for the side being tracked, or None. Until the
        settle window opens the clearest side is chosen afresh every frame;
        from then on the side is locked."""
        for s in (SIDE_LEFT, SIDE_RIGHT):
            if sides.get(s) is None:
                self._smooth[s].reset()
        if self.state == State.READY:
            seen = [s for s in (SIDE_LEFT, SIDE_RIGHT) if sides.get(s) is not None]
            # Nearest the camera = the clearest joints.
            self.side = max(seen, key=lambda s: sides[s]["vis"]) if seen else None
        raw = sides.get(self.side) if self.side is not None else None
        self.current = self._smooth[self.side].push(raw) if raw is not None else None
        return self.current

    def _window_still(self):
        vals = {k: [m[k] for _, m in self._settle] for k in METRICS}
        return (max(vals["elbow"]) - min(vals["elbow"]) <= SETTLE_ELBOW_BAND
                and max(vals["bend"]) - min(vals["bend"]) <= SETTLE_BEND_BAND
                and max(vals["knee"]) - min(vals["knee"]) <= SETTLE_KNEE_BAND
                and max(vals["sway"]) - min(vals["sway"]) <= SETTLE_SWAY_BAND)

    def _start_hold(self, now):
        self.baseline = {k: float(np.median([m[k] for _, m in self._settle]))
                         for k in METRICS}
        self.hold_start = self._settle[0][0]
        self._last_seen = now
        self._last_good = now
        self._bad_since = None
        self._settle.clear()
        self.state = State.HOLDING
        self.status_msg = "Clock running - hold the position"

    def _broken(self, m):
        """The first rule this frame breaks, or None."""
        b = self.baseline
        if m["elbow"] < b["elbow"] - ELBOW_BEND_DEG or m["elbow"] < HOLD_ELBOW_MIN_DEG:
            return "elbow"
        if abs(m["bend"] - b["bend"]) > BEND_MOVE_DEG or abs(m["bend"]) > HOLD_BEND_MAX_DEG:
            return "bend"
        if m["knee"] < b["knee"] - KNEE_BEND_DEG or m["knee"] < HOLD_KNEE_MIN_DEG:
            return "knee"
        if abs(m["sway"] - b["sway"]) > SWAY_MOVE:
            return "sway"
        return None

    def _check_hold(self, m, now):
        why = self._broken(m)
        if why is None:
            self._bad_since = None
            self._bad_reason = None
            self._last_good = now
            self.status_msg = "Clock running - hold the position"
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
        position was still correct. No hold started -> None: NOT
        COMPLETED (not a zero). Already done -> that result unchanged."""
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


def draw_hud(frame, tracker, body_pts=None):
    """Draw the live state onto the frame. body_pts: the tracked side's six
    joints in pixels (shoulder, elbow, wrist, hip, knee, ankle), or None."""
    h, w = frame.shape[:2]
    st = tracker.state
    colour = {State.READY: C_WARN, State.SETTLING: C_WARN,
              State.HOLDING: C_OK, State.DONE: C_OK}.get(st, C_WARN)
    if tracker._bad_since is not None and st == State.HOLDING:
        colour = C_BAD

    # The lines being measured, so the coach can see what the camera sees.
    if body_pts is not None:
        s, e, wr, hp, k, a = (tuple(np.int32(p)) for p in body_pts)
        for p1, p2 in ((s, e), (e, wr), (s, hp), (hp, k), (k, a)):
            cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)
        for p in (e, hp, k):
            cv2.circle(frame, p, 8, colour, -1, cv2.LINE_AA)

    # Top bar: time held out of the 40 s maximum.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = tracker.elapsed_s / tracker.max_hold_s if tracker.max_hold_s else 0.0
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * min(1.0, prog)), 10), C_OK, -1)

    draw_panel(frame, 12, 22, 430, 196)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    line(f"ISOMETRIC PUSH-UP  [{st.name}]", colour, 0.68, 30)
    line(f"Time: {tracker.elapsed_s:.1f} / {tracker.max_hold_s:.0f} s")
    m, b = tracker.current, tracker.baseline
    if m is not None:
        sb = (lambda k, f="{:.0f}": f"  (start {f.format(b[k])})") if b else (lambda k, f="": "")
        line(f"Elbow: {m['elbow']:.0f} deg{sb('elbow')}")
        line(f"Hips off line: {m['bend']:+.0f} deg{sb('bend', '{:+.0f}')}")
        line(f"Knee: {m['knee']:.0f} deg{sb('knee')}")
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


def read_sides(lms, w, h):
    """{SIDE_LEFT: metrics|None, SIDE_RIGHT: metrics|None} for one frame."""
    if lms is None:
        return {SIDE_LEFT: None, SIDE_RIGHT: None}
    return {s: side_metrics(lms, s, w, h) for s in (SIDE_LEFT, SIDE_RIGHT)}


def tracked_points(tracker, lms, w, h):
    """Pixel joints of the side the tracker is using, for the HUD."""
    if lms is None or tracker.side is None or tracker.current is None:
        return None
    return [lm_px(lms, i, w, h) for i in SIDE_IDX[tracker.side]]


def _stage_isometric_pushup(session_id, result, rating=None):
    """Stages the result on the server for review/save on screen3 - does NOT
    write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/isometric_push_up",
            json={
                "session_id":   session_id,
                "hold_sec":     result.get("hold_sec"),
                "held_exact_s": result.get("held_exact_s"),
                "end_reason":   result.get("end_reason"),
                "rating":       rating,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] isometric_push_up staged for review -> "
              f"{result.get('hold_sec')} s  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Isometric Push-Up Test")
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

    print(f"[INFO] Hold the up position of a push-up, max {MAX_HOLD_S:.0f} s. "
          f"R = reset  |  Q = quit\n")

    tracker = IsometricPushUpTracker()
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

    win = "Isometric Push-Up Test"
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

            outcome = tracker.update(read_sides(lms, fw, fh), now=now)
            if outcome is not None and not staged and args.session_id:
                _stage_isometric_pushup(args.session_id, outcome)
                staged = True

            if args.debug:
                m = tracker.current
                print(f"t={now - stream_start:6.2f} state={tracker.state.name:8s} "
                      f"side={tracker.side} clock={tracker.elapsed_s:5.2f} "
                      + (" ".join(f"{k}={m[k]:.2f}" for k in METRICS) if m else "no body"))

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
        _stage_isometric_pushup(args.session_id, outcome)
    print("\n-- Isometric Push-Up ------------------------")
    if tracker.resolved:
        res = tracker.result()
        print(f"  Score  : {res['hold_sec']} s  (exact {res['held_exact_s']} s)")
        print(f"  Ended  : {res['end_reason']}")
    else:
        print("  Result : NOT COMPLETED (the clock never started)")


if __name__ == "__main__":
    main()
