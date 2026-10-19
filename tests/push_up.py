"""
Push-Up Test - as many push-ups as possible at 1 every 3 s, one trial
=====================================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.80-81)
--------------------------------------------------------------------
  Prone on a mat, hands under the shoulders, legs straight and slightly
  apart, weight on the tucked TOES. Push up until the arms are straight
  (fig 5.17a), lower by bending the elbows to 90 degrees (fig 5.17b), push
  back up to straight arms. About 1 push-up every 3 seconds ("approximately
  20 push-ups in 1 minute", with some latitude). ONE trial.

  The score is the number of correct push-ups. One counts each time the
  arms bend and come back to straight. The test ends when the participant
  can't keep the cadence, stops to rest, or quits. Incorrect push-ups are
  not counted: knees touching the floor, arms not straight at the top, arms
  not bent to 90 degrees, jerky or one-sided movement, or the back not kept
  reasonably straight.

What the camera checks (owner's choices, 2026-10-14)
--------------------------------------------------------------------
  Camera SIDE-ON at floor level, whole body in frame. On the side nearest
  the camera:

    elbow   shoulder-elbow-wrist angle       (straight arm ~ 165-180)
    bend    hip off the shoulder-ankle line   (+ piked, - sagging, 0 straight)
    knee    hip-knee-ankle angle              (straight leg ~ 165-180)
    knee_h  knee height above the hands-to-toes line, in thigh lengths
    body_tilt  shoulder-ankle line against horizontal

  A REP is one cycle from the top and back:
    TOP     elbow >= UP_DEG (160)    - "arms straight"
    BOTTOM  elbow <= DOWN_DEG (100)  - 90 degrees, with 10 for camera noise
  counted when the elbow gets back to UP_DEG, unless during that cycle:
    * the knees were down (KNEES), or
    * the back was not reasonably straight (BACK),
  for at least FAULT_MIN_S. Those reps show "Not counted - <why>".
  Coming back up without reaching the bottom is "not down to 90 degrees" -
  but only if the elbow dipped below SHALLOW_DIP_DEG (140) on the way. A
  smaller dip is still resting at the top: a straight arm that reads near
  160 wobbles across the line, and each wobble was logged as a not-counted
  push-up and restarted the 6 s clock, so a student resting at the top
  never ended the test (arm at 160-162: 0 of 20 runs ended). Going back
  down from partway up is "arms not straightened".
  The cycle starts when the elbow leaves the top, so a student resting on
  the knees BETWEEN reps loses nothing once the knees come up again.

  KNEES is two readings. The knee bent below KNEE_MIN_DEG (145) catches a
  knee push-up with the feet raised (~90-120). Isometric push-up's straight
  legs read 163-178 and slightly soft knees 147-162: those still count -
  strict, but not so strict that a student on the toes with soft knees
  loses push-ups (owner, 2026-10-19). Without the toes in view the knee
  height can't be read, so the angle is the only knee check and the line
  is KNEE_MIN_NO_TOES_DEG (150) instead, to catch the shins-flat knee
  push-up below. Which line applies is decided once per PUSH-UP, not per
  frame: 145 if the toes were seen for FAULT_MIN_S while the height check
  runs (elbow >= 130), else 150. Per frame, the pose model dropping the toe
  now and then switched soft knees (147) to the 150 line: with the toes
  missing in 10% of frames, 74% of those push-ups were lost (now 0). With the shins flat on the mat a knee
  push-up's knee only bends to ~140-150 at the top and straightens toward
  ~170 at the bottom, so the knee HEIGHT catches it too: kneeling, the knee
  sits on the floor line through the hands and toes (~0.1 thigh lengths -
  the joint is a few cm up); on the toes it is ~0.4-0.5 thigh lengths up at
  the top. The line follows the floor at any camera height (points on the
  floor stay on one line in the picture). At the BOTTOM of a good push-up
  the knee is legitimately low too (~0.15), so the height is only checked
  while the elbow is >= KNEE_CHECK_DEG (130). It needs the toes in view;
  without them only the angle is used (at 150, see above), and the screen
  WARNS "Toes not in view" once they have been missing for most of the
  last TOE_WINDOW_S. The warning never changes the count.

  BACK: the hip more than BACK_MAX_DEG off the shoulder-ankle line, either
  way ("reasonably straight"). The isometric push-up's hold limit is 28.

  Not checked here (the coach's call): jerky or one-sided movement - one
  side-on camera can't see the far arm.

  Before START the phases are tracked too, so the camera already knows the
  student is at the top. A student not at the top when START is pressed
  must get there first: the first rep starts from straight arms.

  The body must be roughly horizontal (body_tilt <= BODY_TILT_MAX_DEG, 50)
  for anything to count - a student standing (~80-90) or kneeling up (~65+)
  is "not in the push-up position" and the rep under way is dropped. The
  top of a push-up reads ~20-30 on a level phone; 50 leaves room for a phone
  tilted ~20 degrees (at 40, a 10-15 degree tilt stopped the counting).

When the test ends
--------------------------------------------------------------------
  * No push-up for GAP_S (6 s = two 3 s beats), from START or from the
    last one: "can't keep the cadence / stops to rest". A push-up that was
    NOT counted (knees, back, depth...) still resets it: the student is
    still going, and counting only correct ones made a single bad rep in a
    good set end the test (good, bad 3 s later, next good lands on 6 s).
    A student who keeps going with every rep wrong is the coach's STOP.
    Going faster than the beat is fine.
  * The coach presses STOP.
  The gap clock only runs on frames where the body is seen, and each frame
  adds at most MAX_FRAME_DT, so a Wi-Fi gap can't end the test by itself.
  The 3 s beat itself is played by the browser (screen4); this file only
  ends the test.

Usage
-----
  python push_up.py
  python push_up.py --source clip.mp4 --debug

  Keys : SPACE = start counting ("GO")
         R     = reset
         Q     = quit and print the summary

Camera placement
----------------
  Phone on the floor (or just above it), SIDE-ON to the participant,
  1.5-2.5 m away, with the whole body from the hands to the toes in frame.
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
CADENCE_S = 3.0    # "1 push-up every 3 seconds" (the browser beeps it)
GAP_S     = 6.0    # no counted push-up for two beats ends the test

# ── Detection tuning ─────────────────────────────────────────────────────────────
UP_DEG          = 160.0   # elbow at least this = arms straight (the top)
DOWN_DEG        = 100.0   # elbow at most this = down to 90 degrees
SHALLOW_DIP_DEG = 140.0   # a dip from the top must go below this to be a
                          # "not down to 90" attempt; less is resting at the
                          # top (see the docstring)
REBOUND_DEG     = 130.0   # from the bottom up past this, then back down to
                          # DOWN_DEG = the arms were not straightened
KNEE_MIN_DEG    = 145.0   # knee bent more than this = knees down (toes seen)
KNEE_MIN_NO_TOES_DEG = 150.0  # ...the same without the toes (no height check)
KNEE_LOW        = 0.25    # knee lower than this above the hands-toes line, in
                          # thigh lengths = kneeling (see the docstring)
KNEE_CHECK_DEG  = 130.0   # ...checked only while the elbow is at least this
BACK_MAX_DEG    = 25.0    # hip further off the shoulder-ankle line = back bent
FAULT_MIN_S     = 0.2     # a fault must be seen this long within one rep
BODY_TILT_MAX_DEG = 50.0  # shoulder-ankle line flatter than this = in position
MAX_FRAME_DT    = 0.5     # most one frame may add to any clock (Wi-Fi gaps)
SIDE_SWITCH_VIS = 0.10    # the other side must be this much clearer to switch
TOE_WINDOW_S    = 1.5     # "Toes not in view" shows when the toes were missing
TOE_WARN_FRAC   = 0.5     # in more than this share of the tracked frames over
TOE_CLEAR_FRAC  = 0.2     # the window; it clears below TOE_CLEAR_FRAC

SMOOTH_FRAMES   = 3       # median window, as in the other rep counters
MIN_VIS         = 0.50    # landmark visibility floor

SIDE_LEFT, SIDE_RIGHT = "left", "right"
SIDES = (SIDE_LEFT, SIDE_RIGHT)

# Landmark indices (33-point BlazePose topology)
SIDE_IDX = {
    #            shoulder elbow wrist hip knee ankle toe
    SIDE_LEFT:  (11, 13, 15, 23, 25, 27, 31),
    SIDE_RIGHT: (12, 14, 16, 24, 26, 28, 32),
}

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (24, 26), (25, 27), (26, 28),             # legs
    (27, 31), (28, 32),                                 # feet
]

# Phases of the push-up
PH_UNKNOWN  = "unknown"    # not yet seen at the top
PH_UP       = "up"         # arms straight
PH_LOWERING = "lowering"   # left the top, not down to 90 yet
PH_DOWN     = "down"       # reached the bottom; counts once back at the top

# Why a rep was not counted (key -> plain words)
NOT_COUNTED = {
    "shallow":      "not down to 90 degrees",
    "not_straight": "arms not straightened at the top",
    "knees":        "knees touched the floor",
    "back":         "back not straight",
}

END_GAP     = "no push-up for 6 s"
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
    """Everything the test compares, for one side of the body - or None when
    any of its six main joints is unclear or outside the picture (MediaPipe
    still returns a confident-looking guess for joints off the edge). The toe
    is optional: without it knee_h is None."""
    idxs = SIDE_IDX[side]
    if not all(_clear(lms[i]) for i in idxs[:6]):
        return None
    s, e, wr, hp, k, a = (lm_px(lms, i, w, h) for i in idxs[:6])
    thigh = float(np.linalg.norm(hp - k))
    if thigh < 1.0 or abs(a[0] - s[0]) < 1.0:
        return None

    # Hip off the shoulder-ankle line, signed by screen height (above the
    # line = piked), as in isometric_pushup.py.
    line_y = s[1] + (a[1] - s[1]) * (hp[0] - s[0]) / (a[0] - s[0])
    bend = 180.0 - calc_angle(s, hp, a)
    if hp[1] > line_y:
        bend = -bend

    # Knee above the floor line through the hand and the toe tip.
    knee_h = None
    if _clear(lms[idxs[6]]):
        toe = lm_px(lms, idxs[6], w, h)
        if abs(toe[0] - wr[0]) >= 1.0:
            floor_y = wr[1] + (toe[1] - wr[1]) * (k[0] - wr[0]) / (toe[0] - wr[0])
            knee_h = float((floor_y - k[1]) / thigh)

    return {
        "elbow": calc_angle(s, e, wr),
        "bend": bend,
        "knee": calc_angle(hp, k, a),
        "knee_h": knee_h,
        "body_tilt": math.degrees(math.atan2(abs(a[1] - s[1]), abs(a[0] - s[0]))),
        "vis": float(np.mean([lms[i].visibility for i in idxs[:6]])),
    }


def read_sides(lms, w, h):
    return {sd: side_metrics(lms, sd, w, h) for sd in SIDES}


class MedianSmoother:
    """Short median per measurement - kills single-frame landmark spikes.
    knee_h may be missing (toes out of view): its median is over the frames
    that had it, and None while this frame doesn't."""

    KEYS = ("elbow", "bend", "knee", "body_tilt")

    def __init__(self, n=SMOOTH_FRAMES):
        self.n = int(n)
        self.bufs = {k: deque(maxlen=self.n) for k in self.KEYS + ("knee_h",)}

    def push(self, m):
        out = dict(m)
        for k in self.KEYS:
            self.bufs[k].append(float(m[k]))
            out[k] = float(np.median(self.bufs[k]))
        self.bufs["knee_h"].append(m["knee_h"])
        kh = [v for v in self.bufs["knee_h"] if v is not None]
        out["knee_h"] = float(np.median(kh)) if (m["knee_h"] is not None and kh) else None
        return out

    def reset(self):
        for buf in self.bufs.values():
            buf.clear()

    @property
    def full(self):
        return len(self.bufs["elbow"]) == self.n


class PushUpTest:
    """The whole trial: phases, faults, the count, the gap clock and the end.
    Driven entirely through update(sides, now), so it can be exercised with
    made-up readings - no camera needed."""

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
        self.side = None           # the side being read (MediaPipe naming)
        self._sm = MedianSmoother()
        self.m = None              # smoothed readings this frame (None = not seen)
        self.in_position = False   # body roughly horizontal
        self.problem = None        # what is wrong right now, for the screen
        self._fault_t = {"knees": 0.0, "back": 0.0}
        self._knee_soft_t = 0.0    # knee between the two lines (145-150), this rep
        self._toes_seen_t = 0.0    # toes seen while the height check runs, this rep
        self._peak = None          # highest elbow since reaching the bottom
        self._dip = None           # lowest elbow since leaving the top
        self.no_toes = False       # the "toes not in view" warning is showing
        self._toes = deque()       # (t, toes seen) on tracked frames
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

    # ── control ─────────────────────────────────────────────────────────
    def start(self, now=None):
        if self.started:
            return False
        self.started = True
        self.start_t = time.time() if now is None else now
        self.gap_elapsed = 0.0
        # The first push-up starts from straight arms. Caught partway, the
        # student has to get to the top first.
        if self.phase != PH_UP:
            self.phase = PH_UNKNOWN
        self._clear_faults()
        self.status_msg = "Go - one push-up every beep"
        return True

    def stop(self, reason=END_STOPPED, now=None):
        if self.stopped or not self.started:
            return
        self.stopped = True
        self.end_reason = reason
        self.end_t = time.time() if now is None else now

    def _clear_faults(self):
        for k in self._fault_t:
            self._fault_t[k] = 0.0
        self._knee_soft_t = 0.0
        self._toes_seen_t = 0.0
        self._peak = None

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
        return cands[best]

    def update(self, sides, now=None):
        """sides: {SIDE_LEFT: side_metrics|None, SIDE_RIGHT: ...}.
        Returns the end reason on the frame the test ends, else None."""
        now = time.time() if now is None else now
        dt = 0.0 if self._last_t is None else min(MAX_FRAME_DT, max(0.0, now - self._last_t))
        self._last_t = now
        if self.stopped:
            return None

        raw = self._pick(sides)
        self._note_toes(None if raw is None else raw["knee_h"] is not None, now)
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

        ev = self._step(m, dt, self.started)
        if not self.started:
            return None
        if ev in ("rep", "not_counted"):
            # A push-up attempt, counted or not: still going. One bad rep
            # inside a good set must not end the test.
            self.gap_elapsed = 0.0
            return None
        self.gap_elapsed += dt
        if self.gap_elapsed >= self.gap_s:
            self.stop(END_GAP, now)
            self.status_msg = f"No push-up for {self.gap_s:.0f} s - test over"
            return END_GAP
        return None

    def _note_toes(self, seen, now):
        """Keep the "toes not in view" warning up to date. seen is None when
        the body isn't tracked this frame (that frame says nothing)."""
        if seen is not None:
            self._toes.append((now, seen))
        while self._toes and now - self._toes[0][0] > TOE_WINDOW_S:
            self._toes.popleft()
        if not self._toes:
            self.no_toes = False
            return
        missing = sum(1 for _, s in self._toes if not s) / len(self._toes)
        if missing > TOE_WARN_FRAC:
            self.no_toes = True
        elif missing < TOE_CLEAR_FRAC:
            self.no_toes = False

    def _step(self, m, dt, counting):
        """The phase machine. Returns "rep", "not_counted" or None."""
        self.in_position = m["body_tilt"] <= BODY_TILT_MAX_DEG
        if not self.in_position:
            # Standing, kneeling up, sitting: the rep under way is dropped.
            self.phase = PH_UNKNOWN
            self._clear_faults()
            self.problem = "Get down into the push-up position"
            return None

        e = m["elbow"]
        toes = m["knee_h"] is not None
        knees = (m["knee"] < KNEE_MIN_DEG
                 or (toes and e >= KNEE_CHECK_DEG and m["knee_h"] < KNEE_LOW))
        knee_soft = KNEE_MIN_DEG <= m["knee"] < KNEE_MIN_NO_TOES_DEG
        back = abs(m["bend"]) > BACK_MAX_DEG
        self.problem = ("Knees down - legs straight, on the toes"
                        if (knees or (knee_soft and self.no_toes))
                        else ("Hips too high - body straight" if m["bend"] > 0
                              else "Hips sagging - body straight") if back
                        else None)

        if self.phase == PH_UNKNOWN:
            if e >= UP_DEG:
                self.phase = PH_UP
            return None

        if self.phase == PH_UP:
            if e >= UP_DEG:
                return None
            # Leaving the top starts a new rep: its faults start here.
            self._clear_faults()
            self._dip = e
            self.phase = PH_LOWERING

        # A rep under way: add up how long each fault has been seen.
        if knees:
            self._fault_t["knees"] += dt
        if back:
            self._fault_t["back"] += dt
        if knee_soft:
            self._knee_soft_t += dt
        if toes and e >= KNEE_CHECK_DEG:
            self._toes_seen_t += dt

        if self.phase == PH_LOWERING:
            self._dip = min(self._dip, e)
            if e <= DOWN_DEG:
                self.phase = PH_DOWN
                self._peak = e
            elif e >= UP_DEG:
                self.phase = PH_UP
                if self._dip > SHALLOW_DIP_DEG:
                    return None    # a wobble at the top, not an attempt
                return self._reject("shallow", counting)
            return None

        # PH_DOWN
        if e >= UP_DEG:
            self.phase = PH_UP
            if not counting:
                return None
            # Toes not seen long enough this rep for the height check: the
            # knee angle is the only knee check, so it uses the stricter line.
            if (self._toes_seen_t < FAULT_MIN_S - 1e-9
                    and self._fault_t["knees"] + self._knee_soft_t >= FAULT_MIN_S - 1e-9):
                return self._reject("knees", counting)
            for fault in ("knees", "back"):
                if self._fault_t[fault] >= FAULT_MIN_S - 1e-9:
                    return self._reject(fault, counting)
            self.count += 1
            self.status_msg = f"Push-up {self.count}"
            return "rep"
        self._peak = max(self._peak, e)
        if self._peak >= REBOUND_DEG and e <= DOWN_DEG:
            # Came partway up and went back down: that one never reached
            # straight arms. The next one starts at this bottom.
            self._clear_faults()
            self._peak = e
            return self._reject("not_straight", counting)
        return None

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


def body_points(lms, side, w, h):
    """Shoulder, elbow, wrist, hip, knee, ankle of the side being read."""
    return [lm_px(lms, i, w, h) for i in SIDE_IDX[side][:6]]


def draw_hud(frame, test, pts=None):
    h, w = frame.shape[:2]
    colour = C_BAD if test.problem else (C_OK if test.phase == PH_UP else C_WARN)

    if pts is not None:
        s, e, wr, hp, k, a = (tuple(np.int32(p)) for p in pts)
        for p1, p2 in ((s, e), (e, wr), (s, hp), (hp, k), (k, a)):
            cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)
        cv2.circle(frame, e, 9, colour, -1, cv2.LINE_AA)

    # Top bar: the gap clock filling toward the end.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = test.gap_progress()
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 10), C_BAD if prog > 0.6 else C_WARN, -1)

    draw_panel(frame, 12, 22, 430, 200)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _hud_text(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    state = "DONE" if test.stopped else "COUNTING" if test.started else "READY"
    line(f"PUSH-UP  [{state}]", colour, 0.68, 30)
    line(f"Push-ups: {test.count}", C_OK, 0.9, 34)
    if test.m is not None:
        line(f"Elbow: {test.m['elbow']:.0f}  (top {UP_DEG:.0f}, bottom {DOWN_DEG:.0f})")
        line(f"Phase: {test.phase}   Not counted: {test.total_not_counted}")
    else:
        line("Body not tracked - side-on, hands to toes in frame", C_BAD)
    if test.started and not test.stopped:
        line(f"Since last push-up: {test.gap_elapsed:.1f} / {test.gap_s:.0f} s")

    if test.problem and test.m is not None:
        warn = _hud_text(test.problem)
        size = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        x0 = max(12, (w - size[0]) // 2 - 14)
        draw_panel(frame, x0, 214, min(w - 12, x0 + size[0] + 28), 254, C_BAD, 0.8)
        cv2.putText(frame, warn, (x0 + 14, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    C_TEXT, 2, cv2.LINE_AA)

    if test.no_toes and test.tracked:
        warn = "Toes not in view - move the phone back"
        size = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        x0 = max(12, (w - size[0]) // 2 - 14)
        y0 = 262 if test.problem else 214
        draw_panel(frame, x0, y0, min(w - 12, x0 + size[0] + 28), y0 + 40, C_BAD, 0.8)
        cv2.putText(frame, warn, (x0 + 14, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    C_TEXT, 2, cv2.LINE_AA)

    if test.status_msg:
        msg = _hud_text(test.status_msg)
        draw_panel(frame, 12, h - 54, min(w - 12, 12 + 13 * len(msg) + 24), h - 12)
        cv2.putText(frame, msg, (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 2, cv2.LINE_AA)
    return frame


def _stage_push_up(session_id, res):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/push_up",
            json={
                "session_id":   session_id,
                "reps":         res["reps"],
                "duration_sec": res["duration_sec"],
                "end_reason":   res["end_reason"],
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] push_up staged for review → {res['reps']} reps  "
              f"session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Push-Up Test")
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

    print("[INFO] Side-on camera at floor level, hands to toes in frame. "
          "SPACE = start, R = reset, Q = quit (no cadence beep here)\n")

    test = PushUpTest()
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

    win = "Push-Up Test"
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
                _stage_push_up(args.session_id, test.result(now))
                staged = True

            pts = None
            if test.tracked and lms is not None and test.side:
                pts = body_points(lms, test.side, fw, fh)
            if args.debug and test.m is not None:
                m = test.m
                print(f"side={test.side} phase={test.phase} elbow={m['elbow']:.0f} "
                      f"bend={m['bend']:+.0f} knee={m['knee']:.0f} "
                      f"knee_h={'-' if m['knee_h'] is None else round(m['knee_h'], 2)} "
                      f"tilt={m['body_tilt']:.0f} reps={test.count} gap={test.gap_elapsed:.1f}")
            draw_hud(frame, test, pts)
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
        _stage_push_up(args.session_id, res)
    print("\n── Push-Up ──────────────────────────────────")
    print(f"  Push-ups      : {res['reps']}")
    print(f"  Time          : {res['duration_sec']} s")
    print(f"  Ended         : {res['end_reason'] or '--'}")
    for k, n in res["not_counted"].items():
        if n:
            print(f"  Not counted   : {n} x {NOT_COUNTED[k]}")


if __name__ == "__main__":
    main()
