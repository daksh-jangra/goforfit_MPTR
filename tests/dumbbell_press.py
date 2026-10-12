"""
Dumbbell Press Test – overhead presses, max 50, one trial
=========================================================
Libraries : opencv-python, mediapipe>=0.10.21, numpy

Protocol (Brockport Physical Fitness Test Manual p.73)
--------------------------------------------------------------------
  Seated in a wheelchair or sturdy chair, the participant holds a 15 lb
  (6.8 kg) dumbbell in the dominant hand, elbow bent, the weight close to and
  slightly in front of the shoulder (fig 5.9a). They extend the elbow and
  flex the shoulder so the weight goes straight up above the shoulder
  (fig 5.9b), then bring it back to the start. Steady pace, 3-4 s per rep.
  ONE trial. A rep counts each time the weight goes above the shoulder with
  complete elbow extension. Scoring ends at 50 reps, a rest of more than
  4 s between reps, or when they can no longer lift with full extension.

  Modifications: a student who can't fully straighten the elbow is scored
  at their maximal range, and reps slower than 4 s are allowed "as long as
  the participant is working to lift the weight".

What counts as a rep - the HAND HEIGHT, not the elbow angle
--------------------------------------------------------------------
  UP    the wrist at least WRIST_UP_UA (1.3) upper-arm lengths above the
        shoulder. Counted the moment it gets there.
  DOWN  the wrist back within WRIST_DOWN_UA (0.8) of shoulder height.
        Needed before the next rep can count. 0.8, not lower: with the
        elbow pushed forward at the start the hand sits at about chin
        height (0.6+), and at 0.5 such a student never re-armed, so nothing
        counted. The manual's start photo already reads up to 0.44. A student
        who only lowers to the chin between reps is the coach's call.

  Height is in upper-arm lengths so it doesn't depend on camera distance.
  It stands in for "complete elbow extension": with the upper arm upright,
  a straight arm puts the wrist ~1.85 above the shoulder, a 110 deg elbow
  ~1.3, a 90 deg elbow ~1.0. So 1.3 rejects a half press but accepts the
  manual's own raised photo, whose arm is not fully straight (the spotter
  holds the wrist): in the live engine it peaks at 1.34-1.58 per run, so
  1.5 counted it in only 3 of 9 runs and 1.3 counts all 9. Owner's choice
  (2026-10-12), over 1.4 (7 of 9) and keeping 1.5.

  The upper-arm length used is the LONGEST it has looked (for at least
  UA_SEEN_S in total, so a short tracking glitch doesn't count) in the
  last UA_WINDOW_S seconds, not this frame's. Halfway up a forward press the
  elbow points at the camera and the upper arm looks very short; dividing
  by that made a bent-elbow half press (hand at forehead height) read
  3-6 upper arms high, so it counted. The bone can't change length - only
  its picture shrinks - and it is seen full length at the start, where it
  hangs down. On the manual's photos the upper arm reads about the same at
  the start (80 px) and raised (72 px).

  The pressing arm is measured against the LONGER of its own upper-arm
  length and the FREE arm's. With the elbow pushed forward at the start and
  only half presses, the pressing upper arm is never seen full length, so
  its own yardstick stayed short and a half press with the elbow at 95 deg
  counted (4 of 4). The free arm usually rests on the armrest or lap and is
  seen full length. On the manual's photos it reads 0-12% longer than the
  pressing arm, and the raised photo still counts. If the free arm is out of
  view the pressing arm's own length is all there is.

  Why not the elbow angle: the press is shoulder FLEXION - the weight goes
  forward and up - so from the front the forearm points at the camera and
  the 2-D elbow angle is squashed. Played through the live (video-mode)
  engine, the manual's own photos read:
    fig 5.9a start   elbow   5-30 deg (really ~90), wrist -0.3..+0.1
    fig 5.9b raised  elbow 95-135 deg (never 140),  wrist +1.5..+2.3
                     (against that frame's upper arm; 1.28-1.58 against the
                     longest one, which is what is used now)
    idle arm on the armrest: elbow ~145 deg, wrist -1.1..-1.6
  So the elbow says nothing useful front-on - it even reads "straight" on
  the idle arm and "bent" on the raised one - while the wrist height is
  steady. The owner first chose a 140 deg elbow line, then, shown these
  numbers, chose hand height only (2026-10-12). A side-on camera (elbow in
  the picture plane) was the other option; not chosen. 3-D world landmarks
  were also tried: they read the bent start as 106-132 deg.

When the test ends
--------------------------------------------------------------------
  * 50 reps.
  * REST_S (4 s) without the arm working, counted from the first rep on.
    "Working" means a new rep, getting back down to the start after a rep,
    a press still climbing (the wrist rising another PROGRESS_UA beyond
    the highest it has been since the arm was last back at the start), or
    the weight still coming down after a rep (the wrist dropping another
    PROGRESS_UA below the lowest it has been since that rep). So a slow
    press or a slow lowering that keeps going never times out, while
    sitting still - at the bottom, holding the weight up, or stopped
    partway down - does, and so does a string of short presses that never
    get any higher (the protocol's "no longer able to lift").
  * The coach presses STOP.

  The rest clock only runs on frames where the arm is actually seen, and
  each frame adds at most MAX_FRAME_DT: a Wi-Fi gap or the spotter stepping
  in front can't end the test by itself.

What the coach judges (not scored here)
--------------------------------------------------------------------
  The 3-4 s cadence, the dumbbell weight, the seated posture and spotting.

Which arm
--------------------------------------------------------------------
  The dominant hand, which the participant picks. In AUTO (the default)
  both arms are watched until the first counted rep; that arm is locked for
  the rest of the trial, so the other hand can't add reps. The catch: if
  the FREE hand goes overhead first (a wave), it is the one locked, and the
  real presses are ignored. So the coach can pin the arm instead (LEFT /
  RIGHT, the student's own hand, DumbbellPressTest.pinned); then the other
  arm's movements are never looked at - only its upper-arm length (below).
  Owner's choice (2026-10-12): a picker with AUTO as the default, over
  always-pick and switching arms automatically.

Camera placement
----------------
  FRONT-ON, at chest height, 2-3 m away, head and both arms in frame
  (as in fig 5.9). The spotter stands BEHIND the chair - with the tester
  in front of the student, as in the manual's photos, the pose model found
  no one at all.

Usage
-----
  python dumbbell_press.py
  python dumbbell_press.py --source clip.mp4 --debug
  python dumbbell_press.py --arm right    # watch only the right arm

  Keys : SPACE = start counting ("GO")
         R     = reset
         Q     = quit and print the summary
"""

import argparse
import sys
import time
from collections import deque

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
MAX_REPS = 50
REST_S   = 4.0     # "rests for more than 4 seconds between repetitions"

# ── Detection tuning ─────────────────────────────────────────────────────────────
WRIST_UP_UA    = 1.3    # wrist this many upper arms above the shoulder = arm
                        # extended overhead. The raised photo peaks at
                        # 1.34-1.58; a 90 deg elbow with the upper arm
                        # upright ~1.0. See the docstring.
WRIST_DOWN_UA  = 0.8    # wrist back within this of shoulder height = back at
                        # the start. The start photo reads -0.3..+0.44; an
                        # elbow-forward start ~0.6. Owner's choice (2026-10-12).
PROGRESS_UA    = 0.25   # wrist rising this much counts as still working,
                        # measured from shoulder height at the lowest: a still
                        # arm at the start wobbles about +-0.25 in the live
                        # engine, and a climb measured from the bottom of a
                        # wobble let one frame push the 4 s end out to ~5.5 s.
MAX_FRAME_DT   = 0.5    # most one frame may add to the rest clock (Wi-Fi gaps)
UA_WINDOW_S    = 10.0   # upper-arm length = the longest seen this long back.
                        # Covers a slow 8 s press; short enough to follow the
                        # student moving the chair closer or further.
UA_SEEN_S      = 0.4    # ...that it has shown for at least this long in total.
                        # The single longest frame let a 2-3 frame tracking
                        # glitch stretch the yardstick for 10 s and hide the
                        # next reps. 0.4 s is more than 3 frames even at 8 fps;
                        # the real length shows longer, at the bottom or top.

SMOOTH_FRAMES  = 3      # median window, as in the other rep counters
MIN_VIS        = 0.50   # landmark visibility floor for the arm joints

ARM_LEFT, ARM_RIGHT = "left", "right"
ARMS = (ARM_LEFT, ARM_RIGHT)
ARM_AUTO = "auto"

# Landmark indices (33-point BlazePose topology)
ARM_IDX = {
    ARM_LEFT:  (11, 13, 15),   # shoulder, elbow, wrist
    ARM_RIGHT: (12, 14, 16),
}

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                       # torso
]

# Phases of one arm
PH_UNKNOWN  = "unknown"    # not yet seen at the start or the top
PH_DOWN     = "down"       # at the start position, ready to press
PH_PRESSING = "pressing"   # on the way up, not extended yet
PH_UP       = "up"         # extended overhead; must come down before the next rep

END_MAX      = "50 reps"
END_REST     = "rested 4 s"
END_STOPPED  = "stopped by coach"

# HUD palette (matches reverse_curl.py)
C_DARK  = (28, 28, 28)
C_OK    = (80, 220, 120)
C_WARN  = (60, 190, 250)
C_BAD   = (80, 80, 240)
C_TEXT  = (245, 245, 245)


# ── small helpers ────────────────────────────────────────────────────────────────
def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def arm_reading(lms, arm, w, h):
    """(rise_px, upper_px) for one arm: how far the wrist is above the
    shoulder, and how long the upper arm looks, both in pixels. None if the
    joints aren't clear or aren't in the picture. The counter turns these
    into upper-arm lengths against the longest upper arm it has seen."""
    idxs = ARM_IDX[arm]
    if not all(lms[i].visibility >= MIN_VIS for i in idxs):
        return None
    # MediaPipe guesses joints outside the picture "confidently"; those
    # produce made-up heights.
    if not all(0.0 <= lms[i].x <= 1.0 and 0.0 <= lms[i].y <= 1.0 for i in idxs):
        return None
    s, e, wr = (lm_px(lms, i, w, h) for i in idxs)
    upper = float(np.linalg.norm(s - e))
    if upper < 1.0:
        return None
    return float(s[1] - wr[1]), upper


class MedianSmoother:
    """Short median window — kills single-frame landmark spikes."""

    def __init__(self, n=SMOOTH_FRAMES):
        self.buf = deque(maxlen=int(n))

    def push(self, v):
        self.buf.append(float(v))
        return float(np.median(self.buf))

    def reset(self):
        self.buf.clear()

    @property
    def full(self):
        return len(self.buf) == self.buf.maxlen


class ArmCounter:
    """Rep phases for ONE arm. Pure state machine: feed it smoothed numbers
    with update() and it says what happened. Counting only happens when the
    caller says the test is running, but the phase is tracked before that
    too, so at START the arm is already known to be down (or up)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = PH_UNKNOWN
        self.count = 0
        self.short_presses = 0
        self.wrist_up = None
        self._sw = MedianSmoother()
        self._su = MedianSmoother()
        self._lengths = deque()    # (t, smoothed upper-arm px), last UA_WINDOW_S

    def upper_ref(self, upper_px, now):
        """The upper arm's real on-screen length: the longest it has looked
        for at least UA_SEEN_S in the last UA_WINDOW_S (see the docstring).
        The length is median-smoothed first, and only a full median goes into
        the window: right after a dropout the "median" is one frame, and one
        bad length there hid the next reps."""
        v = self._su.push(upper_px)
        while self._lengths and now - self._lengths[0][0] > UA_WINDOW_S:
            self._lengths.popleft()
        if self._su.full:
            self._lengths.append((now, v))
        elif not self._lengths:
            # Nothing settled to go on (first sight, or out of view longer
            # than the window): use this frame, but don't keep it — one bad
            # length kept here hid the next reps for 10 s.
            return v
        return self._longest_held()

    def _longest_held(self):
        """Longest length the arm has shown for UA_SEEN_S in total: walk the
        lengths from longest down, adding up how long each was on screen
        (time to the next sample, capped at MAX_FRAME_DT for gaps), and stop
        once UA_SEEN_S is reached. A short glitch never adds up to that.
        Falls back to the longest seen while the window is still shorter."""
        items = list(self._lengths)
        spans = [min(MAX_FRAME_DT, items[i + 1][0] - items[i][0]) for i in range(len(items) - 1)]
        spans.append(spans[-1] if spans else 0.0)    # the newest: as long as the one before
        total = 0.0
        for length, span in sorted(zip((x for _, x in items), spans), reverse=True):
            total += span
            if total >= UA_SEEN_S:
                return length
        return max(x for _, x in items)

    def held(self, now):
        """This arm's settled upper-arm length right now, or None if there is
        none in the window - for the OTHER arm to use as a yardstick."""
        while self._lengths and now - self._lengths[0][0] > UA_WINDOW_S:
            self._lengths.popleft()
        return self._longest_held() if self._lengths else None

    def lost(self):
        """Tracking dropped. The phase is kept — the arm is still wherever it
        was — but stale frames must not blend into the next reading."""
        self._sw.reset()
        self._su.reset()
        self.wrist_up = None

    def forget_reps(self):
        """Drop this arm's phase and reps (the other arm was locked), but keep
        its upper-arm lengths: they are still the other arm's yardstick."""
        self.phase = PH_UNKNOWN
        self.count = 0
        self.short_presses = 0
        self._sw.reset()
        self.wrist_up = None

    def update(self, rise_px, ref_px, counting):
        """ref_px is the yardstick (DumbbellPressTest picks it, see
        upper_ref). Returns "rep", "armed", "short", or None."""
        wrist_up = rise_px / ref_px
        wu = self._sw.push(wrist_up)
        if not self._sw.full:
            # Just back from a dropout (or just seen): the median is still
            # one or two frames, so one wild frame would decide on its own —
            # it counted a rep that never happened. Wait for a full window;
            # until then the arm reads as not tracked.
            return None
        self.wrist_up = wu
        is_up = wu >= WRIST_UP_UA
        is_down = wu <= WRIST_DOWN_UA

        if self.phase == PH_UNKNOWN:
            if is_down:
                self.phase = PH_DOWN
            elif is_up:
                self.phase = PH_UP
            return None

        if self.phase == PH_UP:
            if is_down:
                self.phase = PH_DOWN
                return "armed"
            return None

        # PH_DOWN or PH_PRESSING
        if is_up:
            self.phase = PH_UP
            if counting:
                self.count += 1
                return "rep"
            return None
        if is_down:
            if self.phase == PH_PRESSING:
                self.phase = PH_DOWN
                if counting:
                    self.short_presses += 1
                    return "short"
            return None
        self.phase = PH_PRESSING
        return None


class DumbbellPressTest:
    """The whole trial: both arms watched until the first rep locks one,
    the rep count, the rest clock and the end rule. Driven entirely through
    update(readings, now), so it can be exercised with made-up sequences."""

    def __init__(self, rest_s=REST_S, max_reps=MAX_REPS):
        self.rest_s = float(rest_s)
        self.max_reps = int(max_reps)
        self.pinned = None         # ARM_LEFT / ARM_RIGHT pinned by the coach
        self.reset()

    def reset(self):
        self.counters = {a: ArmCounter() for a in ARMS}
        self.arm = None            # locked working arm (MediaPipe naming)
        # self.pinned (the coach's arm, MediaPipe naming, or None for auto) is
        # deliberately NOT reset: it is set from outside and survives RESET.
        self.started = False
        self.start_t = None
        self.end_t = None
        self.stopped = False
        self.end_reason = None
        self.rest_elapsed = 0.0
        self._last_t = None
        self._best = None          # highest wrist since back at the start
        self._low = None           # lowest wrist since the last rep
        self._climbed = False      # credited a climb since back at the start
        self.status_msg = "Press START when ready"

    # ── read-outs ───────────────────────────────────────────────────────
    @property
    def count(self):
        return self.counters[self.arm].count if self.arm else 0

    @property
    def short_presses(self):
        return self.counters[self.arm].short_presses if self.arm else 0

    def shown_arm(self):
        """The arm to display: the locked one, else whichever is higher."""
        if self.arm:
            return self.arm
        best, best_w = None, None
        for a in ARMS:
            w = self.counters[a].wrist_up
            if w is not None and (best_w is None or w > best_w):
                best, best_w = a, w
        return best

    def elapsed(self, now=None):
        if not self.started:
            return 0.0
        end = self.end_t if self.stopped else (time.time() if now is None else now)
        return max(0.0, end - self.start_t)

    def rest_progress(self):
        if not self.started or self.stopped or self.count == 0:
            return 0.0
        return min(1.0, self.rest_elapsed / self.rest_s)

    # ── control ─────────────────────────────────────────────────────────
    def start(self, now=None):
        if self.started:
            return False
        self.started = True
        self.start_t = time.time() if now is None else now
        self.status_msg = "Press the weight straight up"
        return True

    def stop(self, reason=END_STOPPED, now=None):
        if self.stopped or not self.started:
            return
        self.stopped = True
        self.end_reason = reason
        self.end_t = time.time() if now is None else now

    # ── per frame ───────────────────────────────────────────────────────
    def update(self, readings, now=None):
        """readings: {ARM_LEFT: (rise_px, upper_px)|None, ARM_RIGHT: ...}.
        Returns the end reason on the frame the test ends, else None."""
        now = time.time() if now is None else now
        dt = 0.0 if self._last_t is None else min(MAX_FRAME_DT, max(0.0, now - self._last_t))
        self._last_t = now
        if self.stopped:
            return None
        counting = self.started

        # Each arm's own upper-arm yardstick first, then each is measured
        # against the longer of its own and the other arm's (see docstring).
        own = {}
        for a in ARMS:
            r = readings.get(a)
            if r is None:
                self.counters[a].lost()
                own[a] = None
            else:
                own[a] = self.counters[a].upper_ref(r[1], now)
        events = {a: None for a in ARMS}
        for a in ARMS:
            if own[a] is None:
                continue
            if self.pinned in ARMS and a != self.pinned:
                # The coach pinned the other arm: this one only lends its
                # length. Its movements never lock the test or add reps.
                self.counters[a].wrist_up = None
                continue
            other = self.counters[ARM_RIGHT if a == ARM_LEFT else ARM_LEFT].held(now)
            ref = own[a] if other is None else max(own[a], other)
            events[a] = self.counters[a].update(readings[a][0], ref, counting)

        if self.arm is None:
            # Before the first rep: nothing to rest from. The first arm to
            # complete a rep becomes the working arm.
            for a in ARMS:
                if events[a] == "rep":
                    self.arm = a
                    break
            if self.arm is None:
                return None
            other = ARM_RIGHT if self.arm == ARM_LEFT else ARM_LEFT
            self.counters[other].forget_reps()   # its stray reps never count

        c = self.counters[self.arm]
        ev = events[self.arm]
        if ev == "rep":
            self.rest_elapsed = 0.0
            self._best = None       # nothing to climb until it's back down
            self._low = c.wrist_up  # the lowering is measured from here
            self.status_msg = f"Rep {c.count}"
            if c.count >= self.max_reps:
                self.stop(END_MAX, now)
                self.status_msg = f"{self.max_reps} reps - test complete"
                return END_MAX
            return None
        if ev == "armed":
            self.rest_elapsed = 0.0
            self._best = max(c.wrist_up, 0.0)    # see PROGRESS_UA
            self._climbed = False
            self._low = None
            return None
        if ev == "short":
            self.status_msg = "Not counted - weight not pressed all the way up"

        if c.wrist_up is None:
            return None      # not seen (or just back): can't tell if working
        # Climbing counts from the start position itself, not only once out
        # of the "down" zone, so a slow press is working from its first
        # centimetres. The high persists across short presses, so pressing
        # to the same height over and over isn't progress.
        if c.phase != PH_UP and self._best is not None:
            if not self._climbed and c.wrist_up < self._best:
                # Still coming down into the start position: the climb is
                # measured from the real bottom, not from where the hand
                # crossed the "back down" line - from there a slow press got
                # no credit for most of the way up and timed out.
                self._best = max(c.wrist_up, 0.0)
            elif c.wrist_up >= self._best + PROGRESS_UA:
                self._best = c.wrist_up
                self._climbed = True
                self.rest_elapsed = 0.0
                return None
        # Coming back down after a rep is working too: a slow, controlled
        # lowering took longer than 4 s and ended the test mid-rep. Same
        # ratchet, downward, so stopping partway down is still rest.
        if c.phase == PH_UP and self._low is not None:
            if c.wrist_up <= self._low - PROGRESS_UA:
                self._low = c.wrist_up
                self.rest_elapsed = 0.0
                return None
        self.rest_elapsed += dt
        if self.rest_elapsed >= self.rest_s:
            self.stop(END_REST, now)
            self.status_msg = f"Rested {self.rest_s:.0f} s - test over"
            return END_REST
        return None

    def result(self, now=None):
        return {
            "reps": self.count,
            "arm": self.arm,
            "duration_sec": round(self.elapsed(now), 1),
            "end_reason": self.end_reason,
            "short_presses": self.short_presses,
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


def draw_hud(frame, test, arm_label=None, arm_pts=None, now=None):
    """Draw the live state. arm_label is the person's side for display."""
    h, w = frame.shape[:2]
    arm = test.shown_arm()
    c = test.counters[arm] if arm else None
    colour = C_OK if (c and c.phase == PH_UP) else C_WARN

    if arm_pts is not None:
        s, e, wr = arm_pts
        cv2.line(frame, tuple(np.int32(s)), tuple(np.int32(e)), colour, 3, cv2.LINE_AA)
        cv2.line(frame, tuple(np.int32(e)), tuple(np.int32(wr)), colour, 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.int32(e)), 9, colour, -1, cv2.LINE_AA)

    # Top bar: the rest clock filling toward the 4 s end.
    cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
    prog = test.rest_progress()
    if prog > 0:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 10), C_BAD if prog > 0.6 else C_WARN, -1)

    draw_panel(frame, 12, 22, 430, 200)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += dy

    state = ("DONE" if test.stopped else "COUNTING" if test.started else "READY")
    line(f"DUMBBELL PRESS  [{state}]", colour, 0.68, 30)
    line(f"Reps: {test.count} / {test.max_reps}", C_OK, 0.9, 34)
    line(f"Arm: {arm_label or '--'}{' (locked)' if test.arm else ''}"
         f"   {c.phase if c else ''}")
    if c is not None and c.wrist_up is not None:
        line(f"Hand above shoulder: {c.wrist_up:+.1f} (need {WRIST_UP_UA:.1f})",
             C_OK if c.phase == PH_UP else C_TEXT)
    else:
        line("Arm not tracked", C_BAD)
    if test.started and not test.stopped and test.count > 0:
        line(f"Rest: {test.rest_elapsed:.1f} / {test.rest_s:.0f} s")

    if test.status_msg:
        msg = _hud_text(test.status_msg)
        draw_panel(frame, 12, h - 54, min(w - 12, 12 + 13 * len(msg) + 24), h - 12)
        cv2.putText(frame, msg, (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 2, cv2.LINE_AA)
    return frame


def _stage_dumbbell_press(session_id, res, arm):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/dumbbell_press",
            json={
                "session_id":   session_id,
                "reps":         res["reps"],
                "arm":          arm,
                "duration_sec": res["duration_sec"],
                "end_reason":   res["end_reason"],
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] dumbbell_press staged for review → {res['reps']} reps  "
              f"session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


# ── Main (desktop window) ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dumbbell Press Test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file")
    parser.add_argument("--arm", choices=[ARM_AUTO, ARM_LEFT, ARM_RIGHT], default=ARM_AUTO,
                        help="Watch only this arm (default: the first arm to press)")
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

    # A recording is clocked by its own frame rate, a camera by the wall clock
    # (same reasoning as reverse_curl.py): the 4 s rest must be the
    # participant's, not the decoder's.
    live = isinstance(source, int)
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not live and not (1.0 <= clip_fps <= 240.0):
        clip_fps = 30.0

    print("[INFO] Front-on camera, spotter behind the chair. SPACE = start, "
          "R = reset, Q = quit\n")

    test = DumbbellPressTest()
    test.pinned = args.arm if args.arm in ARMS else None
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

    win = "Dumbbell Press Test"
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

            readings = {ARM_LEFT: None, ARM_RIGHT: None}
            lms = None
            if results.pose_landmarks:
                lms = results.pose_landmarks[0]
                for i, j in _POSE_CONNECTIONS:
                    if lms[i].visibility >= 0.45 and lms[j].visibility >= 0.45:
                        cv2.line(frame,
                                 (int(lms[i].x * fw), int(lms[i].y * fh)),
                                 (int(lms[j].x * fw), int(lms[j].y * fh)),
                                 (200, 200, 200), 2, cv2.LINE_AA)
                readings = {a: arm_reading(lms, a, fw, fh) for a in ARMS}

            ended = test.update(readings, now=now)
            if ended and not staged and args.session_id:
                _stage_dumbbell_press(args.session_id, test.result(now), test.arm)
                staged = True

            arm = test.shown_arm()
            arm_pts = None
            if arm and lms is not None and readings.get(arm) is not None:
                arm_pts = [lm_px(lms, i, fw, fh) for i in ARM_IDX[arm]]
            if args.debug:
                c = test.counters[arm] if arm else None
                print(f"arm={arm} phase={c.phase if c else '-'} "
                      f"wrist={None if not c or c.wrist_up is None else round(c.wrist_up, 2)} "
                      f"reps={test.count} rest={test.rest_elapsed:.1f}")
            draw_hud(frame, test, arm_label=arm, arm_pts=arm_pts, now=now)
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

    test.stop(END_STOPPED)
    res = test.result()
    if test.started and not staged and args.session_id:
        _stage_dumbbell_press(args.session_id, res, test.arm)
    print("\n── Dumbbell Press ───────────────────────────")
    print(f"  Reps          : {res['reps']}")
    print(f"  Arm           : {res['arm'] or '--'}")
    print(f"  Time          : {res['duration_sec']} s")
    print(f"  Ended         : {res['end_reason'] or '--'}")
    print(f"  Short presses : {res['short_presses']}")


if __name__ == "__main__":
    main()
