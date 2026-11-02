"""
Wheelchair Ramp Test - upper-body strength and endurance, scored 0 to 2
=====================================================================
Libraries : opencv-python, numpy (no MediaPipe: the camera watches the wheel)

Protocol (Brockport Physical Fitness Test Manual, p. 87, fig. 5.23)
--------------------------------------------------------------------
  The participant pushes their wheelchair up a standard ramp (1 inch of rise
  per 12 inches of run, at least 8 ft long), any push technique. Lines are
  marked 8 ft (2.4 m), 15 ft (4.6 m) and 30 ft (9.1 m) from the start of the
  incline. The participant starts with the lead (front) wheels off the ramp
  and tries to get the REAR wheels beyond the lines:

      8 ft   the minimal standard
      15 ft  the preferred standard (or the top of a longer ramp the
             participant uses often)

  Not timed; multiple trials are allowed. A spotter walks behind the chair
  in case it rolls back.

  Scored here as one number:  2 = rear wheels past the 15 ft line,
  1 = past the 8 ft line, 0 = not past the 8 ft line. A ramp shorter than
  15 ft has no 15 ft line, so its best is 1. The 30 ft line is left out: a
  phone can't keep a 9 m ramp and the wheel in view.

How it's read from the camera
--------------------------------------------------------------------
  The phone stands SIDE-ON to the ramp on a stand. Before the test the coach
  taps three points on the picture: the START of the incline, the 8 ft line
  and the 15 ft line (the 15 ft one is skipped on a short ramp). The taps
  give where each line is, which way is uphill, the ramp surface (the taps'
  heights) and the scale: 8 ft between the first two taps.

  The app finds the BIG REAR WHEEL, not the student. A spotter has to walk
  right behind the chair, and on side-on frames of wheelchair ramp videos the
  pose model picked the standing helper over the seated student in 3 of 4
  frames with a helper in them (and on the manual's own photo). A wheel, on
  the other hand, is a circle of a known size sitting on the ramp. Circles
  of the wrong size (the front casters) or not resting on the ramp (trees,
  railings, a helper's legs) are thrown out, and of the rest the one with
  an edge running most of the way round wins. On 5 side-on video frames
  (3 with a helper beside the chair) it found the rear wheel in all 5, with
  the back edge within 5 cm on 4; on an old steel chair with a bar across
  the wheel it was ~20 cm out.

  "Rear wheels beyond the line" is read literally: the BACK edge of the rear
  wheel (the downhill side of the tyre) must be past the line. After the
  circle is found, its radius is pushed out to the outermost complete ring,
  the tyre, so the push rim or spokes inside it aren't taken for the wheel's
  edge (the push rim sits ~5 cm in).

  A line counts once the back edge has been past it for PASS_HOLD_S over at
  least PASS_MIN_READS readings, so one bad frame can't pass it. Rolling back
  afterwards doesn't take it away. The test ends by itself when the last
  line is passed; otherwise the coach presses STOP / SAVE once the student
  has had their tries.

What the app scores vs. what the coach judges
--------------------------------------------------------------------
  SCORED: the furthest line the rear wheels get past. The coach can't change
  it (as for the other camera-scored tests), only RESET the run.

  NOT SCORED (the tester's job): that the ramp is safe and to the standard,
  the spotter, and how the student pushes (any technique is allowed).

  The start is checked: before it can score, the rear wheel's front edge must
  be START_BEHIND_M behind the start line, which puts the front wheels off
  the ramp.

Usage
-----
  python wheelchair_ramp.py
  python wheelchair_ramp.py --source clip.mp4 --lines "120,610;520,560;880,520"

  Click the START, 8 ft and 15 ft lines on the picture (or pass --lines in
  pixels).  Keys : S = start   R = reset   C = clear the lines   Q = quit
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np


# ── Protocol constants ───────────────────────────────────────────────────────────
FT_M = 0.3048
LINE_NAMES = ("START", "8 ft", "15 ft")
LINE_M = (0.0, 8 * FT_M, 15 * FT_M)      # distance of each line from the start
MAX_SCORE = 2

SCORE_TEXT = {
    2: "past the 15 ft line",
    1: "past the 8 ft line",
    0: "not past the 8 ft line",
}

# ── Detection tuning ─────────────────────────────────────────────────────────────
# Rear wheels are 20-26 in (tyre radius ~0.25-0.33 m); the search is wider so
# a slightly angled camera (the wheel looks a bit narrow) still finds it. The
# front casters (~0.1 m) are far below the range.
WHEEL_R_MIN_M  = 0.20
WHEEL_R_MAX_M  = 0.40
FLOOR_TOL_M    = 0.15    # wheel bottom within this of the tapped ramp surface
TYRE_GROW_MAX  = 1.45    # tyre search: up to this x the circle's radius ...
TYRE_PEAK_FRAC = 0.8     # ... the outermost ring this complete vs the best
CANDIDATES     = 6       # circles on the ramp checked for the most complete ring
EDGE_MIN_X     = 3.0     # an edge = this many x the picture's median edge strength
WORK_R_MAX     = 80      # px: the biggest wheel's radius in the searched picture
MIN_SPAN_FRAC  = 0.08    # START to 8 ft must be at least this much of the width
MIN_GAP_FRAC   = 0.04    # 8 ft to 15 ft likewise

MAX_SPEED_MPS  = 2.0     # faster than anyone pushes up a 1:12 ramp
TRACK_SLACK_M  = 0.25    # ... plus this much, for a reading's own error
LOST_S         = 1.0     # wheel unseen this long: search the whole ramp again
CONFIRM_M      = 0.15    # a wheel found afresh must be seen twice within this
SMOOTH_N       = 3       # median of this many back-edge readings

PASS_HOLD_S    = 0.3     # back edge past a line this long ...
PASS_MIN_READS = 3       # ... over at least this many readings
START_BEHIND_M = 0.15    # rear wheel's front edge this far behind the start line
START_HOLD_S   = 0.3     # ... for this long before the run can score


# ── Colours (BGR), same as the other tests ───────────────────────────────────────
C_DARK = (28, 28, 28)
C_OK   = (80, 220, 120)
C_WARN = (60, 190, 250)
C_BAD  = (80, 80, 240)
C_TEXT = (245, 245, 245)


# ── The coach's line taps ────────────────────────────────────────────────────────
class RampLines:
    """START, 8 ft and (optionally) 15 ft, tapped on the picture. Kept in
    fractions of the picture, so they survive a change of frame size; the
    engine clears them when the camera itself changes."""

    def __init__(self):
        self.points = []          # [(fx, fy)] in tap order

    def clear(self):
        self.points = []

    @property
    def count(self):
        return len(self.points)

    @property
    def ready(self):
        return len(self.points) >= 2

    @property
    def max_score(self):
        return len(self.points) - 1 if self.ready else 0

    @property
    def next_name(self):
        return LINE_NAMES[len(self.points)] if len(self.points) < len(LINE_NAMES) else None

    def add(self, fx, fy):
        """Add the next line. Returns an error message, or None."""
        if len(self.points) >= len(LINE_NAMES):
            return "All three lines are set - CLEAR LINES to tap them again"
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            return "Tap inside the picture"
        if len(self.points) == 1:
            if abs(fx - self.points[0][0]) < MIN_SPAN_FRAC:
                return "The 8 ft line is too close to START - tap it again"
        elif len(self.points) == 2:
            d = 1 if self.points[1][0] > self.points[0][0] else -1
            if (fx - self.points[1][0]) * d < MIN_GAP_FRAC:
                return "The 15 ft line must be further up the ramp than 8 ft - tap it again"
        self.points.append((float(fx), float(fy)))
        return None

    # Everything below needs ready == True and the frame size.
    def direction(self):
        """+1 when the ramp goes up to the right of the picture, -1 to the left."""
        return 1 if self.points[1][0] > self.points[0][0] else -1

    def px(self, w, h):
        return [(fx * w, fy * h) for fx, fy in self.points]

    def px_per_m(self, w, h):
        p = self.px(w, h)
        return abs(p[1][0] - p[0][0]) / LINE_M[1]

    def to_m(self, x, w, h):
        """Picture x -> metres up the ramp from the start line, piecewise
        between the taps (each stretch keeps its own scale, so a camera not
        quite square to the ramp still puts every line where it was tapped)."""
        p = self.px(w, h)
        d = self.direction()
        knots = [((px_ - p[0][0]) * d, m) for (px_, _), m in zip(p, LINE_M)]
        s = (x - p[0][0]) * d
        k = 1 if s < knots[1][0] or len(knots) == 2 else 2
        (s0, m0), (s1, m1) = knots[k - 1], knots[k]
        return m0 + (s - s0) * (m1 - m0) / (s1 - s0)

    def floor_y(self, x, w, h):
        """The ramp surface at picture x: straight between the taps,
        carried on past the ends."""
        p = sorted(self.px(w, h))
        if x <= p[1][0] or len(p) == 2:
            (x0, y0), (x1, y1) = p[0], p[1]
        else:
            (x0, y0), (x1, y1) = p[-2], p[-1]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# ── Finding the rear wheel ───────────────────────────────────────────────────────
class Wheel:
    __slots__ = ("cx", "cy", "r")

    def __init__(self, cx, cy, r):
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)


_RING_ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
# Leave out the top of the wheel, where the hands and arms are.
_RING_ANGLES = _RING_ANGLES[np.sin(_RING_ANGLES) > -0.6]
_RING_COS, _RING_SIN = np.cos(_RING_ANGLES), np.sin(_RING_ANGLES)


def _ring_scores(gx, gy, cx, cy, radii, edge_min):
    """For each circle (centre in gx's pixels): how much of the way round it
    an edge runs - the share of points whose radial edge is at least
    edge_min. A share, not a mean: a straight edge (a railing, the ramp's
    side) crossing the circle is very strong at two points but covers
    almost none of it, while a tyre's edge goes all the way round."""
    h, w = gx.shape
    out = []
    for r in radii:
        xs = np.round(cx + r * _RING_COS).astype(int)
        ys = np.round(cy + r * _RING_SIN).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if ok.sum() < len(xs) // 3:
            out.append(0.0)
            continue
        rad = np.abs(gx[ys[ok], xs[ok]] * _RING_COS[ok] + gy[ys[ok], xs[ok]] * _RING_SIN[ok])
        out.append(float((rad >= edge_min).sum()) / len(xs))
    return np.array(out)


def find_wheel(frame, lines, near=None, near_m=None):
    """The rear wheel resting on the ramp, or None.

    near / near_m: the last wheel and how far (metres) it can have moved
    since; when given, only circles that close are considered."""
    h, w = frame.shape[:2]
    s = lines.px_per_m(w, h)
    ys = [lines.floor_y(x, w, h) for x in (0, w)]
    top = int(max(0, min(ys) - 2 * WHEEL_R_MAX_M * s - 0.1 * s))
    bot = int(min(h, max(ys) + FLOOR_TOL_M * s + 2))
    # Searched at a size where the biggest wheel is ~WORK_R_MAX px: circle
    # search time grows with the radius, and 1 px there is still ~1 cm.
    k = min(1.0, WORK_R_MAX / (WHEEL_R_MAX_M * s))
    ks = s * k
    rmin, rmax = max(6, int(WHEEL_R_MIN_M * ks)), int(WHEEL_R_MAX_M * ks)
    if (bot - top) * k < 2 * rmin:
        return None
    gray = cv2.cvtColor(frame[top:bot], cv2.COLOR_BGR2GRAY)
    if k < 1.0:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(4, int(0.3 * ks)),
                               param1=120, param2=40, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return None

    def full(cx, cy, r):                   # work pixels -> frame pixels
        return Wheel(cx / k, cy / k + top, r / k)

    def on_floor(wh):
        return abs(wh.cy + wh.r - lines.floor_y(wh.cx, w, h)) <= FLOOR_TOL_M * s

    cands = []
    for cx, cy, r in circles[0]:           # strongest first
        wh = full(cx, cy, r)
        if not on_floor(wh):
            continue
        if near is not None and abs(wh.cx - near.cx) > near_m * s:
            continue
        cands.append((cx, cy, r))
        if len(cands) >= CANDIDATES:
            break
    if not cands:
        return None

    # The circle search ranks by votes, and a too-big circle through a
    # wheel's edge plus the chair's frame can outvote the wheel itself (an
    # old steel chair did). So each candidate is checked by how strong an
    # edge actually runs around it, and the most complete ring wins. Its radius
    # is then pushed out to the tyre: the outermost strong ring.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_min = EDGE_MIN_X * float(np.median(np.hypot(gx, gy))) + 1.0
    best, best_score = None, -1.0
    for cx, cy, r in cands:
        radii = np.arange(int(r * 0.9), int(min(r * TYRE_GROW_MAX, rmax * 1.1)) + 1)
        sc = _ring_scores(gx, gy, cx, cy, radii, edge_min)
        if not len(sc) or sc.max() <= 0:
            continue
        peaks = [i for i in range(len(sc))
                 if sc[i] >= TYRE_PEAK_FRAC * sc.max()
                 and (i == 0 or sc[i] >= sc[i - 1]) and (i == len(sc) - 1 or sc[i] >= sc[i + 1])]
        wh = full(cx, cy, float(radii[peaks[-1]]) if peaks else r)
        if not on_floor(wh):
            wh = full(cx, cy, r)
        if sc.max() > best_score:
            best, best_score = wh, sc.max()
    return best


# ── The run ──────────────────────────────────────────────────────────────────────
class RampTracker:
    """Follows the wheel from frame to frame and keeps the furthest line the
    back of the rear wheel has got past. No MediaPipe, no threads: the web
    engine and the desktop loop both feed it frames."""

    def __init__(self):
        self.lines = RampLines()
        self.reset()

    def reset(self):
        """A new run on the same lines."""
        self.started = False       # coach pressed START
        self.armed = False         # wheel seen behind the start since START
        self.done = False
        self.end_reason = None
        self.best = None           # furthest line passed (0-2), None until armed
        self.reached_m = None      # furthest the back edge got, metres
        self.status_msg = ""
        self._behind_since = None
        self._pass = None          # (line index, since, reads) for the next line
        self._clear_track()

    def _clear_track(self):
        self.wheel = None
        self.back_m = None         # smoothed back-edge position, metres
        self._last_t = None
        self._hist = deque(maxlen=SMOOTH_N)
        self._fresh = None         # (wheel, t) waiting for a second sighting

    def clear_lines(self):
        self.lines.clear()
        self.reset()

    # The score while the run is live, and what gets staged.
    def score(self):
        return self.best

    def result(self):
        return {
            "score": self.best,
            "reached_m": None if self.reached_m is None else round(max(0.0, self.reached_m), 2),
            "max_score": self.lines.max_score,
            "end_reason": self.end_reason,
        }

    def start(self):
        if not self.lines.ready:
            return "Tap the START and 8 ft lines on the picture first"
        if self.started:
            return "Already started - RESET to start again"
        self.reset()
        self.started = True
        self.status_msg = "Roll back so the front wheels are off the ramp"
        return None

    def finish(self, reason="stopped"):
        """STOP pressed. Returns the result if the run had begun (the wheel
        was seen at the start), else None: nothing was measured."""
        if not self.armed:
            return None
        if not self.done:
            self.done = True
            self.end_reason = reason
        return self.result()

    # ── one frame ───────────────────────────────────────────────────────
    def update(self, frame, now):
        """Measure one frame. Returns the result once when the run ends by
        itself (the last line passed), else None."""
        if not self.lines.ready:
            return None
        h, w = frame.shape[:2]
        wheel = self._track(frame, now)
        if wheel is None:
            if self.started and not self.done:
                self.status_msg = ("Can't see the back wheel - keep the side of the chair "
                                   "facing the camera clear (spotter on the far side)")
            return None

        d = self.lines.direction()
        back = self.lines.to_m(wheel.cx - d * wheel.r, w, h)
        front = self.lines.to_m(wheel.cx + d * wheel.r, w, h)
        self._hist.append(back)
        self.back_m = float(np.median(self._hist))
        if not self.started or self.done:
            return None

        if not self.armed:
            if front <= -START_BEHIND_M:
                if self._behind_since is None:
                    self._behind_since = now
                if now - self._behind_since >= START_HOLD_S - 1e-6:
                    self.armed = True
                    self.best = 0
                    self.reached_m = self.back_m
                    self.status_msg = "Ready - GO!"
            else:
                self._behind_since = None
                self.status_msg = "Roll back so the front wheels are off the ramp"
            return None

        self.reached_m = max(self.reached_m, self.back_m)
        nxt = self.best + 1
        if nxt > self.lines.max_score:
            return None
        if self.back_m >= LINE_M[nxt]:
            if self._pass is None or self._pass[0] != nxt:
                self._pass = (nxt, now, 0)
            line, since, reads = self._pass
            self._pass = (line, since, reads + 1)
            if now - since >= PASS_HOLD_S - 1e-6 and reads + 1 >= PASS_MIN_READS:
                self.best = nxt
                self._pass = None
                self.status_msg = f"Rear wheels past the {LINE_NAMES[nxt]} line"
                if self.best >= self.lines.max_score:
                    self.done = True
                    self.end_reason = f"past the {LINE_NAMES[nxt]} line"
                    return self.result()
        else:
            self._pass = None
            self.status_msg = ""
        return None

    def _track(self, frame, now):
        """The wheel this frame, following the last one; None if unseen."""
        if self.wheel is not None and now - self._last_t > LOST_S:
            self._clear_track()
        if self.wheel is not None:
            reach = MAX_SPEED_MPS * (now - self._last_t) + TRACK_SLACK_M
            wheel = find_wheel(frame, self.lines, self.wheel, reach)
            if wheel is not None:
                self.wheel, self._last_t = wheel, now
            return wheel

        # Not tracking: a wheel found afresh must show up twice in about the
        # same place before it's believed, so one stray circle can't jump in.
        wheel = find_wheel(frame, self.lines)
        if wheel is None:
            return None
        h, w = frame.shape[:2]
        s = self.lines.px_per_m(w, h)
        if (self._fresh is not None and now - self._fresh[1] <= LOST_S
                and abs(wheel.cx - self._fresh[0].cx) <= CONFIRM_M * s):
            self._fresh = None
            self.wheel, self._last_t = wheel, now
            return wheel
        self._fresh = (wheel, now)
        return None


# ── Drawing ──────────────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, colour=C_DARK, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _ascii(txt):
    """cv2's Hershey fonts draw '?' for anything outside ASCII."""
    return txt.encode("ascii", "replace").decode("ascii")


def draw_hud(frame, tracker):
    h, w = frame.shape[:2]
    tr = tracker
    lines = tr.lines
    pts = lines.px(w, h)
    s = lines.px_per_m(w, h) if lines.ready else None

    # The tapped lines, upright through each tap, green once passed.
    for i, (x, y) in enumerate(pts):
        passed = tr.best is not None and i > 0 and tr.best >= i
        col = C_OK if passed else (C_TEXT if i == 0 else C_WARN)
        tall = int(1.2 * s) if s else 120
        cv2.line(frame, (int(x), int(y) - tall), (int(x), int(y) + 12), col, 3, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 6, col, -1, cv2.LINE_AA)
        cv2.putText(frame, LINE_NAMES[i], (int(x) + 6, int(y) - tall + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    wh = tr.wheel
    if wh is not None and lines.ready:
        d = lines.direction()
        cv2.circle(frame, (int(wh.cx), int(wh.cy)), int(wh.r), C_OK, 3, cv2.LINE_AA)
        bx = int(wh.cx - d * wh.r)
        cv2.line(frame, (bx, int(wh.cy - wh.r)), (bx, int(wh.cy + wh.r)), C_WARN, 3, cv2.LINE_AA)

    draw_panel(frame, 12, 22, 560, 150)
    y = 48

    def line(txt, col=C_TEXT, scale=0.6, dy=26):
        nonlocal y
        cv2.putText(frame, _ascii(txt), (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2,
                    cv2.LINE_AA)
        y += dy

    line("WHEELCHAIR RAMP TEST", C_OK if tr.best else C_TEXT, 0.64, 30)
    if not lines.ready or (lines.next_name and not tr.started):
        line(f"Tap the {lines.next_name} line" +
             ("  (or START if the ramp is under 15 ft)" if lines.count == 2 else ""), C_WARN)
    if tr.back_m is not None:
        line(f"Back of the rear wheel: {tr.back_m:.1f} m up the ramp")
    if tr.best is not None:
        line(f"Score: {tr.best} ({SCORE_TEXT[tr.best]})", C_OK if tr.best else C_TEXT)

    if tr.status_msg:
        msg = _ascii(tr.status_msg)
        draw_panel(frame, 12, h - 56, w - 12, h - 12)
        cv2.putText(frame, msg, (24, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 2,
                    cv2.LINE_AA)

    if tr.done and tr.best is not None:
        big = f"SCORE {tr.best}"
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
        org = ((w - size[0]) // 2, (h + size[1]) // 2)
        draw_panel(frame, org[0] - 30, org[1] - size[1] - 24,
                   org[0] + size[0] + 30, org[1] + 24, C_DARK, 0.72)
        cv2.putText(frame, big, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, C_OK, 5, cv2.LINE_AA)
    return frame


# ── Desktop run (webcam or clip) ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Brockport Wheelchair Ramp Test")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--lines", default=None,
                    help='START;8ft[;15ft] in pixels, e.g. "120,610;520,560;880,520"')
    ap.add_argument("--start", action="store_true", help="press START at once (with --lines)")
    ap.add_argument("--no-window", action="store_true",
                    help="don't show the video (print the result only)")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source!r}")
    clip = not isinstance(src, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = RampTracker()
    size = {}
    pending_lines = None
    if args.lines:
        pending_lines = [tuple(float(v) for v in p.split(",")) for p in args.lines.split(";")]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and size and not tracker.started:
            err = tracker.lines.add(x / size["w"], y / size["h"])
            if err:
                print(err)

    if not args.no_window:
        cv2.namedWindow("Wheelchair Ramp Test")
        cv2.setMouseCallback("Wheelchair Ramp Test", on_mouse)

    t0 = time.time()
    n = 0
    now = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        fh, fw = frame.shape[:2]
        size.update(w=fw, h=fh)
        if pending_lines:
            for x, y in pending_lines:
                err = tracker.lines.add(x / fw, y / fh)
                if err:
                    sys.exit(err)
            pending_lines = None
            if args.start:
                tracker.start()
        # A clip runs on its own clock, so it scores the same at any speed.
        now = n / fps if clip else time.time() - t0
        if tracker.update(frame, now) is not None:
            print(f"{now:6.1f}s", tracker.result())
        if args.no_window:
            continue
        draw_hud(frame, tracker)
        cv2.imshow("Wheelchair Ramp Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            err = tracker.start()
            if err:
                print(err)
        if key == ord("r"):
            tracker.reset()
        if key == ord("c"):
            tracker.clear_lines()

    cap.release()
    cv2.destroyAllWindows()
    if not tracker.done:
        print(f"{now:6.1f}s", tracker.finish() or "nothing measured")


if __name__ == "__main__":
    main()
