"""
Tests for wheelchair_ramp.py
=============================
Synthetic side-on frames: a 1:12 ramp, a spoked rear wheel with a push rim
inside the tyre, front casters, and a standing spotter right behind the
chair. The wheel is moved along a script and fed to RampTracker frame by
frame, the way the web engine does.

Run:
      python3 tests/test_wheelchair_ramp.py
      pytest tests/test_wheelchair_ramp.py

  Non-zero exit means something broke.
"""

import os
import sys

import cv2
import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import wheelchair_ramp as wr  # noqa: E402

W, H = 960, 540
S = 130.0                 # px per metre
X0 = 200.0                # start line, picture x
FLOOR0 = 470.0            # floor y at the start line
TYRE_M = 0.30             # rear tyre's outer radius
FPS = 10.0


def floor_y(x, d=1):
    """Floor at the start, then the ramp rising 1 in 12 uphill."""
    up = max(0.0, (x - X0) * d) if d == 1 else max(0.0, (X0 - x))
    return FLOOR0 - up / 12.0


def line_x(m, d=1, x0=X0):
    return x0 + d * m * S


def scene(back_m, d=1, x0=X0, spotter=True, wheel=True, seed=0):
    """A frame with the back of the rear wheel `back_m` up the ramp."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 185, np.uint8)
    # The ramp: a darker band under the floor line.
    xs = np.arange(W)
    fl = np.array([floor_y(x, d) if d == 1 else floor_y_left(x, x0) for x in xs])
    for x in xs:
        img[int(fl[x]):, x] = (150, 150, 150)
    # The taped lines on the floor.
    for m in wr.LINE_M:
        lx = int(line_x(m, d, x0))
        if 0 <= lx < W:
            cv2.line(img, (lx, int(fl[min(W - 1, lx)])), (lx, int(fl[min(W - 1, lx)]) + 6),
                     (240, 240, 240), 3)
    if wheel:
        R = TYRE_M * S
        cx = line_x(back_m, d, x0) + d * R
        fy = floor_y(cx, d) if d == 1 else floor_y_left(cx, x0)
        cy = fy - R
        # Spotter: legs and body, standing behind (downhill of) the chair.
        if spotter:
            bx = int(cx - d * 0.75 * S)
            for off in (-10, 12):
                cv2.rectangle(img, (bx + off - 7, int(fy - 0.9 * S)), (bx + off + 7, int(fy)),
                              (55, 55, 70), -1)
            cv2.rectangle(img, (bx - 22, int(fy - 1.6 * S)), (bx + 24, int(fy - 0.9 * S)),
                          (60, 40, 140), -1)
            cv2.circle(img, (bx, int(fy - 1.75 * S)), 18, (90, 120, 170), -1)
        # The student: seat, back, legs (rectangles), above the wheel.
        cv2.rectangle(img, (int(cx - 0.25 * S), int(cy - 0.75 * S)),
                      (int(cx + 0.15 * S), int(cy - 0.05 * S)), (120, 60, 50), -1)
        cv2.rectangle(img, (int(cx + d * 0.0), int(cy - 0.1 * S)),
                      (int(cx + d * 0.45 * S), int(cy + 0.05 * S)), (70, 70, 90), -1)
        # Front casters, well ahead of the rear wheel.
        kx = cx + d * 0.45 * S
        kf = floor_y(kx, d) if d == 1 else floor_y_left(kx, x0)
        cv2.circle(img, (int(kx), int(kf - 0.07 * S)), int(0.07 * S), (40, 40, 40), 4)
        # Rear wheel: spokes, push rim, tyre.
        for a in np.linspace(0, np.pi, 9):
            p = (int(cx + R * 0.85 * np.cos(a)), int(cy + R * 0.85 * np.sin(a)))
            q = (int(cx - R * 0.85 * np.cos(a)), int(cy - R * 0.85 * np.sin(a)))
            cv2.line(img, p, q, (125, 125, 125), 1, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), int(R - 0.05 * S), (95, 95, 95), 3, cv2.LINE_AA)
        t = int(0.035 * S)
        cv2.circle(img, (int(cx), int(cy)), int(R - t / 2), (30, 30, 30), t, cv2.LINE_AA)
    noise = rng.normal(0, 5, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def floor_y_left(x, x0):
    """The ramp rising to the LEFT of the picture from x0."""
    return FLOOR0 - max(0.0, x0 - x) / 12.0


def make_lines(n=3, d=1, x0=X0):
    L = wr.RampLines()
    for m in wr.LINE_M[:n]:
        x = line_x(m, d, x0)
        y = floor_y(x, d) if d == 1 else floor_y_left(x, x0)
        assert L.add(x / W, y / H) is None
    return L


def tracker(n=3, d=1, x0=X0):
    tr = wr.RampTracker()
    tr.lines = make_lines(n, d, x0)
    return tr


def run(tr, path, d=1, x0=X0, t0=0.0, **kw):
    """Feed back-edge positions (metres), one frame each. Returns (now, the
    results update() returned)."""
    out = []
    now = t0
    for i, m in enumerate(path):
        now += 1.0 / FPS
        f = scene(m, d, x0, seed=i, **kw) if m is not None else scene(0, d, x0, wheel=False, seed=i)
        r = tr.update(f, now)
        if r is not None:
            out.append(r)
    return now, out


def ramp_path(a, b, speed=0.6):
    n = max(2, int(abs(b - a) / speed * FPS))
    return list(np.linspace(a, b, n))


# ── Lines ────────────────────────────────────────────────────────────────────────
def test_lines_map_taps_to_metres():
    for d, x0 in ((1, X0), (-1, 760.0)):
        L = make_lines(3, d, x0)
        assert L.direction() == d
        for m in wr.LINE_M:
            assert abs(L.to_m(line_x(m, d, x0), W, H) - m) < 1e-6
        assert abs(L.px_per_m(W, H) - S) < 1e-6
        # Past the ends it carries on at the nearest stretch's scale.
        assert abs(L.to_m(line_x(-0.5, d, x0), W, H) + 0.5) < 1e-6


def test_lines_refuse_bad_taps():
    L = wr.RampLines()
    assert L.add(0.1, 0.8) is None
    assert L.add(0.12, 0.8) is not None          # 8 ft right on top of START
    assert L.add(0.5, 0.78) is None
    assert L.add(0.3, 0.78) is not None          # 15 ft downhill of 8 ft
    assert L.add(0.9, 0.75) is None
    assert L.add(0.95, 0.75) is not None         # a fourth line
    assert L.max_score == 2


# ── Finding the wheel ────────────────────────────────────────────────────────────
def test_finds_the_tyre_not_the_push_rim_or_spotter():
    L = make_lines()
    for back in (-0.6, 1.0, 2.2, 3.5, 4.4):
        f = scene(back)
        wh = wr.find_wheel(f, L)
        assert wh is not None, back
        got = L.to_m(wh.cx - wh.r, W, H)
        assert abs(got - back) < 0.04, (back, got, wh.r / S)


def test_ignores_wheels_not_on_the_ramp():
    """A circle of the right size floating above the ramp is not a wheel."""
    L = make_lines()
    f = scene(0, wheel=False)
    cv2.circle(f, (500, 200), int(TYRE_M * S), (30, 30, 30), 6)
    assert wr.find_wheel(f, L) is None


# ── The run ──────────────────────────────────────────────────────────────────────
def test_full_run_scores_2_and_ends():
    tr = tracker()
    assert tr.start() is None
    _, res = run(tr, [-0.8] * 6 + ramp_path(-0.8, 4.9))
    assert tr.armed and tr.done
    assert len(res) == 1 and res[0]["score"] == 2
    assert res[0]["reached_m"] >= 4.5


def test_stops_short_of_15ft_scores_1():
    tr = tracker()
    tr.start()
    run(tr, [-0.8] * 6 + ramp_path(-0.8, 3.4) + [3.4] * 10 + ramp_path(3.4, 0.5))
    assert tr.best == 1 and not tr.done
    res = tr.finish()
    assert res["score"] == 1 and 3.2 <= res["reached_m"] <= 3.5


def test_short_of_8ft_scores_0():
    tr = tracker()
    tr.start()
    run(tr, [-0.8] * 6 + ramp_path(-0.8, 2.2) + [2.2] * 10)
    assert tr.finish()["score"] == 0


def test_short_ramp_ends_at_8ft():
    tr = tracker(n=2)
    tr.start()
    _, res = run(tr, [-0.8] * 6 + ramp_path(-0.8, 3.0))
    assert tr.done and res[0]["score"] == 1 and res[0]["max_score"] == 1


def test_ramp_going_left():
    tr = tracker(d=-1, x0=760.0)
    tr.start()
    _, res = run(tr, [-0.8] * 6 + ramp_path(-0.8, 4.9), d=-1, x0=760.0)
    assert res and res[0]["score"] == 2


def test_must_start_behind_the_line():
    """Already on the ramp at START: nothing scores until it rolls back."""
    tr = tracker()
    tr.start()
    run(tr, [3.0] * 10)
    assert not tr.armed and tr.best is None
    assert tr.finish() is None                    # nothing measured
    assert "off the ramp" in tr.status_msg


def test_nothing_before_start():
    tr = tracker()
    run(tr, ramp_path(-0.8, 4.9))
    assert tr.best is None and not tr.armed


def test_brief_glitch_past_a_line_doesnt_count():
    tr = tracker()
    tr.start()
    now, _ = run(tr, [-0.8] * 6 + ramp_path(-0.8, 2.0))
    # One frame reads the wheel past 8 ft (too few to count).
    run(tr, [2.6, 2.0, 2.0, 2.0], t0=now)
    assert tr.best == 0


def test_rolling_back_keeps_the_line():
    tr = tracker()
    tr.start()
    run(tr, [-0.8] * 6 + ramp_path(-0.8, 3.0) + [3.0] * 5 + ramp_path(3.0, -0.5))
    assert tr.best == 1


def test_lost_wheel_mid_run():
    """The wheel hidden for a while (spotter in front): the score keeps what
    was seen and carries on when the wheel comes back."""
    tr = tracker()
    tr.start()
    now, _ = run(tr, [-0.8] * 6 + ramp_path(-0.8, 2.0))
    now, _ = run(tr, [None] * 15, t0=now)
    assert "Can't see" in tr.status_msg and tr.best == 0
    _, res = run(tr, ramp_path(2.4, 4.9) + [4.9] * 5, t0=now)
    assert res and res[0]["score"] == 2


def test_reset_keeps_lines():
    tr = tracker()
    tr.start()
    run(tr, [-0.8] * 6 + ramp_path(-0.8, 3.0))
    tr.reset()
    assert tr.lines.ready and tr.best is None and not tr.started
    assert tr.start() is None


def test_start_needs_lines():
    tr = wr.RampTracker()
    assert tr.start() is not None


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
