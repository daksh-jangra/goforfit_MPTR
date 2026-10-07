"""
Logic checks for the Target Stretch test - no camera, no participant.
============================================================================

WHY THIS FILE EXISTS
--------------------
  The score depends on things nobody can see on the screen: which way an
  angle is signed (a bent elbow and an over-straight one both look "not
  180"), the 15-degree rounding, the 1-second hold, the 10-second side
  window and the left -> right hand-over. Get one wrong and the feed still
  looks fine - the score is just wrong.

  Everything here drives the pure geometry and TargetStretchTracker
  directly with made-up points and angles and an injected clock.

HOW TO RUN IT
-------------
      python3 tests/test_target_stretch.py
      pytest tests/test_target_stretch.py

  Non-zero exit means something broke.
"""

import math
import os
import random
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import target_stretch as ts  # noqa: E402

J = ts.JOINTS


def near(a, b, tol=1.0):
    return a is not None and abs(a - b) <= tol


def pol(origin, length, deg_from_down_toward_right):
    """A point `length` from origin, at an angle measured from straight DOWN
    turning toward picture-RIGHT (image y points down)."""
    r = math.radians(deg_from_down_toward_right)
    return (origin[0] + length * math.sin(r), origin[1] + length * math.cos(r))


# ── rounding and scoring (table 5.3) ─────────────────────────────────────────────
def test_clock_rounding():
    assert ts.to_clock(0) == 0
    assert ts.to_clock(7.4) == 0
    assert ts.to_clock(7.5) == 15
    assert ts.to_clock(-7.4) == 0
    assert ts.to_clock(-7.6) == -15
    assert ts.to_clock(-22.4) == -15
    assert ts.to_clock(-22.6) == -30
    assert ts.to_clock(157.5) == 165
    assert ts.to_clock(157.4) == 150


def test_scores_match_table_5_3():
    cases = [
        ("wrist_extension", 60, 2), ("wrist_extension", 45, 1), ("wrist_extension", 30, 1),
        ("wrist_extension", 15, 0),
        ("elbow_extension", 0, 2), ("elbow_extension", 15, 2), ("elbow_extension", -15, 1),
        ("elbow_extension", -30, 0),
        ("shoulder_extension", 60, 2), ("shoulder_extension", 30, 1),
        ("shoulder_extension", 15, 0),
        ("shoulder_abduction", 180, 2), ("shoulder_abduction", 165, 2),
        ("shoulder_abduction", 150, 1), ("shoulder_abduction", 120, 1),
        ("shoulder_abduction", 105, 0),
        ("shoulder_external_rotation", 75, 2), ("shoulder_external_rotation", 60, 1),
        ("shoulder_external_rotation", 30, 1), ("shoulder_external_rotation", 15, 0),
        ("forearm_supination", 90, 2), ("forearm_supination", 75, 1),
        ("forearm_supination", 45, 1), ("forearm_supination", 30, 0),
        ("forearm_pronation", 90, 2), ("forearm_pronation", 30, 0),
        ("knee_extension", 0, 2), ("knee_extension", -15, 1), ("knee_extension", -30, 0),
    ]
    for key, deg, want in cases:
        got = ts.score_for(J[key], deg)
        assert got == want, (key, deg, got, want)


def test_eight_joints():
    assert len(J) == 8


# ── geometry signs ───────────────────────────────────────────────────────────────
SH = (500.0, 300.0)
HIP = (500.0, 600.0)


def test_abduction():
    for deg in (0, 45, 90, 120, 165, 180):
        el = pol(SH, 150, deg)
        assert near(ts.abduction_deg(SH, el, HIP), deg), deg


def test_shoulder_extension_sign():
    # Facing picture-right (+1): BEHIND is picture-left.
    for deg in (0, 30, 60):
        el = pol(SH, 150, -deg)
        assert near(ts.shoulder_extension_deg(SH, el, HIP, +1), deg), deg
        # Mirror image: facing left, behind is picture-right.
        el = pol(SH, 150, deg)
        assert near(ts.shoulder_extension_deg(SH, el, HIP, -1), deg), deg
    # Arm swung FORWARD is not extension.
    assert ts.shoulder_extension_deg(SH, pol(SH, 150, 40), HIP, +1) < 0


def test_elbow_bent_vs_over_straight():
    el = (500.0, 450.0)
    # Facing right: the elbow bends forward = forearm toward picture-right.
    for bend in (0, 15, 30, 90):
        wr = pol(el, 130, bend)
        assert near(ts.elbow_extension_deg(SH, el, wr, +1), -bend), bend
        wr = pol(el, 130, -bend)
        assert near(ts.elbow_extension_deg(SH, el, wr, -1), -bend), bend
    # 10 degrees over-straight reads +10 (scores 2), not -10.
    wr = pol(el, 130, -10)
    assert near(ts.elbow_extension_deg(SH, el, wr, +1), 10)


def test_knee_seated():
    hip, knee = (400.0, 500.0), (600.0, 500.0)     # thigh level, facing right
    for short in (0, 15, 45, 90):
        # straight = shank continues forward; bending drops the foot down/back
        r = math.radians(short)
        ankle = (knee[0] + 180 * math.cos(r), knee[1] + 180 * math.sin(r))
        assert near(ts.knee_extension_deg(hip, knee, ankle, +1), -short), short
    # Facing left: mirror everything.
    hip, knee = (600.0, 500.0), (400.0, 500.0)
    ankle = (knee[0] - 180 * math.cos(math.radians(30)), knee[1] + 180 * math.sin(math.radians(30)))
    assert near(ts.knee_extension_deg(hip, knee, ankle, -1), -30)


def test_wrist_extension_up_is_positive():
    el, wr = (300.0, 400.0), (450.0, 400.0)          # forearm forward, facing right
    for up in (0, 30, 60, 75):
        r = math.radians(up)
        knuckle = (wr[0] + 60 * math.cos(r), wr[1] - 60 * math.sin(r))
        assert near(ts.wrist_extension_deg(el, wr, wr, knuckle, +1), up), up
    # Hand bent DOWN (flexion) is negative.
    knuckle = (wr[0] + 60 * math.cos(math.radians(40)), wr[1] + 60 * math.sin(math.radians(40)))
    assert ts.wrist_extension_deg(el, wr, wr, knuckle, +1) < 0


def test_rotation_from_length():
    for deg in (0, 30, 45, 60, 75, 90):
        L = math.cos(math.radians(deg))
        assert near(ts.rotation_from_length(L, 1.0), deg), deg
    assert ts.rotation_from_length(1.2, 1.0) == 0.0       # longer than start = 0
    assert ts.rotation_from_length(0.5, None) is None


def test_forearm_rotation():
    pinky = (500.0, 520.0)
    # Left arm, un-mirrored front view: it sits on picture-right, so OUTWARD
    # (supination) turns the index knuckle toward picture-right.
    for deg in (0, 45, 90):
        r = math.radians(deg)
        index = (pinky[0] + 40 * math.sin(r), pinky[1] - 40 * math.cos(r))
        assert near(ts.forearm_rotation_deg(pinky, index, +1), deg), deg
        assert near(ts.forearm_rotation_deg(pinky, index, -1), -deg), deg


def test_side_rules():
    # Un-mirrored: LEFT side to the lens faces picture-left.
    assert ts.expected_facing("left", False) == -1
    assert ts.expected_facing("right", False) == 1
    assert ts.expected_facing("left", True) == 1
    # Facing the lens, un-mirrored: their LEFT arm is on picture-right.
    assert ts.expected_image_side("left", False) == 1
    assert ts.expected_image_side("left", True) == -1


def test_facing_from_head():
    sm = (500.0, 400.0)
    # Side-on facing right: nose ahead of the ears.
    assert ts.facing_from_head((560, 300), (515, 300), (512, 302), sm) == 1
    assert ts.facing_from_head((440, 300), (485, 300), (488, 302), sm) == -1
    # Facing the lens: nose between the ears.
    assert ts.facing_from_head((500, 300), (460, 300), (540, 300), sm) is None


def test_front_on():
    assert ts.front_on((500, 300), (420, 400), (580, 400))
    assert not ts.front_on((500, 300), (495, 400), (520, 400))


# ── the hold window ──────────────────────────────────────────────────────────────
def feed(tracker, steps, fps=10, t0=0.0, jitter=0.0, seed=1):
    """steps: (seconds, angle_or_None). Returns (end time, results seen)."""
    rnd = random.Random(seed)
    t, out = t0, []
    for dur, val in steps:
        n = int(round(dur * fps))
        for _ in range(n):
            v = None if val is None else val + rnd.gauss(0, jitter)
            r = tracker.update(v, now=t)
            if r is not None:
                out.append(r)
            t += 1.0 / fps
    return t, out


def test_hold_needs_a_full_second():
    tr = ts.TargetStretchTracker("shoulder_abduction")
    # Rest, then a 0.5 s flick to 170, then back down: never held.
    feed(tr, [(2.0, 10), (0.5, 170), (2.0, 10)])
    assert tr.best is not None and tr.best < 30, tr.best
    # Now a real 1.5 s hold at 170.
    feed(tr, [(1.5, 170)], t0=4.5)
    assert tr.best is not None and tr.best >= 165, tr.best


def test_hold_survives_jitter():
    tr = ts.TargetStretchTracker("shoulder_abduction")
    feed(tr, [(1.0, 0), (2.0, 125)], jitter=3.0, seed=4)
    r = ts.side_result(J["shoulder_abduction"], tr.best)
    assert r.deg == 120 and r.score == 1, (tr.best, r)


def test_gap_restarts_hold():
    tr = ts.TargetStretchTracker("shoulder_abduction")
    # 0.6 s at 170, a 0.8 s blackout, 0.6 s more: no single second seen.
    feed(tr, [(0.6, 170), (0.8, None), (0.6, 170)])
    assert tr.best is None, tr.best


def test_low_fps_still_holds():
    tr = ts.TargetStretchTracker("knee_extension")
    feed(tr, [(3.0, -3)], fps=5)
    assert tr.best is not None
    assert ts.side_result(J["knee_extension"], tr.best).score == 2


# ── a whole run ──────────────────────────────────────────────────────────────────
def test_left_then_right():
    tr = ts.TargetStretchTracker("shoulder_abduction")
    assert tr.side == "left"
    t, out = feed(tr, [(3.0, 0), (3.0, 170), (5.0, 20)])   # 11 s > 10 s window
    assert tr.side == "right" and not out
    assert tr.results["left"].deg == 165 and tr.results["left"].score == 2
    t, out = feed(tr, [(2.0, None), (2.0, 0), (3.0, 125), (6.0, 0)], t0=t)
    assert tr.resolved and len(out) == 1
    res = out[0]
    assert res["left_score"] == 2 and res["right_score"] == 1, res
    assert res["right_deg"] == 120 and res["joint"] == "shoulder_abduction"


def test_window_starts_at_first_reading():
    tr = ts.TargetStretchTracker("elbow_extension")
    # 20 s of nobody in view must not use up the left side's 10 s.
    t, _ = feed(tr, [(20.0, None)])
    assert tr.side == "left" and tr.time_left() is None
    feed(tr, [(2.0, -20)], t0=t)
    assert tr.side == "left" and 7.5 < tr.time_left() <= 8.1


def test_stray_frames_dont_start_the_clock():
    tr = ts.TargetStretchTracker("elbow_extension")
    # Single frames that pass the view check, among frames that don't.
    t = 0.0
    for k in range(100):
        tr.update(-20 if k % 7 == 0 else None, now=t)
        t += 0.1
    assert tr.time_left() is None and tr.phase == ts.Phase.POSITION


def test_coach_next_and_unseen_side():
    tr = ts.TargetStretchTracker("knee_extension")
    t, _ = feed(tr, [(2.0, -5)])
    tr.next_side(now=t)
    assert tr.side == "right" and tr.results["left"].score == 2
    # Right side never seen: unmeasured after the 30 s backstop.
    t, out = feed(tr, [(31.0, None)], t0=t)
    assert tr.resolved and out and out[0]["right_score"] is None
    assert out[0]["left_score"] == 2


def test_stop_mid_run():
    tr = ts.TargetStretchTracker("forearm_supination")
    t, _ = feed(tr, [(2.0, 50)])
    res = tr.finish(now=t)
    assert tr.resolved
    assert res["left_deg"] == 45 and res["left_score"] == 1
    assert res["right_score"] is None
    # Nothing measured at all -> nothing to stage.
    tr2 = ts.TargetStretchTracker("forearm_supination")
    feed(tr2, [(2.0, None)])
    assert tr2.finish(now=2.0) is None


def test_reset():
    tr = ts.TargetStretchTracker("wrist_extension")
    t, _ = feed(tr, [(11.0, 65)])
    assert tr.side == "right"
    tr.reset()
    assert tr.side == "left" and tr.results["left"] is None and tr.best is None


def test_rotation_start_length_is_a_steady_second():
    """The start length comes from a steady second, not one long frame."""
    sl = ts.StartLength()
    assert sl.push(0.0, 0.85) is None                 # no second seen yet
    t = 0.0
    for k in range(12):                               # 1.1 s at 0.85, one spike
        sl.push(t, 1.30 if k == 5 else 0.85)
        t += 0.1
    assert sl.value is not None and abs(sl.value - 0.85) < 1e-9, sl.value
    a = ts.rotation_from_length(0.85 * math.cos(math.radians(75)), sl.value)
    assert ts.to_clock(a) == 75


def _rotation_run(true_deg, noise, seed, fps=10):
    """A whole side through StartLength + the tracker: 2 s at the start
    position, 3 s turned out by true_deg, 2 s back. Length wobble `noise`."""
    rnd = random.Random(seed)
    tr = ts.TargetStretchTracker("shoulder_external_rotation")
    sl = ts.StartLength()
    t = 0.0
    for d in [0] * (2 * fps) + [true_deg] * (3 * fps) + [0] * (2 * fps):
        length = 0.85 * math.cos(math.radians(d)) * (1 + rnd.gauss(0, noise))
        start = sl.push(t, length)
        tr.update(None if start is None else ts.rotation_from_length(length, start), now=t)
        t += 1.0 / fps
    tr.next_side(now=t)
    r = tr.results["left"]
    return None if r is None else r.score


def test_rotation_still_student_scores_zero_under_wobble():
    # 4 % wobble in the forearm's on-screen length, the realistic case.
    scores = [_rotation_run(0, 0.04, s) for s in range(100)]
    assert scores.count(0) >= 99, scores
    scores = [_rotation_run(15, 0.04, s) for s in range(100)]
    assert scores.count(0) >= 95, scores


def test_rotation_real_turns_still_score():
    for true, want in ((35, 1), (60, 1), (80, 2)):
        scores = [_rotation_run(true, 0.04, s) for s in range(50)]
        assert scores.count(want) == 50, (true, scores)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
