"""
Logic checks for the Isometric Push-Up tracker - no camera, no participant.
============================================================================

WHY THIS FILE EXISTS
--------------------
  The score is a clock, and the clock's start and stop depend on timing the
  operator cannot see: a settle window, a break grace, dropouts, the 40 s cap.
  Get one of those wrong and nothing looks broken - the feed streams, the HUD
  ticks - the number just comes out wrong.

  Everything here drives IsometricPushUpTracker directly with made-up body
  readings and an injected clock, so it runs in about a second.

HOW TO RUN IT
-------------
      python3 tests/test_isometric_pushup.py
      pytest tests/test_isometric_pushup.py

  Non-zero exit means something broke. Run it after ANY edit to
  isometric_pushup.py - especially to the thresholds near the top.
"""

import os
import random
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import isometric_pushup as ip  # noqa: E402

L, R = ip.SIDE_LEFT, ip.SIDE_RIGHT

# The manual's photo (fig 5.14) read through the pose model.
GOOD = {"elbow": 168.0, "bend": -2.0, "knee": 171.0, "sway": 0.1,
        "body_tilt": 11.0, "arm_tilt": 5.0, "vis": 0.97}


def pose(**changes):
    m = dict(GOOD)
    m.update(changes)
    return m


def near(m):
    """Only the near (left) side is tracked, as side-on in real footage."""
    return {L: m, R: None}


def feed(tracker, sequence, fps=20, t0=0.0, jitter=0.0, seed=1):
    """Drive the tracker through (seconds, sides_or_None) steps. Returns
    (end_time, first_result)."""
    rnd = random.Random(seed)
    t, dt, first = t0, 1.0 / fps, None
    for duration, sides in sequence:
        for _ in range(max(1, int(round(duration / dt)))):
            s = sides
            if s is not None and jitter:
                s = {k: (None if v is None else
                         {**v, "elbow": v["elbow"] + rnd.uniform(-jitter, jitter),
                          "bend": v["bend"] + rnd.uniform(-jitter, jitter),
                          "knee": v["knee"] + rnd.uniform(-jitter, jitter),
                          "sway": v["sway"] + rnd.uniform(-jitter, jitter) / 100})
                     for k, v in s.items()}
            out = tracker.update(s if s is not None else {L: None, R: None}, now=t)
            if out and first is None:
                first = out
            t += dt
    return t, first


def check_full_hold_caps_at_40():
    for fps in (10, 20, 30):
        tr = ip.IsometricPushUpTracker()
        _, res = feed(tr, [(60, near(GOOD))], fps=fps, jitter=3)
        assert res and res["hold_sec"] == 40 and res["capped"], (fps, res)


def check_settle_second_is_credited():
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(2, near(pose(elbow=120))),       # getting up: arms bent
              (12.0, near(GOOD)),               # 12 s in position
              (1, near(pose(elbow=110)))])      # drops
    assert tr.resolved and tr.score_s == 12, tr.result()
    assert tr.end_reason == "elbows bent", tr.end_reason


def check_each_break_rule():
    cases = {
        "elbows bent":          pose(elbow=140),
        "hips sagged or piked": pose(bend=-20),
        "knees bent":           pose(knee=145),
        "body swayed":          pose(sway=0.45),
    }
    for words, bad in cases.items():
        tr = ip.IsometricPushUpTracker()
        feed(tr, [(8, near(GOOD)), (1, near(bad))])
        assert tr.resolved and tr.end_reason == words, (words, tr.result())
        assert tr.score_s == 8, (words, tr.result())
    # Piking up counts too, not only sagging.
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(8, near(GOOD)), (1, near(pose(bend=16)))])
    assert tr.end_reason == "hips sagged or piked", tr.result()


def check_one_shaky_frame_does_not_end_it():
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(5, near(GOOD)), (0.2, near(pose(elbow=130))), (10, near(GOOD))])
    assert tr.state == ip.State.HOLDING, tr.result()
    assert 14.5 < tr.elapsed_s < 15.5, tr.elapsed_s


def check_small_wobble_inside_tolerance():
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(3, near(GOOD)),
              (3, near(pose(elbow=155, bend=-12, knee=160, sway=0.3))),
              (3, near(GOOD))])
    assert tr.state == ip.State.HOLDING, tr.result()


def check_own_baseline_used():
    """A child whose arms lock at 152 is not failed for it, but still stops
    when those arms bend a further 20."""
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(10, near(pose(elbow=152)))])
    assert tr.state == ip.State.HOLDING
    feed(tr, [(1, near(pose(elbow=128)))], t0=10.0)
    assert tr.resolved and tr.end_reason == "elbows bent"


def check_wrong_positions_never_start():
    wrong = {
        "standing":     pose(body_tilt=80, arm_tilt=5),
        "bottom hold":  pose(elbow=95, arm_tilt=35),
        "knees down":   pose(knee=110),
        "piked":        pose(bend=28),
        "sagging":      pose(bend=-25),
        "hands ahead":  pose(arm_tilt=50),
    }
    for name, m in wrong.items():
        tr = ip.IsometricPushUpTracker()
        feed(tr, [(10, near(m))])
        assert tr.state == ip.State.READY and tr.hold_start is None, name
        assert tr.finish() is None, name        # not completed, not a zero


def check_restless_setup_does_not_start():
    """Moving about in roughly the right shape never settles."""
    tr = ip.IsometricPushUpTracker()
    seq = []
    for i in range(20):
        seq.append((0.3, near(pose(elbow=155 if i % 2 else 175))))
    feed(tr, seq)
    assert tr.hold_start is None, tr.result()


def check_short_dropout_survives_long_one_ends():
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(5, near(GOOD)), (0.4, None), (5, near(GOOD))])
    assert tr.state == ip.State.HOLDING, tr.result()

    tr = ip.IsometricPushUpTracker()
    feed(tr, [(6, near(GOOD)), (3, None), (5, near(GOOD))])
    assert tr.resolved and tr.score_s == 6, tr.result()
    assert "lost sight" in tr.end_reason


def check_frames_stop_arriving():
    tr = ip.IsometricPushUpTracker()
    t, _ = feed(tr, [(7, near(GOOD))])
    tr.update(near(GOOD), now=t + 5.0)          # 5 s gap in the stream
    assert tr.resolved and tr.score_s == 7, tr.result()


def check_stop_mid_hold_and_before():
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(17.3, near(GOOD))])
    res = tr.finish()
    assert res and res["hold_sec"] == 17 and tr.end_reason == "stopped by the coach", res

    # Stopped during a break's grace: time ends where the break began.
    tr = ip.IsometricPushUpTracker()
    feed(tr, [(9, near(GOOD)), (0.2, near(pose(elbow=120)))])
    assert tr.finish()["hold_sec"] == 9


def check_done_is_final_until_reset():
    tr = ip.IsometricPushUpTracker()
    t, _ = feed(tr, [(5, near(GOOD)), (1, near(pose(knee=120)))])
    score = tr.score_s
    feed(tr, [(20, near(GOOD))], t0=t)
    assert tr.score_s == score and tr.resolved
    tr.reset()
    assert tr.state == ip.State.READY and tr.score_s == 0


def check_side_locks_while_holding():
    """Far arm reads ~20 deg more bent side-on. Once the hold starts on the
    near side, the far side turning clearer must not be read as a bend."""
    tr = ip.IsometricPushUpTracker()
    far = pose(elbow=148, vis=0.6)
    feed(tr, [(5, {L: GOOD, R: far})])
    assert tr.side == L and tr.state == ip.State.HOLDING
    feed(tr, [(5, {L: GOOD, R: pose(elbow=148, vis=0.99)})], t0=5.0)
    assert tr.state == ip.State.HOLDING, tr.result()


def check_rounding_to_nearest_second():
    for held, want in ((10.4, 10), (10.6, 11)):
        tr = ip.IsometricPushUpTracker()
        feed(tr, [(held, near(GOOD)), (1, near(pose(elbow=100)))], fps=50)
        assert tr.score_s == want, (held, tr.result())


CHECKS = [v for k, v in sorted(globals().items()) if k.startswith("check_")]


def test_all():
    for c in CHECKS:
        c()


if __name__ == "__main__":
    failed = 0
    for c in CHECKS:
        try:
            c()
            print(f"  ok    {c.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {c.__name__}: {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
