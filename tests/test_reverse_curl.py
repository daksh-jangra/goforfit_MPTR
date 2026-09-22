"""
Logic checks for the Reverse Curl tracker — no camera, no participant.
=======================================================================

WHY THIS FILE EXISTS
--------------------
  The Reverse Curl verdict depends on timing the operator cannot see: a
  2-second hold, a rest baseline, and what happens when tracking drops out
  mid-hold. Break one of those and nothing looks wrong — the feed still
  streams, the HUD still updates, the test just stops passing people it
  should pass. These checks catch that.

  Everything here drives ReverseCurlTracker / ArmSelector directly with made-up
  angle sequences and an injected clock, so it runs in about a second and needs
  no webcam, no model file and no volunteer.

HOW TO RUN IT
-------------
      python3 tests/test_reverse_curl.py     # prints one line per check
      pytest tests/test_reverse_curl.py      # same checks, if you use pytest

  Non-zero exit means something broke. Run it after ANY edit to
  reverse_curl.py — especially to the thresholds near the top of that file.

WHAT IT DOES NOT COVER
----------------------
  The camera path: landmark extraction, the elbow-angle geometry and the HUD.
  Those need real footage to test meaningfully. What is covered is the part
  that fails silently.
"""

import os
import sys
import random

# Import reverse_curl.py from this directory regardless of where this was run.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import reverse_curl as rc  # noqa: E402

L, R = rc.ARM_LEFT, rc.ARM_RIGHT

# A relaxed seated arm reads ~95 deg at the thigh; the manual's up position
# measures ~22 deg, so 40 is a comfortable, realistic "good curl" target.
REST, UP = 95.0, 40.0


def feed(tracker, sequence, fps, t0=0.0):
    """Drive the tracker through (duration_seconds, angle_or_None) steps at a
    fixed frame rate, using an injected clock so the checks are deterministic.
    Returns (end_time, [(t, result), ...])."""
    t = t0
    dt = 1.0 / fps
    emitted = []
    for duration, angle in sequence:
        for _ in range(max(1, int(round(duration / dt)))):
            out = tracker.update(angle, now=t)
            if out:
                emitted.append((t, out))
            t += dt
    return t, emitted


def good_curl(hold=2.4, up=UP):
    """Rest -> lift -> hold -> lower -> back to rest: one correct curl."""
    return [(1.0, REST), (0.3, 70.0), (0.2, 55.0), (hold, up),
            (0.3, 60.0), (0.3, 80.0), (0.6, REST)]


# ── the curl itself ──────────────────────────────────────────────────────────
def test_correct_curl_passes_at_every_frame_rate():
    # A weak phone may only manage ~10 fps; the verdict must not depend on it.
    for fps in (10, 15, 20, 30):
        tr = rc.ReverseCurlTracker()
        _, emitted = feed(tr, good_curl(), fps)
        assert tr.state is rc.State.PASSED, f"{fps} fps: ended in {tr.state}"
        assert tr.passed, f"{fps} fps: not marked passed"
        assert len(emitted) == 1, f"{fps} fps: emitted {len(emitted)} results, want 1"
        assert abs(tr.best_peak_deg - UP) < 1.0, f"{fps} fps: peak {tr.best_peak_deg}"


def test_baseline_is_the_rest_angle_not_a_mid_lift_frame():
    # The classic bug: sampling the baseline as a threshold is crossed, by
    # which point the arm has already travelled most of the way.
    tr = rc.ReverseCurlTracker()
    feed(tr, good_curl(), 20)
    assert tr.baseline_deg is not None, "no baseline captured"
    assert abs(tr.baseline_deg - REST) < 2.0, f"baseline {tr.baseline_deg}, want ~{REST}"


def test_no_baseline_while_the_elbow_is_already_bent():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(2.0, 50.0)], 20)
    assert tr.state is rc.State.READY, f"state {tr.state}"
    assert tr.baseline_deg is None, f"baseline {tr.baseline_deg} taken from a bent arm"


def test_real_attempt_short_of_the_target_fails_by_itself():
    # 95 -> 58 deg is a clear lift (37 deg of bend) that never reaches 45.
    tr = rc.ReverseCurlTracker()
    _, emitted = feed(tr, [(1.0, REST), (0.3, 75.0), (2.5, 58.0), (0.5, REST)], 20)
    assert tr.state is rc.State.FAILED, f"state {tr.state}, want FAILED"
    assert not tr.passed, "58 deg counted as reaching 45"
    assert tr.verdict_by == "auto", tr.verdict_by
    assert len(emitted) == 1 and emitted[0][1]["passed"] is False, emitted


def test_hold_shorter_than_two_seconds_fails_by_itself():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (1.2, UP), (0.4, 70.0), (0.6, REST)], 20)
    assert tr.state is rc.State.FAILED, f"a 1.2 s hold ended in {tr.state}"
    assert 1.0 < tr.hold_best_s < 1.45, f"hold measured {tr.hold_best_s}"


def test_small_movement_is_ignored_not_failed():
    # 95 -> 70 deg is only 25 deg of bend: shifting in the chair, not a try.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (3.0, 70.0), (0.6, REST)], 20)
    assert tr.state is rc.State.SET, f"a fidget ended the trial: {tr.state}"
    assert not tr.passed and not tr.attempts, "a fidget was recorded as an attempt"


def test_fidget_then_a_correct_curl_still_passes():
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.5, 72.0), (0.6, REST)], 20)
    feed(tr, good_curl()[1:], 20, t0=t)
    assert tr.state is rc.State.PASSED, f"the fidget spoiled the trial: {tr.state}"


def test_reaching_the_target_is_always_a_real_try():
    # Someone resting at 72 deg has only 27 deg to give before 45, less than
    # ATTEMPT_MIN_DEG — but reaching 44 is plainly an attempt, and a short
    # hold there must FAIL rather than be waved off as a fidget.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, 72.0), (0.3, 55.0), (1.0, 44.0), (0.3, 60.0), (0.6, 72.0)], 20)
    assert tr.state is rc.State.FAILED, f"state {tr.state}, want FAILED"


def test_no_pass_without_returning_the_weight_to_the_thigh():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (3.0, UP)], 20)
    assert tr.state is rc.State.LOWERING, f"state {tr.state}"
    assert not tr.passed, "passed while the weight was still up"


# ── tracking dropouts ────────────────────────────────────────────────────────
def test_long_dropout_voids_the_hold():
    # Time nobody could see must not be credited toward the 2 seconds.
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 70.0), (1.0, UP)], 20)
    assert tr.state is rc.State.HOLD, f"not holding before the dropout: {tr.state}"
    t, _ = feed(tr, [(1.0, None)], 20, t0=t)
    assert tr.state is rc.State.LIFTING, f"hold survived a 1 s dropout: {tr.state}"
    assert tr.hold_elapsed == 0.0, f"hold_elapsed {tr.hold_elapsed} not cleared"
    # ...and the participant can still complete the trial afterwards.
    feed(tr, [(2.4, UP), (0.3, 70.0), (0.6, REST)], 20, t0=t)
    assert tr.passed, "could not recover after a dropout"


def test_single_dropped_frame_does_not_void_the_hold():
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 70.0), (0.8, UP)], 20)
    t, _ = feed(tr, [(0.05, None)], 20, t0=t)      # one frame at 20 fps
    assert tr.state is rc.State.HOLD, f"one dropped frame broke the hold: {tr.state}"
    feed(tr, [(1.6, UP), (0.3, 70.0), (0.6, REST)], 20, t0=t)
    assert tr.passed, "hold restarted unnecessarily after one dropped frame"


def test_all_none_stream_is_survivable():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(3.0, None)], 20)
    assert tr.state is rc.State.READY, f"state {tr.state}"


# ── noise ────────────────────────────────────────────────────────────────────
def test_jitter_at_rest_never_starts_an_attempt():
    tr = rc.ReverseCurlTracker()
    rnd = random.Random(7)
    t = 0.0
    for _ in range(200):
        tr.update(REST + rnd.uniform(-3, 3), now=t)
        t += 0.05
    assert not tr.attempts, f"landmark jitter created {len(tr.attempts)} attempt(s)"
    assert not tr.passed, "jitter produced a pass"
    assert tr.baseline_deg is not None, "baseline never settled through jitter"


def test_one_frame_above_the_target_keeps_the_hold():
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 70.0), (0.9, UP)], 20)
    t, _ = feed(tr, [(0.05, UP + 8)], 20, t0=t)    # 48 deg, inside the release band
    assert tr.state is rc.State.HOLD, f"a 48 deg blip reset the hold: {tr.state}"


# ── one trial, decided by the camera ─────────────────────────────────────────
def test_verdict_is_final_one_trial_only():
    # A FAIL is not undone by a good curl afterwards...
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 75.0), (2.5, 58.0), (0.5, REST)], 20)
    _, emitted = feed(tr, good_curl()[1:], 20, t0=t)
    assert tr.state is rc.State.FAILED and not emitted, "a second try changed a FAIL"
    # ...and a PASS is not undone by anything that follows.
    tr2 = rc.ReverseCurlTracker()
    t, _ = feed(tr2, good_curl(), 20)
    _, emitted = feed(tr2, [(0.3, 75.0), (2.5, 58.0), (0.5, REST)], 20, t0=t)
    assert tr2.state is rc.State.PASSED and not emitted, "later movement changed a PASS"


def test_no_manual_verdicts():
    # Owner decision: when the camera decides pass/fail, nobody overrides it.
    tr = rc.ReverseCurlTracker()
    assert not hasattr(tr, "mark_pass") and not hasattr(tr, "mark_fail"), \
        "manual PASS/FAIL came back"


# ── reset ────────────────────────────────────────────────────────────────────


def test_reset_clears_everything():
    tr = rc.ReverseCurlTracker()
    feed(tr, good_curl(), 20)
    tr.reset()
    assert tr.state is rc.State.READY and not tr.passed and not tr.attempts


# ── which arm is doing the test ──────────────────────────────────────────────
def test_arm_choice_is_stable_then_locks_onto_the_curling_arm():
    sel = rc.ArmSelector()
    for _ in range(20):
        sel.update({L: 80.0, R: 100.0})   # both resting, at different angles
    first = sel.arm
    for _ in range(20):
        sel.update({L: 80.0, R: 100.0})
    assert sel.arm == first, "choice flickered while both arms rested"
    assert not sel.locked, "locked without a lift"

    for angle in (95, 85, 70, 60):        # the right arm curls
        sel.update({L: 80.0, R: float(angle)})
    assert sel.arm == R, f"did not follow the curling arm: {sel.arm}"
    assert sel.locked, "did not lock once the lift was real"

    for _ in range(30):                   # the other hand now moves a lot
        sel.update({L: 40.0, R: 60.0})
    assert sel.arm == R, "the idle hand stole the test"


def test_arm_can_be_forced_and_survives_missing_landmarks():
    sel = rc.ArmSelector(forced=L)
    for _ in range(10):
        sel.update({L: 95.0, R: 40.0})
    assert sel.arm == L and sel.locked, "forced arm was overridden"

    sel2 = rc.ArmSelector()
    assert sel2.update({L: None, R: None}) is None
    sel2.update({L: None, R: 95.0})
    assert sel2.arm == R, "did not fall back to the visible arm"

    sel3 = rc.ArmSelector()
    sel3.update({L: 95.0, R: 95.0})
    sel3.reset()
    assert sel3.arm is None and not sel3.locked, "reset left the lock in place"


# ── standalone runner (so this works without pytest) ─────────────────────────
def _main():
    checks = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in checks:
        try:
            fn()
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n          {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed.append(name)
            print(f"  ERROR {name}\n          {exc.__class__.__name__}: {exc}")
        else:
            print(f"  PASS  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed"
          + (f" — FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
