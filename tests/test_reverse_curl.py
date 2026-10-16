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


def test_blackout_while_lowering_does_not_pass_on_the_first_frame_back():
    # The "weight is down" window proves the elbow sat still for a second. If
    # tracking dies partway through that second, the window must not be
    # measured from a reading taken before the blackout: for four seconds
    # nobody could see where the weight was. It used to PASS on the first
    # frame back.
    tr = rc.ReverseCurlTracker()
    down = [(1.0, REST), (0.3, 70.0), (2.4, UP), (0.3, 60.0), (0.2, 78.0)]
    feed(tr, down + [(4.0, None), (0.2, 79.0)], 20)
    assert tr.state is rc.State.LOWERING, f"passed across a blackout: {tr.state}"
    # ...and it is only delayed, not broken: a fresh still second still passes.
    feed(tr, [(1.2, 79.0)], 20, t0=100.0)
    assert tr.state is rc.State.PASSED, f"never passed after settling: {tr.state}"


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


# ── STOP before the camera has decided (one trial, per the manual) ───────────
def test_stop_after_a_real_attempt_is_a_fail():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 75.0), (2.0, 60.0)], 20)  # lifted 35 deg, still going
    res = tr.finish()
    assert tr.state is rc.State.FAILED and res and res["passed"] is False, tr.state


def test_stop_mid_hold_is_a_fail():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (1.0, UP)], 20)    # holding, 1 s of 2
    tr.finish()
    assert tr.state is rc.State.FAILED, tr.state


def test_stop_with_no_real_attempt_is_not_completed():
    for seq in ([(1.0, REST)],                              # resting, never lifted
                [(1.0, REST), (0.5, 80.0)],                 # a 15 deg twitch in progress
                [(2.0, 170.0)],                             # never got to rest (film side-on)
                [(2.0, None)]):                             # arm never seen
        tr = rc.ReverseCurlTracker()
        feed(tr, seq, 20)
        assert tr.finish() is None and not tr.resolved, (seq, tr.state)


def test_stop_just_after_setting_the_weight_down_is_a_pass():
    # Hold done; the weight is back on the thigh at 78 but hasn't been still
    # for the full second the camera waits for. The curl happened: PASS.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (2.4, UP), (0.3, 60.0), (0.3, 78.0)], 20)
    assert tr.state is rc.State.LOWERING, tr.state
    res = tr.finish()
    assert tr.state is rc.State.PASSED and res["passed"], tr.state


def test_stop_half_way_down_is_not_a_pass_at_a_high_rest_angle():
    # "Back down" is measured against THIS participant's own rest angle. A
    # student who rests at 125 deg (taller, or a different thigh angle) and is
    # stopped at 85 still has the weight halfway up — but 85 is inside the
    # resting RANGE, so judging by the range alone credited a PASS, while the
    # same arm a few degrees lower FAILED.
    high = [(1.5, 125.0), (0.3, 100.0), (0.2, 60.0), (2.4, UP)]
    tr = rc.ReverseCurlTracker()
    feed(tr, high + [(0.3, 85.0)], 20)
    assert tr.state is rc.State.LOWERING, tr.state
    res = tr.finish()
    assert tr.state is rc.State.FAILED and res["passed"] is False, tr.state
    # Back near that arm's own rest, the same stop is still a PASS.
    tr2 = rc.ReverseCurlTracker()
    feed(tr2, high + [(0.3, 100.0)], 20)
    assert tr2.state is rc.State.LOWERING, tr2.state
    assert tr2.finish()["passed"], tr2.state


def test_stop_with_the_weight_still_up_after_the_hold_is_a_fail():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (2.4, UP), (1.0, 55.0)], 20)
    tr.finish()
    assert tr.state is rc.State.FAILED, tr.state


def test_stop_after_a_verdict_changes_nothing():
    tr = rc.ReverseCurlTracker()
    feed(tr, good_curl(), 20)
    assert tr.finish()["passed"] and tr.state is rc.State.PASSED
    tr2 = rc.ReverseCurlTracker()
    feed(tr2, [(1.0, REST), (0.3, 75.0), (2.5, 58.0), (0.5, REST)], 20)
    assert tr2.finish()["passed"] is False and tr2.state is rc.State.FAILED


# ── reset ────────────────────────────────────────────────────────────────────


def test_reset_clears_everything():
    tr = rc.ReverseCurlTracker()
    feed(tr, good_curl(), 20)
    tr.reset()
    assert tr.state is rc.State.READY and not tr.passed and not tr.attempts


# ── which arm is doing the test ──────────────────────────────────────────────
def test_arm_choice_is_stable_then_locks_onto_the_curling_arm():
    sel, clock = rc.ArmSelector(), [0.0]
    def upd(left, right, n=1):            # 20 fps
        for _ in range(n):
            sel.update({L: left, R: right}, now=clock[0]); clock[0] += 0.05
    upd(80.0, 100.0, 20)                  # both resting, at different angles
    first = sel.arm
    upd(80.0, 100.0, 20)
    assert sel.arm == first, "choice flickered while both arms rested"
    assert not sel.locked, "locked without a lift"

    for angle in (95, 85, 70, 60):        # the right arm curls
        upd(80.0, float(angle))
    assert sel.arm == R, f"did not follow the curling arm: {sel.arm}"
    assert sel.locked, "did not lock once the lift was real"

    upd(40.0, 60.0, 30)                   # the other hand now moves a lot
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


# ── the resting position ─────────────────────────────────────────────────────
def test_hanging_arm_is_never_taken_as_rest():
    # Arm hanging at the side, weight onto the thigh, arm hanging again, and
    # no curl at all. Taking the hanging arm (170) as "rest" made the hand
    # moving onto the thigh look like a lift — and this sequence a FAIL.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, 170.0), (2.0, REST), (1.0, 170.0)], 20)
    assert tr.state is rc.State.SET, f"no curl, yet state {tr.state}"
    assert abs(tr.baseline_deg - REST) < 2, f"rest {tr.baseline_deg}, want the thigh ~{REST}"


def test_both_ends_of_the_resting_range_can_be_moved():
    # Both ends of the accepted start position are knobs, not just the lower
    # one. A student who sits with the forearm flat along the thigh can read
    # much more open than the default 130; widening the range must let that
    # start count, and the default must still refuse it.
    strange_rest = 150.0
    tight = rc.ReverseCurlTracker()
    feed(tight, [(2.0, strange_rest)], 20)
    assert tight.baseline_deg is None, "the default range accepted a straight arm"

    wide = rc.ReverseCurlTracker(rest_max=160.0)
    t, _ = feed(wide, [(2.0, strange_rest)], 20)
    assert wide.baseline_deg is not None, "rest_max=160 still refused a 150 deg start"
    feed(wide, [(0.3, 120.0), (0.2, 80.0), (2.4, UP),
                (0.3, 90.0), (0.6, strange_rest)], 20, t0=t)
    assert wide.state is rc.State.PASSED, f"state {wide.state}"

    # The arm-switch handover reads the same range...
    handover = rc.ReverseCurlTracker(rest_max=160.0)
    handover.adopt_baseline(strange_rest)
    assert handover.state is rc.State.SET, f"handover refused a {strange_rest:.0f} deg rest"

    # ...and so does a Stop pressed just after the weight is back down. 133
    # is on the thigh but not yet near enough to 150 to pass on its own, so
    # the verdict really does come from Stop.
    stopped = rc.ReverseCurlTracker(rest_max=160.0)
    feed(stopped, [(2.0, strange_rest), (0.3, 120.0), (2.4, UP),
                   (0.3, 90.0), (0.3, 133.0)], 20)
    assert stopped.state is rc.State.LOWERING, stopped.state
    assert stopped.finish()["passed"], "Stop refused a finished curl at a wide rest angle"

    # ...and the lower end still works the same way.
    low = rc.ReverseCurlTracker(rest_min=60.0)
    feed(low, [(2.0, 65.0)], 20)
    assert low.baseline_deg is not None, "rest_min=60 refused a 65 deg start"


def test_hanging_first_then_a_perfect_curl_passes():
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, 170.0)], 20)
    feed(tr, good_curl(), 20, t0=t)
    assert tr.state is rc.State.PASSED, f"state {tr.state}"


# ── the hold ─────────────────────────────────────────────────────────────────
def test_hold_must_stay_at_the_target_not_just_touch_it():
    # Touching 45 once and then sitting at 50 used to count as the hold.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.2, 70.0), (0.05, 45.0), (2.3, 50.0), (0.3, 70.0), (0.6, REST)], 20)
    assert not tr.passed, "a hold at 50 deg passed a 45 deg target"


def test_short_wobble_keeps_the_hold_but_a_long_one_breaks_it():
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 70.0), (0.9, UP)], 20)
    t, _ = feed(tr, [(0.25, UP + 8)], 20, t0=t)          # 0.25 s at 48 deg
    assert tr.state is rc.State.HOLD, f"a brief wobble broke the hold: {tr.state}"
    t, _ = feed(tr, [(0.5, UP + 8)], 20, t0=t)           # now 0.75 s above
    assert tr.state is rc.State.LIFTING, f"a long drift above the target kept the hold: {tr.state}"


def test_video_pause_mid_hold_is_not_counted():
    # Frames simply stop (phone or network stall) — no update() calls at all.
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 70.0), (0.5, UP)], 20)
    feed(tr, [(0.1, UP)], 20, t0=t + 1.5)
    assert tr.state is rc.State.HOLD and tr.hold_elapsed < 0.5, \
        f"unseen time was credited: {tr.state}, {tr.hold_elapsed:.2f} s"


def test_a_slow_but_steady_phone_still_passes():
    # ~4 fps leaves 0.25 s between frames: under the pause limit.
    tr = rc.ReverseCurlTracker()
    feed(tr, good_curl(), 4)
    assert tr.state is rc.State.PASSED, f"state {tr.state} at 4 fps"


# ── back down ────────────────────────────────────────────────────────────────
def test_weight_set_down_on_a_different_spot_still_passes():
    # Rest was 95; the weight goes back down at 78 — still on the thigh.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (2.4, UP), (0.3, 60.0), (3.0, 78.0)], 20)
    assert tr.state is rc.State.PASSED, f"state {tr.state}"


def test_hold_done_but_arm_stays_up_does_not_pass():
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (2.4, UP), (3.0, 60.0)], 20)
    assert tr.state is rc.State.LOWERING, f"passed without coming back down: {tr.state}"


def test_a_struggling_lift_is_not_failed_early():
    # Wobbling between 64 and 72 on the way UP is still the lift, not a
    # "return to the thigh" — the relaxed return applies only after the hold.
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST), (0.3, 75.0)], 20)
    for _ in range(6):
        t, _ = feed(tr, [(0.25, 64.0), (0.25, 72.0)], 20, t0=t)
    assert tr.state is rc.State.LIFTING, f"a struggling lift was ended: {tr.state}"
    feed(tr, [(2.4, UP), (0.3, 70.0), (0.6, REST)], 20, t0=t)
    assert tr.state is rc.State.PASSED, f"state {tr.state}"


# ── which arm, and what the camera can see ───────────────────────────────────
def _pipeline(frames, fps=20):
    """Drive rc.step_arm exactly as the engine does. frames: (sec, L, R)."""
    sel, tr, sm = rc.ArmSelector(), rc.ReverseCurlTracker(), rc.MedianSmoother()
    prev, t = None, 0.0
    for dur, left, right in frames:
        for _ in range(max(1, int(round(dur * fps)))):
            prev, _, _ = rc.step_arm(sel, tr, sm, {L: left, R: right}, prev, now=t)
            t += 1.0 / fps
    return sel, tr


def test_rest_angle_follows_the_arm_that_curls():
    # The other arm hangs at the side; the working arm rests on the thigh
    # and curls. The rest reading must be the working arm's own.
    sel, tr = _pipeline([(1.0, 170.0, REST), (0.1, 170.0, 90.0), (0.1, 170.0, 80.0),
                         (0.1, 170.0, 70.0), (0.1, 170.0, 55.0), (2.4, 170.0, UP),
                         (0.3, 170.0, 70.0), (1.0, 170.0, REST)])
    assert sel.arm == R and tr.state is rc.State.PASSED, (sel.arm, tr.state)


def test_rest_angle_switches_with_the_arm():
    # Both hands rest on the thighs, at different angles (left 125, right 85).
    # The camera starts on the left; the right arm then fidgets to 65 and back.
    # Measured against the LEFT arm's rest, that fidget looks like a big lift
    # that never comes back down; against the right arm's own rest it is 20
    # deg — ignored. Then a proper curl with the right arm passes.
    sel, tr = _pipeline([(1.0, 125.0, 85.0), (0.2, 125.0, 75.0), (0.3, 125.0, 65.0),
                         (1.0, 125.0, 85.0)])
    assert tr.state is rc.State.SET, f"a 20 deg fidget wasn't ignored: {tr.state}"
    assert abs(tr.baseline_deg - 85.0) < 3, f"rest {tr.baseline_deg}, want the right arm's ~85"


def test_arm_choice_locks_once_the_lift_starts():
    sel, tr = _pipeline([(1.0, REST, REST), (0.1, REST, 80.0), (0.1, REST, 65.0)])
    assert sel.locked and sel.arm == R, (sel.locked, sel.arm)
    sel2, tr2 = _pipeline([(1.0, REST, REST), (0.1, REST, 80.0), (0.1, REST, 65.0),
                           (0.3, 30.0, 60.0)])         # the other hand now moves a lot
    assert sel2.arm == R, f"the idle hand took over mid-lift: {sel2.arm}"


def test_a_small_shift_does_not_pin_the_arm_chooser():
    # The chooser pins its arm on flexion worth a real attempt, not on the
    # smaller threshold that merely starts watching a lift — 12-15 deg is
    # about what adjusting the grip on the thigh looks like.
    sel = rc.ArmSelector()
    t = 0.0
    for _ in range(30):                                   # both hands resting
        sel.update({L: REST, R: 92.0}, now=t); t += 0.05
    for a in (88.0, 84.0, 81.0, 84.0, 88.0, REST):        # left adjusts its grip
        sel.update({L: a, R: 92.0}, now=t); t += 0.05
    assert not sel.locked, "a 14 deg shift of the idle hand pinned the choice"
    for a in (80.0, 70.0, 60.0):                          # the right arm curls
        sel.update({L: REST, R: a}, now=t); t += 0.05
    assert sel.locked and sel.arm == R, (sel.locked, sel.arm)


def test_a_fidget_does_not_pin_the_camera_to_the_idle_hand():
    # The idle hand shifts ~14 deg on the thigh — adjusting the grip. That is
    # not an attempt, so it must not own the trial: the working arm then does
    # a perfect curl and it has to be scored. This used to end NOT COMPLETED
    # with the camera watching the wrong arm.
    sel, tr = _pipeline([(1.2, REST, 92.0),
                         (0.15, 84.0, 92.0), (0.15, 81.0, 92.0), (0.15, 84.0, 92.0),
                         (1.2, REST, 92.0),                      # back on the thigh
                         (0.15, REST, 80.0), (0.15, REST, 65.0), (0.15, REST, 50.0),
                         (2.4, REST, UP),
                         (0.3, REST, 60.0), (0.3, REST, 80.0), (0.8, REST, 92.0)])
    assert sel.arm == R, f"still watching the idle hand: {sel.arm}"
    assert tr.state is rc.State.PASSED, f"the curl was not scored: {tr.state}"


def test_a_long_hold_is_not_stolen_by_the_other_hand():
    # The release above is time-based, so check it cannot fire mid-attempt:
    # through a 3 s hold — longer than the idle delay — the other hand curls
    # hard and must be ignored.
    sel, tr = _pipeline([(1.2, REST, REST),
                         (0.15, REST, 75.0), (0.15, REST, 60.0), (0.15, REST, 45.0),
                         (3.0, 45.0, UP),                        # other hand at 45 too
                         (0.3, 45.0, 70.0), (0.8, 45.0, REST)])
    assert sel.arm == R, f"the other hand took over during the hold: {sel.arm}"
    assert tr.state is rc.State.PASSED, f"the curl was not scored: {tr.state}"


def test_slow_lift_is_not_stolen_by_the_other_hand():
    # A slow lift: the right arm creeps from 95 to 80 over 1.5 s. The tracker
    # sees the lift start at ~83; the arm chooser, whose own rest reading
    # drifts down with the slow movement, would lock later. If the other hand
    # moves a lot in that gap it must not take over the attempt in progress.
    creep = [(1.5 / 16, REST, REST - i) for i in range(16)]
    sel, tr = _pipeline([(1.0, REST, REST)] + creep + [(0.5, 60.0, 80.0)])
    assert tr.state is rc.State.LIFTING, f"state {tr.state}"
    assert sel.arm == R, f"the other hand took over a lift in progress: {sel.arm}"
    assert abs(tr.baseline_deg - REST) < 2, f"rest drifted with the slow lift: {tr.baseline_deg}"


def _creep(frm, to, deg_per_s, fps=20):
    """A steady slow movement, one frame at a time: [(sec, angle), ...]."""
    step = deg_per_s / fps * (1 if to > frm else -1)
    out, a = [], frm
    while (a < to) if to > frm else (a > to):
        out.append((1.0 / fps, a)); a += step
    return out


def test_slow_curl_keeps_its_true_rest_angle():
    # ~10 deg/s — slow and controlled. The rest reading must stay at the
    # thigh (95), not creep down with the arm while it is still in range.
    tr = rc.ReverseCurlTracker()
    t, _ = feed(tr, [(1.0, REST)] + _creep(REST, 70.0, 10), 20)
    assert abs(tr.baseline_deg - REST) < 2, f"rest drifted to {tr.baseline_deg}"
    feed(tr, _creep(70.0, UP, 10) + [(2.4, UP), (0.3, 70.0), (0.6, REST)], 20, t0=t)
    assert tr.state is rc.State.PASSED, f"state {tr.state}"


def test_slow_struggling_student_is_not_failed_for_sinking_back_a_little():
    # Creeps up to 60, can't go further, sinks back to 75 — still well off
    # the thigh (95). With a drifted rest angle (~72) that read as "back
    # down" and FAILED. Against the true rest it is still the lift.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST)] + _creep(REST, 60.0, 10) + [(1.0, 60.0)] + _creep(60.0, 75.0, 10)
         + [(1.0, 75.0)], 20)
    assert tr.state is rc.State.LIFTING, f"sinking back to 75 ended the trial: {tr.state}"


def test_slow_lowering_does_not_pass_before_the_weight_is_down():
    # A careful ~8 deg/s lowering passes through the resting range for a
    # while; it must not be taken as "set down" while still moving.
    tr = rc.ReverseCurlTracker()
    feed(tr, [(1.0, REST), (0.3, 70.0), (2.4, UP)] + _creep(UP, 79.0, 8), 20)
    assert tr.state is rc.State.LOWERING, f"passed mid-lowering: {tr.state}"


def test_joints_outside_the_picture_are_ignored():
    import types
    lms = [types.SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    def arm_at(elbow_y):
        for i, (x, y) in {12: (0.40, 0.60), 14: (0.42, elbow_y), 16: (0.47, 0.70)}.items():
            lms[i] = types.SimpleNamespace(x=x, y=y, visibility=0.9)
        return rc.elbow_angle(lms, R, 640, 480)
    assert arm_at(1.30) is None, "an elbow below the picture produced an angle"
    assert arm_at(0.85) is not None, "an arm inside the picture was rejected"


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
