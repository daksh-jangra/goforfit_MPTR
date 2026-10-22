"""
Logic checks for the Bench Press counter - no camera, no participant.
=====================================================================

WHY THIS FILE EXISTS
--------------------
  The count depends on things the coach can't see on the screen: the chest
  and top lines, the 4 s rest and 15 s first-rep ends, the 50 / 30 finish,
  dropouts and glitches. Get one wrong and nothing looks broken - the count
  just comes out wrong.

  Everything here drives BenchPressTest directly with made-up arm readings
  and an injected clock: smooth presses (not step jumps), several frame
  rates, jitter and single-frame spikes.

HOW TO RUN IT
-------------
      python3 tests/test_bench_press.py
      pytest tests/test_bench_press.py

  Non-zero exit means something broke. Run it after ANY edit to bench_press.py.
"""

import math
import os
import random
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import bench_press as bp  # noqa: E402

L, R = bp.SIDE_LEFT, bp.SIDE_RIGHT
TOP, CHEST = 170.0, 75.0          # the reference video: 155-179 / 59-82
FPS_LIST = (8, 15, 20, 30)


# ── body model: readings for a given elbow angle ─────────────────────────────────
def lying(elbow, **chg):
    """Side-on, lying on the bench. The upper arm lies along the body at the
    chest (elev ~0) and points almost at the ceiling at the top (~85), as on
    the reference video's side-on frames."""
    f = (elbow - CHEST) / (TOP - CHEST)
    m = {"elbow": elbow, "elev": 85.0 * f, "body_tilt": 5.0, "vis": 0.95}
    m.update(chg)
    return m


def no_hip(elbow, **chg):
    return lying(elbow, elev=None, body_tilt=None, **chg)


def sitting(elbow, **chg):
    return lying(elbow, body_tilt=80.0, **chg)


# ── movement ─────────────────────────────────────────────────────────────────────
def press_curve(t, period, top=TOP, bottom=CHEST):
    """Elbow over one smooth rep lasting `period`: chest -> top -> chest."""
    return bottom + (top - bottom) * (0.5 - 0.5 * math.cos(2 * math.pi * t / period))


def run(test, segments, fps=20, t0=0.0, jitter=0.0, spike_p=0.0, seed=1, side=L,
        far=None):
    """segments: list of (seconds, fn(t_in_segment) -> metrics|None). Each frame
    goes to the near side; `far` (fn(t) -> metrics|None) feeds the other side.
    Returns the time after the last frame; test.end_t says when it ended."""
    rnd = random.Random(seed)
    t, dt = t0, 1.0 / fps
    other = R if side == L else L
    for duration, fn in segments:
        n = max(1, int(round(duration * fps)))
        for i in range(n):
            m = fn(i * dt)
            if m is not None:
                m = dict(m)
                if jitter:
                    m["elbow"] += rnd.uniform(-jitter, jitter)
                    if m["elev"] is not None:
                        m["elev"] += rnd.uniform(-jitter, jitter)
                if spike_p and rnd.random() < spike_p:
                    m["elbow"] = rnd.uniform(40, 180)       # one wild frame
            sides = {L: None, R: None}
            sides[side] = m
            if far is not None:
                sides[other] = far(i * dt)
            test.update(sides, now=t)
            t += dt
    return t


def hold(model=lying, elbow=CHEST, **chg):
    return lambda t: model(elbow, **chg)


def presses(n, period=3.5, model=lying, top=TOP, bottom=CHEST, **chg):
    return [(n * period, lambda t: model(press_curve(t % period, period, top, bottom), **chg))]


def press_up(dur=1.75):
    """From the chest smoothly up to straight arms (half a rep)."""
    return (dur, lambda t: lying(press_curve(t, 2 * dur)))


def started(fps=20, settle=1.0, sex=bp.SEX_BOY, elbow=CHEST, model=lying):
    """A test with the bar at the chest (or wherever) and START pressed."""
    test = bp.BenchPressTest()
    test.set_sex(sex)
    t = run(test, [(settle, hold(model, elbow))], fps=fps)
    assert test.start(now=t)
    return test, t


# ── checks: counting ─────────────────────────────────────────────────────────────
def check_clean_reps_counted_exactly():
    for fps in FPS_LIST:
        for seed in range(5):
            test, t = started(fps)
            run(test, presses(10), fps=fps, t0=t, jitter=4, seed=seed)
            assert test.count == 10, (fps, seed, test.count, test.not_counted)
            assert test.total_not_counted == 0, (fps, seed, test.not_counted)


def check_fast_and_slow_pace():
    for fps in FPS_LIST:
        for period in (2.0, 6.0):
            test, t = started(fps)
            run(test, presses(6, period), fps=fps, t0=t, jitter=3)
            assert test.count == 6, (fps, period, test.count, test.end_reason)


def check_the_video_numbers_count():
    # Shallowest chest touch and the least-straight lockout seen on the video.
    for fps in FPS_LIST:
        test, t = started(fps, elbow=82.0)
        run(test, presses(5, top=157.0, bottom=82.0), fps=fps, t0=t)
        assert test.count == 5, (fps, test.count, test.not_counted)


def check_reps_before_start_do_not_count():
    test = bp.BenchPressTest()
    test.set_sex(bp.SEX_BOY)
    t = run(test, presses(3))
    assert test.count == 0
    test.start(now=t)
    run(test, presses(2), t0=t)
    assert test.count == 2, test.count


def check_unracked_at_the_top_must_touch_the_chest_first():
    test, t = started(elbow=TOP)
    assert test.phase == bp.PH_UP
    # Lower to the chest, then press: one rep each.
    seg = [(3.5, lambda s: lying(TOP - (TOP - CHEST) * (0.5 - 0.5 * math.cos(2 * math.pi * s / 3.5))))]
    run(test, seg * 3, t0=t)
    assert test.count == 3, test.count


def check_half_press_does_not_count():
    test, t = started()
    t = run(test, presses(2), t0=t)
    run(test, presses(1, top=140.0), t0=t)
    assert test.count == 2, test.count
    assert test.not_counted["not_straight"] == 1, test.not_counted


def check_stopping_short_of_the_chest_does_not_count():
    test, t = started()
    t = run(test, presses(1), t0=t)
    # Back to the top, then dips to 110 and presses again: no chest touch.
    t = run(test, [press_up(), (1.0, hold(elbow=TOP))], t0=t)
    seg = [(3.0, lambda s: lying(TOP - (TOP - 110.0) * (0.5 - 0.5 * math.cos(2 * math.pi * s / 3.0))))]
    run(test, seg, t0=t)
    assert test.count == 2, test.count
    assert test.not_counted["not_chest"] == 1, test.not_counted


def check_elbow_bent_but_arm_raised_is_not_the_chest():
    # Bar held high over the face with bent elbows: elbow under 90 but the
    # upper arm still raised 50 deg - never at the chest, so nothing counts.
    test, t = started(elbow=TOP)
    run(test, presses(4, bottom=85.0, elev=50.0), t0=t)
    assert test.count == 0, test.count


def check_without_the_hip_the_elbow_alone_counts():
    test, t = started(model=no_hip)
    run(test, presses(5, model=no_hip), t0=t, jitter=3)
    assert test.count == 5, test.count


def check_wobble_at_the_top_is_not_an_attempt():
    test, t = started()
    t = run(test, presses(1), t0=t)
    t = run(test, [press_up()], t0=t)
    run(test, [(3.0, lambda s: lying(160.0 + 5.0 * math.sin(2 * math.pi * s)))], t0=t, jitter=3)
    assert test.count == 2 and test.total_not_counted == 0, (test.count, test.not_counted)


def check_sitting_up_does_not_count():
    test, t = started(model=sitting)
    run(test, presses(4, model=sitting), t0=t)
    assert test.count == 0
    assert test.problem == "Lie down on the bench"


def check_far_arm_noise_is_ignored():
    rnd = random.Random(3)
    far = lambda s: lying(rnd.uniform(40, 180), vis=0.55)
    test, t = started()
    run(test, presses(6), t0=t, jitter=3, far=far)
    assert test.count == 6, test.count
    assert test.side == L


def check_either_side_facing_the_camera_works():
    test = bp.BenchPressTest()
    test.set_sex(bp.SEX_GIRL)
    t = run(test, [(1.0, hold())], side=R)
    test.start(now=t)
    run(test, presses(4), t0=t, side=R)
    assert test.count == 4 and test.side == R


# ── checks: glitches and dropouts ────────────────────────────────────────────────
def check_single_wild_frames_do_not_change_the_count():
    for fps in FPS_LIST:
        for seed in range(10):
            test, t = started(fps)
            run(test, presses(8), fps=fps, t0=t, jitter=3, spike_p=0.02, seed=seed)
            assert test.count == 8, (fps, seed, test.count, test.not_counted)


def check_two_wild_frames_at_the_top_do_not_count_twice():
    test, t = started()
    t = run(test, presses(1, period=4.0)[:1], t0=t)          # ends at the chest
    # Press up and hold, with two "at the chest" frames in the middle.
    t = run(test, [(1.5, lambda s: lying(press_curve(min(s, 1.5), 3.0)))], t0=t)
    t = run(test, [(0.5, hold(elbow=TOP))], t0=t)
    t = run(test, [(0.1, hold(elbow=70.0))], t0=t)            # 2 frames at 20 fps
    run(test, [(1.0, hold(elbow=TOP))], t0=t)
    assert test.count == 2, test.count


def check_two_wild_frames_at_the_chest_are_not_a_rep():
    test, t = started()
    t = run(test, [(1.0, hold())], t0=t)
    t = run(test, [(0.1, hold(elbow=TOP))], t0=t)
    run(test, [(1.0, hold())], t0=t)
    assert test.count == 0, test.count


def check_dropout_mid_press_still_counts_once():
    test, t = started()
    gone = lambda s: None
    t = run(test, [(0.8, lambda s: lying(press_curve(s, 3.5)))], t0=t)
    t = run(test, [(0.5, gone)], t0=t)
    run(test, [(2.2, lambda s: lying(press_curve(s + 1.3, 3.5)))], t0=t)
    assert test.count == 1, test.count


# ── checks: the end rules ────────────────────────────────────────────────────────
def check_boy_ends_at_50_and_girl_at_30():
    for sex, cap in ((bp.SEX_BOY, 50), (bp.SEX_GIRL, 30)):
        test, t = started(sex=sex)
        run(test, presses(cap + 3, period=2.5), t0=t)
        assert test.count == cap and test.end_reason == bp.END_MAX, (sex, test.count, test.end_reason)


def check_start_waits_for_boy_or_girl():
    test = bp.BenchPressTest()
    t = run(test, [(1.0, hold())])
    assert test.start_problem() == "Pick Boy or Girl first"
    assert not test.start(now=t) and not test.started
    assert test.set_sex(bp.SEX_GIRL)
    assert test.start(now=t)
    assert not test.set_sex(bp.SEX_BOY), "changed mid-trial"
    assert test.max_reps == 30
    test.reset()
    assert test.sex == bp.SEX_GIRL, "a RESET keeps the pick"


def check_rest_after_the_last_rep_ends_at_four_seconds():
    # Lying still at the chest, holding the bar up, stopped partway up.
    for where, move, count in ((CHEST, [], 3), (TOP, [press_up()], 4),
                               (120.0, [(1.0, lambda s: lying(CHEST + 45.0 * s))], 3)):
        test, t = started()
        t = run(test, presses(3) + move, t0=t)
        t_last = t
        run(test, [(10.0, hold(elbow=where))], t0=t, jitter=4)
        assert test.end_reason == bp.END_REST, (where, test.end_reason)
        # From the end of the movement; at the chest the elbow stops moving
        # a little before the cycle ends.
        assert 3.5 <= test.end_t - t_last <= 5.5, (where, test.end_t - t_last)
        assert test.count == count, (where, test.count)


def check_no_rep_in_15_s_scores_zero():
    test, t = started()
    run(test, [(20.0, hold())], t0=t, jitter=3)
    assert test.end_reason == bp.END_NO_REP and test.count == 0
    assert abs(test.end_t - (test.start_t + 15.0)) < 1e-6


def check_not_seen_still_scores_zero_at_15_s():
    test, t = started()
    run(test, [(20.0, lambda s: None)], t0=t)
    assert test.end_reason == bp.END_NO_REP


def check_half_presses_do_not_hold_off_the_zero():
    test, t = started()
    run(test, presses(8, top=140.0), t0=t)
    assert test.end_reason == bp.END_NO_REP and test.count == 0


def check_slow_press_does_not_time_out():
    test, t = started()
    t = run(test, presses(2), t0=t)
    # 8 s up, steadily.
    t = run(test, [(8.0, lambda s: lying(CHEST + (TOP - CHEST) * s / 8.0))], t0=t, jitter=3)
    assert test.count == 3 and not test.stopped, (test.count, test.end_reason)


def check_slow_lowering_does_not_time_out():
    test, t = started()
    t = run(test, presses(1), t0=t)
    t = run(test, [press_up(), (1.0, hold(elbow=TOP))], t0=t)
    t = run(test, [(7.0, lambda s: lying(TOP - (TOP - CHEST) * s / 7.0))], t0=t, jitter=3)
    t = run(test, [press_up()], t0=t)
    assert not test.stopped and test.count == 3, (test.count, test.end_reason)


def check_repeated_half_presses_end_the_test():
    test, t = started()
    t = run(test, presses(2), t0=t)
    run(test, presses(8, top=135.0), t0=t)
    assert test.end_reason == bp.END_REST and test.count == 2, (test.count, test.end_reason)


def check_one_missed_lockout_does_not_end_the_test():
    # Slow reps; the 3rd stops at 150 (just short). Its lowering plus the
    # next press took the clock past 4 s.
    test, t = started()
    t = run(test, presses(2, period=5.0), t0=t)
    t = run(test, presses(1, period=5.0, top=150.0), t0=t)
    t = run(test, [(1.0, hold())], t0=t)
    run(test, presses(3, period=5.0), t0=t)
    assert test.count == 5 and not test.stopped, (test.count, test.end_reason)
    assert test.not_counted["not_straight"] == 1


def check_two_missed_lockouts_in_a_row_end_the_test():
    test, t = started()
    t = run(test, presses(2, period=5.0), t0=t)
    run(test, presses(4, period=5.0, top=150.0), t0=t)
    assert test.end_reason == bp.END_REST and test.count == 2, (test.count, test.end_reason)


def check_a_long_frame_gap_adds_at_most_half_a_second():
    test, t = started()
    t = run(test, presses(1), t0=t)
    before = test.rest_elapsed
    test.update({L: lying(CHEST), R: None}, now=t + 3.0)     # 3 s Wi-Fi gap
    assert test.rest_elapsed - before <= 0.5 + 1e-9 and not test.stopped


def check_stop_and_reset():
    test, t = started()
    t = run(test, presses(3), t0=t)
    test.stop(now=t)
    run(test, presses(2), t0=t)
    assert test.count == 3 and test.end_reason == bp.END_STOPPED
    test.reset()
    assert test.count == 0 and not test.started and not test.stopped
    assert test.total_not_counted == 0


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
