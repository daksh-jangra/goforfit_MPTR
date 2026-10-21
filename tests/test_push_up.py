"""
Logic checks for the Push-Up counter - no camera, no participant.
=================================================================

WHY THIS FILE EXISTS
--------------------
  The count depends on things the coach can't see on the screen: the elbow
  lines, the knee and back faults within each rep, the 6 s gap that ends the
  test, dropouts. Get one wrong and nothing looks broken - the count just
  comes out wrong.

  Everything here drives PushUpTest directly with made-up body readings and
  an injected clock: smooth push-ups (not step jumps), several frame rates,
  jitter and single-frame spikes.

HOW TO RUN IT
-------------
      python3 tests/test_push_up.py
      pytest tests/test_push_up.py

  Non-zero exit means something broke. Run it after ANY edit to push_up.py.
"""

import math
import os
import random
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import push_up as pu  # noqa: E402

L, R = pu.SIDE_LEFT, pu.SIDE_RIGHT
TOP, BOTTOM = 172.0, 85.0
FPS_LIST = (8, 15, 20, 30)


# ── body models: readings for a given elbow angle ────────────────────────────────
def on_toes(elbow, **chg):
    """A correct push-up. The knee sits ~0.45 thigh lengths above the floor
    line at the top and drops to ~0.15 at the bottom."""
    f = (elbow - BOTTOM) / (TOP - BOTTOM)          # 1 at the top, 0 at the bottom
    m = {"elbow": elbow, "bend": -5.0, "knee": 172.0, "knee_h": 0.15 + 0.30 * f,
         "body_tilt": 8.0 + 14.0 * f, "vis": 0.95}
    m.update(chg)
    return m


def on_knees_shins_flat(elbow, **chg):
    """Knee push-up, shins on the mat: the knee bends only to ~145 at the top
    and straightens toward 170 at the bottom; the knee is on the floor line."""
    f = (elbow - BOTTOM) / (TOP - BOTTOM)
    m = {"elbow": elbow, "bend": -15.0, "knee": 170.0 - 25.0 * f, "knee_h": 0.10,
         "body_tilt": 8.0 + 18.0 * f, "vis": 0.95}
    m.update(chg)
    return m


def on_knees_feet_up(elbow, **chg):
    return on_toes(elbow, knee=100.0, knee_h=0.10, **chg)


def standing(elbow, **chg):
    return on_toes(elbow, body_tilt=80.0, **chg)


# ── movement ─────────────────────────────────────────────────────────────────────
def rep_curve(t, period, top=TOP, bottom=BOTTOM):
    """Elbow over one smooth push-up lasting `period`: top -> bottom -> top."""
    return bottom + (top - bottom) * (0.5 + 0.5 * math.cos(2 * math.pi * t / period))


def run(test, segments, fps=20, t0=0.0, jitter=0.0, spike_p=0.0, seed=1, side=L):
    """segments: list of (seconds, fn(t_in_segment) -> metrics|None). Each frame
    goes to the near side only, as side-on in real footage. Returns the time
    after the last frame; test.end_t says when it ended."""
    rnd = random.Random(seed)
    t, dt = t0, 1.0 / fps
    for duration, fn in segments:
        n = max(1, int(round(duration * fps)))
        for i in range(n):
            m = fn(i * dt)
            if m is not None:
                m = dict(m)
                if jitter:
                    m["elbow"] += rnd.uniform(-jitter, jitter)
                    m["bend"] += rnd.uniform(-jitter, jitter)
                    m["knee"] += rnd.uniform(-jitter, jitter)
                    if m["knee_h"] is not None:
                        m["knee_h"] += rnd.uniform(-jitter, jitter) / 100
                if spike_p and rnd.random() < spike_p:
                    m["elbow"] = rnd.uniform(60, 180)       # one wild frame
            sides = {L: None, R: None}
            sides[side] = m
            test.update(sides, now=t)
            t += dt
    return t


def hold(model, elbow=TOP, **chg):
    return lambda t: model(elbow, **chg)


def reps(model, n, period=3.0, top=TOP, bottom=BOTTOM, **chg):
    return [(n * period, lambda t: model(rep_curve(t % period, period, top, bottom), **chg))]


def started(fps=20, settle=1.0, model=on_toes):
    """A test already at the top with START pressed."""
    test = pu.PushUpTest()
    t = run(test, [(settle, hold(model))], fps=fps)
    test.start(now=t)
    return test, t


# ── checks ───────────────────────────────────────────────────────────────────────
def check_clean_reps_counted_exactly():
    for fps in FPS_LIST:
        for seed in range(5):
            test, t = started(fps)
            run(test, reps(on_toes, 10), fps=fps, t0=t, jitter=4, seed=seed)
            assert test.count == 10, (fps, seed, test.count, test.not_counted)
            assert test.total_not_counted == 0, (fps, seed, test.not_counted)


def check_fast_and_slow_cadence():
    for fps in FPS_LIST:
        for period in (1.5, 5.0):
            test, t = started(fps)
            run(test, reps(on_toes, 6, period=period) + [(0.5, hold(on_toes))],
                fps=fps, t0=t, jitter=3)
            assert test.count == 6 and not test.stopped, (fps, period, test.count, test.end_reason)


def check_gap_ends_test_6s_after_last_rep():
    for fps in FPS_LIST:
        test, t = started(fps)
        t = run(test, reps(on_toes, 3), fps=fps, t0=t)
        last_rep_t = t        # each rep ends back at the top at the period's end
        run(test, [(10, hold(on_toes))], fps=fps, t0=t)
        assert test.stopped and test.end_reason == pu.END_GAP, (fps, test.end_reason)
        assert test.count == 3
        assert abs((test.end_t - last_rep_t) - pu.GAP_S) < 0.5, (fps, test.end_t - last_rep_t)


def check_no_reps_at_all_ends_with_zero():
    test, t = started()
    run(test, [(8, hold(on_toes))], t0=t)
    assert test.stopped and test.count == 0 and test.end_reason == pu.END_GAP
    assert test.result()["reps"] == 0


def check_one_bad_rep_in_a_good_set_doesnt_end_it():
    for fps in FPS_LIST:
        test, t = started(fps)
        t = run(test, reps(on_toes, 3), fps=fps, t0=t, jitter=3)
        t = run(test, reps(on_toes, 1, bottom=120.0), fps=fps, t0=t, jitter=3)
        run(test, reps(on_toes, 3) + [(0.5, hold(on_toes))], fps=fps, t0=t, jitter=3)
        assert not test.stopped and test.count == 6, (fps, test.count, test.end_reason)
        assert test.not_counted["shallow"] == 1, (fps, test.not_counted)


def check_resting_at_the_top_near_the_line_ends_it():
    # A straight arm that reads 160-165 wobbles across the top line. That is
    # rest, not push-up attempts: the test must still end 6 s later, with no
    # made-up "not to 90" entries.
    for top in (160.0, 162.0, 165.0):
        for fps in (8, 15, 30):
            for seed in range(5):
                test, t = started(fps)
                t = run(test, reps(on_toes, 2), fps=fps, t0=t)
                run(test, [(10, hold(on_toes, elbow=top))], fps=fps, t0=t, jitter=8, seed=seed)
                assert test.stopped and test.end_reason == pu.END_GAP, (top, fps, seed)
                assert test.count == 2 and test.total_not_counted == 0, (top, fps, seed, test.not_counted)


def check_tilted_phone_still_counts():
    # The phone rolled ~20 degrees adds that to the body's tilt on screen.
    for roll in (10.0, 20.0):
        def tilted(e, **chg):
            m = on_toes(e, **chg)
            m["body_tilt"] += roll
            return m
        test, t = started(model=tilted)
        run(test, reps(tilted, 4) + [(0.5, hold(tilted))], t0=t, jitter=3)
        assert test.count == 4, (roll, test.count)


def check_kneeling_upright_is_not_in_position():
    test = pu.PushUpTest()
    t = run(test, [(1, hold(on_toes, body_tilt=65.0))])
    test.start(now=t)
    run(test, reps(on_toes, 3, body_tilt=65.0), t0=t)
    assert test.count == 0 and test.stopped


def check_resting_at_the_bottom_ends_it():
    test, t = started()
    t = run(test, reps(on_toes, 2), t0=t)
    run(test, [(10, hold(on_toes, elbow=80.0))], t0=t)
    assert test.stopped and test.count == 2


def check_shallow_not_counted():
    for fps in FPS_LIST:
        test, t = started(fps)
        run(test, reps(on_toes, 5, bottom=118.0), fps=fps, t0=t, jitter=3)
        assert test.count == 0, (fps, test.count)
        assert test.not_counted["shallow"] == 5, (fps, test.not_counted)
    # 95 at the bottom (90 plus a little) still counts.
    test, t = started()
    run(test, reps(on_toes, 5, bottom=95.0), t0=t, jitter=3)
    assert test.count == 5, test.not_counted


def check_arms_not_straight_at_top():
    # Bottom to 150 and back down, over and over: never straight, never counted.
    test, t = started()
    seg = [(15, lambda tt: on_toes(rep_curve((tt % 3.0) + 1.5, 3.0, top=150.0)))]
    run(test, seg, t0=t, jitter=2)
    assert test.count == 0, test.count
    assert test.not_counted["not_straight"] >= 3, test.not_counted


def check_half_then_full_rep():
    # Down, up to 140, down again, then all the way up: one push-up, plus one
    # "not straightened".
    test, t = started()

    def path(tt):
        pts = [(0, TOP), (1, 85), (2, 140), (3, 85), (4.5, TOP), (6, TOP)]
        for (a, ea), (b, eb) in zip(pts, pts[1:]):
            if a <= tt <= b:
                return on_toes(ea + (eb - ea) * (tt - a) / (b - a))
        return on_toes(TOP)
    run(test, [(6, path)], t0=t)
    assert test.count == 1 and test.not_counted["not_straight"] == 1, (test.count, test.not_counted)


def check_knee_push_ups_not_counted():
    for model in (on_knees_shins_flat, on_knees_feet_up):
        for fps in FPS_LIST:
            test, t = started(fps, model=model)
            run(test, reps(model, 5), fps=fps, t0=t, jitter=3)
            assert test.count == 0, (model.__name__, fps, test.count)
            assert test.not_counted["knees"] == 5, (model.__name__, fps, test.not_counted)


def check_knee_push_up_caught_without_toes():
    # Toes out of view: only the knee angle is left, which is ~145 at the top.
    test, t = started(model=on_knees_shins_flat)
    run(test, reps(on_knees_shins_flat, 4, knee_h=None), t0=t, jitter=2)
    assert test.count == 0 and test.not_counted["knees"] == 4, test.not_counted


def check_knee_lines_strict_but_not_too_strict():
    # Toes in view: soft knees (147+) count, clearly bent (140) don't.
    for knee, counted in ((150.0, True), (147.0, True), (140.0, False)):
        test, t = started()
        run(test, reps(on_toes, 4, knee=knee) + [(0.5, hold(on_toes, knee=knee))], t0=t, jitter=2)
        assert (test.count == 4) == counted, (knee, test.count, test.not_counted)
        if not counted:
            assert test.not_counted["knees"] == 4, (knee, test.not_counted)
    # Toes out of view: the angle is the only knee check, so it stays at 150.
    for knee, counted in ((155.0, True), (147.0, False)):
        test, t = started()
        run(test, reps(on_toes, 4, knee=knee, knee_h=None)
            + [(0.5, hold(on_toes, knee=knee, knee_h=None))], t0=t, jitter=2)
        assert (test.count == 4) == counted, ("no toes", knee, test.count, test.not_counted)


def check_toes_flickering_keeps_the_softer_knee_line():
    # The pose model drops the toe now and then. Soft knees (147) on the
    # toes must still count, and knee push-ups must still be caught.
    for miss in (0.1, 0.3):
        for fps in (8, 15, 30):
            for seed in range(5):
                rnd = random.Random(seed)

                def flicker(base, **kw):
                    def model(e, **c):
                        m = base(e, **kw, **c)
                        if rnd.random() < miss:
                            m["knee_h"] = None
                        return m
                    return model
                soft = flicker(on_toes, knee=147.0)
                test, t = started(fps, model=soft)
                run(test, reps(soft, 5) + [(0.5, hold(soft))], fps=fps, t0=t, jitter=2, seed=seed)
                assert test.count == 5, ("soft", miss, fps, seed, test.not_counted)
                kneel = flicker(on_knees_shins_flat)
                test, t = started(fps, model=kneel)
                run(test, reps(kneel, 5), fps=fps, t0=t, jitter=2, seed=seed)
                assert test.count == 0, ("kneel", miss, fps, seed, test.count)


def check_toes_warning_comes_and_goes_and_never_changes_the_count():
    test, t = started()
    t = run(test, reps(on_toes, 2), t0=t)
    assert not test.no_toes
    t = run(test, reps(on_toes, 2, knee_h=None), t0=t)       # toes cropped out
    assert test.no_toes
    t = run(test, [(0.3, hold(on_toes))] + reps(on_toes, 2), t0=t)  # back in view
    assert not test.no_toes
    run(test, [(0.5, hold(on_toes))], t0=t)
    assert test.count == 6, test.count
    # Toes flickering out in 30% of frames: counting is fine, so no warning.
    # Missing in 85%: the warning shows.
    for miss, want in ((0.3, False), (0.85, True)):
        rnd = random.Random(1)
        test, t = started()
        shown = 0
        for i in range(15 * 20):
            m = on_toes(rep_curve((i / 20) % 3.0, 3.0))
            if rnd.random() < miss:
                m["knee_h"] = None
            test.update({L: m, R: None}, now=t)
            t += 1 / 20
            shown += test.no_toes
        assert (shown > 150) == want and (shown == 0 or want), (miss, shown)
    # A single frame without toes doesn't raise it.
    test, t = started()
    seq = [(0.05, hold(on_toes, knee_h=None)), (1.0, hold(on_toes))] * 3
    run(test, seq, t0=t)
    assert not test.no_toes


def check_good_push_up_low_knee_at_bottom_counts():
    # The knee is close to the floor at the bottom of a good push-up; that
    # must not read as kneeling.
    test, t = started()
    run(test, reps(on_toes, 5), t0=t, jitter=4)
    assert test.count == 5 and test.not_counted["knees"] == 0


def check_resting_on_knees_between_reps_is_free():
    test, t = started()
    t = run(test, reps(on_toes, 2), t0=t)
    t = run(test, [(1.5, hold(on_knees_shins_flat))], t0=t)   # brief rest
    t = run(test, [(0.5, hold(on_toes))], t0=t)               # back on the toes
    run(test, reps(on_toes, 2), t0=t)
    assert test.count == 4, (test.count, test.not_counted)


def check_back_faults():
    for bend in (-35.0, 35.0):
        test, t = started()
        run(test, reps(on_toes, 4, bend=bend), t0=t, jitter=2)
        assert test.count == 0 and test.not_counted["back"] == 4, (bend, test.not_counted)
    for bend in (-15.0, 15.0):      # "reasonably straight"
        test, t = started()
        run(test, reps(on_toes, 4, bend=bend), t0=t, jitter=2)
        assert test.count == 4, (bend, test.not_counted)


def check_sag_only_at_the_bottom_still_voids():
    def sag_at_bottom(e):
        return on_toes(e, bend=-35.0 if e < 110 else -5.0)
    test, t = started()
    run(test, [(12, lambda tt: sag_at_bottom(rep_curve(tt % 3.0, 3.0)))], t0=t)
    assert test.count == 0 and test.not_counted["back"] == 4, test.not_counted


def check_single_frame_spikes_change_nothing():
    for fps in (15, 20, 30):
        for seed in range(5):
            test, t = started(fps)
            run(test, reps(on_toes, 8), fps=fps, t0=t, jitter=3, spike_p=0.03, seed=seed)
            assert test.count == 8, (fps, seed, test.count, test.not_counted)


def check_two_wild_frames_in_a_row_change_nothing():
    # Two bad readings back to back get past the 3-frame median. Low ones
    # while resting at the top must not add a push-up or a "not to 90";
    # high ones at the bottom must not count one early.
    for fps in (8, 15, 30):
        for wild, at in ((70.0, TOP), (178.0, 88.0)):
            test, t = started(fps)
            t = run(test, reps(on_toes, 2), fps=fps, t0=t)
            if at != TOP:            # go down and stay at the bottom a moment
                t = run(test, [(1.5, lambda tt: on_toes(rep_curve(tt, 3.0)))], fps=fps, t0=t)
            n = 0

            def glitch(tt):
                nonlocal n
                n += 1
                m = on_toes(at)              # the body really is here...
                if n in (5, 6):
                    m["elbow"] = wild        # ...but two readings say otherwise
                return m
            t = run(test, [(1.5, glitch)], fps=fps, t0=t)
            if at != TOP:            # then finish the push-up properly
                t = run(test, [(1.5, lambda tt: on_toes(rep_curve(tt + 1.5, 3.0)))], fps=fps, t0=t)
            run(test, [(0.5, hold(on_toes))], fps=fps, t0=t)
            want = 2 if at == TOP else 3
            assert test.count == want and test.total_not_counted == 0, \
                (fps, wild, test.count, test.not_counted)


def check_reps_before_start_dont_count():
    test = pu.PushUpTest()
    t = run(test, [(0.5, hold(on_toes))] + reps(on_toes, 3))
    assert test.count == 0 and test.total_not_counted == 0
    test.start(now=t)
    run(test, reps(on_toes, 2), t0=t)
    assert test.count == 2


def check_start_at_the_bottom_needs_the_top_first():
    test = pu.PushUpTest()
    t = run(test, [(1, hold(on_toes)), (1, hold(on_toes, elbow=85.0))])
    test.start(now=t)
    # Up from the bottom: not a push-up (it didn't start from straight arms).
    t = run(test, [(1, hold(on_toes, elbow=TOP))], t0=t)
    assert test.count == 0
    run(test, reps(on_toes, 2), t0=t)
    assert test.count == 2


def check_standing_counts_nothing_and_ends():
    test = pu.PushUpTest()
    t = run(test, [(1, hold(standing))])
    test.start(now=t)
    run(test, reps(standing, 3), t0=t)
    assert test.count == 0 and test.stopped and test.end_reason == pu.END_GAP


def check_dropouts():
    # A 1 s dropout mid-rep at the top: the rep still counts. A 10 s
    # dropout doesn't end the test (nobody was seen resting).
    test, t = started()
    t = run(test, reps(on_toes, 2), t0=t)
    t = run(test, [(1.0, lambda tt: None)], t0=t)
    t = run(test, reps(on_toes, 2), t0=t)
    assert test.count == 4, test.count
    t = run(test, [(10, lambda tt: None)], t0=t)
    assert not test.stopped
    t = run(test, reps(on_toes, 1), t0=t)
    assert test.count == 5


def check_scattered_lost_frames_on_a_slow_phone():
    # Single lost frames (glare, blur) at 8 fps must not hide push-ups.
    for fps in (8, 15):
        for seed in range(10):
            rnd = random.Random(seed)
            test, t = started(fps)
            seg = [(18, lambda tt: None if rnd.random() < 0.2
                    else on_toes(rep_curve(tt % 3.0, 3.0))), (1, hold(on_toes))]
            run(test, seg, fps=fps, t0=t, jitter=2, seed=seed)
            assert test.count == 6, (fps, seed, test.count, test.not_counted)


def check_frames_arriving_slowly_dont_end_it():
    # One frame every 0.9 s (a stalling phone) while at the top for 4 s:
    # each frame adds at most MAX_FRAME_DT.
    test, t = started()
    t = run(test, reps(on_toes, 1), t0=t)
    run(test, [(4, hold(on_toes))], fps=1 / 0.9, t0=t)
    assert not test.stopped


def check_side_switch_keeps_counting():
    test, t = started()
    t = run(test, reps(on_toes, 2), t0=t, side=L)
    run(test, reps(on_toes, 2), t0=t, side=R)
    assert test.count == 4, test.count


def check_stop_and_reset():
    test = pu.PushUpTest()
    test.stop(now=1.0)
    assert not test.stopped            # never started: nothing to stop
    test, t = started()
    t = run(test, reps(on_toes, 3), t0=t)
    test.stop(now=t)
    t = run(test, reps(on_toes, 2), t0=t)
    assert test.count == 3 and test.end_reason == pu.END_STOPPED
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
