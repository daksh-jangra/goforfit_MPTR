"""
Logic checks for the Modified Curl-Up counter - no camera, no participant.
==========================================================================

WHY THIS FILE EXISTS
--------------------
  The count depends on things the coach can't see on the screen: the lying
  level and the hands' rest position learned from the student, the reach
  line at the knee, the feet check within each curl-up, the 6 s gap that
  ends the test, dropouts. Get one wrong and nothing looks broken - the
  count just comes out wrong.

  Everything here drives ModifiedCurlUpTest with made-up body POINTS from a
  small side-on body model (so the geometry in measure() is exercised too)
  and an injected clock: smooth curl-ups (not step jumps), several frame
  rates, camera roll, jitter, single-frame spikes and dropouts.

HOW TO RUN IT
-------------
      python3 tests/test_modified_curl_up.py
      pytest tests/test_modified_curl_up.py

  Non-zero exit means something broke. Run it after ANY edit to
  modified_curl_up.py.
"""

import math
import os
import random
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import modified_curl_up as mcu  # noqa: E402

L, R = mcu.SIDE_LEFT, mcu.SIDE_RIGHT
FPS_LIST = (8, 15, 20, 30)
THIGH = 120.0                # pixels
LYING = -8.0                 # trunk reading lying flat (fig 5.7a: -5..-14)
UP = 22.0                    # trunk at the top of a modified curl-up (5.8b: 16-25)
REST = 0.65                  # fingertip at rest, ~10 cm short of the knee
TOP_REACH = 1.0              # fingertip at the kneecap


# ── body model: side-on points for a pose ───────────────────────────────────────
def body(trunk=LYING, reach=REST, heel_up=0.0, toe_up=0.0, roll=0.0,
         head_right=True, scale=1.0, far_foot=True, near_foot=True, far_heel_up=None,
         far_toe_up=None, standing=False):
    """read_points()-shaped dict for a student lying side-on, the LEFT side
    nearest the camera. Lengths in thigh units, trunk in degrees above the
    hip-heel line, roll = camera roll (degrees)."""
    th = THIGH * scale
    sgn = 1.0 if head_right else -1.0
    d = np.array([sgn, 0.0])                    # floor, feet -> head
    n = np.array([0.0, 1.0])                    # up (maths coords)
    if standing:
        d, n = np.array([0.0, 1.0]), np.array([-sgn, 0.0])
    hip = np.zeros(2)
    e = math.radians(trunk)
    shoulder = hip + 1.3 * th * (math.cos(e) * d + math.sin(e) * n)
    a = math.radians(40.0)                      # thigh up from the floor
    knee = hip + th * (-math.cos(a) * d + math.sin(a) * n)
    heel = hip - 1.6 * th * d
    ankle = heel + 0.12 * th * n
    toe = heel - 0.45 * th * d
    index = hip + reach * (knee - hip) + 0.05 * th * n
    wrist = index - 0.4 * th * (knee - hip) / np.linalg.norm(knee - hip)

    def foot(hu, tu):
        return heel + hu * th * n, toe + tu * th * n, ankle + hu * th * n

    def to_img(p):
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        x, y = c * p[0] - s * p[1], s * p[0] + c * p[1]
        return np.array([320.0 + x, 300.0 - y])

    nh, nt, na = foot(heel_up, toe_up)
    fh, ft, fa = foot(heel_up if far_heel_up is None else far_heel_up,
                      toe_up if far_toe_up is None else far_toe_up)
    shift = np.array([3.0, 2.0])                # the far side, a touch off
    near = {"shoulder": shoulder, "hip": hip, "knee": knee, "ankle": na,
            "heel": nh, "toe": nt, "wrist": wrist, "index": index}
    far = {k: v + shift for k, v in near.items()}
    far.update({"ankle": fa + shift, "heel": fh + shift, "toe": ft + shift})
    out = {L: {k: to_img(v) for k, v in near.items()},
           R: {k: to_img(v) for k, v in far.items()}}
    out[L]["vis"], out[R]["vis"] = 0.95, 0.70
    out[R]["index"] = None                      # far hand hidden behind the thighs
    if not near_foot:
        out[L]["heel"] = out[L]["toe"] = None
    if not far_foot:
        out[R]["heel"] = out[R]["toe"] = None
    return out


# ── movement ─────────────────────────────────────────────────────────────────────
def curl(u, rest=REST, top=TOP_REACH, up=UP, lying=LYING):
    """Body readings at fraction u (0 lying .. 1 at the top) of one curl."""
    return {"trunk": lying + (up - lying) * u, "reach": rest + (top - rest) * u}


def rep_u(t, period):
    """0 -> 1 -> 0 over one smooth curl-up lasting `period`."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * t / period)


def run(test, segments, fps=20, t0=0.0, jitter=0.0, spike_p=0.0, drop_p=0.0, seed=1):
    """segments: list of (seconds, fn(t_in_segment) -> body()|None). Returns
    the time after the last frame; test.end_t says when it ended."""
    rnd = random.Random(seed)
    t, dt = t0, 1.0 / fps
    for duration, fn in segments:
        n = max(1, int(round(duration * fps)))
        for i in range(n):
            pts = fn(i * dt)
            if pts is not None and (jitter or spike_p):
                pts = {sd: {k: (v if k == "vis" or v is None
                                else v + np.array([rnd.uniform(-jitter, jitter),
                                                   rnd.uniform(-jitter, jitter)]))
                            for k, v in p.items()} for sd, p in pts.items()}
                if spike_p and rnd.random() < spike_p:
                    k = rnd.choice(["shoulder", "index", "heel", "toe", "knee"])
                    pts[L][k] = pts[L][k] + np.array([rnd.uniform(-60, 60), rnd.uniform(-60, 60)])
            if drop_p and rnd.random() < drop_p:
                pts = None
            test.update(pts, now=t)
            t += dt
    return t


def hold(**kw):
    return lambda t: body(**kw)


def reps(n, period=3.0, rest=REST, top=TOP_REACH, up=UP, lying=LYING, **kw):
    def fn(t):
        c = curl(rep_u(t % period, period), rest, top, up, lying)
        return body(trunk=c["trunk"], reach=c["reach"], **kw)
    return [(n * period, fn)]


def started(fps=20, settle=1.5, **kw):
    """A test with the student lying still and START pressed."""
    test = mcu.ModifiedCurlUpTest()
    t = run(test, [(settle, hold(**kw))], fps=fps)
    test.start(now=t)
    return test, t


# ── checks ───────────────────────────────────────────────────────────────────────
def check_lying_learned_before_start():
    test, _ = started()
    assert test.phase == mcu.PH_DOWN, test.phase
    assert abs(test.lying_ref - LYING) < 0.5, test.lying_ref
    # The hand lies on top of the thigh: it reads a touch further along.
    assert abs(test.rest_reach - REST) < 0.05, test.rest_reach


def check_clean_reps_counted_exactly():
    for fps in FPS_LIST:
        for seed in range(5):
            test, t = started(fps)
            run(test, reps(10), fps=fps, t0=t, jitter=2.0, seed=seed)
            assert test.count == 10, (fps, seed, test.count, test.not_counted)
            assert test.total_not_counted == 0, (fps, seed, test.not_counted)


def check_fast_and_slow_cadence():
    for fps in FPS_LIST:
        for period in (1.5, 5.0):
            test, t = started(fps)
            run(test, reps(6, period=period) + [(0.5, hold())], fps=fps, t0=t, jitter=1.5)
            assert test.count == 6 and not test.stopped, (fps, period, test.count, test.end_reason)


def check_gap_ends_test_6s_after_last_rep():
    for fps in FPS_LIST:
        test, t = started(fps)
        t = run(test, reps(3), fps=fps, t0=t)
        last_down = t - 0.45           # trunk back within 6 deg of lying
        run(test, [(10.0, hold())], fps=fps, t0=t)
        assert test.stopped and test.end_reason == mcu.END_GAP, (fps, test.end_reason)
        assert test.count == 3
        gap = test.end_t - last_down
        assert 5.8 <= gap <= 6.7, (fps, gap)


def check_no_curl_after_start_ends_at_6s():
    test, t = started()
    run(test, [(8.0, hold())], t0=t)
    assert test.stopped and test.end_reason == mcu.END_GAP
    assert 5.9 <= test.end_t - t <= 6.2, test.end_t - t
    assert test.count == 0


def check_short_curl_not_counted_and_keeps_going():
    for fps in FPS_LIST:
        test, t = started(fps)
        # Fingertips stop at 0.78 of the thigh: well short of the kneecap.
        run(test, reps(5, top=0.78), fps=fps, t0=t, jitter=1.5)
        assert test.count == 0, (fps, test.count)
        assert test.not_counted["short"] == 5, (fps, test.not_counted)
        assert not test.stopped, (fps, test.end_reason)


def check_reach_needs_slide_from_rest():
    # Hands resting far forward (0.80): reaching 0.88 is past the line but
    # only 0.08 of slide - not counted; 0.95 (0.15 slide) counts.
    test, t = started(reach=0.80)
    t = run(test, reps(3, rest=0.80, top=0.88), t0=t)
    assert test.count == 0 and test.not_counted["short"] == 3, (test.count, test.not_counted)
    run(test, reps(3, rest=0.80, top=0.95), t0=t)
    assert test.count == 3, test.count


def check_arms_straightening_while_lying_counts_nothing():
    # Elbows bent at rest (hands near the hips), then straightened down the
    # thighs to the knees with no curl: hands slide, trunk stays down.
    test, t = started(reach=0.30)
    def fn(tt):
        u = rep_u(tt % 3.0, 3.0)
        return body(trunk=LYING, reach=0.30 + 0.75 * u)
    run(test, [(9.0, fn)], t0=t)
    assert test.count == 0 and test.total_not_counted == 0, (test.count, test.not_counted)


def check_feet_off_floor_not_counted():
    for fps in FPS_LIST:
        test, t = started(fps)
        def fn(tt):
            u = rep_u(tt % 3.0, 3.0)
            c = curl(u)
            lift = 0.35 * u          # whole foot up ~12 cm at the top
            return body(trunk=c["trunk"], reach=c["reach"], heel_up=lift, toe_up=lift)
        run(test, [(12.0, fn)], fps=fps, t0=t, jitter=1.5)
        assert test.count == 0, (fps, test.count)
        assert test.not_counted["feet"] == 4, (fps, test.not_counted)


def check_heel_only_lift_still_counts():
    test, t = started()
    def fn(tt):
        u = rep_u(tt % 3.0, 3.0)
        c = curl(u)
        return body(trunk=c["trunk"], reach=c["reach"], heel_up=0.35 * u, toe_up=0.0)
    run(test, [(12.0, fn)], t0=t, jitter=1.5)
    assert test.count == 4 and test.total_not_counted == 0, (test.count, test.not_counted)


def check_feet_blip_shorter_than_fault_min_counts():
    # One frame of both feet "up" (a tracking blip) must not cost the rep.
    test, t = started(fps=20)
    def fn(tt):
        u = rep_u(tt % 3.0, 3.0)
        c = curl(u)
        up = 0.4 if abs(tt % 3.0 - 1.5) < 0.04 else 0.0
        return body(trunk=c["trunk"], reach=c["reach"], heel_up=up, toe_up=up)
    run(test, [(9.0, fn)], fps=20, t0=t)
    assert test.count == 3, (test.count, test.not_counted)


def check_far_foot_used_when_near_hidden_and_switch_is_safe():
    # Near foot hidden for the whole test: the far foot is judged.
    test, t = started(near_foot=False)
    def lifted(tt):
        u = rep_u(tt % 3.0, 3.0)
        c = curl(u)
        return body(trunk=c["trunk"], reach=c["reach"], heel_up=0.35 * u, toe_up=0.35 * u,
                    near_foot=False)
    run(test, [(6.0, lifted)], t0=t)
    assert test.count == 0 and test.not_counted["feet"] == 2, (test.count, test.not_counted)
    # The near foot drops in and out mid-rep: each foot against its own
    # rest, so the switch is not a "lift".
    test, t = started()
    def flicker(tt):
        u = rep_u(tt % 3.0, 3.0)
        c = curl(u)
        return body(trunk=c["trunk"], reach=c["reach"], near_foot=int(tt * 4) % 2 == 0)
    run(test, [(12.0, flicker)], t0=t)
    assert test.count == 4 and test.total_not_counted == 0, (test.count, test.not_counted)


def check_feet_not_in_view_still_counts_with_warning():
    test, t = started(near_foot=False, far_foot=False)
    assert test.problem and "Feet not in view" in test.problem, test.problem
    run(test, reps(4, near_foot=False, far_foot=False), t0=t)
    assert test.count == 4, test.count


def check_not_lying_back_down():
    # Reach, sink only part way (trunk stays 12 deg up, hands slide back),
    # reach again, then lie down properly.
    test, t = started()
    def fn(tt):
        if tt < 1.5:
            u = rep_u(tt, 3.0)                        # up to the top
        elif tt < 4.5:
            u = 1.0 - 0.6 * rep_u(tt - 1.5, 3.0)      # down to 40 % and back up
        else:
            u = rep_u(tt - 3.0, 3.0)                  # finish: down to lying
        c = curl(u)
        return body(trunk=c["trunk"], reach=c["reach"])
    run(test, [(6.5, fn), (1.0, hold())], t0=t)
    assert test.not_counted["not_down"] == 1, test.not_counted
    assert test.count == 1, test.count


def check_shaky_fingertip_is_not_a_second_reach():
    # The fingertip point wobbles while the student lies back down. A bob
    # back over the knee line used to read as a second reach ("didn't lie
    # back down"), and that reject wiped the curl-up's feet record - so a
    # curl-up with the feet off the floor could be counted.
    for seed in range(10):
        rnd = random.Random(seed)
        test, t = started()
        def fn(tt, period=4.0):
            u = rep_u(tt % period, period)
            c = curl(u)
            lift = 0.35 * u if (tt // period) % 2 == 0 else 0.0   # every other one: feet off
            p = body(trunk=c["trunk"], reach=c["reach"], heel_up=lift, toe_up=lift)
            p[L]["index"] = p[L]["index"] + np.array([rnd.uniform(-7, 7), rnd.uniform(-7, 7)])
            return p
        run(test, [(24.0, fn)], t0=t, seed=seed)
        assert test.count == 3 and test.not_counted["feet"] == 3, (seed, test.count, test.not_counted)
        assert test.not_counted["not_down"] == 0, (seed, test.not_counted)


def check_lying_level_does_not_creep():
    # Lies back only 5 deg short every time: counts (within 6 deg), and the
    # lying level stays where it was learned...
    test, t = started()
    t = run(test, reps(4, lying=LYING + 5.0), t0=t)
    assert test.count == 4, test.count
    assert abs(test.lying_ref - LYING) < 0.5, test.lying_ref
    # ...so stopping 10 deg short is never "back down".
    run(test, reps(3, lying=LYING + 10.0), t0=t)
    assert test.count == 4, test.count


def check_camera_roll_and_distance():
    for roll in (-25.0, -10.0, 0.0, 15.0, 25.0):
        for scale in (0.6, 1.0, 1.6):
            for head_right in (True, False):
                kw = dict(roll=roll, scale=scale, head_right=head_right)
                test, t = started(**kw)
                run(test, reps(5, **kw), t0=t, jitter=1.0)
                assert test.count == 5, (roll, scale, head_right, test.count, test.not_counted)


def check_spikes_and_dropouts():
    for fps in (15, 30):
        for seed in range(8):
            test, t = started(fps)
            run(test, reps(10), fps=fps, t0=t, jitter=1.5, spike_p=0.03, drop_p=0.05, seed=seed)
            assert test.count == 10, (fps, seed, test.count, test.not_counted)


def check_max_75_ends_test():
    test, t = started(fps=10)
    run(test, reps(80, period=2.0), fps=10, t0=t)
    assert test.count == 75 and test.stopped and test.end_reason == mcu.END_MAX, \
        (test.count, test.end_reason)


def check_start_while_curled_must_lie_down_first():
    test = mcu.ModifiedCurlUpTest()
    t = run(test, [(1.0, hold())], fps=20)
    t = run(test, [(0.8, lambda tt: body(**curl(0.9)))], t0=t)
    test.start(now=t)
    assert test.phase == mcu.PH_UNKNOWN
    # Comes back down from the curl: not a curl-up of its own.
    t = run(test, [(1.0, lambda tt: body(**curl(max(0.0, 0.9 - tt))))], t0=t)
    assert test.count == 0 and test.total_not_counted == 0, (test.count, test.not_counted)
    run(test, [(0.5, hold())] + reps(2), t0=t)
    assert test.count == 2, test.count


def check_standing_is_not_in_position():
    test = mcu.ModifiedCurlUpTest()
    run(test, [(1.0, hold(standing=True))])
    assert not test.in_position and test.phase == mcu.PH_UNKNOWN
    assert test.problem and "Lie on your back" in test.problem


def check_nothing_counted_before_start():
    test = mcu.ModifiedCurlUpTest()
    t = run(test, [(1.0, hold())])
    t = run(test, reps(3), t0=t)
    assert test.count == 0 and test.total_not_counted == 0 and not test.started
    test.start(now=t)
    run(test, reps(2), t0=t)
    assert test.count == 2


def check_side_switch_keeps_counting():
    # The far side becomes the clearer one (student shifted): readings move
    # to it and the count carries on.
    test, t = started()
    t = run(test, reps(2), t0=t)
    def far_clear(tt):
        c = curl(rep_u(tt % 3.0, 3.0))
        p = body(trunk=c["trunk"], reach=c["reach"])
        p[R]["index"] = p[L]["index"] + np.array([3.0, 2.0])
        p[L]["vis"], p[R]["vis"] = 0.6, 0.95
        return p
    run(test, [(6.0, far_clear)], t0=t)
    assert test.count == 4, (test.count, test.not_counted)


def check_stop_and_reset():
    test = mcu.ModifiedCurlUpTest()
    test.stop(now=1.0)
    assert not test.stopped            # never started: nothing to stop
    test, t = started()
    t = run(test, reps(3), t0=t)
    test.stop(now=t)
    t = run(test, reps(2), t0=t)
    assert test.count == 3 and test.end_reason == mcu.END_STOPPED
    test.reset()
    assert test.count == 0 and not test.started and not test.stopped
    assert test.total_not_counted == 0 and test.lying_ref is None


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
