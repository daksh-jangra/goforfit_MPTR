"""
Logic checks for the Dumbbell Press counter — no camera, no participant.
=========================================================================

WHY THIS FILE EXISTS
--------------------
  The score depends on things the coach can't see go wrong: whether a press
  reached the extended position, whether the arm came back down before the
  next one, which arm is working, and the 4-second rest that ends the test.
  Break one and nothing looks wrong — the count just comes out low, or the
  test ends early on a student who was still working.

  Everything here drives DumbbellPressTest directly with made-up arm
  readings and an injected clock.

HOW TO RUN IT
-------------
      python3 tests/test_dumbbell_press.py
      pytest tests/test_dumbbell_press.py

WHAT IT DOES NOT COVER
----------------------
  Landmark extraction and the HUD.
"""

import os
import sys
import random

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import dumbbell_press as dp  # noqa: E402

L, R = dp.ARM_LEFT, dp.ARM_RIGHT

# Wrist height above the shoulder, in upper arms — from the manual's photos
START  = 0.0     # weight at the shoulder (fig 5.9a)
MID    = 1.0     # on the way up
TOP    = 2.0     # extended overhead (fig 5.9b)
IDLE   = -1.4    # the other arm resting on the armrest
HALF   = 1.1     # a press that stops short


UA_PX = 100.0   # the upper arm seen full length


def px(r):
    """A reading as the engine hands it over, (rise_px, upper_px). A plain
    number is a wrist height with the upper arm seen full length."""
    if r is None or isinstance(r, tuple):
        return r
    return (r * UA_PX, UA_PX)


def feed(test, sequence, fps, t0=0.0):
    """Drive the test through (duration_s, left_reading, right_reading) steps
    at a fixed frame rate. Returns (end_time, [(t, end_reason), ...])."""
    t = t0
    dt = 1.0 / fps
    ended = []
    for duration, left, right in sequence:
        for _ in range(max(1, int(round(duration / dt)))):
            out = test.update({L: px(left), R: px(right)}, now=t)
            if out:
                ended.append((t, out))
            t += dt
    return t, ended


def rep(arm=R, top=TOP, other=IDLE):
    """Start -> up -> back to start: one press, ~3.4 s (the manual's pace)."""
    def step(d, r):
        return (d, other, r) if arm == R else (d, r, other)
    return [step(1.0, START), step(0.6, MID), step(0.6, top),
            step(0.4, MID), step(0.8, START)]


def started(t0=0.0):
    test = dp.DumbbellPressTest()
    feed(test, [(0.5, IDLE, START)], 30, t0)
    test.start(now=t0 + 0.5)
    return test


# ── counting ─────────────────────────────────────────────────────────────────
def test_full_reps_count_at_every_frame_rate():
    for fps in (8, 10, 15, 30):
        test = started()
        feed(test, rep() * 5, fps, 0.5)
        assert test.count == 5, f"{fps} fps: counted {test.count} of 5"
        assert not test.stopped, f"{fps} fps: ended mid-set ({test.end_reason})"


def test_reps_before_start_do_not_count():
    test = dp.DumbbellPressTest()
    feed(test, rep() * 3, 30)
    assert test.count == 0 and test.arm is None
    test.start(now=11.0)
    feed(test, rep() * 2, 30, 11.0)
    assert test.count == 2, f"counted {test.count} of 2 after START"


def test_already_up_at_start_must_come_down_first():
    test = dp.DumbbellPressTest()
    feed(test, [(0.5, IDLE, TOP)], 30)
    test.start(now=0.5)
    feed(test, [(1.0, IDLE, TOP)], 30, 0.5)
    assert test.count == 0, "a weight already overhead at START counted"
    feed(test, rep(), 30, 1.5)
    assert test.count == 1


def test_half_press_does_not_count():
    test = started()
    feed(test, rep(), 30, 0.5)
    feed(test, rep(top=HALF), 30, 4.0)
    assert test.count == 1, f"a short press counted ({test.count})"
    assert test.short_presses == 1


def test_idle_arm_never_counts():
    test = started()
    feed(test, [(1.0, START, IDLE), (1.0, IDLE, IDLE)] * 4, 30, 0.5)
    assert test.count == 0 and test.arm is None


def test_just_short_of_the_line_does_not_count():
    test = started()
    feed(test, rep(top=dp.WRIST_UP_UA - 0.05), 30, 0.5)
    assert test.count == 0
    feed(test, rep(top=dp.WRIST_UP_UA + 0.05), 30, 4.0)
    assert test.count == 1


def forward_press(elbow_deg, flexion_deg, ua=UA_PX):
    """A press straight forward-and-up seen from the front: the upper arm at
    flexion_deg (0 hanging, 90 pointing at the camera, 180 straight up),
    the elbow bent to elbow_deg. Returns the (rise_px, upper_px) reading."""
    import math
    th = math.radians(flexion_deg)
    fore = math.radians(flexion_deg + 180 - elbow_deg)
    rise = ua * (-math.cos(th) - 0.85 * math.cos(fore))
    return (rise, max(1.0, ua * abs(math.cos(th))))


def test_bent_elbow_pointing_at_the_camera_does_not_count():
    # Halfway up a forward press the upper arm looks very short. Elbow still
    # at 90, hand at forehead height: measured against the on-screen upper
    # arm this read 3-6 upper arms high and counted.
    for flexion in (80, 90, 100, 110, 120):
        test = started()
        mid = forward_press(90, flexion)
        feed(test, [(1.0, IDLE, START), (0.4, IDLE, forward_press(90, 60)),
                    (1.0, IDLE, mid), (0.4, IDLE, forward_press(90, 60)),
                    (1.0, IDLE, START)], 30, 0.5)
        assert test.count == 0, f"half press at {flexion} deg flexion counted"


def press_path(a, b, secs, steps=12, other=IDLE):
    """The right arm moving smoothly between two (elbow, flexion) poses of a
    forward press, with the free arm resting (seen full length)."""
    import math
    out = []
    for k in range(1, steps + 1):
        f = (1 - math.cos(math.pi * k / steps)) / 2
        e = a[0] + (b[0] - a[0]) * f
        th = a[1] + (b[1] - a[1]) * f
        out.append((secs / steps, other, forward_press(e, th)))
    return out


def half_presses(start, top, n=4, pinned=None, other=IDLE):
    test = dp.DumbbellPressTest()
    test.pinned = pinned
    feed(test, [(2.0, other, forward_press(*start))], 20)
    test.start(now=2.0)
    seq = []
    for _ in range(n):
        seq += [(0.8, other, forward_press(*start))]
        seq += press_path(start, top, 1.5, other=other) + press_path(top, start, 1.5, other=other)
    feed(test, seq, 20, 2.0)
    return test


def test_half_press_from_an_elbow_forward_start_does_not_count():
    # Elbow pushed forward at the start: the pressing upper arm is never seen
    # full length, so measured against itself the yardstick was short and a
    # half press (elbow still at 95) read high enough to count, 4 of 4. The
    # free arm, resting, shows the real length.
    for flexion in (30, 45, 60):
        for pinned in (None, R):
            test = half_presses((70, flexion), (95, 115), pinned=pinned)
            assert test.count == 0, (f"start flexion {flexion}, pinned {pinned}: "
                                     f"{test.count} half presses counted")


def test_full_press_from_an_elbow_forward_start_still_counts():
    for flexion in (30, 45, 60):
        for pinned in (None, R):
            test = half_presses((70, flexion), (175, 165), pinned=pinned)
            assert test.count == 4, f"start flexion {flexion}, pinned {pinned}: {test.count} of 4"


def test_free_arm_looking_a_bit_longer_still_counts_the_manual_photo():
    # On the manual's photos the free arm reads up to 12% longer than the
    # pressing one; the raised photo must still count.
    start, raised = (-12.0, 80.0), (139.0, 72.0)
    free = (-150.0, 90.0)
    for pinned in (None, R):
        test = dp.DumbbellPressTest()
        test.pinned = pinned
        feed(test, [(0.5, free, start)], 15)
        test.start(now=0.5)
        feed(test, [(1.0, free, start), (1.0, free, raised), (1.0, free, start)], 15, 0.5)
        assert test.count == 1, f"pinned {pinned}: {test.count}"


def test_forward_press_to_a_straight_arm_still_counts():
    test = started()
    feed(test, [(1.0, IDLE, START), (0.4, IDLE, forward_press(90, 90)),
                (0.6, IDLE, forward_press(175, 165)), (0.4, IDLE, forward_press(90, 90)),
                (1.0, IDLE, START)], 30, 0.5)
    assert test.count == 1


def test_manual_photo_lengths_still_count():
    # fig 5.9 at 2x: upper arm 80 px at the start, 72 px raised, wrist 139 px
    # above the shoulder raised.
    start, raised = (-12.0, 80.0), (139.0, 72.0)
    test = dp.DumbbellPressTest()
    feed(test, [(0.5, IDLE, start)], 15)
    test.start(now=0.5)
    feed(test, [(1.0, IDLE, start), (1.0, IDLE, raised), (1.0, IDLE, start)], 15, 0.5)
    assert test.count == 1


def test_one_long_frame_does_not_stretch_the_upper_arm():
    test = started()
    feed(test, [(1.0, IDLE, START), (1 / 30, IDLE, (0.0, 400.0))] + rep(), 30, 0.5)
    assert test.count == 1, "a single bad length spoiled the next rep"


def test_short_burst_of_bad_lengths_does_not_hide_the_next_reps():
    # Tracking glitches come in bursts. Two or three frames in a row that
    # make the upper arm look 2-4x longer got past the 3-frame median, went
    # into the "longest upper arm" yardstick, and for 10 s every real press
    # looked too low.
    for fps, burst in ((30, 2), (30, 3), (30, 5), (15, 2), (15, 3), (8, 2), (8, 3)):
        for bad in (200.0, 400.0):
            test = started()
            end, _ = feed(test, rep(), fps, 0.5)
            end, _ = feed(test, [(0.6, IDLE, TOP)] + [(1 / fps, IDLE, (TOP * UA_PX, bad))] * burst
                          + [(0.4, IDLE, TOP), (0.8, IDLE, START)], fps, end)
            feed(test, rep() * 3, fps, end)
            # rep 1, the press up to the held TOP, then 3 more
            assert test.count == 5, f"{fps} fps, {burst} frames at {bad:.0f} px: {test.count} of 5"


def test_upper_arm_follows_the_camera_moving_away():
    # Student wheels back: everything shrinks to 60%. After the window the
    # old, longer length is forgotten and full presses count again.
    def far(v):
        return (v * 60.0, 60.0)
    test = dp.DumbbellPressTest(rest_s=60.0)     # don't end while they move
    feed(test, [(0.5, IDLE, START)], 30)
    test.start(now=0.5)
    # Both arms shrink - the free arm is a yardstick too.
    feed(test, rep() + [(11.0, far(IDLE), far(START))]
         + [(1.0, far(IDLE), far(START)), (0.6, far(IDLE), far(MID)), (0.6, far(IDLE), far(TOP)),
            (0.4, far(IDLE), far(MID)), (0.8, far(IDLE), far(START))], 30, 0.5)
    assert test.count == 2, f"counted {test.count} of 2 after moving away"


def test_jitter_at_the_top_counts_once():
    test = started()
    rng = random.Random(3)
    seq = [(1.0, IDLE, START), (0.5, IDLE, MID)]
    seq += [(1 / 30, IDLE, rng.uniform(1.4, 2.2)) for _ in range(45)]
    seq += [(0.5, IDLE, MID), (1.0, IDLE, START)]
    feed(test, seq, 30, 0.5)
    assert test.count == 1, f"one wobbly top counted {test.count} times"


def test_dropout_mid_press_still_counts_once():
    test = started()
    feed(test, [(1.0, IDLE, START), (0.3, IDLE, MID), (0.4, IDLE, None),
                (0.5, IDLE, TOP), (0.3, None, None), (0.5, IDLE, TOP),
                (0.8, IDLE, START)], 30, 0.5)
    assert test.count == 1


def ramp(a, b, secs, steps=10, other=IDLE):
    """The right hand moving smoothly from a to b, as a real arm does —
    the step sequences above jump between positions."""
    return [(secs / steps, other, a + (b - a) * k / steps) for k in range(1, steps + 1)]


def test_start_position_with_the_hand_at_chin_height_counts():
    # The elbow pushed forward at the start raises the hand. At 0.6 the
    # 0.5 "back down" line was never reached and nothing counted.
    for low in (0.0, 0.4, 0.6, 0.75):
        test = dp.DumbbellPressTest()
        feed(test, [(1.0, IDLE, low)], 30)
        test.start(now=1.0)
        seq = []
        for _ in range(4):
            seq += ramp(low, 1.85, 1.2) + [(0.3, IDLE, 1.85)] + ramp(1.85, low, 1.2) + [(0.6, IDLE, low)]
        feed(test, seq, 30, 1.0)
        assert test.count == 4, f"bottom at {low}: counted {test.count} of 4"


def test_slow_press_after_a_smooth_lowering_does_not_time_out():
    # Lowered smoothly to the shoulder, a pause, then 6 s to press back up.
    # The climb must get credit from the shoulder, not from where the hand
    # crossed the "back down" line on the way down.
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    seq = ramp(0.0, 1.85, 1.2) + [(0.3, IDLE, 1.85)] + ramp(1.85, 0.0, 1.2)
    seq += [(1.5, IDLE, 0.0)] + ramp(0.0, 1.5, 6.0, steps=30) + ramp(1.5, 0.0, 1.0)
    feed(test, seq, 30, end)
    assert not test.stopped, f"slow press timed out ({test.end_reason})"
    assert test.count == 3


def test_smooth_short_presses_still_end_the_test():
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    seq = []
    for _ in range(4):
        seq += ramp(0.0, 1.1, 1.0) + ramp(1.1, 0.0, 1.0) + [(0.5, IDLE, 0.0)]
    feed(test, seq, 30, end)
    assert test.stopped and test.end_reason == dp.END_REST
    assert test.count == 1


def test_bad_length_after_a_long_dropout_does_not_spoil_the_next_reps():
    # Out of sight longer than the 10 s window (the spotter in front), then
    # one wild upper-arm length as tracking comes back.
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    end, _ = feed(test, [(1.0, IDLE, START), (12.0, IDLE, None),
                         (1 / 30, IDLE, (0.0, 400.0))], 30, end)
    feed(test, rep() * 2, 30, end)
    assert test.count == 3, f"counted {test.count} of 3 after a long dropout"


# ── which arm ────────────────────────────────────────────────────────────────
def test_one_bad_frame_after_a_dropout_is_not_a_rep():
    # Tracking comes back (the spotter stepped past) with one wild frame.
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    feed(test, [(1.0, IDLE, START), (0.5, IDLE, None), (1 / 30, IDLE, 1.7),
                (1.0, IDLE, START)], 30, end)
    assert test.count == 1, f"a phantom rep was counted ({test.count})"


def test_one_bad_length_after_a_dropout_does_not_spoil_the_next_reps():
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    feed(test, [(1.0, IDLE, START), (0.5, IDLE, None), (1 / 30, IDLE, (0.0, 400.0))]
         + rep() * 2, 30, end)
    assert test.count == 3, f"counted {test.count} of 3 after a bad length"


def test_one_bad_frame_after_a_dropout_does_not_stall_a_slow_press():
    # A wild high frame mid-press must not become the "highest so far" a
    # slow press then has to beat to count as still working.
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    seq = [(1.0, IDLE, START), (0.4, IDLE, 0.3), (0.4, IDLE, 0.6),
           (0.3, IDLE, None), (1 / 30, IDLE, 1.25)]
    seq += [(0.8, IDLE, 0.6 + 0.1 * k) for k in range(1, 9)]
    seq += [(0.5, IDLE, TOP), (0.8, IDLE, START)]
    feed(test, seq, 30, end)
    assert not test.stopped, f"the slow press timed out ({test.end_reason})"
    assert test.count == 2


def test_first_arm_to_rep_is_locked_and_the_other_is_ignored():
    test = started()
    end, _ = feed(test, rep(arm=L), 30, 0.5)
    assert test.arm == L
    feed(test, rep(arm=R, other=START) * 3, 30, end)
    assert test.count == 1, f"the other hand added reps ({test.count})"


def test_free_hand_raised_first_takes_over_in_auto():
    # The known AUTO behaviour the arm picker exists for.
    test = started()
    end, _ = feed(test, [(1.0, START, START), (0.5, MID, START), (0.5, TOP, START),
                         (1.0, IDLE, START)], 30, 0.5)
    feed(test, rep() * 3, 30, end)
    assert test.arm == L


def test_pinned_arm_ignores_the_free_hand():
    test = dp.DumbbellPressTest()
    test.pinned = R
    feed(test, [(0.5, IDLE, START)], 30)
    test.start(now=0.5)
    feed(test, [(1.0, START, START), (0.5, MID, START), (0.5, TOP, START),
                (1.0, IDLE, START)] + rep() * 3, 30, 0.5)
    assert test.arm == R and test.count == 3, f"arm {test.arm}, {test.count} reps"
    assert not test.stopped


def test_pinned_free_hand_reps_never_count_or_lock():
    test = dp.DumbbellPressTest()
    test.pinned = R
    feed(test, [(0.5, START, START)], 30)
    test.start(now=0.5)
    feed(test, [(1.0, START, START), (0.6, TOP, START), (1.0, START, START)] * 3, 30, 0.5)
    assert test.arm is None and test.count == 0


# ── ending ───────────────────────────────────────────────────────────────────
def test_rest_after_the_last_rep_ends_at_four_seconds():
    for fps in (10, 30):
        test = started()
        end, _ = feed(test, rep() * 3, fps, 0.5)
        _, ended = feed(test, [(6.0, IDLE, START)], fps, end)
        assert test.stopped and test.end_reason == dp.END_REST, f"{fps} fps: never ended"
        # The last rep's 0.8 s at the start already counts as rest.
        t_end = ended[0][0]
        assert 2.8 <= t_end - end <= 3.6, f"{fps} fps: ended {t_end - end:.2f} s into the rest"
        assert test.count == 3


def test_no_end_before_the_first_rep():
    test = started()
    feed(test, [(20.0, IDLE, START)], 15, 0.5)
    assert not test.stopped, "ended before any rep"


def test_holding_the_weight_up_is_rest():
    test = started()
    feed(test, rep()[:3], 30, 0.5)
    feed(test, [(5.0, IDLE, TOP)], 30, 2.7)
    assert test.stopped and test.end_reason == dp.END_REST
    assert test.count == 1


def test_slow_press_that_keeps_climbing_does_not_time_out():
    # 8 s from the start position to extended, still climbing the whole way
    # — allowed for a student who needs longer.
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    slow = [(1.0, IDLE, START)]
    slow += [(0.8, IDLE, 0.15 * k) for k in range(1, 11)]
    slow += [(0.5, IDLE, TOP), (0.8, IDLE, START)]
    feed(test, slow, 30, end)
    assert not test.stopped, f"slow press ended the test ({test.end_reason})"
    assert test.count == 2


def test_slow_lowering_does_not_time_out():
    # Up, a short pause at the top, then 4.5 s to bring the weight down —
    # still working, and the next rep must still count.
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    slow = [(1.0, IDLE, START), (0.6, IDLE, MID), (0.6, IDLE, TOP), (0.5, IDLE, TOP)]
    slow += [(0.45, IDLE, TOP - 0.15 * k) for k in range(1, 11)]
    slow += [(0.5, IDLE, START)] + rep()
    feed(test, slow, 30, end)
    assert not test.stopped, f"slow lowering ended the test ({test.end_reason})"
    assert test.count == 3


def smooth(a, b, secs, fps=20, other=IDLE):
    """The right hand easing from a to b (slow start and finish, like a real
    arm), one step per frame."""
    import math
    n = max(1, round(secs * fps))
    return [(1 / fps, other, a + (b - a) * (1 - math.cos(math.pi * k / n)) / 2)
            for k in range(1, n + 1)]


def test_slow_student_who_pauses_at_the_bottom_is_not_cut_off():
    # From the stress test: lower over 2.9 s, pause at the shoulder, press
    # over 2.4 s. The end of the lowering, below the 0.8 "back down" line,
    # counted as rest (~1.5 s of it), so a 2.2 s pause - well under the 4 s
    # the protocol allows - cut the student off mid-set.
    for fps in (10, 20, 30):
        test = started()
        end, _ = feed(test, rep(), fps, 0.5)
        seq = []
        for _ in range(4):
            seq += smooth(-0.3, 1.8, 2.4, fps) + [(0.12, IDLE, 1.8)]
            seq += smooth(1.8, -0.3, 2.9, fps) + [(2.2, IDLE, -0.3)]
        feed(test, seq, fps, end)
        assert not test.stopped or test.count == 5, (
            f"{fps} fps: cut off after {test.count} reps ({test.end_reason})")
        assert test.count == 5, f"{fps} fps: {test.count} of 5"


def test_sitting_still_after_a_slow_lowering_still_ends():
    test = started()
    end, _ = feed(test, rep(), 20, 0.5)
    seq = smooth(-0.3, 1.8, 1.5) + smooth(1.8, -0.3, 2.9)
    end2, _ = feed(test, seq, 20, end)
    _, ended = feed(test, [(8.0, IDLE, -0.3)], 20, end2)
    assert ended and ended[0][1] == dp.END_REST
    # The rest is counted from when the hand gets back to shoulder height (the
    # start position), so the last bit of the lowering below the shoulder is
    # in it: ~3 s after the hand fully stops here. It was 2.6 s, counted from
    # the 0.8 "back down" line.
    assert 2.8 <= ended[0][0] - end2 <= 4.2, f"ended {ended[0][0] - end2:.1f} s after the lowering"


def test_jitter_after_a_lowering_does_not_drag_the_end_out():
    # Bug 4: with jittery tracking (as in the live engine) a frame or two of
    # jitter looked like the press or the lowering still going and kept
    # restarting the 4 s - up to 8.6 s after the hand stopped.
    for fps in (12, 20, 30):
        worst = 0.0
        for seed in range(30):
            rng = random.Random(seed)
            test = started()
            end, _ = feed(test, rep(), fps, 0.5)
            end2, _ = feed(test, smooth(0.2, 1.8, 1.5, fps) + smooth(1.8, 0.2, 1.5, fps), fps, end)
            seq = [(1 / fps, IDLE, rng.gauss(0.2, 0.1)) for _ in range(fps * 9)]
            _, ended = feed(test, seq, fps, end2)
            assert ended, f"{fps} fps, seed {seed}: never stopped by itself"
            worst = max(worst, ended[0][0] - end2)
        assert worst <= 5.5, f"{fps} fps: stopped {worst:.1f} s after the hand did (worst of 30)"


def test_stopped_partway_down_is_rest():
    test = started()
    end, _ = feed(test, rep()[:3], 30, 0.5)
    _, ended = feed(test, [(0.5, IDLE, 1.6), (6.0, IDLE, 1.0)], 30, end)
    assert test.stopped and test.end_reason == dp.END_REST
    # The drop to 1.0 is a lowering step; it needs a few frames to show as
    # real travel, so the clock starts a touch later (4.7 s).
    assert ended[0][0] - end <= 4.8, f"ended {ended[0][0] - end:.1f} s after the rep"


def test_wobble_at_the_top_still_ends():
    # Holding the weight up while the reading wobbles, as the live engine
    # does on the manual's raised photo (1.28-1.58).
    for seed in range(20):
        test = started()
        end, _ = feed(test, rep()[:2] + [(1 / 30, IDLE, 1.31)], 30, 0.5)
        rng = random.Random(seed)
        seq = [(1 / 30, IDLE, rng.uniform(1.28, 1.58)) for _ in range(30 * 6)]
        _, ended = feed(test, seq, 30, end)
        assert ended, f"seed {seed}: never ended"
        assert ended[0][0] - end <= 4.3, f"seed {seed}: ended {ended[0][0] - end:.1f} s into the hold"


def test_repeated_short_presses_end_the_test():
    # "No longer able to lift the weight with complete elbow extension".
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    feed(test, rep(top=HALF) * 4, 30, end)
    assert test.stopped and test.end_reason == dp.END_REST
    assert test.count == 1


def test_fifty_reps_ends_the_test():
    test = started()
    _, ended = feed(test, rep() * 52, 15, 0.5)
    assert test.count == 50 and test.end_reason == dp.END_MAX
    assert len(ended) == 1


def test_untracked_arm_does_not_run_the_rest_clock():
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    feed(test, [(8.0, IDLE, None)], 30, end)
    assert not test.stopped, "lost tracking ended the test"


def test_a_long_frame_gap_adds_at_most_half_a_second():
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    test.update({L: px(IDLE), R: px(START)}, now=end + 6.0)
    assert not test.stopped, "a 6 s Wi-Fi gap ended the test"
    assert test.rest_elapsed <= 1.4


def test_still_at_the_bottom_with_jitter_still_ends():
    test = started()
    end, _ = feed(test, rep(), 30, 0.5)
    rng = random.Random(7)
    seq = [(1 / 30, IDLE, rng.uniform(-0.15, 0.15)) for _ in range(30 * 7)]
    feed(test, seq, 30, end)
    assert test.stopped and test.end_reason == dp.END_REST
    assert test.count == 1


def test_live_engine_wobble_at_the_bottom_ends_near_four_seconds():
    # The live engine reads a still start position anywhere in -0.3..+0.2,
    # and the low end can be where the arm was re-armed.
    for seed in range(20):
        test = started()
        end, _ = feed(test, rep()[:4] + [(1 / 30, IDLE, -0.3)], 30, 0.5)
        rng = random.Random(seed)
        seq = [(1 / 30, IDLE, rng.uniform(-0.3, 0.2)) for _ in range(30 * 6)]
        _, ended = feed(test, seq, 30, end)
        assert ended, f"seed {seed}: never ended"
        # The lowering gets credit down to shoulder height, so a wobble right
        # there can add a fraction of a second (worst seed 4.6 s).
        assert ended[0][0] - end <= 4.8, f"seed {seed}: ended {ended[0][0] - end:.1f} s into the rest"


def test_stop_records_the_coach():
    test = started()
    end, _ = feed(test, rep() * 2, 30, 0.5)
    test.stop(now=end)
    res = test.result(now=end)
    assert res["reps"] == 2 and res["end_reason"] == dp.END_STOPPED
    assert res["arm"] == R and res["duration_sec"] > 6


def test_reset_clears_everything():
    test = started()
    feed(test, rep() * 2, 30, 0.5)
    test.reset()
    assert test.count == 0 and test.arm is None and not test.started


# ── geometry ─────────────────────────────────────────────────────────────────
def test_arm_reading_geometry():
    import types
    lms = [types.SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    # Right arm straight up: shoulder, elbow 0.1 above, wrist 0.2 above.
    for i, (x, y) in {12: (0.40, 0.50), 14: (0.40, 0.40), 16: (0.40, 0.30)}.items():
        lms[i] = types.SimpleNamespace(x=x, y=y, visibility=0.9)
    rise, upper = dp.arm_reading(lms, R, 640, 480)
    assert abs(rise / upper - 2.0) < 0.01
    assert dp.arm_reading(lms, L, 640, 480) is None      # not visible
    lms[16] = types.SimpleNamespace(x=0.40, y=-0.05, visibility=0.9)
    assert dp.arm_reading(lms, R, 640, 480) is None      # wrist off the picture


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
