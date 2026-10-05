"""
Logic checks for the Seated Push-Up tracker - no camera, no participant.
============================================================================

WHY THIS FILE EXISTS
--------------------
  The score is a clock, and the clock's start and stop depend on things the
  operator cannot see: where the seat was taken, a settle window, a break
  grace, dropouts, the 20 s cap. Get one of those wrong and nothing looks
  broken - the feed streams, the HUD ticks - the number just comes out wrong.

  Everything here drives SeatedPushUpTracker directly with made-up body
  readings and an injected clock, so it runs in about a second.

HOW TO RUN IT
-------------
      python3 tests/test_seated_pushup.py
      pytest tests/test_seated_pushup.py

  Non-zero exit means something broke. Run it after ANY edit to
  seated_pushup.py - especially to the thresholds near the top.
"""

import os
import random
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import seated_pushup as sp  # noqa: E402

TORSO = 200.0          # pixels, mid-shoulder to mid-hip
SEAT_Y = 600.0         # mid-hip height while sitting, pixels
HAND_Y = 620.0         # the hands, planted on the supports a little lower
KEEP = object()        # "the hand is in view" (default)


def body(lift=0.0, elbow_left=150.0, elbow_right=150.0, torso=TORSO,
         wrist_left=KEEP, wrist_right=KEEP, camera=0.0):
    """One frame's readings with the hips `lift` torso lengths above the seat.
    A hand is in view unless its wrist_* is None; an arm whose elbow is None
    is hidden but its hand may still be seen. `camera` shifts the whole
    picture down by that many pixels, as a bumped phone would."""
    def hand(v):
        return None if v is None else (HAND_Y if v is KEEP else v) + camera
    return {"hip_y": SEAT_Y - lift * TORSO + camera, "torso": torso,
            "wrist_left": hand(wrist_left), "wrist_right": hand(wrist_right),
            "elbow_left": elbow_left, "elbow_right": elbow_right}


SIT = body(0.0, 120.0, 120.0)       # wheelchair: elbows bent while sitting
UP = body(0.30, 176.0, 178.0)       # up, arms straight (the photos: 176-178)


def feed(tracker, sequence, fps=20, t0=0.0, jitter=0.0, seed=1):
    """Drive the tracker through (seconds, body_or_None) steps. Returns
    (end_time, first_result). jitter: +/- degrees on the elbows, and
    +/- hundredths of a torso on the hips and each hand."""
    rnd = random.Random(seed)
    t, dt, first = t0, 1.0 / fps, None
    for duration, b in sequence:
        for _ in range(max(1, int(round(duration / dt)))):
            f = b
            if f is not None and jitter:
                f = dict(f)
                for k in ("hip_y", "wrist_left", "wrist_right"):
                    if f[k] is not None:
                        f[k] += rnd.uniform(-jitter, jitter) * TORSO / 100
                for k in ("elbow_left", "elbow_right"):
                    if f[k] is not None:
                        f[k] += rnd.uniform(-jitter, jitter)
            out = tracker.update(f, now=t)
            if out and first is None:
                first = out
            t += dt
    return t, first


def check_full_hold_caps_at_20():
    for fps in (10, 20, 30):
        tr = sp.SeatedPushUpTracker()
        _, res = feed(tr, [(2, SIT), (40, UP)], fps=fps, jitter=3)
        assert res and res["hold_sec"] == 20 and res["capped"], (fps, res)


def check_settle_second_is_credited():
    # Sit 2 s, up at t=2, back down at t=9: the hold ran from t=2 to t=9.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (7, UP), (2, SIT)])
    assert res and res["hold_sec"] == 7, res
    assert res["end_reason"] == sp.BREAK_WORDS["sink"], res


def check_each_break_rule():
    cases = {
        "sink":  body(0.05, 176.0, 178.0),
        "elbow": body(0.30, 150.0, 178.0),          # one arm bent 26 deg
        "stood": body(1.20, 176.0, 178.0),
    }
    for why, bad in cases.items():
        tr = sp.SeatedPushUpTracker()
        _, res = feed(tr, [(2, SIT), (5, UP), (2, bad)])
        assert res and res["end_reason"] == sp.BREAK_WORDS[why], (why, res)
        assert res["hold_sec"] == 5, (why, res)


def check_straight_arms_while_sitting_need_the_lift():
    # Push-up blocks: the arms are already straight sitting down. That alone
    # must never start the clock - only the hips leaving the seat does.
    tr = sp.SeatedPushUpTracker()
    t, res = feed(tr, [(10, body(0.0, 178.0, 178.0))], jitter=3)
    assert res is None and tr.hold_start is None, tr.state
    _, res = feed(tr, [(3, body(0.20, 178.0, 178.0)), (2, body(0.0, 178.0, 178.0))], t0=t)
    assert res and res["hold_sec"] == 3, res


def check_small_lift_does_not_start():
    # ~1.7 cm: shifting in the seat, not a push-up.
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (10, body(0.04, 176.0, 178.0))], jitter=2)
    assert tr.hold_start is None, tr.state
    # ~3 cm, held dead steady: high enough on screen (>= 0.06), so only the
    # lift minimum itself keeps the clock from starting.
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (10, body(0.07, 176.0, 178.0))])
    assert tr.hold_start is None, tr.state


def check_five_cm_lift_always_starts():
    # A real 5 cm lift (0.12 torso) must start the clock every time, wobble
    # and all - with the lift minimum AT 0.12 it did ~1 time in 10.
    for seed in range(10):
        for fps in (10, 20, 30):
            tr = sp.SeatedPushUpTracker()
            feed(tr, [(2, SIT), (6, body(0.12, 176.0, 178.0))], fps=fps, jitter=3, seed=seed)
            assert tr.hold_start is not None, (seed, fps, tr.status_msg)


def check_sitting_still_never_starts():
    # A minute of sitting still with normal wobble never starts the clock.
    for seed in range(10):
        for fps in (10, 20, 30):
            tr = sp.SeatedPushUpTracker()
            feed(tr, [(60, SIT)], fps=fps, jitter=3, seed=seed)
            assert tr.hold_start is None, (seed, fps)


def check_messages_before_and_during_the_hold():
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT)])
    assert "already up? sit down first" in tr.status_msg, tr.status_msg
    assert tr.sink_line is None
    feed(tr, [(4, UP)], t0=2.0)
    assert tr.state == sp.State.HOLDING and tr.sink_line is not None, tr.state


def check_standing_up_never_starts():
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (10, body(1.10, 176.0, 178.0))])
    assert tr.hold_start is None, tr.state


def check_bent_arm_best_is_allowed():
    # A student who can only reach 140 deg: timed from their best, and only
    # a further bend from there ends it.
    best = body(0.20, 140.0, 142.0)
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (6, best), (2, body(0.20, 115.0, 142.0))])
    assert res and res["hold_sec"] == 6, res
    assert res["end_reason"] == sp.BREAK_WORDS["elbow"], res


def check_own_baseline_used():
    # A slow drop that stays above 35% of the starting lift and within 20 deg
    # of the starting elbows is still holding.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (4, UP), (20, body(0.15, 162.0, 164.0))])
    assert res and res["capped"], res


def check_seat_is_the_lowest_still_sit():
    # Already part-way up (0.20) when the camera first sees them still: that
    # reading becomes a provisional seat. Sitting properly replaces it, so the
    # real lift counts rather than being lost.
    tr = sp.SeatedPushUpTracker()
    t, _ = feed(tr, [(2, body(0.20, 170.0, 170.0)), (2, SIT)])
    assert abs(tr.seat_y - SEAT_Y) < 1.0, tr.seat_y     # drawn where the hips sit
    _, res = feed(tr, [(5, UP), (2, SIT)], t0=t)
    assert res and res["hold_sec"] == 5, res


def check_restless_sit_finds_no_seat():
    tr = sp.SeatedPushUpTracker()
    t = 0.0
    for i in range(80):            # hips bouncing 0.2 torso every few frames
        tr.update(body(0.20 * ((i // 3) % 2)), now=t)
        t += 0.05
    assert tr.seat_y is None and tr.hold_start is None, tr.seat_y


def check_one_shaky_frame_does_not_end_it():
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (0.1, SIT), (20, UP)])
    assert res and res["capped"], res


def check_jitter_does_not_end_a_hold():
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (30, UP)], jitter=3, seed=7)
    assert res and res["capped"], res


def check_short_dropout_survives_long_one_ends():
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (0.4, None), (20, UP)])
    assert res and res["capped"], res
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (3, None), (5, UP)])
    assert res and res["hold_sec"] == 5, res
    assert "lost sight" in res["end_reason"], res


def check_one_arm_lost_still_holds():
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (4, UP), (20, body(0.30, 176.0, None))])
    assert res and res["capped"], res
    # ... and with that hand gone too, the other hand carries the lift.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (4, UP), (20, body(0.30, 176.0, None, wrist_right=None))])
    assert res and res["capped"], res


def check_camera_shift_is_not_a_lift():
    # Fix 1. The phone is bumped after the seat is learned: the whole picture
    # moves 30 px (0.15 torso) but the student never pushes. Measured against
    # the hands, nothing moved - the clock must not start.
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (6, body(0.0, 120.0, 120.0, camera=-30.0))])
    assert tr.hold_start is None, (tr.state, tr.lift)
    # Nor may a bump in the other direction mid-hold end a good hold.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (4, UP), (20, body(0.30, 176.0, 178.0, camera=40.0))])
    assert res and res["capped"], res


def check_hands_at_different_heights():
    # Uneven supports (or a tilted phone): the right hand sits 40 px lower.
    # Losing that hand mid-hold must not move the lift reading.
    low = HAND_Y + 40.0
    sit = body(0.0, 120.0, 120.0, wrist_right=low)
    up = body(0.30, 176.0, 178.0, wrist_right=low)
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, sit), (4, up), (20, body(0.30, 176.0, 178.0, wrist_right=None))])
    assert res and res["capped"], res
    assert abs(tr.lift - 0.30) < 0.01, tr.lift


def check_no_hands_no_lift():
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(3, body(0.0, None, None, wrist_left=None, wrist_right=None))])
    assert tr.seat is None and tr.hold_start is None
    # Seat known, then both hands vanish mid-hold: that is losing the student.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (3, body(0.30, None, None, wrist_left=None,
                                                    wrist_right=None))])
    assert res and res["hold_sec"] == 5 and "lost sight" in res["end_reason"], res


def check_no_arm_in_view_never_starts():
    # Fix 2. Hands and hips seen, but no elbow on either side: the hips lift,
    # yet with no arm to check the clock must wait.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, body(0.0, None, None)), (8, body(0.30, None, None))])
    assert res is None and tr.hold_start is None, tr.state
    assert tr.status_msg == "Keep both arms in view", tr.status_msg


def check_all_arms_lost_mid_hold_ends_it():
    # Fix 2, mid-hold: the arms that have marks both vanish - the hold can no
    # longer be checked, so after the dropout grace it ends where last seen.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (3, body(0.30, None, None))])
    assert res and res["hold_sec"] == 5 and "lost sight" in res["end_reason"], res


def check_bend_back_from_best():
    # Fix 3. Starts at 140 deg while still pushing, straightens to 176, then
    # bends back to 150 - 26 deg off the best. That ends it, when the bend
    # began (t = 2 + 3 + 3 = 8 -> 6 s held).
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (3, body(0.20, 140.0, 140.0)),
                       (3, body(0.25, 176.0, 176.0)), (14, body(0.22, 150.0, 150.0))])
    assert res and res["end_reason"] == sp.BREAK_WORDS["elbow"], res
    assert res["hold_sec"] == 6, res
    assert tr.baseline["elbow_left"] >= 175.0, tr.baseline


def check_one_odd_frame_does_not_raise_the_mark():
    # Fix 3. A single spiky reading as an arm comes back into view must not
    # become the new best - or an ordinary angle would then look like a bend.
    tr = sp.SeatedPushUpTracker()
    t, _ = feed(tr, [(2, SIT), (4, UP), (0.5, body(0.30, None, 178.0))])
    t, _ = feed(tr, [(0.05, body(0.30, 179.0, 178.0))], t0=t)     # back in view
    assert tr.baseline["elbow_left"] < 177.0, tr.baseline
    _, res = feed(tr, [(20, UP)], t0=t)
    assert res and res["capped"], res


def stand(elbow=175.0):
    """Stood up: hips a whole torso higher on screen, hands off the supports
    hanging ~0.35 torso below the hips - so against the hands it reads as an
    ordinary-looking lift."""
    hip = SEAT_Y - 1.0 * TORSO
    return {"hip_y": hip, "torso": TORSO,
            "wrist_left": hip + 0.35 * TORSO, "wrist_right": hip + 0.35 * TORSO,
            "elbow_left": elbow, "elbow_right": elbow}


def check_standing_up_is_caught_on_screen():
    # Hands at the side, low on the blocks.
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (8, stand())])
    assert tr.hold_start is None, (tr.state, tr.lift)
    assert tr.status_msg == "Stay seated - push up with the arms", tr.status_msg
    # Wheelchair armrests: hands ABOVE the hips while sitting.
    armrest = body(0.0, 100.0, 100.0, wrist_left=SEAT_Y - 50, wrist_right=SEAT_Y - 50)
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, armrest), (8, stand())])
    assert tr.hold_start is None, (tr.state, tr.lift)


def check_standing_up_mid_hold_ends_it():
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (4, UP), (3, stand())])
    assert res and res["end_reason"] == sp.BREAK_WORDS["stood"] and res["hold_sec"] == 4, res
    # ... even when the hands leave the picture as they stand.
    gone = dict(stand(), wrist_left=None, wrist_right=None, elbow_left=None, elbow_right=None)
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (4, UP), (3, gone)])
    assert res and res["end_reason"] == sp.BREAK_WORDS["stood"] and res["hold_sec"] == 4, res


def check_arms_swapping_view_mid_hold():
    # Only the right arm is seen as the clock starts; mid-hold it goes out of
    # view and the left arm comes in. The left arm takes its own mark, so the
    # hold carries on - and a bend in it still ends the hold.
    tr = sp.SeatedPushUpTracker()
    t, res = feed(tr, [(2, body(0.0, None, 120.0)), (4, body(0.30, None, 176.0)),
                       (6, body(0.30, 176.0, None))])
    assert res is None and tr.state == sp.State.HOLDING, (res, tr.state)
    assert tr.baseline["elbow_left"] is not None, tr.baseline
    _, res = feed(tr, [(2, body(0.30, 150.0, None))], t0=t)
    assert res and res["end_reason"] == sp.BREAK_WORDS["elbow"] and res["hold_sec"] == 10, res


def check_hands_moving_onto_supports_is_not_a_lift():
    # The seat is learned with the hands resting in the lap, level with the
    # hips; then the hands go down onto the rims / blocks. The hips never move
    # - the clock must not start. Then a real push from there is timed, and
    # sitting back down still ends it.
    lap = body(0.0, 90.0, 90.0, wrist_left=SEAT_Y, wrist_right=SEAT_Y)
    for drop in (0.20, 0.30, 0.50):
        low = SEAT_Y + drop * TORSO
        rims = lambda lift, el=170.0: body(lift, el, el, wrist_left=low, wrist_right=low)
        tr = sp.SeatedPushUpTracker()
        t, _ = feed(tr, [(2, lap), (6, rims(0.0))])
        assert tr.hold_start is None, (drop, tr.state, tr.lift)
        _, res = feed(tr, [(5, rims(0.30, 176.0)), (2, rims(0.0))], t0=t)
        assert res and res["hold_sec"] == 5, (drop, res)
        assert res["end_reason"] == sp.BREAK_WORDS["sink"], (drop, res)


def check_small_regrip_then_sitting_down_stops_the_clock():
    # Push-up blocks: arms straight both sitting and lifted. After the seat is
    # learned the hands settle a little lower (a re-grip, 0.04-0.09 torso -
    # too small to look like a lift on its own). Then a small push, then the
    # student sits back down with the arms still straight. Left uncorrected,
    # the re-grip reads as lift and the clock ran on while they sat.
    for drop in (0.05, 0.07, 0.09):
        low = HAND_Y + drop * TORSO
        b = lambda lift: body(lift, 176.0, 176.0, wrist_left=low, wrist_right=low)
        for lift in (0.20, 0.25):     # clear of the lift minimum, so the
            for seed in range(5):      # start is not late from wobble alone
                tr = sp.SeatedPushUpTracker()
                _, res = feed(tr, [(2, body(0.0, 176.0, 176.0)), (3, b(0.0)),
                                   (5, b(lift)), (6, b(0.0))], jitter=3, seed=seed)
                assert res and res["end_reason"] == sp.BREAK_WORDS["sink"], (drop, lift, seed, res)
                assert 4 <= res["hold_sec"] <= 5, (drop, lift, seed, res)
        # The bug itself lives at SMALL lifts, where the re-grip is a big
        # share of the reading. Whenever the clock started there, it must
        # not still be running after they have sat down for 12 s.
        for lift in (0.12, 0.15):
            for seed in range(10):
                tr = sp.SeatedPushUpTracker()
                feed(tr, [(2, body(0.0, 176.0, 176.0)), (3, b(0.0)),
                          (5, b(lift)), (12, b(0.0))], jitter=3, seed=seed)
                running = tr.state == sp.State.HOLDING or (tr.resolved and tr.capped)
                assert not running, (drop, lift, seed, tr.state, round(tr.elapsed_s, 1))
    # The corner left after that: a re-grip just over the re-learn size, which
    # wobble can hide, followed by a very small push (3.4-5 cm). The stop line
    # must sit above anything the re-learn can miss.
    for drop in (0.03, 0.04, 0.05, 0.06, 0.07):
        low = HAND_Y + drop * TORSO
        b = lambda lift: body(lift, 176.0, 176.0, wrist_left=low, wrist_right=low)
        for lift in (0.08, 0.10, 0.12):
            for seed in range(30):
                tr = sp.SeatedPushUpTracker()
                feed(tr, [(2, body(0.0, 176.0, 176.0)), (3, b(0.0)),
                          (5, b(lift)), (12, b(0.0))], jitter=3, seed=seed)
                running = tr.state == sp.State.HOLDING or (tr.resolved and tr.capped)
                assert not running, (drop, lift, seed, tr.state, round(tr.elapsed_s, 1))


def check_screen_rise_is_a_backstop():
    # Re-learning the seat after the hands move needs a still second, and so
    # does starting the clock - today the re-learn wins the tie. Make the seat
    # slower to learn than the clock is to start: the on-screen rise rule
    # alone must still stop the hands' move from starting the clock.
    lap = body(0.0, 90.0, 90.0, wrist_left=SEAT_Y, wrist_right=SEAT_Y)
    low = SEAT_Y + 0.30 * TORSO
    rims = body(0.0, 170.0, 170.0, wrist_left=low, wrist_right=low)
    tr = sp.SeatedPushUpTracker(seat_still_s=2.0)
    feed(tr, [(3, lap), (6, rims)])
    assert tr.hold_start is None, (tr.state, tr.lift)


def check_bumped_phone_then_a_real_lift():
    # The phone is knocked so the picture drops 30 px while the student sits.
    # Their real lift afterwards must still start the clock - the on-screen
    # seat follows the bump once they have sat still for a moment.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (2, body(0.0, 120.0, 120.0, camera=30.0)),
                       (5, body(0.30, 176.0, 178.0, camera=30.0)),
                       (2, body(0.0, 120.0, 120.0, camera=30.0))])
    assert res and res["hold_sec"] == 5, res


def check_stood_still_before_sitting():
    # Standing still in front of the camera first (hands hanging about as low
    # against the hips as they will rest when seated), then sitting down: the
    # on-screen seat must move down to where they sit, so a lift starts the
    # clock and standing up mid-hold reads as standing up.
    low = SEAT_Y + 0.35 * TORSO
    sat = lambda lift, el=160.0: body(lift, el, el, wrist_left=low, wrist_right=low)
    tr = sp.SeatedPushUpTracker()
    t, _ = feed(tr, [(2, stand()), (2, sat(0.0))])
    assert abs(tr.seat_hip_y - SEAT_Y) < 2.0, tr.seat_hip_y
    _, res = feed(tr, [(4, sat(0.30, 176.0)), (3, stand())], t0=t)
    assert res and res["hold_sec"] == 4 and res["end_reason"] == sp.BREAK_WORDS["stood"], res


def check_wifi_stall_keeps_a_good_hold():
    # The phone's frames stop for up to 2 s (school Wi-Fi) while the student
    # stays up: the hold carries on - and the unseen time counts, because
    # they are seen still up afterwards.
    for stall in (0.8, 1.5, 1.9):
        tr = sp.SeatedPushUpTracker()
        t, _ = feed(tr, [(2, SIT), (5, UP)])
        _, res = feed(tr, [(20, UP)], t0=t + stall)
        assert res and res["capped"], (stall, res)
    # Tracking lost (frames still arriving) for 1.5 s - the same.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (1.5, None), (20, UP)])
    assert res and res["capped"], res


def check_long_stall_stops_at_last_good_frame():
    tr = sp.SeatedPushUpTracker()
    t, _ = feed(tr, [(2, SIT), (5, UP)])
    res = tr.update(UP, now=t + 2.5)
    assert res and res["hold_sec"] == 5 and "paused" in res["end_reason"], res


def check_sat_down_during_a_stall():
    # Up for 5 s, then the frames stop for 1.5 s and when they come back the
    # student is sitting. Nothing before the gap may vouch for them: the
    # clock stops at the last frame they were seen up.
    tr = sp.SeatedPushUpTracker()
    t, _ = feed(tr, [(2, SIT), (5, UP)])
    _, res = feed(tr, [(3, SIT)], t0=t + 1.5)
    assert res and res["end_reason"] == sp.BREAK_WORDS["sink"], res
    assert abs(res["held_exact_s"] - 5.0) < 0.1, res


def check_sat_down_then_lost_from_view():
    # They sit back down and the camera loses them a moment later, before
    # the break grace is over. The clock stops at the last frame they were
    # up (not when the camera lost them), with the real reason.
    tr = sp.SeatedPushUpTracker()
    _, res = feed(tr, [(2, SIT), (5, UP), (0.25, SIT), (3, None)])
    assert res and res["end_reason"] == sp.BREAK_WORDS["sink"], res
    assert abs(res["held_exact_s"] - 5.0) < 0.1, res


def check_stop_ends_at_last_good_frame():
    # STOP pressed while a break is under way: the clock stops before it.
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (6, UP), (0.2, SIT)])
    res = tr.finish()
    assert res and abs(res["held_exact_s"] - 6.0) < 0.1, res


def check_sink_reason_wording():
    # Fix 5: the line sits above the seat, so the words must not claim they
    # sat on it.
    assert sp.BREAK_WORDS["sink"] == "came back down towards the seat"


def check_frames_stop_arriving():
    tr = sp.SeatedPushUpTracker()
    t, _ = feed(tr, [(2, SIT), (5, UP)])
    res = tr.update(UP, now=t + 3.0)
    assert res and res["hold_sec"] == 5 and "paused" in res["end_reason"], res


def check_stop_mid_hold_and_before():
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (6, UP)])
    res = tr.finish()
    assert res and res["hold_sec"] == 6 and res["end_reason"] == "stopped by the coach", res
    tr = sp.SeatedPushUpTracker()
    feed(tr, [(2, SIT), (0.5, UP)])
    assert tr.finish() is None and not tr.resolved


def check_done_is_final_until_reset():
    tr = sp.SeatedPushUpTracker()
    t, res = feed(tr, [(2, SIT), (4, UP), (2, SIT)])
    feed(tr, [(10, UP)], t0=t)
    assert tr.resolved and tr.score_s == 4, tr.score_s
    tr.reset()
    assert tr.state == sp.State.READY and tr.seat_y is None and tr.score_s == 0


def check_rounding_to_nearest_second():
    for held, want in ((4.4, 4), (4.6, 5)):
        tr = sp.SeatedPushUpTracker()
        _, res = feed(tr, [(2, SIT), (held, UP), (2, SIT)])
        assert res and res["hold_sec"] == want, (held, res)


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
