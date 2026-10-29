"""
Logic checks for the Modified Apley Test - no camera, no participant.
============================================================================

WHY THIS FILE EXISTS
--------------------
  The score depends on things nobody can see on the screen: which way the
  hand has to cross the head for each arm (from behind, not mirrored), how
  far down counts as the shoulder blade, that every level needs a full
  second, and the left -> right hand-over. Get one wrong and the feed still
  looks fine - the score is just wrong.

  Hands are built in pixels from the positions measured on the Brockport
  demo video (C0xamv81CCE), in palm lengths from the middle of the head.

HOW TO RUN IT
-------------
      python3 tests/test_modified_apley.py
      pytest tests/test_modified_apley.py

  Non-zero exit means something broke.
"""

import os
import random
import sys
from types import SimpleNamespace

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import modified_apley as ap  # noqa: E402

L, R = ap.SIDE_LEFT, ap.SIDE_RIGHT
EARS = np.array([500.0, 300.0])
PALM = 90.0


def hand(wrist, tips, palm=PALM):
    """A (21, 2) hand: wrist at `wrist`, fingers toward `tips` (both in palm
    lengths from the middle of the head, x + = picture right, y + = down).
    The middle knuckle is one palm from the wrist along that line."""
    w = EARS + np.array(wrist, dtype=float) * PALM
    t = EARS + np.array(tips, dtype=float) * PALM
    d = (t - w) / (np.hypot(*(t - w)) or 1.0)
    pts = np.zeros((21, 2))
    for i in range(21):
        pts[i] = w + d * palm * (0.6 + 0.6 * (i % 5) / 4)   # spread along the fingers
    pts[0] = w
    pts[9] = w + d * palm
    for i in ap.H_REACH_TIPS + (4, 20):
        pts[i] = t
    for i in ap.H_PALM[1:]:
        pts[i] = w + d * palm * 0.95
    return pts


# The demo video's held reach of the LEFT arm: wrist at the back of the head,
# fingertips 0.85 palms toward picture-right and 1.65 below the ears.
LEFT_REACH = dict(wrist=(0.05, -0.05), tips=(0.85, 1.65))
RIGHT_REACH = dict(wrist=(-0.25, 0.0), tips=(-1.2, 1.9))


# ── score 3: the opposite shoulder blade ─────────────────────────────────────────
def test_demo_reaches_score_3_on_the_right_arm():
    assert ap.shoulder_blade_touch(hand(**LEFT_REACH), EARS, PALM, L) == (True, None)
    assert ap.shoulder_blade_touch(hand(**RIGHT_REACH), EARS, PALM, R) == (True, None)


def test_reach_with_the_wrong_arm_is_not_3():
    ok, why = ap.shoulder_blade_touch(hand(**LEFT_REACH), EARS, PALM, R)
    assert not ok and "ACROSS" in why and "LEFT shoulder blade" in why


def test_same_side_reach_is_not_3():
    # The FITNESSGRAM-style reach down the hand's own side of the back
    # (BjO9ZtPC7Rw): the fingertips stay on the hand's own side.
    same = hand(wrist=(-1.5, 0.8), tips=(-0.9, 2.6))
    assert not ap.shoulder_blade_touch(same, EARS, PALM, L)[0]
    assert not ap.shoulder_blade_touch(hand(wrist=(0.6, 1.0), tips=(0.5, 2.8)), EARS, PALM, R)[0]


def test_hand_only_behind_the_neck_is_not_3():
    ok, why = ap.shoulder_blade_touch(hand(wrist=(0.0, -0.4), tips=(0.6, 0.8)), EARS, PALM, L)
    assert not ok and "Lower" in why


def test_hand_on_front_of_shoulder_is_not_3():
    # Facing the camera, hand on the front of the other shoulder: the wrist
    # is down at the chest.
    front = hand(wrist=(-0.2, 2.0), tips=(0.9, 2.8))
    assert ap.shoulder_blade_touch(front, EARS, PALM, L) == (False, None)


def test_hand_coming_up_the_back_is_not_3():
    up = hand(wrist=(0.2, 3.7), tips=(0.8, 1.7))
    assert ap.shoulder_blade_touch(up, EARS, PALM, L) == (False, None)


def test_lines_are_in_palms_so_size_does_not_matter():
    for scale in (0.5, 1.0, 2.0):
        h = EARS + (hand(**LEFT_REACH) - EARS) * scale
        assert ap.shoulder_blade_touch(h, EARS, PALM * scale, L)[0], scale


# ── score 2 and 1 ────────────────────────────────────────────────────────────────
def test_head_top():
    on_top = hand(wrist=(-0.9, -1.6), tips=(0.6, -1.4))
    assert ap.head_top_touch(on_top, EARS, PALM)
    forehead = hand(wrist=(-0.5, -0.2), tips=(0.4, -0.5))
    assert not ap.head_top_touch(forehead, EARS, PALM)
    overhead = hand(wrist=(0.0, -3.6), tips=(0.0, -4.6))
    assert not ap.head_top_touch(overhead, EARS, PALM)
    beside = hand(wrist=(2.4, -1.4), tips=(2.4, -2.4))
    assert not ap.head_top_touch(beside, EARS, PALM)


def test_mouth():
    mouth = EARS + np.array([0.0, 0.6]) * PALM
    at_mouth = hand(wrist=(0.3, 2.3), tips=(0.05, 0.65))
    assert ap.mouth_touch(at_mouth, mouth, PALM)
    at_chest = hand(wrist=(0.3, 3.3), tips=(0.1, 1.8))
    assert not ap.mouth_touch(at_chest, mouth, PALM)


def test_pose_fallback_matches_the_manual_photos():
    # Fig. 5.24b/c front-on, measured: head unit 63 px (shoulders to ears).
    ears, unit = np.array([180.0, 146.0]), 63.0
    on_head = {15: (206, 84), 19: (177, 79), 16: (152, 370), 20: (181, 371)}
    assert ap.pose_head_top(on_head, ears, unit)
    hand_down = {15: (206, 300), 19: (190, 330)}
    assert not ap.pose_head_top(hand_down, ears, unit)
    ears_c, unit_c, mouth_c = np.array([181.5, 113.5]), 71.0, np.array([183.5, 141.5])
    at_mouth = {15: (211, 183), 19: (191, 143)}
    assert ap.pose_mouth(at_mouth, mouth_c, unit_c)
    assert not ap.pose_mouth(on_head, mouth_c, unit_c)


def test_short_looking_palm_uses_the_usual_length():
    ys = ap.PalmYardstick()
    for k in range(30):
        ys.push(k * 0.1, PALM)
    assert ys.unit(PALM * 0.4) == PALM       # edge-on hand on top of the head
    assert ys.unit(PALM * 1.2) == PALM * 1.2


# ── the meter on landmark objects ────────────────────────────────────────────────
def _pose(w=1000, h=1000, shoulder_y=420.0):
    lms = [SimpleNamespace(x=0.5, y=0.9, z=0.0, visibility=0.0) for _ in range(33)]

    def put(i, x, y, vis=1.0):
        lms[i] = SimpleNamespace(x=x / w, y=y / h, z=0.0, visibility=vis)
    put(ap.EAR_L, EARS[0] - 50, EARS[1])
    put(ap.EAR_R, EARS[0] + 50, EARS[1])
    put(ap.MOUTH_L, EARS[0] - 15, EARS[1] + 55)
    put(ap.MOUTH_R, EARS[0] + 15, EARS[1] + 55)
    put(ap.SH_L, EARS[0] - 180, shoulder_y)
    put(ap.SH_R, EARS[0] + 180, shoulder_y)
    return lms


def _hand_lms(pts, w=1000, h=1000):
    return [SimpleNamespace(x=p[0] / w, y=p[1] / h, z=0.0) for p in pts]


def test_meter_levels():
    m = ap.ApleyMeter()
    rd = m.measure(_pose(), [_hand_lms(hand(**LEFT_REACH))], 1000, 1000, L, now=0.0)
    assert rd.level == 3
    rd = m.measure(_pose(), [_hand_lms(hand(**LEFT_REACH))], 1000, 1000, R, now=0.1)
    assert rd.level == 0 and "ACROSS" in rd.problem
    rd = m.measure(_pose(), [], 1000, 1000, L, now=0.2)
    assert rd.level is None                   # a dropout, not "can't reach"
    rd = m.measure(None, [], 1000, 1000, L, now=0.3)
    assert rd.level is None and rd.problem


def test_head_is_kept_while_the_hand_is_tracked():
    # From behind, the pose loses the body once the arm is over the head;
    # the hand tracker doesn't.
    m = ap.ApleyMeter()
    reach = [_hand_lms(hand(**LEFT_REACH))]
    assert m.measure(_pose(), reach, 1000, 1000, L, now=0.0).level == 3
    for k in range(1, 40):
        assert m.measure(None, reach, 1000, 1000, L, now=k * 0.1).level == 3
    # Neither the pose nor a hand for over a second: the head is forgotten.
    assert m.measure(None, [], 1000, 1000, L, now=4.0).level is None
    assert m.measure(None, reach, 1000, 1000, L, now=5.5).level is None


def test_made_up_head_is_refused():
    # Head cut off the picture: MediaPipe puts one level with the shoulders,
    # and a hand up the back would read as on top of it.
    m = ap.ApleyMeter()
    up_back = [_hand_lms(hand(wrist=(0.2, 1.0), tips=(0.2, -0.6)))]
    m.measure(_pose(), [_hand_lms(hand(**LEFT_REACH))], 1000, 1000, L, now=0.0)
    flat = _pose(shoulder_y=EARS[1] + 20)
    rd = m.measure(flat, up_back, 1000, 1000, L, now=0.1)
    assert rd.level is None and "Head not in view" in rd.problem
    off = _pose()
    off[ap.EAR_L].y = off[ap.EAR_R].y = -0.2
    assert m.measure(off, up_back, 1000, 1000, L, now=0.2).level is None


def test_meter_pose_fallback_without_hands():
    lms = _pose()
    lms[15] = SimpleNamespace(x=0.52, y=(EARS[1] - 120) / 1000, z=0.0, visibility=0.9)
    lms[19] = SimpleNamespace(x=0.50, y=(EARS[1] - 130) / 1000, z=0.0, visibility=0.9)
    assert ap.ApleyMeter().measure(lms, [], 1000, 1000, L, now=0.0).level == 2


# ── the hold ─────────────────────────────────────────────────────────────────────
def _feed(tr, seq, t=0.0, dt=0.1):
    out = None
    for lv in seq:
        r = tr.update(lv, now=t)
        out = r if r is not None else out
        t += dt
    return t, out


def test_every_level_needs_a_full_second():
    tr = ap.ApleyTracker()
    _feed(tr, [0] * 5 + [2] * 9 + [0] * 5)
    assert tr.best in (None, 0)
    _feed(tr, [2] * 12, t=3.0)
    assert tr.best == 2


def test_dropout_cannot_stretch_a_short_touch():
    tr = ap.ApleyTracker()
    # Touching for 0.2 s, a 0.5 s dropout, then 0.1 s more: 0.8 s from the
    # first touch to the last. The old percentile window counted the dropout
    # as held time and passed this; the hold is timed from the readings.
    seq = [3, 3, 3] + [None] * 4 + [3, 3]
    _feed(tr, seq)
    assert tr.best is None and tr.side == L


def test_jitter_does_not_break_a_hold():
    tr = ap.ApleyTracker()
    _feed(tr, [3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3])
    assert tr.results[L] == 3 and tr.side == R


def test_long_dip_breaks_a_hold():
    tr = ap.ApleyTracker()
    _feed(tr, [3] * 6 + [0] * 5 + [3] * 6)
    assert tr.side == L and not tr.best


def test_low_fps_still_holds():
    tr = ap.ApleyTracker()
    _feed(tr, [3] * 6, dt=0.25)
    assert tr.results[L] == 3


def test_held_3_ends_the_arm_and_runs_left_then_right():
    tr = ap.ApleyTracker()
    assert tr.side == L
    _feed(tr, [3] * 12)
    assert tr.results[L] == 3 and tr.side == R
    t, out = _feed(tr, [1] * 15, t=5.0)
    assert out is None and tr.best == 1
    out = tr.next_side()
    assert out == {"left_score": 3, "right_score": 1, "score": 1, "ended_by": "coach"}


def test_seen_but_nothing_held_is_0_unseen_is_not_tested():
    tr = ap.ApleyTracker()
    _feed(tr, [0] * 20)
    tr.next_side()
    assert tr.results[L] == 0
    _feed(tr, [None] * 20, t=5.0)
    res = tr.next_side()
    assert res["left_score"] == 0 and res["right_score"] is None and res["score"] == 0


def test_stop_mid_run():
    tr = ap.ApleyTracker()
    _feed(tr, [2] * 15)
    res = tr.finish()
    assert res["left_score"] == 2 and res["right_score"] is None and res["score"] == 2
    tr2 = ap.ApleyTracker()
    assert tr2.finish() is None


def test_best_level_is_kept():
    tr = ap.ApleyTracker()
    _feed(tr, [2] * 15 + [1] * 15 + [0] * 10)
    assert tr.best == 2


def test_reset():
    tr = ap.ApleyTracker()
    _feed(tr, [3] * 15)
    tr.reset()
    assert tr.side == L and tr.results == {L: None, R: None} and tr.best is None


def test_random_flicker_never_scores():
    rng = random.Random(4)
    for _ in range(200):
        tr = ap.ApleyTracker()
        seq = []
        for _ in range(80):
            # a level never lasts more than 0.7 s
            seq += [rng.choice([1, 2, 3])] * rng.randint(1, 7) + [0] * rng.randint(4, 8)
        _feed(tr, seq)
        assert not tr.best and tr.side == L


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
