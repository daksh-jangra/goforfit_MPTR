"""
Logic checks for the Shoulder Stretch detector — no camera, no participant.
===========================================================================

WHY THIS FILE EXISTS
--------------------
  This test has no number in it. The whole verdict is one judgement — did
  the fingertips touch — made from landmarks that are seen from behind, that
  jitter, and that disappear for a few frames at a time because the hands are
  hidden by the participant's own back. Every way of getting that wrong is
  silent: the feed still streams, the HUD still updates, the test just starts
  passing gaps you could put a finger through, or failing touches the coach
  watched happen.

  So the detection logic is checked here with made-up landmarks and an
  injected clock, BEFORE any camera is pointed at anybody. It runs in about a
  second and needs no webcam, no model file and no volunteer.

  Two levels are exercised:
    * the geometry helpers (fingertip_gap_ratio / frame_gap_ratio /
      hand_belongs_to_pose / detect_side) against synthetic landmarks
    * the state machine (ShoulderStretchTracker / SideSelector), both from
      bare ratios and through the same pipeline main() runs per frame

HOW TO RUN IT
-------------
      python3 tests/test_shoulder_stretch.py   # prints one line per check
      pytest tests/test_shoulder_stretch.py    # same checks, if you use pytest

  Non-zero exit means something broke. Run it after ANY edit to
  shoulder_stretch.py — especially to the thresholds near the top of that
  file (PASS_RATIO, TOUCH_FRAMES, OWNER_MAX_FRAC, SIDE_MARGIN_FRAC).

WHAT IT DOES NOT COVER
----------------------
  MediaPipe itself, the HUD, and whether a real fingertip landmark lands
  where the finger actually is. Those need real footage. What is covered is
  everything between the landmarks and the verdict — the part that fails
  without looking broken.
"""

import os
import sys
import random

# Import shoulder_stretch.py from this directory regardless of where this was run.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import shoulder_stretch as ss  # noqa: E402

LEFT, RIGHT = ss.SIDE_LEFT, ss.SIDE_RIGHT

# A 960x540 frame, the size web_engines publishes at.
FW, FH = 960, 540


# ── synthetic landmarks ──────────────────────────────────────────────────────
class _LM:
    """One landmark. Pose landmarks carry .visibility; hand landmarks do not
    (shoulder_stretch.vis() is pose-only for exactly that reason), but an
    extra attribute the hand code never reads is harmless here."""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, visibility=0.9):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, visibility


def _norm(p, vis_=0.9):
    """Pixel point -> a landmark in MediaPipe's normalised 0..1 space."""
    return _LM(p[0] / FW, p[1] / FH, vis_)


def make_pose(points, n=33):
    """points: {landmark index: (x_px, y_px)} or {idx: (x_px, y_px, vis)}.
    Anything not named is an invisible landmark at the origin."""
    lms = [_LM(0.0, 0.0, 0.0) for _ in range(n)]
    for idx, p in points.items():
        lms[idx] = _norm((p[0], p[1]), p[2] if len(p) > 2 else 0.9)
    return lms


def make_hand(wrist, palm, tips, down=False):
    """21 hand landmarks: the wrist (0), the middle-finger knuckle (9) one
    palm-length toward the head, and the five fingertips (4/8/12/16/20).
    tips is either one point (all five stacked) or five points."""
    if not isinstance(tips[0], (tuple, list)):
        tips = [tips] * 5
    lms = [_norm((wrist[0], wrist[1])) for _ in range(21)]
    # The knuckle sits one palm-length from the wrist, toward the fingers:
    # above it for a hand pointing up, below it for one pointing down.
    lms[ss.HAND_MID_MCP_IDX] = _norm((wrist[0], wrist[1] + (palm if down else -palm)))
    for i, tip in zip(ss.FINGERTIP_IDX, tips):
        lms[i] = _norm(tip)
    return lms


# Anchor for every synthetic body. `scale` multiplies the whole skeleton, so
# the same geometry can be replayed at a different distance from the camera.
CX, CY = 480.0, 170.0


def body(scale=1.0, over=LEFT, lift=None, drop=None, vis_=0.9):
    """A participant seen from behind: shoulders `sep` apart, the `over` arm's
    elbow raised above its own shoulder and the other elbow below it, and both
    pose wrists between the shoulder blades where the hands actually meet.

    `lift` (px, above the shoulder) and `drop` (px, below it) let a check make
    the side call obvious, marginal, or impossible."""
    sep = 100.0 * scale
    lift = 0.35 * sep if lift is None else lift
    drop = 0.35 * sep if drop is None else drop
    l_sh, r_sh = (CX - sep / 2, CY), (CX + sep / 2, CY)
    up, down = CY - lift, CY + drop
    l_el = (l_sh[0], up if over == LEFT else down)
    r_el = (r_sh[0], up if over == RIGHT else down)
    # The wrists are behind the back: real footage scores them poorly, and
    # nearest_pose_wrist() deliberately ignores visibility there.
    wy = CY + 0.60 * sep
    return make_pose({
        ss.IDX_L_SHOULDER: (l_sh[0], l_sh[1], vis_),
        ss.IDX_R_SHOULDER: (r_sh[0], r_sh[1], vis_),
        ss.IDX_L_ELBOW:    (l_el[0], l_el[1], vis_),
        ss.IDX_R_ELBOW:    (r_el[0], r_el[1], vis_),
        ss.IDX_L_WRIST:    (CX - 0.15 * sep, wy, 0.2),
        ss.IDX_R_WRIST:    (CX + 0.15 * sep, wy, 0.2),
        # Hips give the torso diagonal (~1.72 x sep) every ownership
        # allowance is measured in.
        ss.IDX_L_HIP:      (l_sh[0], CY + 1.4 * sep, vis_),
        ss.IDX_R_HIP:      (r_sh[0], CY + 1.4 * sep, vis_),
    })


def hands(ratio, scale=1.0, palm_frac=0.30, over=LEFT):
    """The participant's two hands, placed so the contact metric comes out at
    exactly `ratio`: each hand's wrist sits on its own pose wrist and the
    fingertips are (ratio * palm) apart between the shoulder blades."""
    sep = 100.0 * scale
    palm = palm_frac * sep
    gap = ratio * palm
    ty = CY + 0.55 * sep
    # The stretch: the top hand comes DOWN over the shoulder (so its wrist sits
    # toward THAT shoulder, which is what names the side), the bottom hand
    # comes UP the back, and the fingertips meet between the shoulder blades.
    dx = (0.40 * sep) if over == RIGHT else (-0.40 * sep)
    return [make_hand((CX + dx, ty - 1.8 * palm), palm, (CX - gap / 2, ty), down=True),
            make_hand((CX, ty + 1.8 * palm), palm, (CX + gap / 2, ty))]


def scene(ratio, scale=1.0, over=LEFT):
    """(pose, hands) for one frame of a participant holding the stretch with a
    fingertip gap of `ratio` palm lengths."""
    return body(scale=scale, over=over), hands(ratio, scale=scale, over=over)


# ── drivers ──────────────────────────────────────────────────────────────────
def feed(tracker, sequence, fps=20, t0=0.0, side=LEFT):
    """Drive the tracker through (duration_seconds, ratio_or_None) steps at a
    fixed frame rate with an injected clock, so the checks are deterministic.
    Returns (end_time, [(t, result), ...])."""
    t = t0
    dt = 1.0 / fps
    emitted = []
    for duration, ratio in sequence:
        for _ in range(max(1, int(round(duration / dt)))):
            out = tracker.update(ratio, now=t, side=side)
            if out:
                emitted.append((t, out))
            t += dt
    return t, emitted


def feed_frames(tracker, frames, fps=20, t0=0.0, smooth=None, selector=None):
    """Drive the tracker through whole FRAMES the way main() does: ownership
    gate + contact metric, the median window reset whenever a frame can't be
    measured, the side vote, then one tracker.update().

    frames: [(pose_lms|None, hand_lms_list), ...]
    Returns (end_time, [(t, result), ...])."""
    t = t0
    dt = 1.0 / fps
    emitted = []
    for pose_lms, hand_list in frames:
        ratio, side = None, (selector.side if selector else None)
        if pose_lms is not None:
            reading = ss.frame_gap_ratio(hand_list, pose_lms, FW, FH)
            if reading is not None:
                # The engine names the arm from the HANDS, with the elbows as
                # a fallback, and only while the stretch is in view.
                if selector is not None:
                    named = ss.side_from_hands(hand_list, pose_lms, FW, FH)
                    if named is None:
                        named = ss.detect_side(pose_lms, FW, FH)
                    side = selector.update(named)
                ratio = smooth.push(reading.ratio) if smooth else reading.ratio
        out = tracker.update(ratio, now=t, side=side)
        if out:
            emitted.append((t, out))
        t += dt
    return t, emitted


def held(n, ratio, scale=1.0, over=LEFT):
    """n identical frames of the stretch being held at `ratio`."""
    return [scene(ratio, scale=scale, over=over) for _ in range(n)]


def both_sides(n=8, ratio=0.12, scale=1.0):
    """A whole passing test: the right arm over, then the left."""
    return held(n, ratio, scale=scale, over=RIGHT) + held(n, ratio, scale=scale, over=LEFT)


def bank(tracker, side, ratio=0.12, t0=0.0, fps=20):
    """Bank one side's touch straight into the tracker (no landmarks)."""
    t = t0
    for _ in range(ss.TOUCH_READINGS):
        tracker.update(ratio, now=t, side=side)
        t += 1.0 / fps
    return t


# ── the contact metric ───────────────────────────────────────────────────────
def test_gap_ratio_is_the_gap_in_palm_lengths():
    # 10 px between the closest fingertips, 50 px palms -> 0.20 exactly, the
    # threshold itself. The manual gives no number, so this arithmetic IS the
    # definition of "touching" and must not drift.
    a = make_hand((300.0, 300.0), 50.0, (400.0, 300.0))
    b = make_hand((600.0, 300.0), 50.0, (410.0, 300.0))
    r = ss.fingertip_gap_ratio(a, b, FW, FH)
    assert r is not None, "no reading from two perfectly good hands"
    assert abs(r.gap_px - 10.0) < 0.5, f"gap {r.gap_px} px, want 10"
    assert abs(r.palm_px - 50.0) < 0.5, f"palm {r.palm_px} px, want 50"
    assert abs(r.ratio - 0.20) < 0.005, f"ratio {r.ratio}, want 0.20"


def test_gap_is_the_closest_fingertip_pair_not_the_nearest_finger():
    # The fingers meet in whatever order they happen to; only the closest of
    # the 5x5 pairs is the contact. Here every pair is far apart except the
    # ring fingers, which are touching.
    a = make_hand((300.0, 300.0), 50.0,
                  [(400.0, 100.0), (400.0, 150.0), (400.0, 200.0),
                   (400.0, 300.0), (400.0, 250.0)])
    b = make_hand((600.0, 300.0), 50.0,
                  [(460.0, 100.0), (460.0, 150.0), (460.0, 200.0),
                   (405.0, 300.0), (460.0, 250.0)])
    r = ss.fingertip_gap_ratio(a, b, FW, FH)
    assert abs(r.gap_px - 5.0) < 0.5, f"gap {r.gap_px} px — not the closest pair"
    assert r.ratio <= ss.PASS_RATIO, f"a 5 px touch read as {r.ratio}"


def test_the_same_geometry_at_two_scales_gives_the_same_ratio():
    # This is the whole reason the metric is normalised by palm length: a
    # child close to the phone and an adolescent across the room must be
    # judged by the same threshold.
    ratios = []
    for scale in (0.6, 1.0, 2.5):
        pose_lms, hand_list = scene(0.12, scale=scale)
        r = ss.frame_gap_ratio(hand_list, pose_lms, FW, FH)
        assert r is not None, f"scale {scale}: frame not measurable"
        ratios.append(r.ratio)
    spread = max(ratios) - min(ratios)
    assert spread < 0.01, f"ratio moved with distance from the camera: {ratios}"
    for r in ratios:
        assert r <= ss.PASS_RATIO, f"a touch at one scale read as {r}"


def test_a_visible_gap_is_a_visible_gap_at_every_scale():
    for scale in (0.6, 1.0, 2.5):
        pose_lms, hand_list = scene(0.96, scale=scale)
        r = ss.frame_gap_ratio(hand_list, pose_lms, FW, FH)
        assert r.ratio > ss.PASS_RATIO, f"scale {scale}: 0.96 read as a touch"


def test_hands_too_small_in_frame_are_unmeasurable_not_a_fail():
    # A tiny hand's landmarks are noise; reporting a huge ratio would read as
    # a confident FAIL, so the frame must be discarded instead.
    a = make_hand((300.0, 300.0), 6.0, (400.0, 300.0))
    b = make_hand((600.0, 300.0), 6.0, (410.0, 300.0))
    assert ss.fingertip_gap_ratio(a, b, FW, FH) is None, \
        "a sub-MIN_PALM_PX hand produced a reading"


# ── the hand-ownership gate ──────────────────────────────────────────────────
def test_a_hand_at_the_participants_wrist_belongs_to_them():
    pose_lms = body()
    near = make_hand((CX - 15.0, CY + 60.0), 30.0, (CX, CY + 55.0))
    assert ss.hand_belongs_to_pose(near, pose_lms, FW, FH)


def test_a_hand_a_body_away_is_not_theirs():
    pose_lms = body()
    far = make_hand((60.0, 470.0), 30.0, (70.0, 470.0))
    assert not ss.hand_belongs_to_pose(far, pose_lms, FW, FH), \
        "a hand across the frame passed the ownership gate"


def test_a_bystanders_two_hands_are_not_a_reading():
    # The real a_088 geometry: the LOCKED body is the participant, and the two
    # hands the detector found are a coach's, clasped in front of his chest
    # standing beside her. They are measurable and nearly touching — only the
    # behind-the-back gate stops them scoring her trial.
    pose_lms = body()
    td = ss.torso_diagonal_px(pose_lms, FW, FH)
    cx, cy = CX + 1.3 * td, CY + 30.0
    coach = [make_hand((cx - 2.0, cy - 50.0), 30.0, (cx - 2.0, cy), down=True),
             make_hand((cx + 2.0, cy + 50.0), 30.0, (cx + 2.0, cy))]
    reading = ss.fingertip_gap_ratio(*coach, FW, FH)
    assert reading is not None and reading.ratio <= ss.PASS_RATIO, "test setup"
    assert ss.frame_gap_ratio(coach, pose_lms, FW, FH) is None, \
        "a bystander's hands scored the participant's trial"


def test_a_chest_height_gesture_is_not_on_the_back():
    # The real a_109 geometry: fingertips nearly meeting, but well below the
    # participant's shoulder blades (she was bent over; the coach gestured).
    pose_lms = body()
    td = ss.torso_diagonal_px(pose_lms, FW, FH)
    y = CY + 0.75 * td
    low = [make_hand((CX - 2.0, y - 50.0), 30.0, (CX - 2.0, y), down=True),
           make_hand((CX + 2.0, y + 50.0), 30.0, (CX + 2.0, y))]
    assert ss.fingertip_gap_ratio(*low, FW, FH).ratio <= ss.PASS_RATIO, "test setup"
    assert ss.frame_gap_ratio(low, pose_lms, FW, FH) is None, \
        "a contact below the back scored as a touch"


def test_one_foreign_hand_discards_the_whole_frame():
    pose_lms, hand_list = scene(0.10)
    stranger = make_hand((900.0, 420.0), 30.0, (896.0, 470.0), down=True)
    assert not ss.hand_belongs_to_pose(stranger, pose_lms, FW, FH)
    assert ss.frame_gap_ratio([hand_list[0], stranger], pose_lms, FW, FH) is None, \
        "the participant's hand plus a stranger's scored as a touch"


def test_a_badly_placed_pose_wrist_does_not_reject_her_own_hand():
    # f_011: the pose wrist behind the back was predicted 90 px off the real
    # hand. The gate no longer looks at pose wrists at all.
    pose_lms, hand_list = scene(0.12)
    pose_lms[ss.IDX_L_WRIST] = _norm((CX - 200.0, CY + 300.0), 0.3)
    pose_lms[ss.IDX_R_WRIST] = _norm((CX + 200.0, CY + 300.0), 0.3)
    assert ss.frame_gap_ratio(hand_list, pose_lms, FW, FH) is not None


# ── whose body: the participant lock ─────────────────────────────────────────
def _shifted(dx):
    pose_lms = body()
    for lm in pose_lms:
        if lm.visibility > 0:
            lm.x += dx / FW
    return pose_lms


def test_the_lock_seeds_only_on_a_lone_body():
    lock = ss.ParticipantLock()
    lock.update([body(), _shifted(300.0)], FW, FH, now=0.0)
    assert not lock.locked, "locked onto one of two bodies"
    lock.update([body()], FW, FH, now=0.1)
    assert lock.locked


def test_a_second_body_cannot_steal_the_lock():
    lock = ss.ParticipantLock()
    lock.update([body()], FW, FH, now=0.0)                # the student, alone
    coach = _shifted(0.9 * ss.torso_diagonal_px(body(), FW, FH))
    got = lock.update([coach], FW, FH, now=0.1)           # only the coach seen
    assert got is None, "the lock jumped to a body 0.9 td away"
    got = lock.update([coach, body()], FW, FH, now=0.2)
    assert ss.shoulder_mid_px(got, FW, FH)[0] == \
        ss.shoulder_mid_px(body(), FW, FH)[0], "the lock followed the coach"


def test_no_pose_reuses_the_body_briefly():
    lock = ss.ParticipantLock()
    lock.update([body()], FW, FH, now=0.0)
    lock.update([], FW, FH, now=0.5)
    assert lock.body_for(now=0.5) is not None, "a 0.5 s pose dropout lost the body"
    lock.update([], FW, FH, now=1.5)
    assert lock.body_for(now=1.5) is None, "a 1.5 s old body was still used"


def test_fewer_than_two_hands_is_not_a_reading():
    pose_lms, hand_list = scene(0.10)
    assert ss.frame_gap_ratio([], pose_lms, FW, FH) is None
    assert ss.frame_gap_ratio(hand_list[:1], pose_lms, FW, FH) is None
    assert ss.frame_gap_ratio(hand_list, None, FW, FH) is None


# ── which side the trial is ──────────────────────────────────────────────────
def test_side_is_the_arm_over_the_shoulder():
    assert ss.detect_side(body(over=LEFT), FW, FH) == LEFT
    assert ss.detect_side(body(over=RIGHT), FW, FH) == RIGHT


def test_a_ten_pixel_margin_is_not_a_side_call():
    # A twisted torso reads both elbows above their shoulders. At a 100 px
    # shoulder separation SIDE_MARGIN_FRAC asks for 12 px, so ~10 px of
    # difference must name nothing rather than name the trial from noise.
    twisted = body(over=LEFT, lift=40.0, drop=-30.0)   # both elbows up, 10 px apart
    assert ss.detect_side(twisted, FW, FH) is None, \
        "a 10 px difference named the trial"
    # ...and a clear difference at the same posture still resolves.
    clear = body(over=LEFT, lift=40.0, drop=-5.0)      # 45 px apart
    assert ss.detect_side(clear, FW, FH) == LEFT


def test_hands_together_in_front_of_the_face_is_not_a_try():
    # The laptop false PASS: sitting facing the camera, fingertips brought
    # together in front of the face. Both hands point UP — not the stretch
    # (one down over the shoulder, one up the back) — so no reading.
    pose_lms = body()
    ty = CY + 0.10 * 100.0
    up_up = [make_hand((CX - 20.0, ty + 60.0), 30.0, (CX - 1.0, ty)),
             make_hand((CX + 20.0, ty + 60.0), 30.0, (CX + 1.0, ty))]
    assert ss.fingertip_gap_ratio(*up_up, FW, FH).ratio <= ss.PASS_RATIO, "test setup"
    assert ss.frame_gap_ratio(up_up, pose_lms, FW, FH) is None, \
        "hands together in front of the face scored a touch"


def test_the_side_is_read_from_the_picture_not_the_labels():
    # From behind MediaPipe's left/right labels come out swapped. Swap them
    # here: the raised elbow is still on the picture's right, so it is RIGHT.
    pose_lms = body(over=RIGHT)
    for a, b in ((ss.IDX_L_SHOULDER, ss.IDX_R_SHOULDER), (ss.IDX_L_ELBOW, ss.IDX_R_ELBOW)):
        pose_lms[a], pose_lms[b] = pose_lms[b], pose_lms[a]
    assert ss.detect_side(pose_lms, FW, FH) == RIGHT


def test_no_elbow_above_its_shoulder_says_nothing():
    arms_down = body(over=LEFT, lift=-20.0, drop=60.0)
    assert ss.detect_side(arms_down, FW, FH) is None


def test_side_needs_the_joints_to_be_visible():
    hidden = body(over=LEFT, vis_=0.1)
    assert ss.detect_side(hidden, FW, FH) is None, \
        "named the trial from landmarks the model could not see"


def test_the_side_voter_rides_out_a_stray_mis_read():
    v = ss.SideVoter()
    assert v.side is None
    for _ in range(3):
        v.update(None)                       # frames that name nothing
    assert v.side is None, "a frame naming nothing named the trial"
    for _ in range(3):
        v.update(RIGHT)
    assert v.side == RIGHT
    v.update(LEFT)                           # one stray frame
    assert v.side == RIGHT, "a single mis-read flipped the side"


def test_the_side_voter_follows_the_student_when_they_switch_arms():
    # Both sides must be banked in one trial, so the label MUST be able to
    # change — the old locking selector would have frozen on the first arm.
    v = ss.SideVoter()
    for _ in range(ss.SideVoter.VOTE_WINDOW):
        v.update(RIGHT)
    assert v.side == RIGHT
    for _ in range(ss.SideVoter.VOTE_WINDOW):
        v.update(LEFT)
    assert v.side == LEFT, "the side could not follow the student to the other arm"


def test_the_side_is_named_from_the_hands():
    # The down-pointing hand came over the shoulder, so it names the arm.
    for over in (LEFT, RIGHT):
        pose_lms, hand_list = scene(0.12, over=over)
        assert ss.side_from_hands(hand_list, pose_lms, FW, FH) == over


def test_the_side_reaches_the_result():
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed_frames(tr, both_sides(), selector=ss.SideVoter())
    assert emitted, "a clean touch on both sides did not resolve"
    res = emitted[0][1]
    assert res["right_passed"] and res["left_passed"], res


def test_a_sustained_touch_on_both_sides_passes():
    # Run it at several frame rates: a weak phone manages ~8 fps in this pose
    # and the verdict must not depend on that.
    for fps in (8, 10, 15, 30):
        tr = ss.ShoulderStretchTracker()
        _, emitted = feed_frames(tr, both_sides(12, 0.12), fps=fps,
                                 smooth=ss.MedianSmoother(), selector=ss.SideVoter())
        assert tr.passed, f"{fps} fps: a held touch on both sides did not pass ({tr.state})"
        assert tr.state is ss.State.PASSED
        assert len(emitted) == 1, f"{fps} fps: emitted {len(emitted)} results, want 1"
        res = emitted[0][1]
        assert res["verdict_by"] == "auto", res
        assert res["right_passed"] and res["left_passed"], res
        assert abs(res["best_gap_ratio"] - 0.12) < 0.01, res


def test_one_side_alone_is_not_a_pass():
    # The whole point of the both-sides rule.
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed_frames(tr, held(20, 0.10, over=RIGHT), selector=ss.SideVoter())
    assert not tr.passed, "one side alone passed the test"
    assert not emitted, "a one-sided trial resolved early"
    res = tr.end_trial()
    assert res["right_passed"] and not res["left_passed"], res
    assert res["passed"] is False


def test_the_trial_is_capped_at_fifteen_seconds_from_the_first_attempt():
    tr = ss.ShoulderStretchTracker()
    # Nothing happens for a while: the clock must NOT be running yet.
    feed(tr, [(30.0, None)])
    assert tr.cap_remaining() is None, "the clock started before the first attempt"
    t = bank(tr, RIGHT, t0=30.0)                 # first attempt -> clock starts
    assert abs(tr.cap_remaining(now=t) - (ss.TRIAL_CAP_S - (t - 30.0))) < 0.01
    # ...and the second side arrives too late.
    res = tr.update(0.10, now=30.0 + ss.TRIAL_CAP_S + 0.1, side=LEFT)
    assert res is not None and res["passed"] is False, res
    assert res["ended_by"] == "cap", res


def test_both_sides_inside_the_cap_pass():
    tr = ss.ShoulderStretchTracker()
    t = bank(tr, RIGHT, t0=0.0)
    res = None
    for _ in range(ss.TOUCH_READINGS):
        res = tr.update(0.10, now=t, side=LEFT) or res
        t += 0.05
    assert res is not None and res["passed"] is True, res
    assert res["ended_by"] == "pass"


def test_a_one_inch_miss_fails():
    # 0.26 is the interpolated one-inch (2.5 cm) gap. It is inside the
    # release band, so it must not break a run — but it must never bank one
    # either, and the trial ends FAIL.
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed_frames(tr, held(60, 0.26), smooth=ss.MedianSmoother())
    assert not emitted, "a one-inch miss resolved as a PASS"
    assert not tr.passed and tr.touch_count == 0, f"touch_count {tr.touch_count}"
    res = tr.end_trial()
    assert res is not None and res["passed"] is False
    assert abs(res["best_gap_ratio"] - 0.26) < 0.01, res
    assert tr.state is ss.State.FAILED


def test_a_clear_miss_fails():
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(3.0, 0.96)])
    assert not tr.passed and tr.state is ss.State.REACHING
    assert tr.touch_count == 0


def test_the_threshold_itself_counts_and_just_above_it_does_not():
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(0.05, ss.PASS_RATIO)])
    assert tr.touch_count == 1, "a gap exactly at the threshold was not counted"
    tr2 = ss.ShoulderStretchTracker()
    feed(tr2, [(0.05, ss.PASS_RATIO + 0.01)])
    assert tr2.touch_count == 0, "a gap over the threshold was counted"


def test_one_lucky_frame_in_a_failing_trial_never_passes():
    # The single most important check in this file: a MINIMUM over the trial
    # would pass this, which is exactly why the verdict counts sustained
    # detections instead. Driven with raw ratios (no median window) so it is
    # the tracker's rule being tested, not the smoother's.
    tr = ss.ShoulderStretchTracker()
    rnd = random.Random(11)
    t = 0.0
    for i in range(120):
        ratio = 0.10 if i in (17, 61, 93) else 0.90 + rnd.uniform(-0.05, 0.05)
        assert tr.update(ratio, now=t) is None, f"resolved on frame {i}"
        t += 0.05
    assert not tr.passed, "three lucky frames spread over a trial passed it"
    assert tr.touch_count <= 1, f"touch_count {tr.touch_count}"
    # ...but the closest gap is still recorded, because the threshold is meant
    # to be retuned later from exactly this number.
    assert abs(tr.best_gap_ratio - 0.10) < 1e-6, tr.best_gap_ratio
    res = tr.end_trial()
    assert res["passed"] is False and abs(res["best_gap_ratio"] - 0.10) < 1e-6


def test_one_frame_short_of_the_run_is_not_a_pass():
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed_frames(tr, held(ss.TOUCH_FRAMES - 1, 0.12) + held(6, 0.90))
    assert not emitted and not tr.passed, "a short touch passed"
    assert tr.touch_count == 0, "a real separation did not clear the run"


def test_a_jittery_frame_does_not_throw_away_the_run():
    # A single frame at 0.22 (between PASS_RATIO and TOUCH_RELEASE_RATIO) is
    # landmark noise on a hand seen from behind, not the hands coming apart.
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(0.15, 0.12)])                       # 3 frames banked at 20 fps
    banked = tr.touch_count
    assert banked == 3, f"banked {banked}"
    feed(tr, [(0.05, 0.22)], t0=0.15)
    assert tr.touch_count == banked, "a 0.22 blip zeroed the run"
    assert tr.state is ss.State.TOUCHING
    feed(tr, [(0.05 * (ss.TOUCH_READINGS - banked), 0.12)], t0=0.20)
    assert tr.side_done(LEFT), "could not finish the side after one jittery frame"


def test_a_real_separation_does_clear_the_run():
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(0.15, 0.12), (0.10, 0.60)])
    assert tr.touch_count == 0, "the hands came apart and the run survived"
    assert tr.state is ss.State.REACHING


# ── intermittent hand detection ──────────────────────────────────────────────
def test_intermittent_hand_detection_still_scores():
    # Dropouts are NORMAL in this pose: the hands are behind the torso and
    # seen from the back, so the detector returns two hands, then one, then
    # none, then two again. The trial must survive that, not restart on it.
    tr = ss.ShoulderStretchTracker()
    frames = []
    for over in (RIGHT, LEFT):
        pose_lms, hand_list = scene(0.12, over=over)
        for i in range(24):
            if i % 3 == 1:
                frames.append((pose_lms, hand_list[:1]))   # one hand found
            elif i % 3 == 2:
                frames.append((pose_lms, []))              # neither hand found
            else:
                frames.append((pose_lms, hand_list))       # both
    _, emitted = feed_frames(tr, frames, fps=20,
                             smooth=ss.MedianSmoother(), selector=ss.SideVoter())
    assert tr.passed, f"a held touch failed on an intermittent feed ({tr.state})"
    assert len(emitted) == 1, f"emitted {len(emitted)} results"


def test_a_short_dropout_keeps_the_banked_touch():
    tr = ss.ShoulderStretchTracker()
    t, _ = feed(tr, [(0.10, 0.12)])                # 2 detections banked
    assert tr.touch_count == 2
    t, _ = feed(tr, [(0.20, None)], t0=t)          # inside DROPOUT_GRACE_S
    assert tr.touch_count == 2, "a 0.2 s dropout voided the run"
    feed(tr, [(0.15, 0.12)], t0=t)
    assert tr.side_done(LEFT), "could not finish the side through a short dropout"


def test_touches_outside_the_window_do_not_add_up():
    # The old counter ratcheted: brief dips minutes apart summed to a PASS.
    tr = ss.ShoulderStretchTracker(cap_s=None)
    t = 0.0
    for _ in range(ss.TOUCH_READINGS + 2):
        t, _ = feed(tr, [(0.05, 0.12)], t0=t)                  # one touch reading
        t, _ = feed(tr, [(ss.TOUCH_WINDOW_S + 0.5, None)], t0=t)
    assert not tr.passed, "touches further apart than the window added up"


def test_the_ratchet_regression_fails():
    # Half the frames above the threshold, never two contact frames in a row.
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(0.05, r) for r in [0.19, 0.25] * 10])
    assert not tr.passed, "alternating 0.19 / 0.25 ratcheted to a PASS"


def test_the_trial_cap_ends_in_a_fail():
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed(tr, [(ss.TRIAL_CAP_S + 1.0, 0.22)])
    assert tr.resolved and not tr.passed
    assert emitted and emitted[0][1]["ended_by"] == "cap"


def test_an_all_none_stream_is_survivable():
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(5.0, None)])
    assert tr.state is ss.State.READY and not tr.passed
    assert tr.best_gap_ratio is None and tr.frames_seen == 0
    res = tr.end_trial()
    assert res["passed"] is False and res["best_gap_ratio"] is None


def test_a_frame_with_no_pose_at_all_is_survivable():
    tr = ss.ShoulderStretchTracker()
    frames = [(None, [])] * 10 + both_sides()
    _, emitted = feed_frames(tr, frames, smooth=ss.MedianSmoother(),
                             selector=ss.SideVoter())
    assert tr.passed and len(emitted) == 1


def test_the_smoother_forgets_across_a_dropout():
    # If the median window survived a dropout the ratio would creep back from
    # stale frames when the hands are re-detected, and a touch that had ended
    # would keep scoring.
    sm = ss.MedianSmoother()
    for _ in range(3):
        sm.push(0.10)
    assert abs(sm.value - 0.10) < 1e-9
    sm.reset()
    assert sm.value is None
    assert abs(sm.push(0.90) - 0.90) < 1e-9, "stale frames leaked through a reset"


# ── trial lifecycle ──────────────────────────────────────────────────────────
def test_a_resolved_trial_is_frozen():
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed_frames(tr, both_sides(), selector=ss.SideVoter())
    assert tr.passed
    verdict = dict(emitted[0][1])
    # More frames arrive after the verdict — the loop keeps running until the
    # coach stops it. None of them may change what was staged.
    feed(tr, [(2.0, 0.95), (1.0, None)], t0=1.0)
    assert tr.passed and tr.state is ss.State.PASSED
    assert tr.result()["passed"] == verdict["passed"]
    assert tr.result()["best_gap_ratio"] == verdict["best_gap_ratio"]
    assert tr.end_trial() is None, "end_trial re-resolved a decided trial"

def test_end_trial_resolves_once():
    tr = ss.ShoulderStretchTracker()
    feed(tr, [(0.5, 0.60)])
    first = tr.end_trial()
    assert first is not None and first["passed"] is False
    assert first["verdict_by"] == "auto", "the coach does not adjudicate this test"
    assert tr.end_trial() is None, "end_trial resolved the same trial twice"


def test_there_is_no_coach_override():
    # reverse_curl has mark_pass/mark_fail because that manual hands the
    # single trial to the tester. This one is decided by the measurement, and
    # a verdict button would quietly contradict the protocol.
    tr = ss.ShoulderStretchTracker()
    assert not hasattr(tr, "mark_pass"), "a coach PASS override crept in"
    assert not hasattr(tr, "mark_fail"), "a coach FAIL override crept in"


def test_reset_clears_everything():
    tr = ss.ShoulderStretchTracker()
    feed_frames(tr, both_sides(), selector=ss.SideVoter())
    assert tr.passed
    tr.reset()
    assert tr.state is ss.State.READY
    assert not tr.passed and not tr.resolved
    assert tr.touch_count == 0 and tr.frames_seen == 0
    assert tr.best_gap_ratio is None and tr.ratio is None and tr.side is None
    assert not tr.side_done(LEFT) and not tr.side_done(RIGHT)
    # ...and the same tracker can run a fresh trial.
    feed_frames(tr, both_sides(), selector=ss.SideVoter())
    assert tr.passed, "a reset tracker could not score again"


def test_result_dict_is_the_shape_the_wiring_expects():
    # app.py's _stage closure and db.save_shoulder_stretch are positional /
    # keyed on exactly these names; a rename here fails silently at save time.
    tr = ss.ShoulderStretchTracker()
    _, emitted = feed_frames(tr, both_sides(), selector=ss.SideVoter())
    res = emitted[0][1]
    assert set(res) == {"passed", "left_passed", "right_passed",
                        "best_gap_left", "best_gap_right", "best_gap_ratio",
                        "threshold_ratio", "required_frames", "window_s",
                        "trial_cap_s", "qualifying_readings_total",
                        "ended_by", "verdict_by"}, res
    assert isinstance(res["passed"], bool)
    assert res["passed"] == (res["left_passed"] and res["right_passed"])
    assert res["threshold_ratio"] == ss.PASS_RATIO
    assert res["required_frames"] == ss.TOUCH_FRAMES
    assert round(res["best_gap_ratio"], 3) == res["best_gap_ratio"], \
        "best_gap_ratio is not rounded at emit time"


def test_touch_progress_tracks_the_run():
    tr = ss.ShoulderStretchTracker()
    assert tr.touch_progress() == 0.0
    feed(tr, [(0.10, 0.12)], side=RIGHT)
    assert 0.0 < tr.touch_progress() < 0.5, tr.touch_progress()
    t = bank(tr, RIGHT, t0=1.0)
    assert abs(tr.touch_progress() - 0.5) < 1e-6, "one banked side is half the test"
    bank(tr, LEFT, t0=t)
    assert tr.passed and tr.touch_progress() == 1.0


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
