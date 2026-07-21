"""Lane identity across the two cameras.

The bug these pin: shuttle_run's two cameras face each other across the
track, so End B sees the lanes in the opposite left-right order to End A.
Both ends feed ONE StudentManager keyed by lane number — so if each end
numbers its columns left-to-right independently, End B's "lane 1" is End
A's lane 4 and every runner's touch count is a blend of two people.

The fix reverses the NUMBERING on End B, not the boundary coordinates.
Mirroring the x-values (the desktop script's old --mirror-b) is a no-op
for a centred layout and lands on empty track for an off-centre one —
test_mirroring_boundaries_is_not_a_correction keeps that from coming back.
"""

import pytest

from conftest import FakeCap  # noqa: F401  (imported for the fake_camera fixture)


LANE_XS = [100, 200, 300, 400, 500]  # 4 lanes, evenly spaced, centred in 600px


class FakeTracker:
    """Records what lane number each column was handed, without paying for a
    MediaPipe Pose instance per lane."""

    def __init__(self, lane_number, end_label, x0, x1, touch_line_y, manager):
        self.lane_number = lane_number
        self.end_label = end_label
        self.x0, self.x1 = x0, x1

    def close(self):
        pass


@pytest.fixture
def shuttle(fake_camera, monkeypatch):
    from dual_feed_engines import ShuttleRunEngine
    import shuttleruntest as srt

    monkeypatch.setattr(srt, "LaneTracker", FakeTracker)
    eng = ShuttleRunEngine("0", "1", touches=4)

    def calibrate(side):
        """Feed the same clicks a coach would make on that side's feed."""
        for x in LANE_XS:
            eng.action("calib_click", {"cam": side, "x": x, "y": 300})
        for y in (400, 402):
            eng.action("calib_click", {"cam": side, "x": 250, "y": y})
        eng._try_finish_calibration(side, (540, 600, 3))

    eng.calibrate = calibrate
    return eng


def _lane_by_column(eng, side):
    """Lane numbers in column order, left to right as displayed."""
    return [t.lane_number for t in sorted(eng.trackers[side], key=lambda t: t.x0)]


def test_end_a_numbers_left_to_right(shuttle):
    shuttle.calibrate("a")
    assert _lane_by_column(shuttle, "a") == [1, 2, 3, 4]


def test_end_b_numbers_right_to_left(shuttle):
    """The cameras face each other, so End B's leftmost column is lane 4."""
    shuttle.calibrate("b")
    assert _lane_by_column(shuttle, "b") == [4, 3, 2, 1]


def test_both_ends_agree_on_the_same_physical_lane(shuttle):
    """The actual invariant: a runner in one physical lane is the same lane
    number at both ends. End A's leftmost column and End B's RIGHTMOST column
    look at the same strip of track."""
    shuttle.calibrate("a")
    shuttle.calibrate("b")
    a = {t.lane_number: (t.x0, t.x1) for t in shuttle.trackers["a"]}
    b = {t.lane_number: (t.x0, t.x1) for t in shuttle.trackers["b"]}

    assert set(a) == set(b) == {1, 2, 3, 4}
    # Same lane number => mirror-image column positions across the frame.
    for lane in a:
        ax0, ax1 = a[lane]
        bx0, bx1 = b[lane]
        assert (ax0, ax1) == (600 - bx1, 600 - bx0)


def test_columns_are_identical_regardless_of_numbering(shuttle):
    """Reversing changes labels only — the crops still sit on the lanes."""
    shuttle.calibrate("b")
    reversed_cols = sorted((t.x0, t.x1) for t in shuttle.trackers["b"])
    shuttle.action("flip_lane_order_b", False)
    straight_cols = sorted((t.x0, t.x1) for t in shuttle.trackers["b"])
    assert reversed_cols == straight_cols


def test_flip_action_relabels_end_b(shuttle):
    """Side-by-side camera rigs need the reversal off."""
    shuttle.calibrate("b")
    assert shuttle.action("flip_lane_order_b", False)["reverse_lanes_b"] is False
    assert _lane_by_column(shuttle, "b") == [1, 2, 3, 4]
    assert shuttle.get_state()["reverse_lanes_b"] is False


def test_mirroring_boundaries_is_not_a_correction():
    """Why the fix reverses numbering instead of coordinates: mirroring the
    boundary x-values does nothing at all to a centred layout."""
    frame_w = 600
    assert sorted(frame_w - x for x in LANE_XS) == LANE_XS


def test_desktop_calibration_exposes_the_same_mapping():
    """shuttleruntest.py's Calibration must agree with the engine, so the
    desktop script and the browser engine can't drift apart again."""
    import shuttleruntest as srt

    calib = srt.Calibration(list(LANE_XS), touch_line_y=400, frame_w=600, frame_h=540)
    assert [calib.lane_number_for_column(i) for i in range(4)] == [1, 2, 3, 4]
    calib.reverse_lane_numbering = True
    assert [calib.lane_number_for_column(i) for i in range(4)] == [4, 3, 2, 1]


@pytest.mark.parametrize("lanes", [1, 2, 4, 5, 7, 9])
def test_dash50m_columns_tile_the_frame_for_any_lane_count(fake_camera, lanes):
    """960px does not divide by 7 or 9 — both offered by the setup form. The
    old reshape() raised inside the run loop, killing the test mid-race."""
    from dual_feed_engines import Dash50mEngine

    eng = Dash50mEngine("0", "1", lanes=lanes)
    width = 960
    bounds = [eng._column_bounds(i, width) for i in range(lanes)]

    assert bounds[0][0] == 0, "first column must start at the frame edge"
    assert bounds[-1][1] == width, "last column must reach the frame edge"
    for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
        assert prev_end == next_start, "columns must not gap or overlap"
    assert all(x1 > x0 for x0, x1 in bounds), "every lane needs a non-empty column"


@pytest.mark.parametrize("lanes,runner_col", [(4, 0), (4, 3), (7, 0), (7, 3), (7, 6), (9, 8)])
def test_dash50m_attributes_the_finish_to_the_right_lane(fake_camera, lanes, runner_col):
    """End-to-end through _process_pair: warm the background up, whistle,
    then put a runner in one overhead column and check the time lands on
    that lane. Catches both the reshape crash and the uninitialised overhead
    model, which silently pinned every finish to whichever column argmax
    reached first."""
    import numpy as np
    from dual_feed_engines import Dash50mEngine

    h, w = 540, 960
    eng = Dash50mEngine("0", "1", lanes=lanes)
    eng._finish_strip = (400, 560)

    blank = lambda: np.zeros((h, w, 3), np.uint8)
    for _ in range(30):  # background learns an empty track, pre-whistle
        eng._process_pair(blank(), blank())
    eng._on_whistle()

    finish, overhead = blank(), blank()
    finish[:, 400:560] = 255                                  # body crosses the strip
    x0, x1 = eng._column_bounds(runner_col, w)
    overhead[:, x0:x1] = 255                                  # ...in this lane
    eng._process_pair(finish, overhead)

    timed = sorted(lane for lane, t in eng.lane_times.items() if t is not None)
    assert timed == [eng._lane_for_column(runner_col)]


def test_dash50m_uses_a_separate_subtractor_per_camera(fake_camera):
    """The finish and overhead cameras see different scenes; one shared
    background model learns a blend of both and neither mask means anything."""
    from dual_feed_engines import Dash50mEngine

    eng = Dash50mEngine("0", "1", lanes=4)
    assert eng._bg_subtractor is not eng._bg_overhead


def test_dash50m_lane_labels_follow_the_flip(fake_camera):
    """dash_50m has one overhead camera, so nothing can infer the right
    order — but the flip must relabel columns consistently."""
    from dual_feed_engines import Dash50mEngine

    eng = Dash50mEngine("0", "1", lanes=4)
    assert [eng._lane_for_column(i) for i in range(4)] == [1, 2, 3, 4]
    eng.action("flip_lane_order")
    assert [eng._lane_for_column(i) for i in range(4)] == [4, 3, 2, 1]
    assert eng.get_state()["reverse_lanes"] is True
