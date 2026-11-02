"""
Logic checks for the Sit and Reach scale - no camera, no participant.

The reach in cm is fingertip pixels divided by px_per_cm, and px_per_cm comes
from the leg on screen against the hip-to-ankle length the coach typed in.
If the leg reads short, every pixel reads as more cm and the reach comes out
too big. These checks pin down that the leg is measured hip -> knee -> ankle,
so a bent knee does not shrink it.

      python3 tests/test_sit_reach.py
      pytest tests/test_sit_reach.py
"""

import math
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import snr  # noqa: E402

FW, FH = 960, 540


class _LM:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, visibility=0.9):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, visibility


def make_pose(points):
    """points: {landmark index: (x_px, y_px[, vis])}; the rest invisible."""
    lms = [_LM(0.0, 0.0, 0.0) for _ in range(33)]
    for idx, p in points.items():
        vis_ = p[2] if len(p) > 2 else 0.9
        lms[idx] = _LM(p[0] / FW, p[1] / FH, vis_)
    return lms


HIP, KNEE, ANKLE = snr._LM.LEFT_HIP, snr._LM.LEFT_KNEE, snr._LM.LEFT_ANKLE
THIGH = SHIN = 200.0


def leg(knee_bend_deg, knee_vis=0.9):
    """Seated side-on leg: hip at (700, 400), shin flat along the floor toward
    the left. knee_bend_deg 0 = straight; the knee rises as it bends."""
    hip = (700.0, 400.0)
    a = math.radians(knee_bend_deg / 2.0)          # thigh and shin tilt equally
    knee = (hip[0] - THIGH * math.cos(a), hip[1] - THIGH * math.sin(a))
    ankle = (knee[0] - SHIN * math.cos(a), knee[1] + SHIN * math.sin(a))
    return make_pose({HIP: hip, KNEE: (*knee, knee_vis), ANKLE: ankle})


def test_straight_leg_reads_its_full_length():
    assert abs(snr.hip_ankle_px(leg(0), FW, FH) - 400.0) < 0.5


def test_bent_knee_does_not_shrink_the_leg():
    # The straight hip-ankle line would read ~283 px at 90 deg (a 41% bigger
    # reach); hip -> knee -> ankle stays 400.
    for bend in (20, 45, 90):
        got = snr.hip_ankle_px(leg(bend), FW, FH)
        assert abs(got - 400.0) < 0.5, (bend, got)


def test_unseen_knee_falls_back_to_the_straight_line():
    got = snr.hip_ankle_px(leg(90, knee_vis=0.1), FW, FH)
    straight = 2 * THIGH * math.cos(math.radians(45))
    assert abs(got - straight) < 0.5, got


def test_no_leg_seen_gives_none():
    assert snr.hip_ankle_px(make_pose({}), FW, FH) is None


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
        else:
            print(f"  PASS  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
