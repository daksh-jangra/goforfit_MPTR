"""Stall watchdog: a wedged capture must be visible in /state.

The bug this covers: the capture stops delivering frames but the engine
still reports running:true, so screen4 shows a black rectangle with no
explanation and only a server restart clears it. get_state() must now
carry the reason while the engine is still nominally running.

Timeouts are shrunk per-engine so the suite stays fast — the production
values (5s / 15s) are only defaults on the mixin.
"""

import time

import numpy as np
import pytest

from conftest import FakeCap  # noqa: F401  (imported for the fake_camera fixture)


FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def _make_engine(publish=True):
    """A BaseWebEngine whose loop publishes only while `pump` is set — so a
    test can wedge the feed without touching the camera fake."""
    from web_engines import BaseWebEngine

    class Engine(BaseWebEngine):
        STOP_JOIN_TIMEOUT = 0.1
        STALL_TIMEOUT_S = 0.15
        STALL_STARTUP_GRACE_S = 0.3

        def __init__(self):
            super().__init__()
            self.pump = publish

        def _run_loop(self):
            while self._running:
                if self.pump:
                    self._publish(FRAME)
                time.sleep(0.01)

        def _state_locked(self):
            return {}

    return Engine()


@pytest.fixture
def engine(fake_camera):
    """Started engine, always stopped — no loop thread outlives its test."""
    made = []

    def _start(publish=True):
        eng = _make_engine(publish=publish)
        made.append(eng)
        eng.start()
        return eng

    yield _start
    for eng in made:
        eng.stop()


def _wait_for_error(eng, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        err = eng.get_state().get("error")
        if err:
            return err
        time.sleep(0.02)
    return eng.get_state().get("error")


def test_healthy_feed_reports_no_error(engine):
    eng = engine()
    time.sleep(0.4)  # well past both the grace period and the stall budget
    state = eng.get_state()
    assert state["running"] is True
    assert not state.get("error")


def test_feed_that_never_starts_is_reported(engine):
    eng = engine(publish=False)
    err = _wait_for_error(eng)
    assert err and "never delivered a frame" in err
    assert eng.get_state()["running"] is True  # the point: running, but broken


def test_startup_grace_is_longer_than_the_stall_budget(engine):
    """A slow first frame (MediaPipe warm-up) must not read as a stall."""
    eng = engine(publish=False)
    time.sleep(0.2)  # past STALL_TIMEOUT_S, inside STALL_STARTUP_GRACE_S
    assert not eng.get_state().get("error")


def test_feed_that_wedges_mid_run_is_reported(engine):
    eng = engine()
    time.sleep(0.1)
    eng.pump = False  # capture wedges
    err = _wait_for_error(eng)
    assert err and "stalled" in err


def test_recovered_feed_clears_the_stall(engine):
    eng = engine()
    time.sleep(0.1)
    eng.pump = False
    assert _wait_for_error(eng)
    eng.pump = True  # frames come back
    time.sleep(0.1)
    assert not eng.get_state().get("error")


def test_real_error_wins_over_the_stall_message(engine):
    eng = engine(publish=False)
    eng._error = "model file missing"
    assert _wait_for_error(eng) == "model file missing"


def test_stopped_engine_reports_no_stall(engine):
    eng = engine(publish=False)
    eng.stop()
    time.sleep(0.4)
    assert not eng.get_state().get("error")


def test_dual_feed_engine_reports_a_stall(fake_camera):
    """Both cameras dead → the loop `continue`s forever without publishing,
    which is the shape of a dropped phone stream."""
    from dual_feed_engines import DualFeedEngine

    class Engine(DualFeedEngine):
        STOP_JOIN_TIMEOUT = 0.1
        STALL_TIMEOUT_S = 0.15
        STALL_STARTUP_GRACE_S = 0.2

        def _run_loop(self):
            while self._running:
                time.sleep(0.01)  # never publishes

    eng = Engine("0", "1")
    eng.start()
    try:
        err = _wait_for_error(eng)
        assert err and "never delivered a frame" in err
    finally:
        eng.stop()
