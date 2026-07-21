"""Result-staging semantics: engines stage their own partial/best-available
result during stop(), exactly once, without ever blocking the shutdown.

Covers the synthetic base-class contract plus the two standalone engines
(WingspanEngine, VerticalJumpEngine) whose staging logic is real code here,
not a test double.
"""

import time

import pytest

from conftest import FakeCap  # noqa: F401  (fixture wiring)


def _partial_engine(fake_camera, staged, count):
    from web_engines import BaseWebEngine

    class Engine(BaseWebEngine):
        STOP_JOIN_TIMEOUT = 0.1

        def __init__(self):
            super().__init__(on_stage=lambda *a: staged.append(a))
            self.count = count

        def _stage_on_stop(self):
            if self.count > 0:
                self._safe_stage(self.count)

        def _run_loop(self):
            while self._running:
                time.sleep(0.01)

        def _state_locked(self):
            return {}

    return Engine()


def test_partial_result_staged_exactly_once(fake_camera):
    staged = []
    eng = _partial_engine(fake_camera, staged, count=7)
    eng.start()
    eng.stop()
    assert staged == [(7,)]
    eng.stop()  # double-stop must not restage
    assert staged == [(7,)]


def test_worker_staged_result_is_not_overwritten(fake_camera):
    staged = []
    eng = _partial_engine(fake_camera, staged, count=7)
    eng.start()
    eng._safe_stage(3)  # simulates the worker staging at natural test end
    assert eng._staged
    eng.stop()
    assert staged == [(3,)], "stop must not restage over the worker's result"


def test_zero_count_stages_nothing(fake_camera):
    staged = []
    eng = _partial_engine(fake_camera, staged, count=0)
    eng.start()
    eng.stop()
    assert staged == []


def test_failing_stage_callback_does_not_block_stop(fake_camera):
    staged = []
    eng = _partial_engine(fake_camera, staged, count=5)
    eng.on_stage = lambda *a: (_ for _ in ()).throw(RuntimeError("sink down"))
    eng.start()
    eng.stop()  # must not raise
    assert fake_camera.instances[0].released
    assert "Stage failed" in (eng._error or "")


# ── WingspanEngine (real class) ─────────────────────────────────────────────

def _wingspan(staged):
    from wingspan_web_engine import WingspanEngine

    return WingspanEngine(on_stage=lambda cm: staged.append(cm))


def test_wingspan_stages_live_reading_on_stop():
    staged = []
    eng = _wingspan(staged)
    eng.result_cm = 171.24
    eng._stage_on_stop()
    assert staged == [171.2]
    eng._stage_on_stop()  # idempotent
    assert staged == [171.2]


def test_wingspan_prefers_completed_snapshot():
    staged = []
    eng = _wingspan(staged)
    eng.result_cm = 150.0
    eng.snap = {"phase": "DONE", "final_cm": 180.55}
    eng._stage_on_stop()
    assert staged == [pytest.approx(180.5, abs=0.1)]


def test_wingspan_stages_nothing_without_a_reading():
    staged = []
    eng = _wingspan(staged)
    eng._stage_on_stop()
    assert staged == []


# ── VerticalJumpEngine (real class) ─────────────────────────────────────────

def _vjump(staged):
    from vertical_jump_web_engine import VerticalJumpEngine

    return VerticalJumpEngine(on_stage=lambda **kw: staged.append(kw))


def test_vertical_jump_stages_best_attempt():
    staged = []
    eng = _vjump(staged)
    eng.attempt_results_cm = [30.5, 32.25]
    eng.standing_reach_cm = 200.0
    eng._maybe_stage()
    assert len(staged) == 1
    assert staged[0]["jump_cm"] == 32.25
    assert staged[0]["attempts"] == [30.5, 32.25]
    eng._maybe_stage()  # idempotent unless forced
    assert len(staged) == 1
    eng._maybe_stage(force=True)
    assert len(staged) == 2


def test_vertical_jump_stages_nothing_without_attempts():
    staged = []
    eng = _vjump(staged)
    eng._maybe_stage()
    assert staged == []
