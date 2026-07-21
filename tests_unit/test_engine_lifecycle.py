"""Camera lifecycle invariants for BaseWebEngine and DualFeedEngine.

The three rules under test (each was a real production bug):
1. A failed start() must release every capture/model it acquired — a leaked
   capture keeps the device busy for the life of the process.
2. stop() must never release a capture while the loop thread might still be
   using it — that is a native use-after-free that kills the whole process.
   A hung thread is handed to a reaper that waits for the loop to exit.
3. A clean start/stop cycle releases everything exactly once.
"""

import time

import pytest

from conftest import FakeCap, wait_until


def _make_single(fake_camera, gate=None, fail_models=False):
    from web_engines import BaseWebEngine

    class Engine(BaseWebEngine):
        STOP_JOIN_TIMEOUT = 0.1

        def _load_models(self):
            self.models_released = False
            if fail_models:
                raise RuntimeError("model file missing")

        def _release_models(self):
            self.models_released = True

        def _run_loop(self):
            if gate is not None:
                gate.wait()  # simulates being stuck inside cap.read()
                return
            while self._running:
                time.sleep(0.01)

        def _state_locked(self):
            return {}

    return Engine()


def _make_dual(fake_camera, gate=None, fail_models=False, finish_log=None):
    from dual_feed_engines import DualFeedEngine

    class Engine(DualFeedEngine):
        STOP_JOIN_TIMEOUT = 0.1

        def _load_models(self):
            if fail_models:
                raise RuntimeError("boom")

        def _release_models(self):
            pass

        def maybe_finish(self):
            if finish_log is not None:
                finish_log.append(1)

        def _run_loop(self):
            if gate is not None:
                gate.wait()
                return
            while self._running:
                time.sleep(0.01)

        def _state_locked(self):
            return {}

    return Engine("0", "1")


# ── Rule 1: failed start releases everything ────────────────────────────────

def test_failed_model_load_releases_camera(fake_camera):
    eng = _make_single(fake_camera, fail_models=True)
    with pytest.raises(RuntimeError):
        eng.start()
    assert fake_camera.instances[0].released
    assert eng.cap is None
    assert eng.models_released
    assert not eng.is_running()


def test_dual_failed_model_load_releases_both_cameras(fake_camera):
    eng = _make_dual(fake_camera, fail_models=True)
    with pytest.raises(RuntimeError):
        eng.start()
    assert all(cap.released for cap in fake_camera.instances)
    assert eng.cap_a is None and eng.cap_b is None


def test_dual_camera_b_failure_releases_camera_a(fake_camera):
    fake_camera.fail_sources = {1}  # camera B ("1") never opens, retries included
    eng = _make_dual(fake_camera)
    with pytest.raises(RuntimeError, match="camera B"):
        eng.start()
    assert fake_camera.instances[0].released
    assert eng.cap_a is None and eng.cap_b is None


# ── Rule 2: no release under a live loop thread ─────────────────────────────

def test_hung_thread_defers_release_to_reaper(fake_camera, gate):
    eng = _make_single(fake_camera, gate=gate)
    eng.start()
    t0 = time.monotonic()
    eng.stop()
    assert time.monotonic() - t0 < 1.0, "stop() must return promptly"
    assert not eng.is_running()
    assert not fake_camera.instances[0].released, (
        "capture must NOT be released while the loop thread is alive"
    )
    gate.set()  # loop finally exits
    assert wait_until(lambda: fake_camera.instances[0].released), (
        "reaper must release the capture once the loop exits"
    )


def test_dual_hung_thread_defers_release_to_reaper(fake_camera, gate):
    eng = _make_dual(fake_camera, gate=gate)
    eng.start()
    eng.stop()
    assert not any(cap.released for cap in fake_camera.instances)
    gate.set()
    assert wait_until(lambda: all(cap.released for cap in fake_camera.instances))


# ── Rule 3: clean cycle releases exactly once ───────────────────────────────

def test_clean_start_stop_releases_camera(fake_camera):
    eng = _make_single(fake_camera)
    eng.start()
    assert eng.is_running()
    eng.stop()
    assert not eng.is_running()
    assert fake_camera.instances[0].released


def test_dual_clean_start_stop_releases_both(fake_camera):
    eng = _make_dual(fake_camera)
    eng.start()
    eng.stop()
    assert all(cap.released for cap in fake_camera.instances)


def test_dual_stop_calls_maybe_finish(fake_camera):
    log = []
    eng = _make_dual(fake_camera, finish_log=log)
    eng.start()
    eng.stop()
    assert log == [1], "stop() must stage via maybe_finish without the HTTP layer's help"


def test_get_state_reports_running_consistently(fake_camera):
    eng = _make_single(fake_camera)
    eng.start()
    assert eng.get_state()["running"] is True
    eng.stop()
    assert eng.get_state()["running"] is False
