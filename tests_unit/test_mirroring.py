"""EVERY engine must publish a MIRRORED (selfie-style) preview.

A subject facing the camera must see their real right hand on the right of
the screen — and the flip must happen BEFORE inference/detection so
landmarks, overlays and browser calibration clicks all live in the same
flipped space.

This drifted twice in practice: first curl_up and sit_reach streamed
un-mirrored while bmi/wingspan/etc. mirrored, then the dual-camera field
tests (shuttle_run/dash_50m/run_600m) were left un-mirrored while everything
else mirrored — which is what a coach reported as "I raise my right hand and
it shows on the left". These tests pin the contract for both families.

Slower than the other tests: each engine loads its real MediaPipe model and
runs real inference on synthetic frames (~2-3s per engine).
"""

import sys
import time

import numpy as np
import pytest

from conftest import wait_until


class SynthCap:
    """Fake camera producing a bright square in the LEFT third of the frame.
    A mirrored pipeline must publish it in the RIGHT third."""

    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return True

    def set(self, *a):
        pass

    def get(self, *a):
        return 30.0

    def release(self):
        pass

    def read(self):
        f = np.zeros((540, 960, 3), dtype=np.uint8)
        f[200:340, 40:280] = 255
        return True, f


def _published_square_side(engine):
    import cv2

    engine._open_camera = lambda: setattr(engine, "cap", SynthCap())
    engine.start()
    try:
        assert wait_until(lambda: engine.get_frame()[0] is not None, timeout=8.0), \
            "engine never published a frame"
        time.sleep(0.3)  # let inference settle past the first raw frame
        jpg, _ = engine.get_frame()
    finally:
        engine.stop()

    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    band = img[h // 2 - 40:h // 2 + 40]  # mid-height band, away from HUD text
    left, right = band[:, : w // 3].mean(), band[:, -w // 3:].mean()
    return "right" if right > left else "left"


@pytest.mark.parametrize("engine_name,kwargs", [
    ("BmiEngine", {}),
    ("CurlUpEngine", {}),
    ("SitReachEngine", {"leg_cm": 87.0}),
    ("WallTossEngine", {}),
])
def test_self_facing_engine_publishes_mirrored_preview(engine_name, kwargs):
    import web_engines

    engine = getattr(web_engines, engine_name)(**kwargs)
    assert _published_square_side(engine) == "right", (
        f"{engine_name} published an un-mirrored frame — the left-side square "
        "stayed on the left. Previews must be selfie-mirrored, with "
        "cv2.flip(frame, 1) BEFORE inference."
    )


@pytest.mark.parametrize("module_name,engine_name", [
    ("wingspan_web_engine", "WingspanEngine"),
    ("vertical_jump_web_engine", "VerticalJumpEngine"),
])
def test_standalone_engine_publishes_mirrored_preview(monkeypatch, module_name, engine_name):
    """Wingspan and vertical_jump were the two engines this file never
    covered — they predate BaseWebEngine and open their capture inline, so
    the _open_camera override the other tests use doesn't reach them. That
    is precisely why they could have drifted un-mirrored without failing
    anything, so they get patched at the cv2 level instead.
    """
    import cv2
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(cv2, "VideoCapture", SynthCap)

    engine = getattr(module, engine_name)()
    engine.start()
    try:
        assert wait_until(lambda: engine.get_frame()[0] is not None, timeout=10.0), \
            f"{engine_name} never published a frame"
        time.sleep(0.5)
        jpg, _ = engine.get_frame()
    finally:
        engine.stop()

    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    band = img[h // 2 - 40:h // 2 + 40]
    left, right = band[:, : w // 3].mean(), band[:, -w // 3:].mean()
    assert right > left, (
        f"{engine_name} published an un-mirrored frame — the left-side square "
        "stayed on the left."
    )


def test_dual_feed_engine_mirrors_before_processing(monkeypatch):
    """The field tests share one flip in DualFeedEngine._run_loop. Assert the
    frames handed to _process_pair are already mirrored — that ordering is
    what keeps detection, overlays and calibration clicks in one space."""
    import cv2
    import dual_feed_engines as dfe

    seen = {}

    class Probe(dfe.DualFeedEngine):
        def _load_models(self):
            pass

        def _release_models(self):
            pass

        def _process_pair(self, frame_a, frame_b):
            if "a" not in seen:
                band = frame_a[frame_a.shape[0] // 2]
                left = band[: frame_a.shape[1] // 3].mean()
                right = band[-frame_a.shape[1] // 3:].mean()
                seen["a"] = "right" if right > left else "left"
            return frame_a, frame_b

        def _state_locked(self):
            return {}

    monkeypatch.setattr(cv2, "VideoCapture", SynthCap)
    eng = Probe("0", "1")
    eng.start()
    try:
        assert wait_until(lambda: "a" in seen, timeout=8.0), "_process_pair never ran"
    finally:
        eng.stop()

    assert seen["a"] == "right", (
        "DualFeedEngine handed _process_pair an un-mirrored frame — the "
        "left-side square stayed on the left. Set MIRROR/flip before "
        "_process_pair so detection and clicks share the displayed space."
    )


# ── the opt-out, for cameras that already mirror ─────────────────────────

def test_mirror_false_publishes_the_frame_unflipped(monkeypatch):
    """Some sources mirror themselves — iPhone Continuity Camera, several
    phone IP-cam apps. Flipping those again cancels it out and the coach
    steps left while the picture goes right. Nothing can detect that from
    the pixels, so it's a per-camera setting; this pins that setting off
    genuinely means no flip."""
    import web_engines

    engine = web_engines.BmiEngine(mirror=False)
    assert _published_square_side(engine) == "left", (
        "mirror=False still flipped the frame — the opt-out does nothing, so a "
        "self-mirroring camera stays backwards."
    )


def test_mirror_defaults_to_on(monkeypatch):
    """A plain webcam must keep mirroring without anyone configuring it."""
    import web_engines

    assert web_engines.BmiEngine().mirror_preview is True
    assert web_engines.WingspanEngine().mirror_preview is True
