"""Every self-facing engine must publish a MIRRORED (selfie-style) preview.

The athlete watches themselves on screen4, so their real-world right must
appear on screen right — and the flip must happen BEFORE inference so
landmarks/overlays live in the same flipped space. This was inconsistent
once (curl_up and sit_reach streamed un-mirrored while bmi/wingspan/etc.
mirrored); this test pins the contract.

Field tests (shuttle_run/dash_50m/run_600m) are deliberately NOT mirrored —
coach-view, direction-sensitive lane logic — see DualFeedEngine's note.

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
        "stayed on the left. Self-facing previews must be selfie-mirrored, "
        "with cv2.flip(frame, 1) BEFORE inference."
    )
