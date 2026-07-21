"""Shared fixtures for the engine unit tests.

These tests import the REAL engine modules (so mediapipe/opencv must be
installed — use the project .venv) but replace cv2.VideoCapture with a fake,
so no camera hardware is touched and capture acquire/release can be asserted.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

# The engine modules live in tests/ (which holds CV engines, not tests) and
# import each other by bare name, so that directory must be importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


class FakeCap:
    """Stands in for cv2.VideoCapture: tracks open/release so tests can
    assert exactly when the 'camera' is acquired and freed."""

    instances: list["FakeCap"] = []
    fail_sources: set = set()  # sources that must never open (incl. retries)

    def __init__(self, *args, **kwargs):
        self.source = args[0] if args else None
        self.released = False
        self.opened = self.source not in FakeCap.fail_sources
        FakeCap.instances.append(self)

    def isOpened(self):
        return self.opened and not self.released

    def set(self, *args):
        pass

    def get(self, *args):
        return 30.0

    def release(self):
        self.released = True

    def read(self):
        return True, object()


@pytest.fixture
def fake_camera(monkeypatch):
    """Replace cv2.VideoCapture with FakeCap for the duration of a test."""
    import cv2

    FakeCap.instances.clear()
    FakeCap.fail_sources = set()
    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    return FakeCap


def wait_until(predicate, timeout=2.0, interval=0.02):
    """Poll until predicate() is true or timeout — for reaper-thread asserts."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def gate():
    """An Event a test engine's loop can block on, simulating a hung cap.read().
    Always released at teardown so no loop thread outlives its test."""
    g = threading.Event()
    yield g
    g.set()
