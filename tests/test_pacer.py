import os
import sys
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.join(_ROOT, "tests")
for _path in (_ROOT, _TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import app as app_module
import db
from dual_feed_engines import PacerEngine, PacerLineTracker


class _FakePoseApi:
    class PoseLandmark:
        LEFT_ANKLE = 0
        RIGHT_ANKLE = 1

    def __init__(self, ankle_y):
        self.ankle_y = iter(ankle_y)

    def Pose(self, **kwargs):
        api = self

        class FakePose:
            def process(self, _frame):
                y = next(api.ankle_y)
                return SimpleNamespace(pose_landmarks=SimpleNamespace(
                    landmark=[
                        SimpleNamespace(y=y, visibility=1.0),
                        SimpleNamespace(y=y, visibility=1.0),
                    ]
                ))

            def close(self):
                pass

        return FakePose()


class _FakeCounter:
    def __init__(self):
        self.ends = []

    def register_touch(self, _lane, end):
        self.ends.append(end)
        return True


def test_pacer_line_tracker_detects_crossings_in_both_directions():
    counter = _FakeCounter()
    tracker = PacerLineTracker("B", 50, counter, _FakePoseApi([0.4, 0.6, 0.4]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _ in range(3):
        tracker.process(frame)

    assert counter.ends == ["B", "B"]
    tracker.close()


def test_pacer_counts_alternating_crossings_and_stages_selected_distance(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("dual_feed_engines.time.time", lambda: now[0])
    staged = []
    engine = PacerEngine("camera-a", "camera-b", distance_m=15, on_stage=staged.append)
    engine.calib = {"a": object(), "b": object()}

    assert engine.action("start_test")["ok"]
    assert engine.manager.register_touch(1, "B")
    now[0] += 1
    assert engine.manager.register_touch(1, "A")
    engine.maybe_finish()

    assert staged == [{"session_id": None, "distance_m": 15, "completed_laps": 2}]


def test_pacer_refuses_start_until_both_lines_are_calibrated():
    engine = PacerEngine("camera-a", "camera-b")

    assert not engine.action("start_test")["ok"]
    assert not engine.started


def test_pacer_save_api_and_flat_results(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "pacer-test.db")
    db.init_db()
    session_id = db.db.new_session(student_name="PACER Test")
    client = TestClient(app_module.app)

    response = client.post("/api/save/pacer", json={
        "session_id": session_id,
        "distance_m": 20,
        "completed_laps": 12,
    })

    assert response.status_code == 200
    assert response.json()["ok"]
    row = db.db.get_all("pacer", session_id=session_id)[0]
    assert row["completed_laps"] == 12
    assert row["distance_m"] == 20
    flat = db.db.get_all_results()
    assert any(r["test"] == "pacer" and r["primary_result"] == 12 for r in flat)


def test_pacer_setup_and_live_feed_pages_render():
    client = TestClient(app_module.app)

    setup = client.get("/screen3?test=pacer")
    live = client.get("/screen4?test=pacer&cam_a_url=0&cam_b_url=1&distance_m=15")

    assert setup.status_code == 200
    assert "20 metres" in setup.text and "15 metres" in setup.text
    assert live.status_code == 200
    assert "/api/pacer/feed_a" in live.text
    assert "START PACER" in live.text
