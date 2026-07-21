"""Two coaches, one server.

Every test here is a scenario that used to go wrong, in order of how badly:

  - Raj's screen3 picked up Priya's finished reading and saved it against
    HER student. Silent, and the number looked plausible.
  - Either coach could stop the other's test mid-measurement.
  - The second coach to press START was refused outright, naming a test they
    had never launched.

The engines are faked: these assert the ownership and guard logic in app.py,
which is where all three bugs lived. Nothing here opens a camera.
"""

import importlib
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIYA = {"X-GoForFit-Coach": "Priya"}
RAJ = {"X-GoForFit-Coach": "Raj"}


class FakeEngine:
    """Stands in for a camera engine: records stop() so a test can prove
    whose engine a request actually reached."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stopped = False
        self._running = True

    def start(self):
        pass

    def stop(self):
        self.stopped = True
        self._running = False

    def is_running(self):
        return self._running

    def get_state(self):
        return {"running": self._running}


@pytest.fixture
def app_mod(monkeypatch):
    monkeypatch.setenv("GOFORFIT_AUTH", "off")  # the gate is covered in test_auth
    for name in ("auth", "app"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("app")

    # Every engine becomes a FakeEngine, and every builder a trivial one, so
    # launching touches no hardware. The builder still receives the body, so
    # the coach/camera injection is exercised for real.
    built = []

    def fake_builder(body):
        eng = FakeEngine(**body)
        built.append(eng)
        return eng

    for slot in mod._ENGINES.values():
        monkeypatch.setattr(slot, "builder", fake_builder)
        monkeypatch.setattr(slot, "cls", FakeEngine)

    mod._built = built
    yield mod


@pytest.fixture
def client(app_mod):
    with TestClient(app_mod.app, follow_redirects=False) as c:
        yield c


def _launch(client, headers, test_id="bmi", camera="0"):
    client.cookies.set("gf_camera", camera)
    return client.post(f"/api/launch/{test_id}", json={}, headers=headers)


# ── the wrong-student bug ────────────────────────────────────────────────

def test_one_coachs_staged_result_is_invisible_to_the_other(client):
    """The worst of the three: silent, and the number looks right."""
    client.post("/api/stage/bmi", json={"session_id": 1, "height_cm": 160}, headers=PRIYA)

    assert client.get("/api/pending/bmi", headers=PRIYA).json()["pending"]["height_cm"] == 160
    assert client.get("/api/pending/bmi", headers=RAJ).json()["pending"] is None


def test_clearing_one_coachs_result_leaves_the_others(client):
    client.post("/api/stage/bmi", json={"height_cm": 160}, headers=PRIYA)
    client.post("/api/stage/bmi", json={"height_cm": 150}, headers=RAJ)

    client.post("/api/pending/bmi/clear", headers=RAJ)

    assert client.get("/api/pending/bmi", headers=PRIYA).json()["pending"]["height_cm"] == 160
    assert client.get("/api/pending/bmi", headers=RAJ).json()["pending"] is None


def test_a_relaunch_only_clears_its_own_stale_result(client):
    client.post("/api/stage/bmi", json={"height_cm": 150}, headers=RAJ)
    _launch(client, PRIYA, "bmi", camera="0")
    assert client.get("/api/pending/bmi", headers=RAJ).json()["pending"]["height_cm"] == 150


# ── stopping someone else's test ─────────────────────────────────────────

def test_a_coach_cannot_stop_the_others_test(app_mod, client):
    _launch(client, PRIYA, "bmi", camera="0")
    priya_engine = app_mod._built[-1]

    client.post("/api/stop/bmi", headers=RAJ)

    assert not priya_engine.stopped, "Raj stopped Priya's running test"
    assert client.get("/api/status/bmi", headers=PRIYA).json()["running"] is True


def test_a_coach_can_stop_their_own(app_mod, client):
    _launch(client, PRIYA, "bmi", camera="0")
    priya_engine = app_mod._built[-1]

    client.post("/api/stop/bmi", headers=PRIYA)

    assert priya_engine.stopped
    assert client.get("/api/status/bmi", headers=PRIYA).json()["running"] is False


def test_state_and_actions_only_reach_your_own_engine(client):
    _launch(client, PRIYA, "bmi", camera="0")

    assert client.get("/api/bmi/state", headers=PRIYA).json()["running"] is True
    assert client.get("/api/bmi/state", headers=RAJ).json()["running"] is False
    # Raj has nothing running, so his action is refused rather than applied
    # to Priya's engine.
    assert client.post("/api/bmi/action", json={"action": "scan"}, headers=RAJ).status_code == 409


# ── concurrency ──────────────────────────────────────────────────────────

def test_two_coaches_with_different_cameras_run_at_once(client):
    """The whole point. This was a flat 409 before."""
    assert _launch(client, PRIYA, "bmi", camera="0").json()["ok"] is True
    assert _launch(client, RAJ, "curl_up", camera="http://192.168.1.9:4747/video").json()["ok"] is True

    assert client.get("/api/status/bmi", headers=PRIYA).json()["running"] is True
    assert client.get("/api/status/curl_up", headers=RAJ).json()["running"] is True


def test_two_coaches_on_the_same_camera_are_refused(client):
    """One device genuinely cannot serve two engines — refuse, and say who
    has it, instead of a confusing open failure."""
    _launch(client, PRIYA, "bmi", camera="0")
    r = _launch(client, RAJ, "curl_up", camera="0")

    assert r.status_code == 409
    assert r.json()["held_by"] == "Priya"
    assert "Priya" in r.json()["error"]


def test_the_same_camera_spelled_differently_still_collides(client):
    """'0' and 0 and '0/' are one device; the guard must not be fooled."""
    _launch(client, PRIYA, "bmi", camera="0")
    assert _launch(client, RAJ, "curl_up", camera=" 0 ").status_code == 409


def test_a_coach_still_runs_only_one_test_at_a_time(client):
    _launch(client, PRIYA, "bmi", camera="0")
    r = _launch(client, PRIYA, "curl_up", camera="0")
    assert r.status_code == 409
    assert r.json()["running"] == "bmi"


def test_stopping_frees_the_camera_for_the_other_coach(client):
    _launch(client, PRIYA, "bmi", camera="0")
    assert _launch(client, RAJ, "curl_up", camera="0").status_code == 409

    client.post("/api/stop/bmi", headers=PRIYA)

    assert _launch(client, RAJ, "curl_up", camera="0").json()["ok"] is True


def test_a_failed_launch_does_not_keep_holding_the_camera(app_mod, client, monkeypatch):
    """A builder that rejects the body must not leave the camera claimed —
    that would lock everyone out of a device nobody is using."""
    from fastapi.responses import JSONResponse

    slot = app_mod._ENGINES["bmi"]
    monkeypatch.setattr(slot, "builder",
                        lambda body: JSONResponse({"ok": False, "error": "nope"}, status_code=400))

    assert _launch(client, PRIYA, "bmi", camera="0").status_code == 400
    assert _launch(client, RAJ, "curl_up", camera="0").json()["ok"] is True


# ── the coach reaches the engine and the database ────────────────────────

def test_the_engine_is_built_with_the_coachs_own_camera(app_mod, client):
    _launch(client, PRIYA, "bmi", camera="http://phone.local/video")
    assert app_mod._built[-1].kwargs["_camera"] == "http://phone.local/video"
    assert app_mod._built[-1].kwargs["_coach"] == "Priya"


def test_a_launch_cannot_claim_to_be_another_coach(app_mod, client):
    """The body is page-supplied; identity must come from the cookie/header."""
    client.cookies.set("gf_camera", "0")
    client.post("/api/launch/bmi", json={"_coach": "Somebody Else"}, headers=PRIYA)
    assert app_mod._built[-1].kwargs["_coach"] == "Priya"


def test_sessions_are_stamped_and_scoped(client):
    created = client.post("/api/session", json={"student_name": "Ayaan"}, headers=PRIYA).json()
    assert created["ok"] is True

    mine = client.get("/api/sessions", headers=PRIYA).json()["sessions"]
    theirs = client.get("/api/sessions", headers=RAJ).json()["sessions"]
    everyone = client.get("/api/sessions?everyone=1", headers=RAJ).json()["sessions"]

    ids = lambda rows: {r["id"] for r in rows}
    assert created["session_id"] in ids(mine)
    assert created["session_id"] not in ids(theirs)
    assert created["session_id"] in ids(everyone), "?everyone=1 must still show it"

    # don't leave the probe row behind
    client.delete(f"/api/session/{created['session_id']}", headers=PRIYA)
