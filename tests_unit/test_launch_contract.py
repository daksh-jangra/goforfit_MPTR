"""What /api/launch may and may not do.

The server used to carry a second launch path that spawned the original
desktop OpenCV script as a subprocess. Every registered test is served by an
in-process engine, so that path could never run — it was untested code that
read like a working fallback, and when it HAD run it produced the "camera
opens in another window and nothing syncs back" reports.

It is gone. test_every_registered_test_has_an_engine is what keeps its
absence honest: add a test with no engine slot and this fails, pointing at
the decision rather than letting a launch quietly 503 in the field.
"""

import importlib
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def app_mod(monkeypatch):
    monkeypatch.setenv("GOFORFIT_AUTH", "off")  # gate is covered in test_auth
    monkeypatch.delenv("GOFORFIT_TOKEN", raising=False)
    for name in ("auth", "app"):
        sys.modules.pop(name, None)
    return importlib.import_module("app")


def test_every_registered_test_has_an_engine(app_mod):
    missing = [tid for tid in app_mod.TESTS if tid not in app_mod._ENGINES]
    assert not missing, (
        f"{missing} have no engine slot. There is no subprocess fallback any "
        "more — either give them an engine or write that launcher back "
        "deliberately, with tests."
    )


def test_every_registered_test_is_live_feed(app_mod):
    """The registry flag and the engine registry must not disagree."""
    not_live = [tid for tid, cfg in app_mod.TESTS.items() if not cfg.get("live_feed")]
    assert not not_live, f"{not_live} claim to need a desktop window"


def test_launch_refuses_when_the_engine_failed_to_import(app_mod, monkeypatch):
    """The one path that reaches the old fallback: a broken engine module.
    It must refuse, not degrade into spawning a desktop window."""
    slot = app_mod._ENGINES["bmi"]
    monkeypatch.setattr(slot, "cls", None)

    with TestClient(app_mod.app) as client:
        r = client.post("/api/launch/bmi", json={})

    assert r.status_code == 503
    body = r.json()
    assert body["ok"] is False
    assert "unavailable" in body["error"]


def test_launch_never_spawns_a_subprocess(app_mod, monkeypatch):
    """Belt and braces: nothing under /api/launch may reach Popen."""
    import subprocess

    def explode(*args, **kwargs):
        raise AssertionError("launch tried to spawn a subprocess")

    monkeypatch.setattr(subprocess, "Popen", explode)
    slot = app_mod._ENGINES["bmi"]
    monkeypatch.setattr(slot, "cls", None)

    with TestClient(app_mod.app) as client:
        assert client.post("/api/launch/bmi", json={}).status_code == 503


def test_unknown_test_is_a_404(app_mod):
    with TestClient(app_mod.app) as client:
        r = client.post("/api/launch/not_a_test", json={})
    assert r.status_code == 404


def test_pyaudio_is_not_a_required_dependency():
    """It needs portaudio at the OS level and only the two unreachable legacy
    scripts import it — the live engines use sounddevice."""
    lines = [l.split("#")[0].strip()
             for l in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()]
    assert not [l for l in lines if l.startswith("pyaudio")]
    assert [l for l in lines if l.startswith("sounddevice")]
