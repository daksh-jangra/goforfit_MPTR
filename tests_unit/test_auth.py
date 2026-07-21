"""Shared-token gate on every route.

What these protect: the coach setup is GOFORFIT_HOST=0.0.0.0, so before this
gate existed anyone on the school Wi-Fi could read student records and DELETE
sessions. The gate is whole-app rather than per-endpoint precisely so a new
route can't be added unprotected — test_every_route_is_gated is the check
that keeps that true as routes are added.

app.py reads its token at import, so each test imports the module fresh with
the environment it wants.
"""

import importlib
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOKEN = "s3cret-code"


def _load_app(monkeypatch, token=TOKEN, auth=None):
    monkeypatch.delenv("GOFORFIT_TOKEN", raising=False)
    monkeypatch.delenv("GOFORFIT_AUTH", raising=False)
    if token is not None:
        monkeypatch.setenv("GOFORFIT_TOKEN", token)
    if auth is not None:
        monkeypatch.setenv("GOFORFIT_AUTH", auth)
    for name in ("auth", "app"):
        sys.modules.pop(name, None)
    return importlib.import_module("app")


@pytest.fixture
def client(monkeypatch):
    app_mod = _load_app(monkeypatch)
    with TestClient(app_mod.app, follow_redirects=False) as c:
        yield c


# ── the gate itself ──────────────────────────────────────────────────────

def test_page_without_token_redirects_to_unlock(client):
    r = client.get("/screen3?test=sit_reach")
    assert r.status_code == 303
    assert r.headers["location"].startswith("/unlock?")
    # and it remembers where the coach was going
    assert "screen3" in r.headers["location"]


def test_api_without_token_is_401_not_a_redirect(client):
    """fetch() needs a status it can act on, not an HTML login page."""
    r = client.get("/api/sessions")
    assert r.status_code == 401
    assert r.json()["ok"] is False


def test_destructive_routes_are_gated(client):
    """The concrete harm: an unauthenticated stranger deleting a session."""
    assert client.delete("/api/session/1").status_code == 401
    assert client.delete("/api/result/bmi/1").status_code == 401


def test_student_data_is_not_readable_without_a_token(client):
    for path in ("/api/sessions", "/api/results", "/api/export/bmi", "/results"):
        r = client.get(path)
        assert r.status_code in (401, 303), f"{path} leaked with {r.status_code}"


def test_wrong_token_is_rejected(client):
    r = client.get("/api/sessions", headers={"X-GoForFit-Token": "not-it"})
    assert r.status_code == 401


def test_every_route_is_gated(client):
    """No route may answer un-authenticated except the unlock page itself."""
    exempt = {"/unlock", "/favicon.ico"}
    checked = 0
    for route in client.app.routes:
        path, methods = getattr(route, "path", None), getattr(route, "methods", None)
        if not path or not methods or path in exempt or "{" in path:
            continue
        for method in methods & {"GET", "POST", "DELETE"}:
            r = client.request(method, path)
            assert r.status_code in (303, 401), f"{method} {path} answered {r.status_code}"
            checked += 1
    assert checked > 10, "route sweep found almost nothing — did the app fail to build?"


# ── ways in ──────────────────────────────────────────────────────────────

def test_header_token_is_accepted(client):
    assert client.get("/api/sessions", headers={"X-GoForFit-Token": TOKEN}).status_code == 200


def test_cookie_token_is_accepted(client):
    client.cookies.set("gf_token", TOKEN)
    assert client.get("/api/sessions").status_code == 200


def test_query_token_is_swapped_for_a_cookie(client):
    """A phone following a QR code arrives with ?token=…; it must not stay in
    the URL, where it would sit in history and leak via Referer."""
    r = client.get("/screen2?category=All+Tests&token=" + TOKEN)
    assert r.status_code == 303
    assert "token" not in r.headers["location"]
    assert "category=All+Tests" in r.headers["location"]
    assert "gf_token" in r.cookies


def test_unlock_page_is_reachable_without_a_token(client):
    assert client.get("/unlock").status_code == 200


def test_unlock_with_the_right_code_sets_the_cookie(client):
    r = client.post("/unlock", data={"token": TOKEN, "next": "/results"})
    assert r.status_code == 303
    assert r.headers["location"] == "/results"
    assert r.cookies.get("gf_token") == TOKEN


def test_unlock_with_the_wrong_code_does_not(client):
    r = client.post("/unlock", data={"token": "nope", "next": "/"})
    assert r.status_code == 401
    assert "gf_token" not in r.cookies


def test_non_ascii_token_is_refused_not_a_500(client):
    """secrets.compare_digest raises TypeError on non-ASCII str, so a code
    typed with an accent used to come back as a traceback instead of a
    refusal. (The header route reaches the same comparison but httpx won't
    send a non-ASCII header at all — curl will, which is how this surfaced.)"""
    assert client.post("/unlock", data={"token": "café", "next": "/"}).status_code == 401


def test_tokens_match_is_total():
    """The comparison must never raise, whatever arrives in a cookie."""
    import auth

    for junk in ("café", "", "🎽", "x" * 10_000, None, 42, b"bytes"):
        assert auth.tokens_match(junk, "the-real-token") is False
    assert auth.tokens_match("the-real-token", "the-real-token") is True


def test_a_stale_cookie_does_not_block_a_valid_token_link(client):
    """The common case, not an edge case: an unconfigured server mints a new
    token every restart, so every previously-unlocked device holds a dead
    cookie. A fresh ?token= link must still get them in."""
    client.cookies.set("gf_token", "token-from-a-previous-run")
    r = client.get("/screen2?token=" + TOKEN)

    assert r.status_code == 303
    assert "/unlock" not in r.headers["location"]
    assert r.cookies.get("gf_token") == TOKEN, "stale cookie was not replaced"


def test_the_token_never_reaches_the_unlock_page(client):
    """A rejected request must not carry the attempted token into `next`,
    where it would sit in the form, the address bar and history."""
    r = client.get("/screen2?category=All&token=whatever-was-tried")
    assert r.status_code == 303
    assert "whatever-was-tried" not in r.headers["location"]
    assert "category" in r.headers["location"]


def test_a_valid_header_wins_over_a_stale_cookie(client):
    """Checking only the first credential PRESENT would fail this."""
    client.cookies.set("gf_token", "stale")
    r = client.get("/api/sessions", headers={"X-GoForFit-Token": TOKEN})
    assert r.status_code == 200


def test_unlock_will_not_bounce_to_another_host(client):
    """`next` is attacker-controllable — it must stay same-site."""
    r = client.post("/unlock", data={"token": TOKEN, "next": "//evil.example/x"})
    assert r.headers["location"] == "/"


# ── configuration ────────────────────────────────────────────────────────

def test_auth_off_opens_everything(monkeypatch):
    app_mod = _load_app(monkeypatch, token=None, auth="off")
    with TestClient(app_mod.app, follow_redirects=False) as c:
        assert c.get("/api/sessions").status_code == 200


def test_token_is_generated_when_unconfigured(monkeypatch):
    """Forgetting to set anything must fail CLOSED, with a token you can read
    off the console — not fall back to no auth at all."""
    app_mod = _load_app(monkeypatch, token=None)
    assert app_mod.ACCESS_TOKEN and app_mod.ACCESS_TOKEN_GENERATED
    with TestClient(app_mod.app, follow_redirects=False) as c:
        assert c.get("/api/sessions").status_code == 401
        assert c.get("/api/sessions",
                     headers={"X-GoForFit-Token": app_mod.ACCESS_TOKEN}).status_code == 200
