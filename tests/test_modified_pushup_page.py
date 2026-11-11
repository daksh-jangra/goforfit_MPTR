"""
Checks for the live site's Modified Push-Up tracker page - no camera, no
participant, nothing written to the database.

The page at /modified_pushup_tracker came from the live server's copy of the
app; it must still load, and its save call must answer without a server error
whatever it is sent, because the page ignores the reply and syncs to the
website either way.

      python3 tests/test_modified_pushup_page.py
      pytest tests/test_modified_pushup_page.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402

_client = TestClient(app_module.app)


def test_tracker_page_loads():
    r = _client.get("/modified_pushup_tracker?student_id=5&session_id=1")
    assert r.status_code == 200, r.status_code
    assert "/api/save/modified_pushup" in r.text


def test_save_without_session_is_refused():
    r = _client.post("/api/save/modified_pushup", json={"reps": 3})
    assert r.status_code == 400, r.status_code


def test_save_for_unknown_session_is_refused_not_a_crash():
    r = _client.post("/api/save/modified_pushup",
                     json={"session_id": 987654321, "reps": 3, "duration_sec": 30})
    assert r.status_code == 400, (r.status_code, r.text)


def test_tracker_is_not_listed_as_a_test():
    assert "modified_pushup" not in app_module.TESTS
    assert _client.get("/modified_pushup", follow_redirects=False).status_code == 404


def _main():
    checks = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in checks:
        try:
            fn()
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n          {exc}")
        else:
            print(f"  PASS  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
