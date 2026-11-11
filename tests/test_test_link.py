"""
Logic checks for website links that name the test in the path - no camera,
no participant.

The website opens a test as /fms/<test_id>?id=...&name=...&roll=..., while the
camera page lives at /screen4?test=<test_id>. Such a link is sent on to screen4
with every query value kept; anything that is not a test id stays Not Found.

      python3 tests/test_test_link.py
      pytest tests/test_test_link.py
"""

import os
import sys
from urllib.parse import parse_qsl, urlsplit

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402

_client = TestClient(app_module.app)

WEBSITE_QUERY = ("test=isometric_push_up&id=13268&name=Divit+Varma&roll=8"
                 "&class_info=Class+5-A&ex_id=1031&trainerId=42")


def _go(path):
    return _client.get(path, follow_redirects=False)


def test_website_link_goes_to_screen4_with_every_value():
    r = _go("/isometric_push_up?" + WEBSITE_QUERY)
    assert r.status_code == 307, r.status_code
    loc = urlsplit(r.headers["location"])
    assert loc.path == "screen4", loc
    assert dict(parse_qsl(loc.query)) == dict(parse_qsl(WEBSITE_QUERY)), loc.query


def test_redirect_is_relative_so_it_stays_under_fms():
    r = _go("/isometric_push_up?" + WEBSITE_QUERY)
    assert not r.headers["location"].startswith("/"), r.headers["location"]


def test_path_test_id_wins_over_a_different_query_test():
    r = _go("/push_up?test=bench_press&id=1")
    pairs = parse_qsl(urlsplit(r.headers["location"]).query)
    assert [v for k, v in pairs if k == "test"] == ["push_up"], pairs


def test_link_without_query_still_opens_the_test():
    r = _go("/wheelchair_ramp")
    assert r.status_code == 307, r.status_code
    assert r.headers["location"] == "screen4?test=wheelchair_ramp", r.headers["location"]


def test_every_test_id_has_a_link():
    for tid in app_module.TESTS:
        r = _go("/" + tid)
        assert r.status_code == 307, (tid, r.status_code)


def test_unknown_name_is_still_not_found():
    r = _go("/not_a_test?id=1")
    assert r.status_code == 404, r.status_code
    assert r.json() == {"detail": "Not Found"}


def test_existing_pages_are_not_redirected():
    for path in ("/", "/screen2", "/screen3?test=push_up", "/screen4?test=push_up", "/results"):
        r = _go(path)
        assert r.status_code == 200, (path, r.status_code)


def test_followed_link_lands_on_the_right_test_page():
    r = _client.get("/isometric_push_up?" + WEBSITE_QUERY)
    assert r.status_code == 200, r.status_code
    title = app_module.TESTS["isometric_push_up"]["title"]
    assert title in r.text, title


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
