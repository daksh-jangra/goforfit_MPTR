"""
Logic checks for what the Partial Curl-Up hands to screen3 on STOP - no
camera, no participant.

The 30 s timer stages its count, 0 included. STOP before the timer used to
stage only a count above 0, so a student stopped early on 0 left nothing to
review. Now any started trial stages its count; a STOP before START stages
nothing.

      python3 tests/test_curl_up.py
      pytest tests/test_curl_up.py
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import web_engines  # noqa: E402


def _engine():
    staged = []
    eng = web_engines.CurlUpEngine(on_stage=lambda *a: staged.append(a))
    return eng, staged


def test_stop_after_start_with_zero_stages_zero():
    eng, staged = _engine()
    assert eng.action_start()["ok"]
    eng.stop()
    assert staged == [(0,)], staged


def test_stop_after_start_stages_the_count():
    eng, staged = _engine()
    eng.action_start()
    eng.count = 7
    eng.stop()
    assert staged == [(7,)], staged


def test_stop_before_start_stages_nothing():
    eng, staged = _engine()
    eng.stop()
    assert staged == [], staged


def test_reset_then_stop_before_start_stages_nothing():
    eng, staged = _engine()
    eng.action_start()
    eng.action_reset()
    eng.stop()
    assert staged == [], staged


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
