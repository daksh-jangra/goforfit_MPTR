"""
Logic checks for what Wall Toss hands to screen3 on STOP - no camera, no
participant.

The 30 s timer stages its catch count, 0 included. STOP before the timer used
to stage only a count above 0, so a student stopped early on 0 left nothing
to review. Now a running trial stages its count; a STOP before START stages
nothing.

      python3 tests/test_wall_toss.py
      pytest tests/test_wall_toss.py
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import web_engines  # noqa: E402


def _engine():
    staged = []
    eng = web_engines.WallTossEngine(on_stage=lambda *a: staged.append(a))
    eng._load_models()
    return eng, staged


def _preset(eng):
    return eng.counter._preset_name


def test_stop_after_start_with_zero_stages_zero():
    eng, staged = _engine()
    assert eng.action_start()["ok"]
    eng.stop()
    assert staged == [(0, _preset(eng))], staged


def test_stop_after_start_stages_the_count():
    eng, staged = _engine()
    eng.action_start()
    eng.counter.toss_count = 7
    eng.stop()
    assert staged == [(7, _preset(eng))], staged


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
