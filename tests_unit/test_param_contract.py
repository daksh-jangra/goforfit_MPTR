"""`required` in the TESTS registry must match what the builders enforce.

Three places read that flag — the registry itself, screen3's pre-handoff
validation, and screen4's pre-launch guard — and nothing tied any of them to
the builder that actually decides. Get it wrong in either direction and the
failure is quiet:

  marked required, builder defaults it  -> screen3 refuses to hand off and
                                           screen4 shows NOT CONFIGURED for a
                                           value the server would have filled
                                           in; the coach is stuck with no way
                                           to proceed.
  marked optional, builder demands it   -> both screens wave the coach
                                           through and the launch 400s at the
                                           one moment they can't act on it.

So rather than trusting the flag, this asks each builder directly: drop one
param and see whether it still builds.
"""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def app_mod():
    import os

    os.environ["GOFORFIT_AUTH"] = "off"
    for name in ("auth", "app"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("app")
    yield mod
    os.environ.pop("GOFORFIT_AUTH", None)


def _sample(param):
    """A plausible value of the right shape. Builders coerce rather than
    range-check, so the exact number doesn't matter — only that it parses."""
    if param.get("type") == "number":
        return "1"
    return "0"  # works as a camera index and as a one-item comma list


def _full_body(cfg):
    return {p["name"]: _sample(p) for p in cfg.get("params", [])}


def _cases():
    """(test_id, param_name, required) for every param of every test."""
    import os

    os.environ.setdefault("GOFORFIT_AUTH", "off")
    app_mod = importlib.import_module("app")
    for test_id, cfg in app_mod.TESTS.items():
        for param in cfg.get("params", []):
            yield pytest.param(test_id, param["name"], param.get("required", True),
                               id=f"{test_id}:{param['name']}")


def test_a_full_body_builds_every_engine(app_mod):
    """Baseline — if this fails the per-param results below mean nothing."""
    for test_id, cfg in app_mod.TESTS.items():
        slot = app_mod._ENGINES[test_id]
        built = slot.builder(_full_body(cfg))
        assert not isinstance(built, JSONResponse), (
            f"{test_id} rejected a fully-populated body: {built.body}")


@pytest.mark.parametrize("test_id,param_name,required", list(_cases()))
def test_required_flag_matches_what_the_builder_enforces(app_mod, test_id, param_name, required):
    body = _full_body(app_mod.TESTS[test_id])
    body.pop(param_name)

    built = app_mod._ENGINES[test_id].builder(body)
    rejected = isinstance(built, JSONResponse)

    if required:
        assert rejected, (
            f"{test_id}.{param_name} is marked required but the builder "
            "defaults it — screen3 and screen4 will block a launch the server "
            'would have accepted. Mark it "required": False.')
    else:
        assert not rejected, (
            f'{test_id}.{param_name} is marked "required": False but the '
            "builder rejects it — the coach gets waved through and the launch "
            "400s. Drop the flag so both screens ask for it.")
