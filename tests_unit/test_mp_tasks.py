"""The Tasks-API migration, and the pin it removed.

mediapipe 0.10.30 deleted the `solutions` namespace. This project was pinned
below that, which also pinned it to Python <= 3.12 (no sub-0.10.30 wheels for
3.13). Every engine is now on the Tasks API, so both pins are gone.

test_no_engine_imports_the_legacy_api is the guard: it fails the moment
anything reaches for `mp.solutions` again, which would silently re-impose the
version ceiling — silently, because the pinned dev venv still HAS that API,
so the offending code would work locally and only break on a fresh install.
"""

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PROJECT_ROOT / "tests"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))


def _engine_sources():
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.endswith(".bak"):
            continue
        yield path


def _legacy_uses(tree):
    """Line numbers of real `solutions` usage — parsed, not grepped, so the
    docstrings and comments that EXPLAIN the migration don't trip it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "solutions":
            if isinstance(node.value, ast.Name) and node.value.id in ("mp", "mediapipe"):
                yield node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "mediapipe":
            if any(a.name == "solutions" for a in node.names):
                yield node.lineno


def test_no_engine_imports_the_legacy_api():
    offenders = []
    for path in _engine_sources():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))
        offenders += [f"{path.relative_to(PROJECT_ROOT)}:{n}" for n in _legacy_uses(tree)]
    assert not offenders, (
        "mediapipe.solutions was removed in 0.10.30 — use tests/mp_tasks.py "
        "instead. Offending lines: " + ", ".join(offenders)
    )


def test_requirements_no_longer_pin_below_the_cutoff():
    lines = [l.split("#")[0].strip()
             for l in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()]
    mediapipe = [l for l in lines if l.startswith("mediapipe")]
    assert mediapipe == ["mediapipe>=0.10.30"], (
        f"expected a single un-ceilinged mediapipe requirement, got {mediapipe}")


# ── the helper module itself ─────────────────────────────────────────────

def test_connection_pairs_are_index_tuples():
    """Drawing iterates these as (a, b) index pairs into the landmark list."""
    import mp_tasks

    for connections in (mp_tasks.HAND_CONNECTIONS, mp_tasks.POSE_CONNECTIONS):
        assert connections, "empty connection list would draw nothing"
        for pair in connections:
            assert len(pair) == 2 and all(isinstance(i, int) for i in pair)


def test_video_clock_strictly_advances():
    """VIDEO mode rejects a timestamp that doesn't move forward, and two
    frames can land in the same millisecond — so the clock counts frames
    rather than reading the wall clock."""
    import mp_tasks

    clock = mp_tasks.VideoClock()
    stamps = [clock.tick() for _ in range(100)]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_each_clock_is_independent():
    """shuttle_run builds one detector per lane per camera; a shared counter
    would have them fighting over the timeline."""
    import mp_tasks

    a, b = mp_tasks.VideoClock(), mp_tasks.VideoClock()
    a.tick(); a.tick()
    assert b.tick() < a.tick()


def test_pose_landmark_indices_match_mediapipes_ordering():
    """There is no PoseLandmark enum under tasks, so these are hardcoded —
    if they drift, shuttle_run silently tracks the wrong body part."""
    import mp_tasks

    assert (mp_tasks.LEFT_ANKLE, mp_tasks.RIGHT_ANKLE) == (27, 28)
    assert (mp_tasks.LEFT_SHOULDER, mp_tasks.RIGHT_SHOULDER) == (11, 12)
    assert (mp_tasks.LEFT_HIP, mp_tasks.RIGHT_HIP) == (23, 24)


def test_drawing_handles_landmarks_without_visibility():
    """Hand landmarks carry no .visibility; pose landmarks do. One drawing
    path serves both, so the missing attribute must not raise."""
    import numpy as np
    import mp_tasks

    class Bare:
        x, y, z = 0.5, 0.5, 0.0

    frame = np.zeros((60, 60, 3), np.uint8)
    mp_tasks.draw_hand_skeleton(frame, [[Bare() for _ in range(21)]])
    assert frame.any(), "nothing was drawn"


def test_drawing_tolerates_empty_results():
    """A frame with no hand or pose in it is the common case, not an error."""
    import numpy as np
    import mp_tasks

    frame = np.zeros((60, 60, 3), np.uint8)
    mp_tasks.draw_hand_skeleton(frame, [])
    mp_tasks.draw_hand_skeleton(frame, None)
    mp_tasks.draw_pose_skeleton(frame, type("R", (), {"pose_landmarks": []})())
    assert not frame.any()


@pytest.mark.parametrize("name", ["hand_landmarker.task", "pose_landmarker_lite.task"])
def test_models_resolve_next_to_the_engines(name):
    """Resolved from the module's own directory, not cwd — the web app starts
    the engines from a different working directory than a manual run."""
    import mp_tasks

    resolved = Path(mp_tasks.model_path(name))
    assert resolved.is_absolute() and resolved.exists()
    assert resolved.parent == ENGINE_DIR
