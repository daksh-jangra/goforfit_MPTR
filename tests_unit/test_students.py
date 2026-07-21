"""The student roster.

Until this existed, screen3.html carried the entire roster as a hardcoded
two-entry `mockDb` object, and the name written onto every saved measurement
came from that template rather than from any record. These cover the table
that replaced it and the rules that stop one child's result landing on
another's row.
"""

import importlib
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AUTH = {"X-GoForFit-Coach": "Priya"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Runs against a throwaway database — these tests insert and delete
    students, and must never touch the real goforfit.db."""
    monkeypatch.setenv("GOFORFIT_AUTH", "off")
    for name in ("auth", "app", "db"):
        sys.modules.pop(name, None)

    import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    db_mod.init_db()

    app_mod = importlib.import_module("app")
    with TestClient(app_mod.app) as c:
        yield c


def _add(client, **kwargs):
    return client.post("/api/students", json=kwargs, headers=AUTH)


def test_a_new_install_has_an_empty_roster(client):
    """No seeded demo children — a real school's database must not arrive
    pre-populated with people who don't exist."""
    assert client.get("/api/classes", headers=AUTH).json()["classes"] == []
    assert client.get("/api/students", headers=AUTH).json()["students"] == []


def test_adding_a_student_makes_their_class_appear(client):
    r = _add(client, name="Ayaan Yadav", grade="Class 4-A", roll_no="5", reg_no="REG-1")
    assert r.json()["ok"] is True
    assert client.get("/api/classes", headers=AUTH).json()["classes"] == ["Class 4-A"]


def test_students_are_listed_only_for_their_own_class(client):
    _add(client, name="Ayaan", grade="Class 4-A", roll_no="5")
    _add(client, name="Meera", grade="Class 1-D", roll_no="12")

    in_4a = client.get("/api/students?grade=Class 4-A", headers=AUTH).json()["students"]
    assert [s["name"] for s in in_4a] == ["Ayaan"]
    assert len(client.get("/api/students", headers=AUTH).json()["students"]) == 2


def test_roll_numbers_sort_as_numbers(client):
    """'10' must not come before '9' — a coach scanning the list for a roll
    number gives up on a register that isn't in register order."""
    for roll in ("10", "9", "2"):
        _add(client, name=f"Child {roll}", grade="Class 2-B", roll_no=roll)

    listed = client.get("/api/students?grade=Class 2-B", headers=AUTH).json()["students"]
    assert [s["roll_no"] for s in listed] == ["2", "9", "10"]


def test_a_duplicate_roll_in_one_class_is_refused(client):
    """Two children in a class can share a name; the roll number is what tells
    them apart, so a duplicate would make the register ambiguous."""
    _add(client, name="Ayaan", grade="Class 4-A", roll_no="5")
    r = _add(client, name="Someone Else", grade="Class 4-A", roll_no="5")

    assert r.status_code == 409
    assert "already used" in r.json()["error"]


def test_the_same_roll_in_a_different_class_is_fine(client):
    assert _add(client, name="A", grade="Class 4-A", roll_no="5").json()["ok"]
    assert _add(client, name="B", grade="Class 1-D", roll_no="5").json()["ok"]


def test_a_student_needs_a_name_and_a_class(client):
    assert _add(client, name="   ", grade="Class 4-A").status_code == 400
    assert _add(client, name="Ayaan", grade="  ").status_code == 400


def test_deleting_a_student_leaves_their_measurements(client):
    """Removing a child from the register must never quietly erase results
    already recorded against them."""
    import db as db_mod

    sid = db_mod.db.new_session(student_name="Ayaan", grade="Class 4-A", coach="Priya")
    db_mod.db.save_sit_reach(sid, reach_cm=31.5, leg_cm=87)
    student_id = _add(client, name="Ayaan", grade="Class 4-A", roll_no="5").json()["student"]["id"]

    assert client.delete(f"/api/students/{student_id}", headers=AUTH).json()["ok"] is True

    results = client.get("/api/results", headers=AUTH).json()["results"]
    assert [r["student_name"] for r in results] == ["Ayaan"]


def test_deleting_an_unknown_student_is_a_404(client):
    assert client.delete("/api/students/99999", headers=AUTH).status_code == 404


def test_the_roster_is_shared_between_coaches(client):
    """A class belongs to the school, not to whoever entered it — two coaches
    testing the same class must see the same children."""
    _add(client, name="Ayaan", grade="Class 4-A", roll_no="5")

    other = client.get("/api/students?grade=Class 4-A",
                       headers={"X-GoForFit-Coach": "Raj"}).json()["students"]
    assert [s["name"] for s in other] == ["Ayaan"]


def test_screen3_no_longer_hardcodes_a_roster():
    """The template used to BE the database. If mockDb comes back, the name on
    a saved measurement stops coming from a record again."""
    screen3 = (PROJECT_ROOT / "templates" / "screen3.html").read_text()
    assert "mockDb" not in screen3
    assert "/api/students" in screen3 and "/api/classes" in screen3
