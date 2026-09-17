"""
GoForFit NEP FMS — Database Layer
===================================
SQLite-backed storage with one table per fitness test.
All tables share a common `sessions` parent via session_id (FK).

Usage (standalone):
    python db.py          # creates goforfit.db and prints schema info

Usage (from app.py):
    from db import db
    sid = db.new_session(student_name="Riya", age=14, gender="F", school="DPS")
    db.save_bmi(sid, height_cm=163.2, weight_kg=52.0)
    db.save_wingspan(sid, wingspan_cm=162.5)
    rows = db.get_all("bmi")
"""

import sqlite3
import datetime
import json
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "goforfit.db"


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row          # dict-like rows
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
-- ── Shared session / student info ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT,
    age          INTEGER,
    gender       TEXT CHECK(gender IN ('M','F','Other',NULL)),
    school       TEXT,
    grade        TEXT,
    notes        TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Student roster (the list screen3 picks from) ──────────────────────────
-- A class roster belongs to the school, so it is deliberately NOT scoped per
-- coach: two coaches testing the same class must see the same children.
CREATE TABLE IF NOT EXISTS students (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    grade      TEXT NOT NULL,          -- the "class", e.g. "Class 4-A"
    roll_no    TEXT,
    reg_no     TEXT,
    school     TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),
    -- Two children in one class can share a name; the roll number is what the
    -- school uses to tell them apart, so that pair is what must be unique.
    UNIQUE (grade, roll_no)
);

-- ── BMI / Height Estimator ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bmi (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    height_cm        REAL,
    weight_kg        REAL,
    bmi_value        REAL,
    bmi_category     TEXT,          -- Underweight / Normal / Overweight / Obese
    calibration_used INTEGER DEFAULT 0,   -- 1 = calibrated, 0 = uncalibrated
    recorded_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Wingspan Measurement ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wingspan (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    wingspan_cm             REAL,
    height_cm               REAL,         -- snapshot height for ratio
    wingspan_ratio          REAL,         -- wingspan / height  (NULL if height missing)
    calibration_distance_cm REAL,
    recorded_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Partial Curl-Up Counter ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS curl_up (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    reps         INTEGER,        -- total curl-ups in 30 seconds
    duration_sec INTEGER DEFAULT 30,
    rating       TEXT,           -- Excellent / Good / Average / Poor (optional)
    recorded_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Modified Pull-Up Counter ──────────────────────────────────────────────
-- One untimed trial; duration_sec is how long the attempt took, recorded for
-- context only (the Brockport protocol scores reps alone).
CREATE TABLE IF NOT EXISTS pull_up (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    reps         INTEGER,        -- total correct modified pull-ups
    duration_sec REAL,
    rating       TEXT,
    recorded_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Wall Toss Coordination Test ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wall_toss (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    catches        INTEGER,      -- alternating catches in 30 seconds
    duration_sec   INTEGER DEFAULT 30,
    ball_colour    TEXT,         -- colour preset used (e.g. "orange")
    rating         TEXT,
    recorded_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── 4×10 m Shuttle Run Timer ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shuttle_run (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    lane         INTEGER DEFAULT 1,  -- lane number 1–4
    time_sec     REAL,               -- total elapsed time for 4×10 m
    rating       TEXT,
    recorded_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── 50 m Dash Timer ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dash_50m (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    lane         INTEGER DEFAULT 1,
    time_sec     REAL,
    start_mode   TEXT DEFAULT 'whistle',  -- 'whistle' | 'manual'
    rating       TEXT,
    recorded_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Sit and Reach Test ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sit_reach (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    reach_cm        REAL,        -- forward reach (positive = past feet)
    leg_cm          REAL,        -- hip-to-ankle length used for calibration
    rating          TEXT,
    recorded_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Vertical Jump Test ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vertical_jump (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    jump_cm           REAL,        -- best (max) jump height across attempts
    average_jump_cm   REAL,
    standing_reach_cm REAL,
    attempts_json     TEXT,        -- JSON array of per-attempt jump heights (cm)
    rating            TEXT,
    recorded_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── 600 m Run (multi-lane colour tracking) ────────────────────────────────
CREATE TABLE IF NOT EXISTS run_600m (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    lane         INTEGER DEFAULT 1,   -- lane number 1–5
    time_sec     REAL,                -- total elapsed time to complete 600 m
    lap_count    INTEGER,             -- number of finish-line crossings recorded
    rating       TEXT,
    recorded_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

-- ── Convenience view: all results flat ────────────────────────────────────
DROP VIEW IF EXISTS all_results;
CREATE VIEW IF NOT EXISTS all_results AS
    SELECT 'bmi' AS test, b.id AS row_id, b.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           b.height_cm, b.weight_kg, b.bmi_value AS primary_result, b.bmi_category AS rating, b.recorded_at
    FROM bmi b LEFT JOIN sessions s ON b.session_id = s.id
UNION ALL
    SELECT 'wingspan', w.id, w.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, w.wingspan_cm, NULL, w.recorded_at
    FROM wingspan w LEFT JOIN sessions s ON w.session_id = s.id
UNION ALL
    SELECT 'curl_up', c.id, c.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, c.reps, c.rating, c.recorded_at
    FROM curl_up c LEFT JOIN sessions s ON c.session_id = s.id
UNION ALL
    SELECT 'pull_up', p.id, p.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, p.reps, p.rating, p.recorded_at
    FROM pull_up p LEFT JOIN sessions s ON p.session_id = s.id
UNION ALL
    SELECT 'wall_toss', t.id, t.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, t.catches, t.rating, t.recorded_at
    FROM wall_toss t LEFT JOIN sessions s ON t.session_id = s.id
UNION ALL
    SELECT 'shuttle_run', r.id, r.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, r.time_sec, r.rating, r.recorded_at
    FROM shuttle_run r LEFT JOIN sessions s ON r.session_id = s.id
UNION ALL
    SELECT 'dash_50m', d.id, d.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, d.time_sec, d.rating, d.recorded_at
    FROM dash_50m d LEFT JOIN sessions s ON d.session_id = s.id
UNION ALL
    SELECT 'sit_reach', sr.id, sr.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, sr.reach_cm, sr.rating, sr.recorded_at
    FROM sit_reach sr LEFT JOIN sessions s ON sr.session_id = s.id
UNION ALL
    SELECT 'vertical_jump', vj.id, vj.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, vj.jump_cm, vj.rating, vj.recorded_at
    FROM vertical_jump vj LEFT JOIN sessions s ON vj.session_id = s.id
UNION ALL
    SELECT 'run_600m', rm.id, rm.session_id, s.student_name, s.age, s.gender, s.school, s.grade,
           NULL, NULL, rm.time_sec, rm.rating, rm.recorded_at
    FROM run_600m rm LEFT JOIN sessions s ON rm.session_id = s.id;
"""


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25.0: return "Normal"
    elif bmi < 30.0: return "Overweight"
    else:            return "Obese"


def init_db():
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript(_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
# Public API class
# ─────────────────────────────────────────────────────────────────────────────

class GoForFitDB:
    def __init__(self):
        init_db()

    # ── Sessions ─────────────────────────────────────────────────────────

    def new_session(
        self,
        student_name: str = None,
        age: int = None,
        gender: str = None,
        school: str = None,
        grade: str = None,
        notes: str = None,
    ) -> int:
        """Create a new session row and return its id."""
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO sessions (student_name, age, gender, school, grade, notes)
                   VALUES (?,?,?,?,?,?)""",
                (student_name, age, gender, school, grade, notes),
            )
            return cur.lastrowid

    def get_session(self, session_id: int) -> dict:
        with _conn() as con:
            row = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with _conn() as con:
            rows = con.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # ── Students (roster) ────────────────────────────────────────────────

    def list_classes(self) -> list[str]:
        with _conn() as con:
            return [r["grade"] for r in con.execute(
                "SELECT DISTINCT grade FROM students ORDER BY grade")]

    def list_students(self, grade: str = None) -> list[dict]:
        q = "SELECT * FROM students"
        params = ()
        if grade is not None:
            q += " WHERE grade=?"
            params = (grade,)
        # Roll numbers are strings but read as numbers, so "10" must not sort
        # before "9". CAST puts the numeric ones in human order and leaves
        # anything non-numeric (e.g. "12B") to fall back on the text sort.
        q += " ORDER BY CAST(roll_no AS INTEGER), roll_no, name"
        with _conn() as con:
            return [dict(r) for r in con.execute(q, params)]

    def get_student(self, student_id: int) -> dict:
        with _conn() as con:
            row = con.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
            return dict(row) if row else None

    def add_student(self, name: str, grade: str, roll_no: str = None,
                    reg_no: str = None, school: str = None) -> int:
        """Returns the new id, or raises sqlite3.IntegrityError if that class
        already has a student with this roll number."""
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO students (name, grade, roll_no, reg_no, school)
                   VALUES (?,?,?,?,?)""",
                (name, grade, roll_no, reg_no, school),
            )
            return cur.lastrowid

    def delete_student(self, student_id: int) -> bool:
        """Removes a student from the roster. Their past results are NOT
        touched — those live in sessions keyed by name, and deleting a child
        from the register must never quietly erase measurements already taken.
        """
        with _conn() as con:
            return con.execute("DELETE FROM students WHERE id=?", (student_id,)).rowcount > 0

    # ── BMI ──────────────────────────────────────────────────────────────

    def save_bmi(
        self,
        session_id: int,
        height_cm: float,
        weight_kg: float = None,
        calibration_used: bool = False,
    ) -> int:
        bmi_val  = None
        category = None
        if weight_kg and height_cm:
            bmi_val  = round(weight_kg / ((height_cm / 100) ** 2), 2)
            category = _bmi_category(bmi_val)

        with _conn() as con:
            cur = con.execute(
                """INSERT INTO bmi (session_id, height_cm, weight_kg, bmi_value, bmi_category, calibration_used)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, height_cm, weight_kg, bmi_val, category, int(calibration_used)),
            )
            return cur.lastrowid

    # ── Wingspan ─────────────────────────────────────────────────────────

    def save_wingspan(
        self,
        session_id: int,
        wingspan_cm: float,
        height_cm: float = None,
        calibration_distance_cm: float = None,
    ) -> int:
        ratio = round(wingspan_cm / height_cm, 3) if height_cm else None
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO wingspan (session_id, wingspan_cm, height_cm, wingspan_ratio, calibration_distance_cm)
                   VALUES (?,?,?,?,?)""",
                (session_id, wingspan_cm, height_cm, ratio, calibration_distance_cm),
            )
            return cur.lastrowid

    # ── Curl-Up ──────────────────────────────────────────────────────────

    def save_curl_up(
        self,
        session_id: int,
        reps: int,
        duration_sec: int = 30,
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO curl_up (session_id, reps, duration_sec, rating)
                   VALUES (?,?,?,?)""",
                (session_id, reps, duration_sec, rating),
            )
            return cur.lastrowid

    # ── Modified Pull-Up ─────────────────────────────────────────────────

    def save_pull_up(
        self,
        session_id: int,
        reps: int,
        duration_sec: float = None,
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO pull_up (session_id, reps, duration_sec, rating)
                   VALUES (?,?,?,?)""",
                (session_id, reps, duration_sec, rating),
            )
            return cur.lastrowid

    # ── Wall Toss ────────────────────────────────────────────────────────

    def save_wall_toss(
        self,
        session_id: int,
        catches: int,
        duration_sec: int = 30,
        ball_colour: str = None,
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO wall_toss (session_id, catches, duration_sec, ball_colour, rating)
                   VALUES (?,?,?,?,?)""",
                (session_id, catches, duration_sec, ball_colour, rating),
            )
            return cur.lastrowid

    # ── Shuttle Run ──────────────────────────────────────────────────────

    def save_shuttle_run(
        self,
        session_id: int,
        time_sec: float,
        lane: int = 1,
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO shuttle_run (session_id, lane, time_sec, rating)
                   VALUES (?,?,?,?)""",
                (session_id, lane, time_sec, rating),
            )
            return cur.lastrowid

    # ── 50 m Dash ────────────────────────────────────────────────────────

    def save_dash_50m(
        self,
        session_id: int,
        time_sec: float,
        lane: int = 1,
        start_mode: str = "whistle",
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO dash_50m (session_id, lane, time_sec, start_mode, rating)
                   VALUES (?,?,?,?,?)""",
                (session_id, lane, time_sec, start_mode, rating),
            )
            return cur.lastrowid

    # ── Sit and Reach ────────────────────────────────────────────────────

    def save_sit_reach(
        self,
        session_id: int,
        reach_cm: float,
        leg_cm: float = None,
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO sit_reach (session_id, reach_cm, leg_cm, rating)
                   VALUES (?,?,?,?)""",
                (session_id, reach_cm, leg_cm, rating),
            )
            return cur.lastrowid
        
    # ── Vertical Jump ────────────────────────────────────────────────────

    def save_vertical_jump(
        self,
        session_id: int,
        jump_cm: float,
        average_jump_cm: float = None,
        standing_reach_cm: float = None,
        attempts: list = None,
        rating: str = None,
    ) -> int:
        attempts_json = json.dumps(attempts) if attempts else None
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO vertical_jump
                       (session_id, jump_cm, average_jump_cm, standing_reach_cm, attempts_json, rating)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, jump_cm, average_jump_cm, standing_reach_cm, attempts_json, rating),
            )
            return cur.lastrowid

    # ── 600 m Run ────────────────────────────────────────────────────────

    def save_run_600m(
        self,
        session_id: int,
        time_sec: float,
        lane: int = 1,
        lap_count: int = None,
        rating: str = None,
    ) -> int:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO run_600m (session_id, lane, time_sec, lap_count, rating)
                   VALUES (?,?,?,?,?)""",
                (session_id, lane, time_sec, lap_count, rating),
            )
            return cur.lastrowid

    # ── Generic fetch helpers ────────────────────────────────────────────

    VALID_TABLES = {
        "bmi", "wingspan", "curl_up", "pull_up", "wall_toss",
        "shuttle_run", "dash_50m", "sit_reach", "run_600m", "vertical_jump",
    }

    def get_all(self, table: str, session_id: int = None) -> list[dict]:
        """Fetch all rows from a test table, optionally filtered by session."""
        if table not in self.VALID_TABLES:
            raise ValueError(f"Unknown table '{table}'")
        q = f"SELECT * FROM {table}"
        params = ()
        if session_id is not None:
            q += " WHERE session_id=?"
            params = (session_id,)
        q += " ORDER BY recorded_at DESC"
        with _conn() as con:
            return [dict(r) for r in con.execute(q, params).fetchall()]

    def get_all_results(self) -> list[dict]:
        """Flat view across all tests — useful for dashboards."""
        with _conn() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM all_results ORDER BY recorded_at DESC"
            ).fetchall()]

    def get_student_summary(self, student_name: str) -> list[dict]:
        """All results for a particular student name."""
        with _conn() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM all_results WHERE student_name=? ORDER BY recorded_at DESC",
                (student_name,),
            ).fetchall()]

    def delete_result(self, test: str, row_id: int) -> bool:
        """Delete a single test result row by its id. Returns True if a row was deleted."""
        if test not in self.VALID_TABLES:
            raise ValueError(f"Unknown table '{test}'")
        with _conn() as con:
            cur = con.execute(f"DELETE FROM {test} WHERE id=?", (row_id,))
            return cur.rowcount > 0

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its test rows (CASCADE). Returns True if deleted."""
        with _conn() as con:
            cur = con.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            return cur.rowcount > 0

    def export_csv(self, table: str, out_path: str = None) -> str:
        """Export a table to CSV. Returns the file path written."""
        import csv
        rows = self.get_all(table)
        if not rows:
            return None
        out_path = out_path or str(Path(__file__).parent / f"{table}_export.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (import and use directly)
# ─────────────────────────────────────────────────────────────────────────────
db = GoForFitDB()


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Database : {DB_PATH}")

    # Create a demo session
    sid = db.new_session(
        student_name="Demo Student",
        age=15,
        gender="M",
        school="GoForFit Academy",
        grade="10",
    )
    print(f"Session  : {sid}")

    # Insert one row into every test table
    db.save_bmi(sid, height_cm=170.0, weight_kg=60.0, calibration_used=True)
    db.save_wingspan(sid, wingspan_cm=172.5, height_cm=170.0, calibration_distance_cm=200.0)
    db.save_curl_up(sid, reps=22, rating="Good")
    db.save_wall_toss(sid, catches=18, ball_colour="orange", rating="Average")
    db.save_shuttle_run(sid, time_sec=12.4, lane=1, rating="Good")
    db.save_dash_50m(sid, time_sec=8.1, lane=1, start_mode="whistle", rating="Excellent")
    db.save_sit_reach(sid, reach_cm=5.3, leg_cm=87.0, rating="Average")

    print("\n── BMI ──────────────────────────────────────")
    for r in db.get_all("bmi"):
        print(r)

    print("\n── All Results (flat view) ──────────────────")
    for r in db.get_all_results():
        print(r)

    print("\nDone. Tables created and demo data inserted.")