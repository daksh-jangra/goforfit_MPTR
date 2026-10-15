# GoForFit NEP FMS — AI Fitness Assessment App

**Project:** Computer-vision fitness testing for school students
**Battery:** Indian FMS (Fitness365) tests, plus adapted tests from the Brockport Physical Fitness Test Manual
**Scope:** A coach runs each test from a phone or laptop browser, and the camera scores it
**Stack:** FastAPI · OpenCV · MediaPipe · SQLite · Jinja templates
**Category:** Computer Vision | Pose Estimation | Sports Science

---

## Executive Summary

GoForFit turns a phone camera into a fitness tester. The coach picks a student,
opens a test and points the camera. The app watches the movement with
MediaPipe pose, hand and ball tracking. It counts reps, times holds, measures
distances and gives pass/fail verdicts. Then it saves the result and sends it
to the GoForFit dashboard.

The app is designed to:

- Run **18 fitness tests** across anthropometry, strength, endurance, flexibility, speed, agility, power and coordination
- Score straight from the **coach's phone browser**: no desktop window, no keyboard and no app install
- Follow each test's **written protocol** for its start, stop, rep and verdict rules
- Leave the coach in charge of judging form, while the camera decides the score
- Keep every result in a local database and **sync confirmed scores** to the Laravel dashboard

---

## Repository Navigation

```
goforfit_MPTR/
├── README.md                     ← This file
├── requirements.txt              ← Python dependencies
├── app.py                        ← FastAPI server: test registry, routes, engine control
├── db.py                         ← SQLite schema + queries (one table per test)
│
├── templates/                    ← The coach's screens
│   ├── index.html                ← Home: test categories
│   ├── screen2.html              ← Test list for a category
│   ├── screen3.html              ← Student picker, manual entry, dashboard sync
│   ├── screen4.html              ← Live camera view with on-screen controls
│   └── results.html              ← Saved results, per test
│
└── tests/                        ← Camera engines and per-test logic
    ├── web_engines.py            ← Browser-camera engines for single-camera tests
    ├── dual_feed_engines.py      ← Two-camera engines (shuttle run, 50 m dash, 600 m)
    ├── camera.py                 ← Turns a camera index or IP-camera URL into a capture
    ├── stall_watchdog.py         ← Flags a frozen camera feed instead of a black screen
    ├── wingspan_web_engine.py    ← Wingspan in-browser engine
    ├── vertical_jump_web_engine.py
    │
    ├── bmi__3_.py  wingspan2.py  parcurlup.py  wtt.py  snr.py  vertical_jump.py
    ├── pullup.py  pullup_full.py  trunk_lift.py  reverse_curl.py
    ├── shoulder_stretch.py  isometric_pushup.py  seated_pushup.py
    ├── dumbbell_press.py  target_stretch.py
    ├── shuttleruntest.py  sprint_timer.py  50m_dash_mod.py
    ├── 600m_new/                 ← 600 m run: colour tracking, lanes, lap counting
    │
    ├── test_*.py                 ← Logic checks (no camera needed)
    └── *.task                    ← MediaPipe pose / hand model files
```

---

## Architecture Overview

GoForFit is a **single FastAPI server** with a separate engine for each test.
The phone browser sends camera frames to the server, the engine scores them,
and the browser polls for the live result.

| Component | Role |
| --- | --- |
| **FastAPI Server** (`app.py`) | Test registry, page routes, starts/stops engines, save/export APIs |
| **Coach Screens** (`templates/`) | Category → test → student → live camera flow, built for a phone |
| **Browser-Camera Engines** (`tests/web_engines.py`) | Take frames from the phone (`POST /api/<test>/frame`), run MediaPipe, publish state |
| **Per-Test Logic** (`tests/<test>.py`) | Each protocol's rules: rep counting, hold timing, angles, verdicts |
| **Dual-Camera Engines** (`tests/dual_feed_engines.py`) | Shuttle run and 50 m dash from two IP cameras; 600 m from one camera |
| **Stall Watchdog** (`tests/stall_watchdog.py`) | Detects a frozen feed and tells the coach |
| **Results Store** (`db.py`) | SQLite: sessions, the student roster, one results table per test |
| **Dashboard Sync** (`screen3.html`) | Sends a confirmed score to the Laravel parent window that opened the app |

---

## Tests

| ID | Test | Category | Camera | Score |
| --- | --- | --- | --- | --- |
| `bmi` | BMI / Height Estimator | Anthropometry | Phone | cm |
| `wingspan` | Wingspan Measurement | Anthropometry | Phone | cm |
| `curl_up` | Partial Curl-Up | Muscular Endurance | Phone | reps |
| `pull_up` | Modified Pull-Up | Muscular Endurance | Phone | reps |
| `full_pull_up` | Pull-Up (full hang) | Muscular Endurance | Phone | reps |
| `isometric_push_up` | Isometric Push-Up | Muscular Endurance | Phone | seconds (max 40) |
| `seated_push_up` | Seated Push-Up | Muscular Endurance | Phone | seconds (max 20) |
| `dumbbell_press` | Dumbbell Press | Muscular Endurance | Phone | reps (max 50) |
| `wall_toss` | Wall Toss | Coordination | Phone | catches |
| `sit_reach` | Sit and Reach | Flexibility | Phone | cm |
| `trunk_lift` | Trunk Lift | Musculoskeletal | Phone | cm (max 30) |
| `reverse_curl` | Reverse Curl | Musculoskeletal | Phone | pass / fail |
| `shoulder_stretch` | Shoulder Stretch | Musculoskeletal | Phone | pass / fail |
| `target_stretch` | Target Stretch (8 joints) | Musculoskeletal | Phone | 0 / 1 / 2 per side |
| `vertical_jump` | Vertical Jump | Power | Phone | cm |
| `run_600m` | 600 m Run | Endurance | One camera | seconds |
| `shuttle_run` | 10 m Shuttle Run | Agility | Two IP cameras | seconds |
| `dash_50m` | 50 m Dash | Speed | Two IP cameras | seconds |

Notes:
- For `sit_reach`, enter the student's hip-to-ankle length on the setup screen first.
- `vertical_jump` measures how far the hips rise from standing to the top of the
  jump. The older fingertip-reach method is still available: set
  `MEASURE_METHOD = "reach"` in `tests/vertical_jump_web_engine.py`.
- `target_stretch` measures the left side first, then the right. The coach picks
  the joint on the camera screen.

---

## Protocol Sources

Each test follows a published protocol:

- **Indian FMS battery (Fitness365)**: the main reference. Where its demo video and the Brockport manual disagree, the video wins (for example, the Modified Pull-Up).
- **Brockport Physical Fitness Test Manual**: Pull-Up, Trunk Lift, Reverse Curl, Shoulder Stretch, Isometric Push-Up, Seated Push-Up, Dumbbell Press and Target Stretch
- **Coach's role**: the app scores; the coach judges form cues that the protocol leaves to the tester

---

## Setup & Run

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Then open `http://localhost:8080`. To test from a phone, open the laptop's address
on the same Wi-Fi. FastAPI's interactive API docs are at `http://localhost:8080/docs`.

1. Start on the home page and pick a category
2. Pick a test, then the student, on `screen3`
3. Tap **Switch to AI** to open `screen4` and point the camera
4. Back on `screen3`, review the result and tap **Save to Database**. If the dashboard opened the app, the score is synced back
5. Review or export results at `/results`

### Dashboard sync

Set `GOFORFIT_PARENT_ORIGIN` to the dashboard's origin in production. It defaults
to `*`, which is fine locally because sync only runs when a dashboard window
opened the app.

### Logic checks

The rep, hold and verdict rules can be tested without a camera:

```bash
python3 -m pytest tests/test_*.py
```
