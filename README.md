# GoForFit Integrated Fitness App

This package combines:

- the `fitness.zip` four-screen frontend flow
- the `goforfitfinal.zip` AI/OpenCV fitness test scripts
- a backend now running on **FastAPI** (migrated from Flask)

## What is integrated

- `/` shows the fitness365 assessment category screen.
- `/screen2` lists backend-backed fitness tests by category.
- `/screen3` keeps the student/configuration and manual timer screen.
- `/screen4` keeps the AI viewport layout and launches the real backend test through `POST /api/launch/<test_id>`.
- Existing backend endpoints are preserved:
  - `POST /api/launch/<id>`
  - `GET /api/status`
  - `GET /api/status/<id>`
  - `POST /api/stop/<id>`
  - `POST /api/session`, `GET /api/session/<id>`, `GET /api/sessions`
  - `POST /api/save/<test_id>`, `GET /api/results`, `GET /api/results/<test_id>`, `GET /api/export/<test_id>`

The AI tests still open in their own OpenCV windows, as the original backend was designed to do — **except `wingspan`**, which now runs entirely in the browser (see below).

## Wingspan: live feed in the browser (no keyboard needed)

Every other test still needs the original physical-keyboard controls
(SPACE/C/M/S/Q/etc.) inside a desktop OpenCV window, which doesn't work from
a phone. Wingspan was the one test that needed several distinct actions
(calibrate → enter real cm → measure → snapshot → save), so it's been
rebuilt to run without any OpenCV window or keyboard at all:

- `tests/wingspan_web_engine.py` runs the camera + MediaPipe processing loop
  **in-process**, in a background thread, instead of as a subprocess. It
  reuses all the geometry/drawing code from `tests/wingspan2.py` (imported,
  not duplicated) — only the keyboard-driven control loop was replaced.
- `screen4` for `wingspan` shows a live `<img>` MJPEG feed plus on-screen
  buttons (CALIBRATE / MEASURE MODE / SNAPSHOT / SAVE READING) and a numeric
  input for typing your real wingspan during calibration — these replace the
  C / M / SPACE / S keys one-for-one.
- New endpoints: `GET /api/wingspan/feed` (MJPEG stream), `GET
  /api/wingspan/state` (polled to drive button enable/disable + status
  text), `POST /api/wingspan/action` (`{"action": "...", "value": ...}`).
- `POST /api/launch/wingspan` and `POST /api/stop/wingspan` are unchanged
  from the frontend's point of view — they now start/stop the in-process
  engine instead of a subprocess when wingspan is the test.
- `tests/wingspan2.py` is untouched and still works standalone (`python
  wingspan2.py`) if you ever want the original desktop/keyboard version.
- If `wingspan_web_engine.py` fails to import for any reason (e.g. missing
  dependency), `wingspan` automatically falls back to the old
  subprocess/OpenCV-window behavior so the app never breaks.

## Migration notes (Flask → FastAPI)

- All routes, JSON response shapes, and status codes (400/404/409/500) are preserved exactly.
- The four screen templates (`index.html`, `screen2.html`, `screen3.html`, `screen4.html`) are **unchanged** — a small `url_for` compatibility shim in `app.py` lets them keep using Flask-style calls like `url_for('screen2', category='All Tests')`.
- `db.py` is unchanged — it's plain `sqlite3` with no Flask dependency.
- The `/api/save/*` and `/api/session` endpoints now use Pydantic models for the request body instead of Flask's untyped `request.get_json()`. Fields are still all optional with the same names/defaults, but values now get type-checked (e.g. `height_cm` must be numeric). If your `tests/*.py` scripts POST to these endpoints, double check the payloads match these types — see "What I still need from you" below.
- New: interactive API docs at `http://localhost:8080/docs` (Swagger UI), courtesy of FastAPI.
- Server entry point now uses `uvicorn` instead of Flask's built-in dev server.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open:

```text
http://localhost:8080
```

## Test IDs

| ID | Test | Camera | Status |
| --- | --- | --- | --- |
| `bmi` | BMI / Height Estimator | Browser (phone) | ✅ Integrated |
| `wingspan` | Wingspan Measurement | Browser (phone) | ✅ Integrated |
| `curl_up` | Partial Curl-Up Counter | Browser (phone) | ✅ Integrated |
| `wall_toss` | Wall Toss Coordination Test | Browser (phone) | ✅ Integrated |
| `sit_reach` | Sit and Reach Test | Browser (phone) | ✅ Integrated |
| `vertical_jump` | Vertical Jump Test | Browser (phone) | ✅ Integrated — hip-displacement method |
| `shuttle_run` | 10 m Shuttle Run Timer | Two IP cameras | ✅ Integrated |
| `dash_50m` | 50 m Dash Timer | Two IP cameras | ✅ Integrated |
| `run_600m` | 600 m Run | Two IP cameras | ✅ Integrated |

"Integrated" = wired up and running in the app (browser live-feed engines, or the
dual-camera engines). It does not by itself certify measurement accuracy on every
device — validate each test live before relying on the numbers.

For `sit_reach`, enter hip-to-ankle length on the configuration screen before launching AI.

`vertical_jump` measures the rise of the hip centre (center-of-mass) between the
resting stance and the jump apex, using the same known-height (cm-per-pixel)
calibration as `bmi`. The legacy fingertip reach-differential method is kept behind
`MEASURE_METHOD = "reach"` in `tests/vertical_jump_web_engine.py`.