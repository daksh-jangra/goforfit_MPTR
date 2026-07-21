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
- New: interactive API docs at `http://localhost:5000/docs` (Swagger UI), courtesy of FastAPI.
- Server entry point now uses `uvicorn` instead of Flask's built-in dev server.

## Requirements

- **Python 3.10 – 3.12.** Python 3.13 will NOT work: our `mediapipe<0.10.30`
  pin (see requirements.txt for why) has no wheels built for 3.13, so
  `pip install` fails with "No matching distribution found for mediapipe".
  On a Mac with only Python 3.13: `brew install python@3.12`.
- A webcam (phone IP-cams work for the dual-camera tests via their URL params).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a byte-for-byte reproduction of the known-working environment (pinned
transitive dependencies included), use the lockfile instead:

```bash
pip install -r requirements.lock
```

## Run

```bash
python app.py
```

Open:

```text
http://localhost:5001
```

The server defaults to **localhost only, port 5001, no auto-reload**:

- **Port 5001, not 5000** — on macOS, AirPlay Receiver (ControlCenter)
  listens on port 5000, so binding there fails or silently answers with the
  wrong service.
- **Localhost by default** — to use the app from a phone on the same network
  (the normal coach setup), opt in explicitly:

  ```bash
  GOFORFIT_HOST=0.0.0.0 python app.py
  ```

- **Access code** — this server serves student names, ages, schools and
  measurements, and can delete sessions, so every route requires a shared
  token. The server prints one at startup:

  ```
  ──────────────────────────────────────────────────────────────
    GoForFit access code:  PBtAQMNz7MK8
    (generated for this run — set GOFORFIT_TOKEN to pin your own)
  ──────────────────────────────────────────────────────────────
  ```

  Open the app, type the code once, and that device stays unlocked for 12
  hours. Set `GOFORFIT_TOKEN=…` to choose your own so it survives restarts —
  otherwise a new one is generated each launch and every device must re-enter
  it. Scripts can send `X-GoForFit-Token`, and a QR code can carry `?token=…`
  (swapped for the cookie and stripped from the URL on arrival).

  `GOFORFIT_AUTH=off` disables the gate entirely. Only do that on localhost —
  with `GOFORFIT_HOST=0.0.0.0` it means anyone on the school network can read
  and delete student records.

- Overrides: `GOFORFIT_HOST`, `GOFORFIT_PORT`, `GOFORFIT_RELOAD=1`
  (auto-reload restarts the server on any file change — never use it during a
  live test, the restart kills the camera engine and loses staged results),
  `GOFORFIT_TOKEN`, `GOFORFIT_AUTH`.

## Running the tests

Unit tests live in `tests_unit/` (NOT `tests/` — that directory holds the
OpenCV/MediaPipe engine scripts; `pytest.ini` scopes collection so pytest
never imports them). The suite fakes `cv2.VideoCapture`, so no camera is
needed, but the real engine modules are imported — run it inside the venv:

```bash
pip install -r requirements-dev.txt
pytest
```

It covers the engine lifecycle invariants (a failed start releases the
camera; stop never releases a capture under a live loop thread; clean cycles
release exactly once) and the result-staging contract (partial results stage
exactly once on stop, worker-staged results are never overwritten, a failing
stage callback can't block shutdown).

## Test IDs

| ID | Test |
| --- | --- |
| `bmi` | BMI / Height Estimator |
| `wingspan` | Wingspan Measurement |
| `curl_up` | Partial Curl-Up Counter |
| `wall_toss` | Wall Toss Coordination Test |
| `shuttle_run` | 4x10 m Shuttle Run Timer |
| `dash_50m` | 50 m Dash Timer |
| `sit_reach` | Sit and Reach Test |

For `sit_reach`, enter hip-to-ankle length on the configuration screen before launching AI.

## STOP/SAVE and the Laravel integration

Every `live_feed` test's `screen4.html` STOP/SAVE button runs `handleStopClick()`, which:

1. Stops the camera engine (`POST /api/stop/<test_id>`).
2. Reads the last polled state and builds a small payload via that test's
   `syncFields()` (e.g. wall_toss → `{catches, ball_colour}`).
3. If this screen was opened as a popup/child window by Laravel
   (`window.opener` is set), it posts that payload to the parent window with
   `window.opener.postMessage({type: 'AI_SYNC_DATA', testId, sessionId, ...fields}, 'https://nep.goforfit.in')`.
   If the page was opened directly (no `window.opener`), this step is
   skipped — there is **no direct HTTP call to a Laravel backend endpoint**,
   the whole integration is a `postMessage` handshake with a parent window.
4. Redirects to `/screen3` either way, where the coach reviews the result
   and the existing `POST /api/save/<test_id>` endpoints write it to the
   local `goforfit.db` SQLite database (this local save always happens,
   independent of Laravel).

Field names in each engine's `syncFields()` are still placeholders — several
are marked `// TODO: confirm exact Laravel field name with integration
owner` — and the allowed origin `https://nep.goforfit.in` should be
double-checked with whoever owns the Laravel app before go-live.

Note: an outdated code comment in `screen4.html` says only `wingspan` syncs
directly to Laravel ("bypasses screen3"); in the current code every
`live_feed` test defines `syncFields()`, so `handleStopClick()` will attempt
the `postMessage` sync for **all** of them, not just wingspan. Worth
flagging to the integration owner.

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**

Google removed the legacy `mediapipe.solutions` API (used by `Hands`,
`drawing_utils`, etc.) starting in mediapipe **0.10.30**
([tracking issue](https://github.com/google-ai-edge/mediapipe/issues/6192)).
`tests/wtt.py` (wall_toss), `tests/parcurlup.py` (curl_up), and
`tests/shuttleruntest.py` (shuttle_run) all still use this legacy API, so a
`pip install` that resolves to mediapipe ≥0.10.30 breaks those three tests.

Fixed by pinning `mediapipe>=0.10.9,<0.10.30` in `requirements.txt`. If you
already have a newer mediapipe installed, re-run:

```bash
pip install -r requirements.txt --upgrade
```

Longer-term, these three scripts should be migrated to the MediaPipe Tasks
API (`HandLandmarker` / `PoseLandmarker`) the way `wingspan2.py` and the
`600m_new` scripts already were (see the `.task` model files already in
`tests/`) — that removes the version ceiling entirely.

**`AttributeError: module 'mediapipe.tasks.python.vision' has no attribute 'PoseLandmark'`**

Separate, unrelated bug (not a version issue) found in `tests/bmi__3_.py`
after the mediapipe pin above was applied. `bmi__3_.py` already uses the
newer Tasks API and had:

```python
PoseLandmark = mp.tasks.vision.PoseLandmark   # <- does not exist
```

`mp.tasks.vision` only exposes `PoseLandmarker`, `PoseLandmarkerOptions`,
and `PoseLandmarksConnections` — there is no `PoseLandmark` index-enum
under the Tasks API. This line was dead code (the file hardcodes its own
`IDX_NOSE`, `IDX_LEFT_SHOULDER`, etc. constants a few lines below and never
actually reads the `PoseLandmark` variable), so it's been removed. This was
crashing the import of `web_engines.py`, which in turn disabled **every**
browser AI test (including wingspan), not just BMI — that's why the app
logged the `[FATAL] Live-feed camera engines failed to import` banner for
all tests and every `/api/launch/*` call returned `503`.