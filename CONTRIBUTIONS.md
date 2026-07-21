# CONTRIBUTIONS.md — GoForFit Integrated Fitness App

A summary of who worked on what across the project. Sidhant and Saksham's
sections are longer because they carried the two largest, most
cross-cutting pieces of work — the initial full-stack app, and later the
web-engine rebuild + Laravel routing + debugging — not because the rest of
the team contributed less. Everyone else owns a complete, working AI test
end-to-end, which is the hardest part of this project.

---

## Sidhant Bhutani — Initial App Foundation & Wingspan Engine

Built the first working end-to-end version of GoForFit — the routing
skeleton, the first draft of the `TESTS` registry, the original
launch/session/save API surface, the first `db.py` schema, and the first
`README.md` — everything later stages were built on top of. Also owns the
**Wingspan** test engine (calibrate → measure → snapshot pipeline) and the
**Screen 3** configuration/review UI (student setup, manual timer, pending-
result review, Laravel sync from that screen).
**Technologies:** Python, Flask, SQLite, Jinja2, MediaPipe (Hand + Pose
Landmarker), OpenCV, JavaScript, HTML/Tailwind CSS.

## Saksham — Web-Engine Rebuild, Laravel Routing & Debugging

Took the app from Sidhant's initial working version to fully polished:
rebuilt every keyboard-driven desktop test into an in-process browser
engine (the shared `BaseWebEngine` layer, the Wingspan/Vertical-Jump/dual-
camera engines), implemented the Laravel `postMessage` integration end-to-
end, and did the bulk of the debugging — including tracking down and fixing
the mediapipe version regression, a broken import in the BMI script, a
broken drawing-utils import in the Wingspan script, and a model-path bug in
Vertical Jump. Also owns the **Vertical Jump** engine and the **Results**
dashboard.
**Technologies:** Python, MediaPipe Tasks API (PoseLandmarker, MJPEG
frame streaming), OpenCV, multithreading, FastAPI, JavaScript
(`postMessage` API), pip/dependency debugging.

## Ashish Jain & Sparsh Kedia — Backend Architecture (joint)

Together handled the core backend architecture: the Flask → FastAPI
migration, typed Pydantic request models, the growing `TESTS` registry, and
`db.py`'s schema/CRUD layer as new tests were added over time.
**Technologies:** Python, FastAPI, Uvicorn, Pydantic, SQLite.

- **Ashish** individually owns the **BMI** and **Partial Curl-Up** test
  engines. **Technologies:** Python, MediaPipe Pose Landmarker, OpenCV.
- **Sparsh** individually owns the **Wall Toss** test engine (HSV ball
  detection, alternating-catch state machine, colour-preset tuning) and
  the **Sit-and-Reach** script, and helped wire both into the browser
  live-feed layer. **Technologies:** Python, MediaPipe Hands, OpenCV
  (HSV colour-space detection).

## Shivam — Shuttle Run Engine & Live Capture UI

Owns the **Shuttle Run** test (dual-camera, whistle-triggered, multi-lane
timing) and built out **Screen 4**, the AI capture viewport — the generic
live-feed layout, calibration overlay, on-screen button system, and the
STOP/SAVE flow used by every test.
**Technologies:** Python, OpenCV, MediaPipe Pose, JavaScript, HTML/Tailwind
CSS.

## Aarika — 50 m Dash Engine & Landing/Navigation UI

Owns the **50 m Dash** test (sprint start/finish detection, lane timing)
and built the **landing page** and **category list screen** (Screen 1/2)
that a coach starts from.
**Technologies:** Python, OpenCV, Jinja2, HTML/Tailwind CSS.

## Akash — 600 m Run Package & Shared CV Conventions

Owns the **600 m Run** test, built as a cleanly modularized package
(detection, lap counting, direction tracking, state machine, UI, config —
split into separate files rather than one script). Also established the
shared on-screen HUD and keyboard-control conventions reused across all the
desktop test scripts.
**Technologies:** Python, OpenCV, modular Python package design.

---

## Shared / Cross-cutting work

A few things were genuinely built together rather than owned by one person:
deciding which tests got the full browser rebuild vs. staying keyboard-
driven; keeping the three parallel per-test configs (`TESTS`,
`LIVE_ENGINES`, `EXERCISE_SYNC_MAP`) in sync with each other; designing the
stage → review → save flow end-to-end; and the ongoing manual QA pass
across every test, camera path, and Laravel sync payload.
