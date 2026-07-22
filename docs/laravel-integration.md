# GoForFit ⇄ Laravel integration contract

How the Laravel site (`nep.goforfit.in`) drives the GoForFit AI camera app,
and what it gets back. Written for the Laravel side; everything here is what
the app **already does** unless marked otherwise.

Verified against the code on 2026-07-22. File/line references are into this
repo so anything below can be checked against the source.

---

## 1. The model: the app is a popup child of Laravel

Laravel opens the app in a **popup window** (`window.open`, not an iframe and
not a `rel="noopener"` link). The app relies on `window.opener` pointing back
at the Laravel page, so:

- **iframe** → `window.opener` is null → results never come back. Don't use it.
- **`target="_blank"` / `rel="noopener"`** → same problem.
- **`window.open(url, name, features)`** → correct. This is the only supported way.

The coach runs the test in the popup; on save, the score is posted back to the
Laravel window via `postMessage`, and the popup closes itself.

---

## 2. Opening the app — the URL Laravel must build

Point the popup at **`/screen4`** (the AI test screen) with these query params.
Values are taken straight from what the app reads today
([screen3.html `goToAI`](../templates/screen3.html), which is the reference
for the param names):

| Param         | Required | Meaning                                        |
|---------------|----------|------------------------------------------------|
| `test`        | ✅       | Which test — see the test-id table in §5       |
| `session_id`  | ✅ *     | The DB session the result attaches to — see §3 |
| `id`          | ✅       | Your student id (passed through, for your side)|
| `name`        | ✅       | Student full name (recorded on the result)     |
| `class_info`  | ⬜       | Class / grade, e.g. `Class 4-A`                |
| `roll`        | ⬜       | Roll number                                    |
| `trainerId`   | ⬜       | Coach id (passed through)                       |
| `traiName`    | ⬜       | Coach name (passed through)                     |
| `exerdseName` | ⬜       | Human test title (display only) *(sic — that is the real param name in the code)* |

\* `session_id` is required for the result to be **saved to the app's own DB**.
If you only care about the `postMessage` result (§4) and not the app's local
copy, it can be omitted — but see the open question in §6.

**Per-test extra params.** Three tests use two IP cameras and need their URLs
supplied as params (the app cannot guess them):

| test          | extra required params                                  |
|---------------|--------------------------------------------------------|
| `shuttle_run` | `cam_a_url`, `cam_b_url`  (optional: `touches`)         |
| `dash_50m`    | `finish_cam_url`, `overhead_cam_url` (opt: `lanes`, `lane_width`) |
| `run_600m`    | `lane_colors` (comma-separated) (opt: `camera_url`, `camera_distance_m`) |

The other six tests (`bmi`, `wingspan`, `curl_up`, `wall_toss`, `sit_reach`,
`vertical_jump`) use the coach's single camera and need no camera params.

Example:

```
https://<goforfit-host>/screen4?test=wingspan&session_id=812&id=9105
    &name=Ayaan+Yadav&class_info=Class+4-A&roll=5
    &trainerId=995&traiName=Hareram+Singh&exerdseName=Wingspan+Measurement
```

---

## 3. The session — who creates it

A **session** is the app's DB row that a result attaches to (it carries the
student name, class, and coach). Two options, and **we need to agree which**:

- **A — the app creates it.** Before opening `/screen4`, Laravel (or the app's
  own screen3) calls `POST /api/session` with `{student_name, grade, notes}`;
  the response gives `{session_id}`, which then goes in the `/screen4` URL.
  This is what the app does today in standalone mode.
- **B — Laravel owns the id.** Laravel passes its own identifier and the app
  treats the whole run as fire-and-forget, relying only on the `postMessage`
  result (§4) and never writing its own DB row.

Pick one. **This is the first open question for Bhawani** — it decides whether
the app keeps a local record or Laravel is the sole system of record.

---

## 4. What comes back — the `postMessage` result

⚠️ **There are currently TWO different payload shapes**, from two different
save paths. This is a real inconsistency in the app and must be reconciled —
flagged here so your listener isn't surprised.

Both post to the origin configured by `GOFORFIT_PARENT_ORIGIN` (default
`https://nep.goforfit.in`) and both carry `type: "AI_SYNC_DATA"`. **Always
check `event.origin` before trusting a message.**

### Shape 1 — the review-then-save path (most tests)
Sent from screen3 after the coach reviews and clicks **Save to Database**
([screen3.html `syncResultToLaravel`](../templates/screen3.html)):

```js
{ type: "AI_SYNC_DATA", exerciseType: "Wingspan", wingspanScore: 168.5 }
```
One `exerciseType` string + one score field whose **name** is per-test (§5).

### Shape 2 — the direct path (STOP/SAVE on screen4)
Sent from screen4 when a test syncs straight back without the review screen
([screen4.html `handleStopClick`](../templates/screen4.html)):

```js
{ type: "AI_SYNC_DATA", testId: "wingspan", sessionId: 812, wingspanScore: 168.5 }
```
Carries `testId` + `sessionId` instead of `exerciseType`, plus the same
per-test score field(s).

**Recommendation to reconcile:** standardise on one shape carrying *both*
`exerciseType` and `testId` and `sessionId`. Tell us which fields you want and
we will make both paths emit exactly that. Until then, a listener must handle
both.

### Multi-lane tests are not a single score
`shuttle_run`, `dash_50m`, `run_600m` produce a **time per lane**, not one
number. Their payload carries a `lanes`/`laps` object keyed by lane. The exact
shape is unconfirmed — **second open question for Bhawani:** what do you want
for multi-runner results?

---

## 5. Test ids and score field names

`type: "AI_SYNC_DATA"`, one row per test. **Only `wingspan` has been confirmed
against your sample code — every other `scoreKey` below is our guess and needs
your confirmation.** (Source of truth: `LARAVEL_SYNC` in `app.py`.)

| `test` id       | `exerciseType`  | score field (`scoreKey`) | confirmed? |
|-----------------|-----------------|--------------------------|------------|
| `wingspan`      | `Wingspan`      | `wingspanScore`          | ✅ yes      |
| `bmi`           | `BMI`           | `bmiScore`               | ❓ guess    |
| `curl_up`       | `CurlUp`        | `curlUpScore`            | ❓ guess    |
| `wall_toss`     | `WallToss`      | `wallTossScore`          | ❓ guess    |
| `sit_reach`     | `SitReach`      | `sitReachScore`          | ❓ guess    |
| `vertical_jump` | `VerticalJump`  | `verticalJumpScore`      | ❓ guess    |
| `shuttle_run`   | `ShuttleRun`    | `shuttleRunScore`        | ❓ guess + multi-lane |
| `dash_50m`      | `Dash50m`       | `dash50mScore`           | ❓ guess + multi-lane |
| `run_600m`      | `Run600m`       | `run600mScore`           | ❓ guess + multi-lane |

**Third open question for Bhawani:** confirm or correct every row. One reply
fixing this table and we update `LARAVEL_SYNC`, and every test syncs correctly
instead of just wingspan.

---

## 6. Authentication — how the popup gets past the app's own gate

The app has its own access gate (a shared code, then a coach-name screen).
When Laravel opens the popup, that gate is in the way unless handled. Options:

- **The popup is served with the gate off** (`GOFORFIT_AUTH=off`) because
  Laravel has *already* authenticated the user. Simplest if the app is only
  ever reached through Laravel and never exposed directly.
- **Laravel appends the access code**: open `/screen4?...&token=<code>`. The
  app swaps it for a cookie and strips it from the URL automatically. The
  coach still picks their name once per device.
- **The coach unlocks the app once per device** and the 12-hour cookie carries
  subsequent opens.

The app's gate is **not** a substitute for Laravel's own login — it stops
casual LAN access to the raw app, nothing more. Real parent/coach identity,
passwords, and the parent→child link all live on the Laravel side; the app
never sees them. **Fourth open question:** which of the three above do you want?

---

## 7. Origin configuration

`postMessage` delivers **only** to the origin set in `GOFORFIT_PARENT_ORIGIN`
(default `https://nep.goforfit.in`). For a staging Laravel, start the app with
that env var pointed at the staging origin. There is a local test harness at
`/dev/parent` (only when started with `GOFORFIT_DEV=1`) that stands in for the
Laravel window and logs what it receives — useful for proving the handoff
without the real dashboard.

---

## Open questions — the whole list for one reply

1. **Session ownership** (§3): app creates the session, or Laravel owns the id?
2. **Payload shape** (§4): reconcile the two shapes — which fields do you want?
3. **Score field names** (§5): confirm/correct every row of the table.
4. **Multi-lane results** (§4): what shape for `shuttle_run`/`dash_50m`/`run_600m`?
5. **Auth** (§6): gate off, token-in-URL, or unlock-per-device?
6. **How you open the popup** (§1): confirm `window.open` (not iframe / noopener).
