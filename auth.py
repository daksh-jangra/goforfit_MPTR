"""Shared-token access control for the whole app.

Why this exists: the normal coach setup is GOFORFIT_HOST=0.0.0.0 so a phone
on the school Wi-Fi can reach the app. Until now that also meant anyone else
on that network could read every student's name, age, school and measurements
and DELETE sessions, with no credential of any kind. That is the only part of
this project that can hurt a real person.

The model is deliberately the simplest thing that closes it: ONE token,
shared by everyone who is allowed in, checked on every request. It is not
per-user auth and does not pretend to be — there is no user model here, and
inventing one would be a much larger change for a tool that a single coach
runs on a laptop at the side of a field.

    GOFORFIT_TOKEN=letmein   uvicorn app:app --host 0.0.0.0
    GOFORFIT_AUTH=off        uvicorn app:app          # opt out, localhost dev

With neither set, a random token is generated and printed at startup — so
the failure mode of forgetting to configure anything is "I need to read the
token off the laptop", not "the whole school can browse student records".

How a request proves itself, in order:
  1. the gf_token cookie          — set by /unlock, rides on fetch() and on
                                    the <img> MJPEG feeds, which cannot send
                                    custom headers
  2. X-GoForFit-Token header      — for scripts and the Laravel integration
  3. ?token=… in the query string — the entry path for a phone following a
                                    QR code; immediately exchanged for the
                                    cookie and redirected away so the token
                                    does not sit in history or in referrers
"""

import os
import secrets
from urllib.parse import urlencode

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

COOKIE_NAME = "gf_token"
HEADER_NAME = "X-GoForFit-Token"
QUERY_NAME = "token"

# Who is operating this browser. Deliberately NOT a security boundary — the
# access code above is. This only says which coach's work a request belongs
# to, so that two coaches sharing a server don't collide. Anyone holding the
# access code can sign in as any name; that was the explicit trade for not
# making coaches manage a second secret.
COACH_COOKIE = "gf_coach"
COACH_HEADER = "X-GoForFit-Coach"

# Reachable without a token, or nobody could ever get one.
PUBLIC_PATHS = {"/unlock", "/favicon.ico"}

# Reachable with a token but before choosing a coach — otherwise the sign-in
# screen would redirect to itself.
NO_COACH_NEEDED = PUBLIC_PATHS | {"/whoami"}


def resolve_token():
    """(token, was_generated). None means auth is switched off entirely."""
    if os.environ.get("GOFORFIT_AUTH", "").strip().lower() in ("off", "0", "false", "no"):
        return None, False
    configured = os.environ.get("GOFORFIT_TOKEN", "").strip()
    if configured:
        return configured, False
    # token_urlsafe(9) → 12 chars: long enough that guessing it over a LAN is
    # hopeless, short enough to read off a screen and type on a phone.
    return secrets.token_urlsafe(9), True


def tokens_match(supplied, expected):
    """Constant-time compare that survives arbitrary input.

    secrets.compare_digest raises TypeError on a str containing non-ASCII,
    so a coach typing a code with an accent — or any junk in a cookie —
    used to come back as a 500 with a traceback instead of a clean refusal.
    Comparing the UTF-8 bytes keeps it constant-time and total.
    """
    if not isinstance(supplied, str) or not expected:
        return False
    return secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _safe_next(raw):
    """Only same-site relative paths — an attacker-supplied `next` must not
    be able to bounce the coach off to another host after unlocking."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/"
    return raw


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Gates every route. Deliberately a whole-app gate rather than a
    dependency on each endpoint: there are 30-odd routes and the cost of
    forgetting one is a student-data leak, so the default must be closed."""

    def __init__(self, app, token):
        super().__init__(app)
        self.token = token

    def _authorised(self, request):
        # Any of the three may be stale while another is valid, so check them
        # all rather than stopping at the first one that is merely PRESENT.
        return any(tokens_match(supplied, self.token) for supplied in (
            request.cookies.get(COOKIE_NAME),
            request.headers.get(HEADER_NAME),
            request.query_params.get(QUERY_NAME),
        ))

    def _clean_target(self, request):
        """Where this request wanted to go, with the token stripped out — it
        must never reach the unlock page's `next`, a Referer, or history."""
        remaining = [(k, v) for k, v in request.query_params.multi_items() if k != QUERY_NAME]
        return request.url.path + (f"?{urlencode(remaining)}" if remaining else "")

    async def dispatch(self, request, call_next):
        if self.token is None or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if not self._authorised(request):
            if request.url.path.startswith("/api/"):
                return JSONResponse(
                    {"ok": False, "error": "Unauthorised — unlock this server first."},
                    status_code=401,
                )
            target = urlencode({"next": self._clean_target(request)})
            return RedirectResponse(f"/unlock?{target}", status_code=303)

        # Arrived with ?token=… — trade it for the cookie and strip it back
        # out of the URL. This runs even when a cookie is ALREADY set, because
        # the usual reason a link like this gets used is that the existing
        # cookie is stale: an unconfigured server mints a fresh token every
        # restart, so every previously-unlocked device is holding a dead one.
        # Skipping the swap in that case left the QR-code path permanently
        # broken for exactly the devices most likely to use it.
        if request.query_params.get(QUERY_NAME) \
                and request.method == "GET" and not request.url.path.startswith("/api/"):
            response = RedirectResponse(self._clean_target(request), status_code=303)
            set_token_cookie(response, self.token)
            return response

        # Past the gate. Resolve WHO this is — every request that touches an
        # engine or a result needs it, so it is attached once here rather
        # than re-read in thirty endpoints.
        request.state.coach = coach_of(request)
        if not request.state.coach and request.url.path not in NO_COACH_NEEDED:
            if request.url.path.startswith("/api/"):
                return JSONResponse(
                    {"ok": False, "error": "No coach selected — open the app and sign in."},
                    status_code=409,
                )
            target = urlencode({"next": self._clean_target(request)})
            return RedirectResponse(f"/whoami?{target}", status_code=303)

        return await call_next(request)


def print_startup_banner(token, generated):
    """Printed at import so it appears however the app was started — the
    `uvicorn app:app` path never runs app.py's __main__ block."""
    line = "─" * 62
    if token is None:
        print(f"\n{line}\n  GoForFit: ACCESS CONTROL IS OFF (GOFORFIT_AUTH=off).")
        print("  Anyone who can reach this port can read and delete student")
        print(f"  records. Only safe on localhost.\n{line}\n")
        return
    print(f"\n{line}")
    if generated:
        print(f"  GoForFit access code:  {token}")
        print("  (generated for this run — set GOFORFIT_TOKEN to pin your own)")
    else:
        print("  GoForFit access code:  set from GOFORFIT_TOKEN")
    print(f"{line}\n")


COACH_MAX_LEN = 40


def clean_coach_name(raw):
    """Normalise a submitted coach name, or None if it isn't usable.

    The name is a dictionary key for engine ownership and a value written to
    the sessions table, so it gets bounded and stripped of control characters
    rather than trusted as typed.
    """
    if not isinstance(raw, str):
        return None
    name = " ".join(raw.split())  # collapse whitespace, drop tabs/newlines
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:COACH_MAX_LEN] or None


def coach_of(request):
    """Signed-in coach for this request, or None."""
    return (clean_coach_name(request.cookies.get(COACH_COOKIE))
            or clean_coach_name(request.headers.get(COACH_HEADER)))


MIRROR_COOKIE = "gf_mirror"
CAMERA_COOKIE = "gf_camera"
DEFAULT_CAMERA = "0"


def camera_of(request):
    """Which camera this browser drives. Kept in a cookie rather than server
    state because it describes where the coach is physically standing, and
    because it then survives a server restart — the same restart that
    regenerates the access code."""
    raw = (request.cookies.get(CAMERA_COOKIE) or "").strip()
    return raw or DEFAULT_CAMERA


def mirror_of(request):
    """Should the engines selfie-flip this camera's frames? Default yes — a
    plain webcam hands over an un-mirrored image. Set to "off" for a source
    that mirrors itself (iPhone Continuity Camera, many phone IP-cam apps),
    where flipping again cancels it out."""
    return (request.cookies.get(MIRROR_COOKIE) or "on").strip().lower() != "off"


def set_mirror_cookie(response, enabled):
    response.set_cookie(
        MIRROR_COOKIE, "on" if enabled else "off",
        httponly=True, samesite="lax", path="/", max_age=60 * 60 * 12,
    )


def set_camera_cookie(response, camera):
    response.set_cookie(
        CAMERA_COOKIE, camera or DEFAULT_CAMERA,
        httponly=True, samesite="lax", path="/", max_age=60 * 60 * 12,
    )


def set_coach_cookie(response, coach):
    """Not HttpOnly-critical the way the token is — it carries no authority —
    but kept HttpOnly anyway so page scripts can't quietly rewrite whose work
    a request counts as."""
    response.set_cookie(
        COACH_COOKIE, coach, httponly=True, samesite="lax", path="/", max_age=60 * 60 * 12,
    )


def set_token_cookie(response, token):
    """HttpOnly so page scripts can't read it back out; Lax so the Laravel
    parent window can still open screen4 as a popup and stay unlocked. Not
    Secure — this server speaks plain HTTP on a LAN by design."""
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, samesite="lax", path="/", max_age=60 * 60 * 12,
    )
