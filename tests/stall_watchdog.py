"""Stall watchdog shared by every browser engine.

The failure this exists for: the capture wedges (camera unplugged, phone
stream drops, cap.read() blocks forever) and the engine keeps reporting
`running: true` while publishing zero frames. screen4 then shows a black
void with no explanation, and only a server restart clears it.

The engines already stamp every frame they hand to the MJPEG buffer, so
detecting the stall costs one monotonic read at publish time and one
subtraction when /state is polled — no watchdog thread, nothing to shut
down. The verdict is COMPUTED in _stall_error() rather than stored in
self._error, so a feed that recovers clears its own warning.

Wiring an engine up is three lines:

    class MyEngine(StallWatchdogMixin, ...):
        def start(self):
            ...
            self._watchdog_reset()          # before the loop thread starts
        def _publish(self, frame):
            with self._lock:
                ...
                self._watchdog_mark()       # caller already holds the lock
        def get_state(self):
            with self._lock:
                return {..., "error": self._error or self._stall_error()}
"""

import time


class StallWatchdogMixin:
    # Steady-state budget between published frames. Generous next to the
    # ~30 fps the engines actually run at, so only a real wedge trips it.
    STALL_TIMEOUT_S = 5.0

    # The first frame costs more than the rest: MediaPipe warms up its graph
    # on the initial inference, and the dual-feed engines are pulling from
    # phones over Wi-Fi. Don't call that a stall.
    STALL_STARTUP_GRACE_S = 15.0

    # Class-level defaults so the mixin needs no __init__ — engines that
    # never call _watchdog_reset() simply never report a stall.
    _last_publish = None
    _published_any = False

    def _watchdog_reset(self):
        """Start the stall clock. Call from start(), before the loop thread —
        so a capture that opens but never delivers a frame still trips."""
        self._last_publish = time.monotonic()
        self._published_any = False

    def _watchdog_mark(self):
        """A frame just reached the MJPEG buffer. Assigns without locking:
        call it from inside the engine's existing `with self._lock` block
        (WingspanEngine's lock is a plain Lock, not an RLock)."""
        self._last_publish = time.monotonic()
        self._published_any = True

    def _stall_error(self):
        """Human-readable reason if the loop has gone quiet, else None.
        Call with self._lock held — it reads shared fields directly."""
        last = self._last_publish
        if last is None or not self._running:
            return None
        quiet_for = time.monotonic() - last
        budget = self.STALL_TIMEOUT_S if self._published_any else self.STALL_STARTUP_GRACE_S
        if quiet_for < budget:
            return None
        if not self._published_any:
            return (f"Camera never delivered a frame ({quiet_for:.0f}s). "
                    "Press STOP, check the camera is connected and not in use "
                    "by another app, then start again.")
        return (f"Camera stalled — no new frame for {quiet_for:.0f}s. "
                "Press STOP and start again; restart the server if it keeps happening.")
