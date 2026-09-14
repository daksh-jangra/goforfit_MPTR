"""
timer.py
--------
Timing primitives for the fitness test.

Design note: instead of relying on wall-clock timestamps (which get messy
when a timer needs to be paused/resumed across STOPPED periods), every timer
here accumulates elapsed time via an explicit `update(dt)` call that the main
loop invokes once per frame with the measured delta-time. This makes
start/stop/reset trivial to reason about and keeps all timers perfectly in
sync with the video processing loop.
"""


class GlobalTimer:
    """Tracks the overall test duration."""

    def __init__(self) -> None:
        self.total_elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self.total_elapsed = 0.0
        self._running = False

    def update(self, dt: float) -> None:
        if self._running:
            self.total_elapsed += dt

    @property
    def is_running(self) -> bool:
        return self._running


class LaneTimer:
    """
    Tracks per-lane statistics:
      - total_elapsed: time since the test started for this lane
      - current_lap_elapsed: time since the last recorded lap (i.e. the
        "Current Timer" shown in the stats panel for the lap in progress)
      - last_lap_time: duration of the most recently completed lap
      - lap_count: number of laps completed
    """

    def __init__(self) -> None:
        self.total_elapsed: float = 0.0
        self.current_lap_elapsed: float = 0.0
        self.last_lap_time: float = 0.0
        self.lap_count: int = 0
        self._running: bool = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self.total_elapsed = 0.0
        self.current_lap_elapsed = 0.0
        self.last_lap_time = 0.0
        self.lap_count = 0
        self._running = False

    def update(self, dt: float) -> None:
        if self._running:
            self.total_elapsed += dt
            self.current_lap_elapsed += dt

    def record_lap(self) -> None:
        """Called exactly once per valid finish-line crossing."""
        self.last_lap_time = self.current_lap_elapsed
        self.current_lap_elapsed = 0.0
        self.lap_count += 1

    @property
    def is_running(self) -> bool:
        return self._running
