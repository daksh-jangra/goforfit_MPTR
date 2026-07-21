"""
state_machine.py
-----------------
A small, explicit finite state machine controlling the overall application
lifecycle. Centralising this logic means the UI, detection, and timing
modules never have to guess "are we currently allowed to count a lap" --
they simply ask the state machine.

States
------
READY    - webcam running, detection active, nothing is being timed/counted.
RUNNING  - a live test is in progress; laps are being counted and timed.
STOPPED  - the test has been paused/ended; all values are frozen for review.

Transitions are driven exclusively by the Spacebar (see main.py):
  READY   --SPACE--> RUNNING
  RUNNING --SPACE--> STOPPED
  STOPPED --SPACE--> (full reset) --> RUNNING
"""

from enum import Enum
from typing import Callable, Optional


class AppState(Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class StateMachine:
    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        on_start : called whenever a test transitions into RUNNING
        on_stop  : called whenever a test transitions into STOPPED
        on_reset : called right before a STOPPED->RUNNING restart re-arms everything
        """
        self._state = AppState.READY
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_reset = on_reset

    @property
    def state(self) -> AppState:
        return self._state

    def is_running(self) -> bool:
        return self._state == AppState.RUNNING

    def handle_space(self) -> None:
        """Advance the state machine in response to a Spacebar press."""
        if self._state == AppState.READY:
            self._state = AppState.RUNNING
            if self._on_start:
                self._on_start()

        elif self._state == AppState.RUNNING:
            self._state = AppState.STOPPED
            if self._on_stop:
                self._on_stop()

        elif self._state == AppState.STOPPED:
            if self._on_reset:
                self._on_reset()
            self._state = AppState.RUNNING
            if self._on_start:
                self._on_start()

    def status_text(self) -> str:
        return self._state.value
