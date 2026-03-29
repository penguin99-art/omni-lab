"""StateMachine with explicit valid transitions."""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


class State:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PERCEIVING = "PERCEIVING"
    ROUTING = "ROUTING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    CAPTURING = "CAPTURING"


VALID_TRANSITIONS: dict[str, set[str]] = {
    State.IDLE:       {State.LISTENING, State.CAPTURING},
    State.LISTENING:  {State.PERCEIVING, State.IDLE, State.CAPTURING},
    State.PERCEIVING: {State.LISTENING, State.ROUTING, State.CAPTURING},
    State.ROUTING:    {State.PERCEIVING, State.THINKING, State.LISTENING, State.CAPTURING},
    State.THINKING:   {State.SPEAKING, State.THINKING},
    State.SPEAKING:   {State.LISTENING, State.CAPTURING},
    State.CAPTURING:  {State.LISTENING, State.PERCEIVING, State.ROUTING, State.IDLE},
}


class InvalidTransition(Exception):
    pass


class StateMachine:
    def __init__(self) -> None:
        self.state = State.IDLE
        self._listeners: list[Callable] = []

    def transition(self, new_state: str) -> None:
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(f"{self.state} -> {new_state}")
        old = self.state
        self.state = new_state
        for cb in self._listeners:
            try:
                cb(old, new_state)
            except Exception:
                log.exception("StateMachine listener error")

    def on_change(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def __repr__(self) -> str:
        return f"StateMachine(state={self.state!r})"
