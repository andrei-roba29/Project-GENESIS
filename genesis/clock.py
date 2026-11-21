"""Simulation clock.

Default rate: 1 real hour = 100 simulated years
(1 simulated year ~= 36 real seconds, 1 simulated day ~= 0.099 real seconds).
Simulation time is independent of rendering frame rate; the engine advances
in scheduled sub-steps (weather daily, flora weekly, fauna monthly, soil
daily, climate seasonal) and never recomputes the whole planet per frame.
"""
from __future__ import annotations


class SimClock:
    """Tracks simulated time and the update scheduler."""

    # Update periods in simulated days.
    PERIODS = {
        "weather": 1.0,
        "soil": 1.0,
        "flora": 7.0,
        "fauna": 30.0,
        "climate": 30.0,
        "resources": 30.0,
        "stats": 30.0,
    }

    def __init__(self, sim_day: float = 0.0):
        self.sim_day = float(sim_day)
        self._next = {k: 0.0 for k in self.PERIODS}

    def due(self, system: str) -> bool:
        return self.sim_day >= self._next[system] - 1e-9

    def mark(self, system: str) -> None:
        self._next[system] = self.sim_day + self.PERIODS[system]

    def advance(self, days: float) -> None:
        self.sim_day += days

    def to_dict(self) -> dict:
        return {"sim_day": self.sim_day, "next": dict(self._next)}

    @classmethod
    def from_dict(cls, d: dict) -> "SimClock":
        c = cls(d.get("sim_day", 0.0))
        c._next.update(d.get("next", {}))
        return c
