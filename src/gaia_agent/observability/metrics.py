from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict


class Metrics:
    """
    Collects application metrics.

    Responsible only for recording and retrieving metrics.
    It does not handle logging, tracing, tokens, or costs.
    """

    def __init__(self) -> None:
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._durations: DefaultDict[str, list[float]] = defaultdict(list)

        self._lock = Lock()

    def increment(
        self,
        name: str,
        value: int = 1,
    ) -> None:
        """Increment a counter metric."""

        with self._lock:
            self._counters[name] += value

    def record_duration(
        self,
        name: str,
        duration: float,
    ) -> None:
        """Record a duration value in seconds."""

        with self._lock:
            self._durations[name].append(duration)

    def get_counter(self, name: str) -> int:
        """Return the current value of a counter."""

        with self._lock:
            return self._counters[name]

    def get_durations(self, name: str) -> list[float]:
        """Return recorded durations."""

        with self._lock:
            return list(self._durations[name])

    def get_average_duration(self, name: str) -> float:
        """Return the average duration for a metric."""

        with self._lock:
            durations = self._durations[name]

            if not durations:
                return 0.0

            return sum(durations) / len(durations)