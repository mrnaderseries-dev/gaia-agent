from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict


class Metrics:
    def __init__(self) -> None:
        self._counters: DefaultDict[str, int] = (
            defaultdict(int)
        )

        self._durations: DefaultDict[
            str,
            list[float],
        ] = defaultdict(list)

        self._lock = Lock()

    def increment(
        self,
        name: str,
        value: int = 1,
    ) -> None:
        if not name:
            raise ValueError(
                "metric name must not be empty"
            )

        if value < 0:
            raise ValueError(
                "metric increment must be >= 0"
            )

        with self._lock:
            self._counters[name] += value

    def record_duration(
        self,
        name: str,
        duration: float,
    ) -> None:
        if not name:
            raise ValueError(
                "metric name must not be empty"
            )

        if duration < 0:
            raise ValueError(
                "duration must be >= 0"
            )

        with self._lock:
            self._durations[name].append(
                duration
            )

    def get_counter(
        self,
        name: str,
    ) -> int:
        with self._lock:
            return self._counters[name]

    def get_durations(
        self,
        name: str,
    ) -> list[float]:
        with self._lock:
            return list(
                self._durations[name]
            )

    def get_average_duration(
        self,
        name: str,
    ) -> float:
        with self._lock:
            durations = self._durations[name]

            if not durations:
                return 0.0

            return sum(durations) / len(
                durations
            )

    def snapshot_counters(
        self,
    ) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def snapshot_durations(
        self,
    ) -> dict[str, list[float]]:
        with self._lock:
            return {
                name: list(values)
                for name, values
                in self._durations.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._durations.clear()