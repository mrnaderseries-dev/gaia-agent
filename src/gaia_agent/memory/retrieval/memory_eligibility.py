from __future__ import annotations

from datetime import datetime, timezone

from ..models import Memory


class MemoryEligibilityFilter:
    """
    Filters memories before retrieval.

    This is NOT a ranking component.
    It only decides whether a memory is eligible
    to participate in retrieval.
    """

    def __init__(
        self,
        min_confidence: float = 0.30,
        min_importance: float = 0.10,
        max_age_days: float | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_importance = min_importance
        self.max_age_days = max_age_days

    def filter(
        self,
        memories: list[Memory],
    ) -> list[Memory]:

        eligible: list[Memory] = []

        for memory in memories:

            if not memory.active:
                continue

            if memory.confidence < self.min_confidence:
                continue

            if memory.importance < self.min_importance:
                continue

            if self.max_age_days is not None:
                if not self._is_recent(memory):
                    continue

            eligible.append(memory)

        return eligible

    def _is_recent(self, memory: Memory) -> bool:

        now = datetime.now(timezone.utc)

        age = now - memory.updated_at

        return age.total_seconds() <= (
            self.max_age_days * 24 * 60 * 60
        )
    