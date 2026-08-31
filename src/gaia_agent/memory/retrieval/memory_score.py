from __future__ import annotations

from datetime import datetime, timezone

from .models import RetrievalCandidate


class MemoryScoreCalculator:

    def __init__(
        self,
        relevance_weight: float = 0.60,
        importance_weight: float = 0.15,
        confidence_weight: float = 0.10,
        recency_weight: float = 0.10,
        usage_weight: float = 0.05,
    ) -> None:

        total = (
            relevance_weight
            + importance_weight
            + confidence_weight
            + recency_weight
            + usage_weight
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Memory score weights must sum to 1.0"
            )

        self.relevance_weight = relevance_weight
        self.importance_weight = importance_weight
        self.confidence_weight = confidence_weight
        self.recency_weight = recency_weight
        self.usage_weight = usage_weight

    def calculate(
        self,
        candidate: RetrievalCandidate,
    ) -> float:

        memory = candidate.memory

        # Use reranker score if a reranker has actually
        # produced one. Otherwise use hybrid relevance.
        relevance_score = (
            candidate.reranker_score
            if candidate.reranker_score > 0.0
            else candidate.hybrid_score
        )

        recency_score = self._recency_score(
            memory.updated_at
        )

        usage_score = self._usage_score(
            memory.access_count
        )

        score = (
            self.relevance_weight * relevance_score
            + self.importance_weight * memory.importance
            + self.confidence_weight * memory.confidence
            + self.recency_weight * recency_score
            + self.usage_weight * usage_score
        )

        return max(
            0.0,
            min(1.0, score),
        )

    def score(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:

        for candidate in candidates:
            candidate.memory_score = self.calculate(
                candidate
            )

        return sorted(
            candidates,
            key=lambda candidate: candidate.memory_score,
            reverse=True,
        )

    @staticmethod
    def _recency_score(
        updated_at: datetime,
    ) -> float:

        now = datetime.now(timezone.utc)

        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(
                tzinfo=timezone.utc
            )

        age_days = (
            now - updated_at
        ).total_seconds() / 86400

        if age_days <= 0:
            return 1.0

        # 30-day half-life.
        return 2.0 ** (-age_days / 30.0)

    @staticmethod
    def _usage_score(
        access_count: int,
    ) -> float:

        if access_count <= 0:
            return 0.0

        return min(
            1.0,
            access_count / 10.0,
        )