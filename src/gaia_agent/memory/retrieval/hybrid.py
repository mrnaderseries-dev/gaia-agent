from __future__ import annotations

from .models import RetrievalCandidate


class HybridScorer:

    def __init__(
        self,
        dense_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> None:

        if dense_weight < 0 or lexical_weight < 0:
            raise ValueError(
                "Weights must be non-negative"
            )

        if dense_weight + lexical_weight == 0:
            raise ValueError(
                "At least one weight must be greater than 0"
            )

        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def score(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:

        if not candidates:
            return []

        lexical_scores = [
            candidate.lexical_score
            for candidate in candidates
        ]

        min_score = min(lexical_scores)
        max_score = max(lexical_scores)

        for candidate in candidates:

            normalized_lexical = self._normalize(
                score=candidate.lexical_score,
                min_score=min_score,
                max_score=max_score,
            )

            candidate.lexical_score = normalized_lexical

            candidate.hybrid_score = (
                self.dense_weight
                * candidate.dense_score
                +
                self.lexical_weight
                * normalized_lexical
            )

        return candidates

    @staticmethod
    def _normalize(
        score: float,
        min_score: float,
        max_score: float,
    ) -> float:

        if max_score == min_score:
            return 0.0

        normalized = (
            (score - min_score)
            / (max_score - min_score)
        )

        return max(
            0.0,
            min(1.0, normalized),
        )