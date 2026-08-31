from __future__ import annotations

from .models import RetrievalCandidate


class Ranker:

    def rank(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:

        return sorted(
            candidates,
            key=lambda candidate: candidate.hybrid_score,
            reverse=True,
        )