from __future__ import annotations

import re
from typing import Sequence

from .models import RetrievalCandidate


class Reranker:
    """
    Re-ranks retrieval candidates using query-memory lexical overlap.

    The reranker does not perform retrieval.
    It only refines the ordering of candidates that were
    already produced by the hybrid retrieval stage.
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_tokens = set(
            self._tokenize(query)
        )

        for candidate in candidates:

            memory_tokens = set(
                self._tokenize(
                    candidate.memory.content
                )
            )

            overlap_score = self._overlap(
                query_tokens,
                memory_tokens,
            )

            # Hybrid remains the main relevance signal.
            # Lexical overlap provides an additional refinement.
            candidate.reranker_score = (
                0.70 * candidate.hybrid_score
                + 0.30 * overlap_score
            )

        ranked = sorted(
            candidates,
            key=lambda candidate: candidate.reranker_score,
            reverse=True,
        )

        return ranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    @staticmethod
    def _overlap(
        query_tokens: set[str],
        memory_tokens: set[str],
    ) -> float:

        if not query_tokens:
            return 0.0

        return (
            len(query_tokens & memory_tokens)
            / len(query_tokens)
        )