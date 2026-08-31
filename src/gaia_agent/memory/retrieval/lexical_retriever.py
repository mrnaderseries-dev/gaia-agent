from __future__ import annotations

import math
import re
from typing import Sequence

from ..models import Memory
from .models import RetrievalCandidate


class LexicalRetriever:
    """
    Performs lexical retrieval using BM25.

    Responsible only for calculating lexical relevance.
    It does not calculate hybrid or final memory scores.
    """

    def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:

        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not memories:
            return []

        tokenized_memories = [
            self._tokenize(memory.content)
            for memory in memories
        ]

        query_tokens = self._tokenize(query)

        scores = self._bm25_scores(
            query_tokens,
            tokenized_memories,
        )

        candidates = [
            RetrievalCandidate(
                memory=memory,
                lexical_score=score,
            )
            for memory, score in zip(memories, scores)
        ]

        candidates.sort(
            key=lambda candidate: candidate.lexical_score,
            reverse=True,
        )

        return candidates[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    @staticmethod
    def _bm25_scores(
        query_tokens: list[str],
        documents: list[list[str]],
    ) -> list[float]:

        if not documents:
            return []

        document_count = len(documents)

        average_length = (
            sum(len(document) for document in documents)
            / document_count
        )

        if average_length == 0:
            return [0.0] * document_count

        document_frequency: dict[str, int] = {}

        for document in documents:
            unique_terms = set(document)

            for term in unique_terms:
                document_frequency[term] = (
                    document_frequency.get(term, 0) + 1
                )

        k1 = 1.5
        b = 0.75

        scores: list[float] = []

        for document in documents:

            document_length = len(document)

            term_frequency: dict[str, int] = {}

            for term in document:
                term_frequency[term] = (
                    term_frequency.get(term, 0) + 1
                )

            score = 0.0

            for term in query_tokens:

                tf = term_frequency.get(term, 0)

                if tf == 0:
                    continue

                df = document_frequency.get(term, 0)

                idf = math.log(
                    1.0
                    + (
                        document_count - df + 0.5
                    )
                    / (
                        df + 0.5
                    )
                )

                denominator = (
                    tf
                    + k1
                    * (
                        1
                        - b
                        + b
                        * document_length
                        / average_length
                    )
                )

                score += (
                    idf
                    * (
                        tf
                        * (k1 + 1)
                        / denominator
                    )
                )

            scores.append(score)

        return scores