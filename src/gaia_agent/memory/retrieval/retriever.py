from __future__ import annotations

from typing import Sequence

from ..models import Memory
from .dense_retriver import CandidateRetriever
from .lexical_retriever import LexicalRetriever
from .hybrid import HybridScorer
from .models import RetrievalCandidate
from .ranker import Ranker
from .reranker import Reranker
from .memory_score import MemoryScoreCalculator


class MemoryRetriever:

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
        lexical_retriever: LexicalRetriever,
        hybrid_scorer: HybridScorer | None = None,
        ranker: Ranker | None = None,
        reranker: Reranker | None = None,
        memory_score_calculator: MemoryScoreCalculator | None = None,
    ) -> None:

        self.candidate_retriever = candidate_retriever
        self.lexical_retriever = lexical_retriever

        self.hybrid_scorer = (
            hybrid_scorer
            if hybrid_scorer is not None
            else HybridScorer()
        )

        self.ranker = (
            ranker
            if ranker is not None
            else Ranker()
        )

        self.reranker = (
            reranker
            if reranker is not None
            else Reranker()
        )

        self.memory_score_calculator = (
            memory_score_calculator
            if memory_score_calculator is not None
            else MemoryScoreCalculator()
        )

    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:

        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not memories:
            return []

        # 1. Dense retrieval
        dense_candidates = (
            await self.candidate_retriever.retrieve(
                query=query,
                memories=memories,
                top_k=top_k,
            )
        )

        # 2. Lexical retrieval
        lexical_candidates = (
            self.lexical_retriever.retrieve(
                query=query,
                memories=memories,
                top_k=top_k,
            )
        )

        lexical_by_memory_id = {
            candidate.memory.memory_id: candidate
            for candidate in lexical_candidates
        }

        # 3. Merge dense + lexical
        candidates: list[RetrievalCandidate] = []

        for dense_candidate in dense_candidates:

            lexical_candidate = (
                lexical_by_memory_id.get(
                    dense_candidate.memory.memory_id
                )
            )

            lexical_score = (
                lexical_candidate.lexical_score
                if lexical_candidate is not None
                else 0.0
            )

            candidates.append(
                RetrievalCandidate(
                    memory=dense_candidate.memory,
                    dense_score=dense_candidate.dense_score,
                    lexical_score=lexical_score,
                )
            )

        if not candidates:
            return []

        # 4. Hybrid
        candidates = self.hybrid_scorer.score(
            candidates
        )

        # 5. Initial ranking
        candidates = self.ranker.rank(
            candidates
        )

        # 6. Reranking
        candidates = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=len(candidates),
        )

        # 7. Memory score
        candidates = self.memory_score_calculator.score(
            candidates
        )

        # 8. Final top-k
        return candidates[:top_k]
    