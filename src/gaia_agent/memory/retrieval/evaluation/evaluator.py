from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_memory_ids: set[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float


class RetrievalEvaluator:

    @staticmethod
    def evaluate(
        results: Sequence[tuple[object, float]],
        case: RetrievalCase,
    ) -> RetrievalMetrics:

        if not results:
            return RetrievalMetrics(
                precision_at_k=0.0,
                recall_at_k=0.0,
                mrr=0.0,
            )

        retrieved_ids = [
            str(memory.memory_id)
            for memory, _ in results
        ]

        relevant_ids = case.relevant_memory_ids

        relevant_count = sum(
            1
            for memory_id in retrieved_ids
            if memory_id in relevant_ids
        )

        precision = (
            relevant_count / len(retrieved_ids)
        )

        recall = (
            relevant_count / len(relevant_ids)
            if relevant_ids
            else 0.0
        )

        mrr = 0.0

        for rank, memory_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if memory_id in relevant_ids:
                mrr = 1.0 / rank
                break

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
        )