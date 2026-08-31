from dataclasses import dataclass
from typing import Sequence

from ..models import Memory


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
        results: Sequence[tuple[Memory, float]],
        case: RetrievalCase,
    ) -> RetrievalMetrics:

        if not case.relevant_memory_ids:
            raise ValueError(
                "relevant_memory_ids must not be empty"
            )

        retrieved_ids = [
            str(memory.memory_id)
            for memory, _ in results
        ]

        relevant_retrieved = sum(
            memory_id in case.relevant_memory_ids
            for memory_id in retrieved_ids
        )

        precision = (
            relevant_retrieved / len(retrieved_ids)
            if retrieved_ids
            else 0.0
        )

        recall = (
            relevant_retrieved
            / len(case.relevant_memory_ids)
        )

        reciprocal_rank = 0.0

        for rank, memory_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if memory_id in case.relevant_memory_ids:
                reciprocal_rank = 1.0 / rank
                break

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=reciprocal_rank,
        )