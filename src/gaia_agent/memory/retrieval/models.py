from __future__ import annotations

from dataclasses import dataclass

from ..models import Memory


@dataclass
class RetrievalCandidate:
    memory: Memory
    dense_score: float = 0.0
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    reranker_score:float=0.0
    memory_score:float=0.0
@dataclass(frozen=True)
class RetrievalResult:
    memory: Memory
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievalRequest:
    """
    Input parameters for a retrieval operation.
    """

    query: str
    top_k: int = 5