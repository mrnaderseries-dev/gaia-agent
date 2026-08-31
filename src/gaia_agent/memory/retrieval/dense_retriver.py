from __future__ import annotations

from typing import Sequence

from ..models import Memory
from .embedding import EmbeddingProvider
from .models import RetrievalCandidate


class CandidateRetriever:

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    async def retrieve(
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

        query_embedding = await self.embedding_provider.embed(query)

        scored_memories = []

        for memory in memories:
            memory_embedding = await self.embedding_provider.embed(
                memory.content
            )

            similarity = self._cosine_similarity(
                query_embedding,
                memory_embedding,
            )

            scored_memories.append((memory, similarity))

        scored_memories.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [ RetrievalCandidate(
            memory=memory,
            dense_score=score)
            for memory , score in scored_memories[:top_k]
        ]

    @staticmethod
    def _cosine_similarity(
        vector_a: Sequence[float],
        vector_b: Sequence[float],
    ) -> float:

        if len(vector_a) != len(vector_b):
            raise ValueError("Embedding vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        magnitude_a = sum(a * a for a in vector_a) ** 0.5

        magnitude_b = sum(b * b for b in vector_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            raise ValueError("Embedding vector cannot have zero magnitude")

        return dot_product / (magnitude_a * magnitude_b)
