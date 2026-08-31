from datetime import datetime, timezone
from uuid import uuid4

import pytest

from gaia_agent.memory.models import Memory
from gaia_agent.memory.retrieval.embedding import EmbeddingProvider
from gaia_agent.memory.retrieval.retriever import MemoryRetriever


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Fake embedding provider used only for testing.

    It returns deterministic vectors so that we can test
    retrieval and cosine similarity without Ollama.
    """

    def __init__(self):
        self.embeddings = {
            "The user prefers Python for backend development.": [
                1.0,
                0.0,
                0.0,
            ],
            "The user is learning Python async programming.": [
                0.95,
                0.05,
                0.0,
            ],
            "The user uses FastAPI for backend APIs.": [
                0.90,
                0.10,
                0.0,
            ],
            "The user is building the GAIA AI agent.": [
                0.0,
                1.0,
                0.0,
            ],
            "The user is studying AI agent architecture.": [
                0.0,
                0.95,
                0.05,
            ],
            "The user is learning LangGraph.": [
                0.0,
                0.90,
                0.10,
            ],
            "The user is learning RAG and vector databases.": [
                0.0,
                0.0,
                1.0,
            ],
            "The user is studying embeddings and vector search.": [
                0.05,
                0.0,
                0.95,
            ],
            "The user is using PostgreSQL for persistence.": [
                0.10,
                0.0,
                0.90,
            ],
            "The user is learning Docker for deployment.": [
                0.30,
                0.0,
                0.70,
            ],
        }

        self.query_embeddings = {
            "What programming language does the user prefer?": [
                1.0,
                0.0,
                0.0,
            ],
            "What is the user building?": [
                0.0,
                1.0,
                0.0,
            ],
            "What is the user learning about RAG?": [
                0.0,
                0.0,
                1.0,
            ],
            "What database does the user use?": [
                0.0,
                0.0,
                1.0,
            ],
        }

    async def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if text in self.query_embeddings:
            return self.query_embeddings[text]

        return self.embeddings[text]


def create_memory(content: str) -> Memory:
    now = datetime.now(timezone.utc)

    return Memory(
        user_id="test-user",
        content=content,
        memory_id=uuid4(),
        memory_type="fact",
        source="test",
        importance=0.8,
        confidence=0.9,
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
        access_count=0,
        active=True,
        metadata={},
    )


@pytest.fixture
def memories() -> list[Memory]:
    return [
        create_memory(
            "The user prefers Python for backend development."
        ),
        create_memory(
            "The user is learning Python async programming."
        ),
        create_memory(
            "The user uses FastAPI for backend APIs."
        ),
        create_memory(
            "The user is building the GAIA AI agent."
        ),
        create_memory(
            "The user is studying AI agent architecture."
        ),

create_memory(
            "The user is learning LangGraph."
        ),
        create_memory(
            "The user is learning RAG and vector databases."
        ),
        create_memory(
            "The user is studying embeddings and vector search."
        ),
        create_memory(
            "The user is using PostgreSQL for persistence."
        ),
        create_memory(
            "The user is learning Docker for deployment."
        ),
    ]


@pytest.fixture
def retriever() -> MemoryRetriever:
    provider = FakeEmbeddingProvider()

    return MemoryRetriever(
        embedding_provider=provider
    )


@pytest.mark.asyncio
async def test_retrieves_python_memories(
    retriever: MemoryRetriever,
    memories: list[Memory],
):
    results = await retriever.retrieve(
        query="What programming language does the user prefer?",
        memories=memories,
        top_k=3,
    )

    assert len(results) == 3

    assert (
        results[0][0].content
        == "The user prefers Python for backend development."
    )

    assert results[0][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_retrieves_agent_memories(
    retriever: MemoryRetriever,
    memories: list[Memory],
):
    results = await retriever.retrieve(
        query="What is the user building?",
        memories=memories,
        top_k=3,
    )

    assert results[0][0].content == (
        "The user is building the GAIA AI agent."
    )

    assert results[0][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_retrieves_rag_memories(
    retriever: MemoryRetriever,
    memories: list[Memory],
):
    results = await retriever.retrieve(
        query="What is the user learning about RAG?",
        memories=memories,
        top_k=3,
    )

    assert results[0][0].content == (
        "The user is learning RAG and vector databases."
    )

    assert results[0][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_top_k_limits_results(
    retriever: MemoryRetriever,
    memories: list[Memory],
):
    results = await retriever.retrieve(
        query="What programming language does the user prefer?",
        memories=memories,
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_empty_query_raises_error(
    retriever: MemoryRetriever,
    memories: list[Memory],
):
    with pytest.raises(ValueError):
        await retriever.retrieve(
            query="   ",
            memories=memories,
        )


@pytest.mark.asyncio
async def test_invalid_top_k_raises_error(
    retriever: MemoryRetriever,
    memories: list[Memory],
):
    with pytest.raises(ValueError):
        await retriever.retrieve(
            query="What is the user building?",
            memories=memories,
            top_k=0,
        )


@pytest.mark.asyncio
async def test_empty_memories_returns_empty_list(
    retriever: MemoryRetriever,
):
    results = await retriever.retrieve(
        query="What is the user building?",
        memories=[],
        top_k=5,
    )

    assert results == []