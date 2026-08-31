import pytest

from gaia_agent.memory.retrieval.embedding import (
    OllamaEmbeddingProvider,
)


@pytest.mark.asyncio
async def test_ollama_embedding():

    provider = OllamaEmbeddingProvider()

    embedding = await provider.embed(
        "The user prefers Python for backend development."
    )

    assert embedding
    assert all(
        isinstance(value, float)
        for value in embedding
    )