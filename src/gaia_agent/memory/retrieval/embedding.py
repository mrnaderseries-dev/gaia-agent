from abc import ABC, abstractmethod
from typing import Sequence

import httpx


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed(self, text: str) -> Sequence[float]:
        raise NotImplementedError


class OllamaEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def embed(self, text: str) -> Sequence[float]:

        if not text.strip():
            raise ValueError("text must not be empty")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
            )

            response.raise_for_status()

            data = response.json()

        embedding = data.get("embedding")

        if not embedding:
            raise ValueError(
                "Ollama returned no embedding"
            )

        return embedding