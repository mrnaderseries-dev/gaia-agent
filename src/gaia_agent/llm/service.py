from __future__ import annotations

from typing import Any, Sequence, TypeVar

from .client import LLMClient
from .model import LLMModel


T = TypeVar("T")


class LLMService:

    def __init__(
        self,
        client: LLMClient,
        model: LLMModel,
    ) -> None:
        self.client = client
        self.model = model

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T | str:

        return await self.client.generate(
            messages,
            model=self.model,
            output_schema=output_schema,
            tools=tools,
            **kwargs,
        )
    