from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar

from .model import LLMModel


T = TypeVar("T")


Message = dict[str, Any]


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: Sequence[Message],
        *,
        model: LLMModel,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T | str:

        raise NotImplementedError