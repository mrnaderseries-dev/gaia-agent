from future import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar

from .model import LLMModel


T = TypeVar("T")


Message = dict[str, Any]


class LLMClient(ABC):
    """
    Provider-independent LLM interface.

    The client is responsible for communicating with the actual
    model provider.

    The service layer above this class is responsible for
    higher-level operations such as image analysis.
    """

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