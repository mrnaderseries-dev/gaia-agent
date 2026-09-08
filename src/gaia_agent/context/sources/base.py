from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import ContextRequest


class ContextSource(ABC):

    @abstractmethod
    async def get(
        self,
        request: ContextRequest,
    ) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def is_available(
        self,
        request: ContextRequest,
    ) -> bool:
        raise NotImplementedError