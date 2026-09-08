from __future__ import annotations

from typing import Any

from .base import ContextSource
from ..models import ContextRequest


class MemorySource(ContextSource):
    async def get(
        self,
        request: ContextRequest,
    ) -> list[Any]:

        return []

    def is_available(
        self,
        request: ContextRequest,
    ) -> bool:
        return False