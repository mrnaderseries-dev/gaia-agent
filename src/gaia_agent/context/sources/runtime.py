from __future__ import annotations

from typing import Any

from .base import ContextSource
from ..models import (
    ContextRequest,
    RuntimeContext,
)


class RuntimeSource(ContextSource):

    async def get(
        self,
        request: ContextRequest,
    ) -> list[RuntimeContext]:

        return [
            RuntimeContext(
                iteration=request.iteration,
                tool_name=request.tool_name,
                blocked=request.blocked,
                tool_result=request.tool_result,
                tool_error=request.tool_error,
            )
        ]

    def is_available(
        self,
        request: ContextRequest,
    ) -> bool:

        return (
            request.iteration > 0
            or request.tool_name is not None
            or request.blocked
            or request.tool_result is not None
            or request.tool_error is not None
        )