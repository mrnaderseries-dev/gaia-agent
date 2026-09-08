from __future__ import annotations

from .base import ContextSource
from ..models import (
    ContextRequest,
    HistoryContext,
)


class HistorySource(ContextSource):

    async def get(
        self,
        request: ContextRequest,
    ) -> list[HistoryContext]:

        return [
            HistoryContext(
                plan=list(request.plan),
                current_step=request.current_step,
                completed_steps=list(
                    request.completed_steps
                ),
                current_action=request.current_action,
                step_type=request.step_type,
            )
        ]

    def is_available(
        self,
        request: ContextRequest,
    ) -> bool:

        return (
            bool(request.plan)
            or request.current_step is not None
            or bool(request.completed_steps)
            or request.current_action is not None
            or request.step_type is not None
        )