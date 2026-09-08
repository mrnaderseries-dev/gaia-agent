from __future__ import annotations

from .base import ContextSource
from ..models import (
    ContextRequest,
    ConversationContext,
)


class ConversationSource(ContextSource):

    async def get(
        self,
        request: ContextRequest,
    ) -> list[ConversationContext]:

        return [
            ConversationContext(
                messages=list(request.conversation)
            )
        ]

    def is_available(
        self,
        request: ContextRequest,
    ) -> bool:

        return bool(request.conversation)