from __future__ import annotations

from .base import ContextSource
from ..models import Attachment, ContextRequest


class AttachmentSource(ContextSource):
    async def get(
        self,
        request: ContextRequest,
    ) -> list[Attachment]:
        return list(request.attachments)

    def is_available(
        self,
        request: ContextRequest,
    ) -> bool:
        return bool(request.attachments)