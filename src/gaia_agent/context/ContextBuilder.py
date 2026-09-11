from __future__ import annotations

from .ContextBudget import ContextBudget
from .ContextCompressor import ContextCompressor
from .ContextPolicy import ContextPolicy
from .ContextValidator import ContextValidator
from .models import ContextRequest, FinalContext

from .sources.attachments import AttachmentSource
from .sources.conversation import ConversationSource
from .sources.history import HistorySource
from .sources.memory import MemorySource
from .sources.runtime import RuntimeSource


class ContextBuilder:
    def __init__(
        self,
        policy: ContextPolicy,
        budget: ContextBudget,
        validator: ContextValidator,
        compressor: ContextCompressor,
        attachment_source: AttachmentSource,
        conversation_source: ConversationSource,
        history_source: HistorySource,
        memory_source: MemorySource,
        runtime_source: RuntimeSource,
    ) -> None:
        self.policy = policy
        self.budget = budget
        self.validator = validator
        self.compressor = compressor

        self.attachment_source = attachment_source
        self.conversation_source = conversation_source
        self.history_source = history_source
        self.memory_source = memory_source
        self.runtime_source = runtime_source

    async def build(
        self,
        request: ContextRequest,
    ) -> FinalContext:

        context: list[object] = []

        if self.policy.include_attachments:
            if self.attachment_source.is_available(
                request
            ):
                context.extend(
                    await self.attachment_source.get(
                        request
                    )
                )

        if self.policy.include_conversation:
            if self.conversation_source.is_available(
                request
            ):
                context.extend(
                    await self.conversation_source.get(
                        request
                    )
                )
        if self.policy.include_history:
            if self.history_source.is_available(
                request
            ):
                context.extend(
                    await self.history_source.get(
                        request
                    )
                )
        if self.policy.include_memory:
            if self.memory_source.is_available(
                request
            ):
                context.extend(
                    await self.memory_source.get(
                        request
                    )
                )

        if self.policy.include_runtime:
            if self.runtime_source.is_available(
                request
            ):
                context.extend(
                    await self.runtime_source.get(
                        request
                    )
                )

        context = await self.compressor.compress(
            context
        )
        validation = self.validator.validate(
            context
        )

        if not validation.valid:
            raise ValueError(
                f"Invalid context: {validation.errors}"
            )

        token_count = self.budget.count_tokens(
            context
        )
        return FinalContext(
            items=context,
            token_count=token_count,
        )