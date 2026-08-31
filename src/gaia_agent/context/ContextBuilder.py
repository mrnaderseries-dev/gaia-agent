from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gaia_agent.core.agent_state import AgentState

from .ContextBudget import ContextBudget
from .ContextCompressor import ContextCompressor
from .ContextPolicy import ContextPolicy
from .ContextValidator import ContextValidator

from .sources.conversation import ConversationSource
from .sources.history import HistorySource
from .sources.memory import MemorySource
from .sources.runtime import RuntimeSource


@dataclass(slots=True)
class FinalContext:

    items: list[Any]
    token_count: int


class ContextBuilder:

    def __init__(
        self,
        policy: ContextPolicy,
        budget: ContextBudget,
        validator: ContextValidator,
        compressor: ContextCompressor,
        conversation_source: ConversationSource,
        history_source: HistorySource,
        memory_source: MemorySource,
        runtime_source: RuntimeSource,
    ) -> None:
        self.policy = policy
        self.budget = budget
        self.validator = validator
        self.compressor = compressor

        self.conversation_source = conversation_source
        self.history_source = history_source
        self.memory_source = memory_source
        self.runtime_source = runtime_source

    async def build(
        self,
        state: AgentState,
    ) -> FinalContext:

        context: list[Any] = []

        if (
            self.policy.include_conversation
            and self.conversation_source.is_available(state)
        ):
            context.extend(
                await self.conversation_source.get(state)
            )

        if (
            self.policy.include_history
            and self.history_source.is_available(state)
        ):
            context.extend(
                await self.history_source.get(state)
            )

        if (
            self.policy.include_memory
            and self.memory_source.is_available(state)
        ):
            context.extend(
                await self.memory_source.get(state)
            )

        if (
            self.policy.include_runtime
            and self.runtime_source.is_available(state)
        ):
            context.extend(
                await self.runtime_source.get(state)
            )

        context = await self.compressor.compress(context)

        validation = self.validator.validate(context)

        if not validation.valid:
            raise ValueError(
                f"Invalid context: {validation.errors}"
            )

        token_count = self.budget.count_tokens(context)

        return FinalContext(
            items=context,
            token_count=token_count,
        )