from __future__ import annotations

from typing import Any

from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel

from .ContextBudget import ContextBudget
from .ContextPolicy import ContextPolicy, ContextPriority


class ContextCompressionError(Exception):
    """
    Raised when context cannot be compressed
    enough to fit within the token budget.
    """

    def __init__(
        self,
        message: str,
        *,
        recoverable: bool = True,
        requires_human_approval: bool = False,
    ) -> None:
        super().__init__(message)
        self.recoverable = recoverable
        self.requires_human_approval = requires_human_approval


class ContextCompressor:

    MAX_ATTEMPTS = 2

    def __init__(
        self,
        client: LLMClient,
        model: LLMModel,
        budget: ContextBudget,
        policy: ContextPolicy,
    ) -> None:
        self.client = client
        self.model = model
        self.budget = budget
        self.policy = policy

    async def compress(
        self,
        context: list[Any],
    ) -> list[Any]:

        if self.budget.fits(context):
            return context

        current_context = list(context)

        for i in range(self.MAX_ATTEMPTS):

            current_context = await self._compress_once(
                current_context
            )

            if self.budget.fits(current_context):
                return current_context

        token_count = self.budget.count_tokens(
            current_context
        )

        raise ContextCompressionError(
            (
                "Context could not be compressed "
                f"within the token budget after "
                f"{self.MAX_ATTEMPTS} attempts. "
                f"tokens={token_count}, "
                f"max={self.budget.max_tokens}"
            ),
            recoverable=False,
            requires_human_approval=True,
        )

    async def _compress_once(
        self,
        context: list[Any],
    ) -> list[Any]:

        preserved: list[Any] = []
        compressible: list[Any] = []

        for item in context:

            priority = self._get_priority(item)

            if priority == ContextPriority.PRESERVE:
                preserved.append(item)

            elif priority == ContextPriority.COMPRESS:
                compressible.append(item)

            elif priority == ContextPriority.DISCARD:
                continue

        result = list(preserved)

        if compressible:
            compressed = await self._compress_items(
                compressible
            )
            result.extend(compressed)

        return result

    async def _compress_items(
        self,
        items: list[Any],
    ) -> list[Any]:

        prompt = self._build_prompt(items)

        response = await self.client.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a context compression component "
                        "for an AI agent. Compress the provided "
                        "context while preserving facts, decisions, "
                        "constraints, and information required to "
                        "complete the task. Do not invent information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=self.model,
        )

        return [response]

    def _build_prompt(
        self,
        items: list[Any],
    ) -> str:

        return (
            "Compress the following agent context.\n\n"
            "Rules:\n"
            "- Do not invent information.\n"
            "- Do not change factual meaning.\n"
            "- Remove redundancy.\n"
            "- Remove irrelevant details.\n"
            "- Preserve important decisions.\n"
            "- Preserve constraints.\n"
            "- Preserve information needed for the current task.\n\n"
            f"Context:\n{items}"
        )

    def _get_priority(
        self,
        item: Any,
    ) -> ContextPriority:

        name = type(item).__name__

        if name == "ConversationContext":
            return self.policy.conversation_priority

        if name == "MemoryContext":
            return self.policy.memory_priority

        if name == "HistoryContext":
            return self.policy.history_priority

        if name == "RuntimeContext":
            return self.policy.runtime_priority

        return ContextPriority.COMPRESS
    