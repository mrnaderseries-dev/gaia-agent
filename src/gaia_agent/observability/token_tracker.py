from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict

from gaia_agent.llm.usage import TokenUsage


class TokenTracker:
    """
    Thread-safe, provider-independent token usage tracker.

    The tracker receives normalized TokenUsage objects.
    It does not know anything about Ollama, OpenAI, Anthropic,
    or provider-specific response fields.
    """

    def __init__(self) -> None:
        self._usage: DefaultDict[str, TokenUsage] = defaultdict(TokenUsage)
        self._lock = Lock()

    def record(
        self,
        operation: str,
        *,
        usage: TokenUsage,
    ) -> None:
        """
        Record normalized token usage for an operation.
        """

        if not operation:
            raise ValueError("operation must not be empty")

        if usage.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")

        if usage.completion_tokens < 0:
            raise ValueError("completion_tokens must be >= 0")

        with self._lock:
            current = self._usage[operation]

            self._usage[operation] = TokenUsage(
                prompt_tokens=(
                    current.prompt_tokens + usage.prompt_tokens
                ),
                completion_tokens=(
                    current.completion_tokens + usage.completion_tokens
                ),
            )

    def get(self, operation: str) -> TokenUsage:
        """
        Return accumulated usage for one operation.
        """

        with self._lock:
            return self._usage.get(operation, TokenUsage())

    def total(self) -> TokenUsage:
        """
        Return accumulated usage across all operations.
        """

        with self._lock:
            prompt_tokens = sum(
                usage.prompt_tokens
                for usage in self._usage.values()
            )

            completion_tokens = sum(
                usage.completion_tokens
                for usage in self._usage.values()
            )

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def snapshot(self) -> dict[str, TokenUsage]:
        """
        Return a copy of all accumulated usage.
        """

        with self._lock:
            return dict(self._usage)

    def reset(self) -> None:
        """
        Clear all tracked usage.
        """

        with self._lock:
            self._usage.clear()