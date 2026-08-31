from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import DefaultDict
from collections import defaultdict


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:

    def __init__(self) -> None:
        self._usage: DefaultDict[str, TokenUsage] = defaultdict(
            TokenUsage
        )
        self._lock = Lock()

    def record(
        self,
        operation: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:

        if prompt_tokens < 0:
            raise ValueError(
                "prompt_tokens cannot be negative."
            )

        if completion_tokens < 0:
            raise ValueError(
                "completion_tokens cannot be negative."
            )

        with self._lock:
            current = self._usage[operation]

            self._usage[operation] = TokenUsage(
                prompt_tokens=(
                    current.prompt_tokens
                    + prompt_tokens
                ),
                completion_tokens=(
                    current.completion_tokens
                    + completion_tokens
                ),
            )

    def get_usage(
        self,
        operation: str,
    ) -> TokenUsage:

        with self._lock:
            return self._usage[operation]

    def get_total(self) -> TokenUsage:

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

    def reset(self) -> None:

        with self._lock:
            self._usage.clear()