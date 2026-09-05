from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TokenUsage:
   

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTrackerProtocol(Protocol):
    def record(
        self,
        operation: str,
        *,
        usage: TokenUsage,
    ) -> None:
        ...