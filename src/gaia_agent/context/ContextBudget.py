from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import tiktoken
@dataclass(slots=True)
class ContextBudget:
    max_tokens: int

    def count_tokens(self, context: list[Any]) -> int:
        text = self._serialize(context)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def fits(self, context: list[Any]) -> bool:
        return self.count_tokens(context) <= self.max_tokens

    @staticmethod
    def _serialize(context: list[Any]) -> str:
        return "\n".join(
            str(item)
            for item in context
            if item is not None
        )