from __future__ import annotations

from enum import IntEnum


class ContextPriority(IntEnum):
    DISCARD = 0
    COMPRESS = 1
    PRESERVE = 2


class ContextPolicy:

    def __init__(
        self,
        *,
        include_memory: bool = False,
        include_conversation: bool = True,
        include_history: bool = True,
        include_runtime: bool = True,
    ) -> None:
        self.include_memory = include_memory
        self.include_conversation = include_conversation
        self.include_history = include_history
        self.include_runtime = include_runtime

        self.conversation_priority = ContextPriority.PRESERVE
        self.memory_priority = ContextPriority.COMPRESS
        self.history_priority = ContextPriority.COMPRESS
        self.runtime_priority = ContextPriority.COMPRESS