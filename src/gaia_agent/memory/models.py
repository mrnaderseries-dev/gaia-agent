from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    SKILL = "skill"
    CONSTRAINT = "constraint"
    EVENT = "event"


class MemorySource(str, Enum):
    USER = "user"
    CONVERSATION = "conversation"
    SYSTEM = "system"
    AGENT = "agent"


@dataclass(slots=True)
class Memory:
    """
    Represents a persistent piece of user-related information.
    """

    user_id: int
    content: str

    memory_id: UUID = field(
        default_factory=uuid4
    )

    memory_type: MemoryType = MemoryType.FACT

    source: MemorySource = (MemorySource.CONVERSATION)

    importance: float = 0.5
    confidence: float = 0.5

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    last_accessed_at: datetime | None = None

    access_count: int = 0

    active: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError(
                "Memory user_id cannot be empty."
            )

        if not self.content.strip():
            raise ValueError(
                "Memory content cannot be empty."
            )

        self._validate_score(
            self.importance,
            "importance",
        )

        self._validate_score(
            self.confidence,
            "confidence",
        )

        if self.access_count < 0:
            raise ValueError(
                "Memory access_count cannot be negative."
            )

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Memory {name} must be between "
                "0.0 and 1.0."
            )

    def update_content(
        self,
        content: str,
    ) -> None:

        if not content.strip():
            raise ValueError(
                "Memory content cannot be empty."
            )

        self.content = content
        self.updated_at = datetime.now(timezone.utc)

    def update_importance(
        self,
        importance: float,
    ) -> None:

        self._validate_score(
            importance,
            "importance",
        )

        self.importance = importance
        self.updated_at = datetime.now(timezone.utc)

    def update_confidence(
        self,
        confidence: float,
    ) -> None:

        self._validate_score(
            confidence,
            "confidence",
        )

        self.confidence = confidence
        self.updated_at = datetime.now(timezone.utc)

    def record_access(self) -> None:
        """
        Record that this memory was retrieved/used.
        """

        self.access_count += 1
        self.last_accessed_at = (
            datetime.now(timezone.utc)
        )

    def deactivate(self) -> None:
        self.active = False
        self.updated_at = datetime.now(timezone.utc)

    def activate(self) -> None:
        self.active = True
        self.updated_at = datetime.now(timezone.utc)