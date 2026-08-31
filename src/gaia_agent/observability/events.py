from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EventType(str, Enum):
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    LLM_REQUEST_STARTED = "llm_request_started"
    LLM_REQUEST_COMPLETED = "llm_request_completed"
    LLM_REQUEST_FAILED = "llm_request_failed"

    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    RETRY_STARTED = "retry_started"
    FALLBACK_USED = "fallback_used"


@dataclass
class ObservabilityEvent:
    event_type: EventType
    correlation_id: UUID

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    agent_id: UUID | None = None
    iteration: int | None = None
    latency: float | None = None
    error: str | None = None
def create_event(
    event_type: EventType,
    correlation_id: UUID,
    metadata: dict[str, Any] | None = None,
    agent_id: UUID | None = None,
    iteration: int | None = None,
    latency: float | None = None,
    error: str | None = None,
) -> ObservabilityEvent:
    return ObservabilityEvent(
        event_type=event_type,
        correlation_id=correlation_id,
        metadata=metadata or {},
        agent_id=agent_id,
        iteration=iteration,
        latency=latency,
        error=error,
    )    