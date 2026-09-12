from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any
from uuid import UUID, uuid4


class EventType(str, Enum):
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    ORCHESTRATION_STARTED = "orchestration_started"
    ORCHESTRATION_COMPLETED = "orchestration_completed"
    ORCHESTRATION_FAILED = "orchestration_failed"

    PLAN_CREATED = "plan_created"
    PLAN_REPLANNED = "plan_replanned"

    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"

    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"

    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"

    APPROVAL_REQUIRED = "approval_required"

    TERMINATION = "termination"

    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"

    LLM_REQUEST_STARTED = "llm_request_started"
    LLM_REQUEST_COMPLETED = "llm_request_completed"
    LLM_REQUEST_FAILED = "llm_request_failed"

    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    RETRY_STARTED = "retry_started"
    FALLBACK_USED = "fallback_used"


@dataclass(frozen=True, slots=True)
class ObservabilityEvent:
    event_type: EventType
    correlation_id: UUID

    event_id: UUID = field(
        default_factory=uuid4
    )

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    agent_id: UUID | None = None

    run_id: UUID | None = None

    iteration: int | None = None

    step_id: int | None = None

    attempt: int | None = None

    phase: str | None = None

    latency: float | None = None

    error: str | None = None


def create_event(
    event_type: EventType,
    correlation_id: UUID,
    *,
    metadata: dict[str, Any] | None = None,
    agent_id: UUID | None = None,
    run_id: UUID | None = None,
    iteration: int | None = None,
    step_id: int | None = None,
    attempt: int | None = None,
    phase: str | None = None,
    latency: float | None = None,
    error: str | None = None,
) -> ObservabilityEvent:
    return ObservabilityEvent(
        event_type=event_type,
        correlation_id=correlation_id,
        metadata=dict(metadata or {}),
        agent_id=agent_id,
        run_id=run_id,
        iteration=iteration,
        step_id=step_id,
        attempt=attempt,
        phase=phase,
        latency=latency,
        error=error,
    )


class EventLogger:
    def init(self) -> None:
        self._events: list[ObservabilityEvent] = []
        self._lock = Lock()

    def log(
        self,
        event: ObservabilityEvent,
    ) -> None:
        if not isinstance(
            event,
            ObservabilityEvent,
        ):
            raise TypeError(
                "event must be an ObservabilityEvent"
            )

        with self._lock:
            self._events.append(event)

    def get_events(
        self,
    ) -> list[ObservabilityEvent]:
        with self._lock:
            return list(self._events)

    def get_events_by_type(
        self,
        event_type: EventType,
    ) -> list[ObservabilityEvent]:
        with self._lock:
            return [
                event
                for event in self._events
                if event.event_type is event_type
            ]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def len(self) -> int:
        with self._lock:
            return len(self._events)