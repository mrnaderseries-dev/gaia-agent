from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import UUID

from .events import (
    EventLogger,
    EventType,
    ObservabilityEvent,
    create_event,
)
from .metrics import Metrics
from .tracer import Span, Tracer


class ObservabilityFacade:
    def __init__(
        self,
        *,
        event_logger: EventLogger,
        tracer: Tracer,
        metrics: Metrics,
    ) -> None:
        self._event_logger = event_logger
        self._tracer = tracer
        self._metrics = metrics

    def emit(
        self,
        event_type: EventType,
        *,
        correlation_id: UUID,
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
        event = create_event(
            event_type,
            correlation_id,
            metadata=metadata,
            agent_id=agent_id,
            run_id=run_id,
            iteration=iteration,
            step_id=step_id,
            attempt=attempt,
            phase=phase,
            latency=latency,
            error=error,
        )

        self._event_logger.log(event)

        return event

    def increment(
        self,
        name: str,
        value: int = 1,
    ) -> None:
        self._metrics.increment(
            name,
            value,
        )

    def record_duration(
        self,
        name: str,
        duration: float,
    ) -> None:
        self._metrics.record_duration(
            name,
            duration,
        )

    def start_span(
        self,
        operation: str,
        correlation_id: UUID | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        return self._tracer.start_span(
            operation,
            correlation_id,
            attributes=attributes,
        )

    def end_span(
        self,
        span: Span,
        error: str | Exception | Any | None = None,
    ) -> None:
        self._tracer.end_span(
            span,
            error=error,
        )

    @contextmanager
    def span(
        self,
        operation: str,
        correlation_id: UUID | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        with self._tracer.span(
            operation,
            correlation_id,
            attributes=attributes,
        ) as span:
            yield span