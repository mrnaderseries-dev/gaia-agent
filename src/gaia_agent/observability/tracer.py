from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator
from uuid import UUID, uuid4

_current_span: ContextVar[
    UUID | None
] = ContextVar(
    "gaia_current_span",
    default=None,
)


@dataclass
class Span:
    operation: str = ""

    span_id: UUID = field(
        default_factory=uuid4
    )

    correlation_id: UUID | None = None

    parent_span_id: UUID | None = None

    start_time: float = field(
        default_factory=perf_counter
    )

    end_time: float | None = None

    duration: float | None = None

    error: str | None = None

    status: str = "started"

    attributes: dict[str, Any] = field(
        default_factory=dict
    )


class Tracer:
    def __init__(self) -> None:
        self._spans: list[Span] = []

    def start_span(
        self,
        operation: str,
        correlation_id: UUID | None = None,
        *,
        parent_span_id: UUID | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        if not operation:
            raise ValueError(
                "operation must not be empty"
            )

        if parent_span_id is None:
            parent_span_id = _current_span.get()

        span = Span(
            operation=operation,
            correlation_id=correlation_id,
            parent_span_id=parent_span_id,
            attributes=dict(attributes or {}),
        )

        self._spans.append(span)

        return span

    def end_span(
        self,
        span: Span,
        error: str | Exception | Any | None = None,
    ) -> None:
        if span.end_time is not None:
            return

        span.end_time = perf_counter()

        span.duration = (
            span.end_time
            - span.start_time
        )

        if error is not None:
            span.error = str(error)
            span.status = "error"
        else:
            span.status = "ok"

    @contextmanager
    def span(
        self,
        operation: str,
        correlation_id: UUID | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        parent_span_id = _current_span.get()

        span = self.start_span(
            operation,
            correlation_id,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )

        token = _current_span.set(
            span.span_id
        )

        try:
            yield span

        except Exception as exc:
            self.end_span(
                span,
                error=exc,
            )
            raise

        else:
            self.end_span(span)

        finally:
            _current_span.reset(token)

    def get_spans(self) -> list[Span]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()

    @staticmethod
    def current_span_id() -> UUID | None:
        return _current_span.get()