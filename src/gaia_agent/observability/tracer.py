from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from uuid import UUID, uuid4

@dataclass
class Span:
    operation: str = ""  
    span_id: UUID = field(default_factory=uuid4)
    correlation_id: UUID | None = None
    start_time: float = field(default_factory=perf_counter)
    duration: float | None = None
    error: str | None = None


class Tracer:
    def __init__(self) -> None:
        self._spans: list[Span] = []

    def start_span(self, operation: str, correlation_id: UUID | None = None) -> Span:
        span = Span(operation=operation, correlation_id=correlation_id)
        self._spans.append(span)
        return span

    def end_span(self, span: Span, error: str | None = None) -> None:
        span.duration = perf_counter() - span.start_time  # Added () to perf_counter
        span.error = error

    def get_spans(self) -> list[Span]:
        return list(self._spans)
