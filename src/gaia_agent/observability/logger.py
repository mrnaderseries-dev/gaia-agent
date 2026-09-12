from __future__ import annotations

import logging

from .events import ObservabilityEvent


class LoggingEventSink:
    def __init__(
        self,
        name: str = "gaia.observability",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

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

        self._logger.log(
            logging.INFO,
            self._format_event(event),
        )

    @staticmethod
    def _format_event(
        event: ObservabilityEvent,
    ) -> str:
        return (
            f"event={event.event_type.value} "
            f"event_id={event.event_id} "
            f"correlation_id={event.correlation_id} "
            f"run_id={event.run_id} "
            f"agent_id={event.agent_id} "
            f"iteration={event.iteration} "
            f"step_id={event.step_id} "
            f"attempt={event.attempt} "
            f"phase={event.phase} "
            f"latency={event.latency} "
            f"error={event.error} "
            f"metadata={event.metadata}"
        )