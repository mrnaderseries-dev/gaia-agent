from __future__ import annotations

import logging

from .events import ObservabilityEvent


class EventLogger:

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

        self._logger.log(
            logging.INFO,
            self._format_event(event),
        )

    def _format_event(
        self,
        event: ObservabilityEvent,
    ) -> str:

        return (
            f"event={event.event_type.value} "
            f"event_id={event.event_id} "
            f"correlation_id={event.correlation_id} "
            f"timestamp={event.timestamp.isoformat()} "
            f"agent_id={event.agent_id} "
            f"iteration={event.iteration} "
            f"latency={event.latency} "
            f"error={event.error} "
            f"metadata={event.metadata}"
        )