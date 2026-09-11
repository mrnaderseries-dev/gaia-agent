from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    recovered: bool
    result: Any | None = None
    reason: str = ""
    changed: bool = False
    error: AgentError | None = None


class Recovery:
    def __init__(
        self,
        *,
        error_handler: ErrorHandler | None = None,
    ) -> None:
        self.error_handler = error_handler or ErrorHandler()

    async def execute(
        self,
        *,
        error: AgentError,
        operation: Callable[
            [AgentError],
            Awaitable[Any],
        ],
        change_detector: Callable[
            [Any],
            bool,
        ] | None = None,
    ) -> RecoveryResult:
        try:
            result = await operation(error)
        except Exception as exception:
            recovery_error = self.error_handler.handle(
                exception,
                source="recovery",
                operation="recovery",
                attempt=1,
            )

            return RecoveryResult(
                recovered=False,
                result=None,
                changed=False,
                reason="Recovery operation failed.",
                error=recovery_error,
            )

        if change_detector is None:
            return RecoveryResult(
                recovered=True,
                result=result,
                changed=True,
                reason="Recovery produced a meaningful change.",
                error=None,
            )

        try:
            changed = bool(
                change_detector(result)
            )
        except Exception as exception:
            validation_error = self.error_handler.handle(
                exception,
                source="recovery",
                operation="change_detection",
                attempt=1,
            )

            return RecoveryResult(
                recovered=False,
                result=result,
                changed=False,
                reason="Recovery change validation failed.",
                error=validation_error,
            )

        if not changed:
            return RecoveryResult(
                recovered=False,
                result=result,
                changed=False,
                reason="Recovery produced no meaningful change.",
                error=None,
            )

        return RecoveryResult(
            recovered=True,
            result=result,
            changed=True,
            reason="Recovery produced a meaningful change.",
            error=None,
        )