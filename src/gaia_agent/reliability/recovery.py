from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gaia_agent.reliability.errors import AgentError


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    recovered: bool
    result: Any | None = None
    error: AgentError | None = None
    reason: str = ""


class Recovery:

    async def execute(
        self,
        *,
        error: AgentError,
        operation: Callable[[AgentError], Awaitable[Any]],
    ) -> RecoveryResult:

        try:

            # Recovery callbacks are shaped like
            #   async def recover(error: AgentError) -> Any
            # so the AgentError must be forwarded.
            result = await operation(error)

            return RecoveryResult(
                recovered=True,
                result=result,
                reason="Recovery succeeded.",
            )

        except Exception as exc:

            # ---------------------------------------------------------
            # IMPORTANT: AgentError is a frozen dataclass, NOT an
            # Exception subclass. `except AgentError:` used to raise
            #   TypeError: catching classes that do not inherit from
            #   BaseException is not allowed
            # whenever a recovery callback failed, which silently
            # destroyed the real error and killed every replan.
            # ---------------------------------------------------------
            if isinstance(exc, AgentError):

                return RecoveryResult(
                    recovered=False,
                    error=exc,
                    reason=exc.message,
                )

            recovery_error = AgentError(
                error_type=type(exc).__name__,
                message=str(exc),
                source="recovery",
                operation="recovery",
                recoverable=False,
                original_exception=exc,
            )

            return RecoveryResult(
                recovered=False,
                error=recovery_error,
                reason=str(exc),
            )
