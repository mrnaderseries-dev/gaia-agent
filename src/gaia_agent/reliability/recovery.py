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
    changed: bool = False


class Recovery:
    async def execute(
        self,
        *,
        error: AgentError,
        operation: Callable[[AgentError], Awaitable[Any]],
        change_detector: Callable[[Any], bool] | None = None,
    ) -> RecoveryResult:
   

        try:
            result = await operation(error)

            changed = True

            if change_detector is not None:
                try:
                    changed = bool(change_detector(result))
                except Exception as exc:
                    recovery_error = AgentError(
                        error_type=type(exc).__name__,
                        message=str(exc),
                        source="recovery",
                        operation="change_detection",
                        recoverable=False,
                        original_exception=exc,
                    )

                    return RecoveryResult(
                        recovered=False,
                        error=recovery_error,
                        reason=(
                            "Recovery succeeded but change validation failed."
                        ),
                        changed=False,
                    )

            if not changed:
                return RecoveryResult(
                    recovered=False,
                    result=result,
                    reason=(
                        "Recovery produced no meaningful change; "
                        "replanning would repeat the same execution."
                    ),
                    changed=False,
                )

            return RecoveryResult(
                recovered=True,
                result=result,
                reason="Recovery succeeded with a meaningful change.",
                changed=True,
            )

        except Exception as exc:
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
                changed=False,
            )