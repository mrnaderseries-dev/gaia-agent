from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.failure_classifier import (
    FailureClass,
    FailureClassifier,
)
from gaia_agent.reliability.retry import Retry
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.policies.recovery_policy import (
    RecoveryAction,
    RecoveryPolicy,
)

from gaia_agent.reliability.policies.retry_policy import RetryPolicy

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ReliabilityResult(Generic[T]):
    success: bool
    result: T | None = None
    error: AgentError | None = None
    attempts: int = 0
    recovery_attempted: bool = False
    reason: str = ""


class ReliabilityEngine:
    def __init__(
        self,
        *,
        error_handler: ErrorHandler,
        failure_classifier: FailureClassifier,
        retry_policy: RetryPolicy,
        recovery_policy: RecoveryPolicy,
        retry: Retry,
        recovery: Recovery,
    ) -> None:
        self.error_handler = error_handler
        self.failure_classifier = failure_classifier
        self.retry_policy = retry_policy
        self.recovery_policy = recovery_policy
        self.retry = retry
        self.recovery = recovery

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        operation_name: str,
        source: str,
        validator: Callable[[T], bool] | None = None,
        recovery_operation: Callable[[AgentError], Awaitable[Any]] | None = None,
        max_attempts: int = 3,
    ) -> ReliabilityResult[T]:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
            raise TypeError("max_attempts must be an integer.")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero.")

        last_error: AgentError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = await operation()

                if validator is not None and not validator(result):
                    raise ValueError(
                        f"{operation_name} returned an invalid result."
                    )

                return ReliabilityResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    recovery_attempted=False,
                    reason="Operation succeeded.",
                )

            except Exception as exc:
                error = self.error_handler.handle(
                    exc,
                    source=source,
                    operation=operation_name,
                    attempt=attempt,
                )

                last_error = error

                failure_class = self.failure_classifier.classify(error)

                if attempt < max_attempts:
                    retry_decision = self.retry_policy.evaluate(
                        failure_class,
                        current_attempt=attempt,
                    )

                    if retry_decision.should_retry:
                        await self.retry.delay(retry_decision.delay)
                        continue

                recovery_decision = self.recovery_policy.evaluate(
                    failure_class
                )

                if (
                    recovery_decision.action == RecoveryAction.REPLAN
                    and recovery_operation is not None
                ):
                    return await self._execute_recovery(
                        error=error,
                        attempts=attempt,
                        recovery_operation=recovery_operation,
                    )

                return ReliabilityResult(
                    success=False,
                    error=last_error,
                    attempts=attempt,
                    recovery_attempted=False,
                    reason=(
                        recovery_decision.reason
                        or last_error.message
                    ),
                )

        return ReliabilityResult(
            success=False,
            error=last_error,
            attempts=max_attempts,
            recovery_attempted=False,
            reason=(
                last_error.message
                if last_error is not None
                else "Operation failed."
            ),
        )

    async def _execute_recovery(
        self,
        *,
        error: AgentError,
        attempts: int,
        recovery_operation: Callable[[AgentError], Awaitable[Any]],
    ) -> ReliabilityResult[T]:
        recovery_result = await self.recovery.execute(
            error=error,
            operation=recovery_operation,
        )

        if not recovery_result.recovered:
            recovery_error = recovery_result.error or error

            return ReliabilityResult(
                success=False,
                error=recovery_error,
                attempts=attempts,
                recovery_attempted=True,
                reason=(
                    recovery_result.reason
                    or recovery_error.message
                ),
            )

        return ReliabilityResult(
            success=False,
            result=None,
            error=error,
            attempts=attempts,
            recovery_attempted=True,
            reason=(
                recovery_result.reason
                or "Recovery completed; task execution must continue."
            ),
        )