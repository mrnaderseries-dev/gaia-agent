from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.policies.recovery_policy import (
    RecoveryAction,
    RecoveryPolicy,
)
from gaia_agent.reliability.retry import Retry
from gaia_agent.reliability.recovery import Recovery


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
        recovery_operation: (
            Callable[[AgentError], Awaitable[Any]] | None
        ) = None,
        max_attempts: int | None = None,
    ) -> ReliabilityResult[T]:

        attempts = 0
        last_error: AgentError | None = None

        # Bounded retry budget. Without an explicit budget a
        # transient failure can spin the operation forever
        # (the "Infinite loop signature detected" failure mode).
        retry_budget = (
            max_attempts
            if max_attempts is not None
            else self.retry_policy.max_attempts
        )

        loop_guard = 0

        while True:

            # Hard safety valve against unbounded spinning loops.
            loop_guard += 1

            if loop_guard > retry_budget + 2:
                return ReliabilityResult(
                    success=False,
                    result=None,
                    error=last_error,
                    attempts=attempts,
                    recovery_attempted=False,
                    reason="Execution exceeded its retry budget.",
                )

            attempts += 1

            try:

                result = await operation()

                # --------------------------------------------------
                # Validation
                # --------------------------------------------------

                if validator is not None:

                    valid = validator(result)

                    if not valid:

                        raise ValueError(
                            f"{operation_name} "
                            "returned an invalid result."
                        )

                return ReliabilityResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                )

            except Exception as exc:

                error = self.error_handler.handle(
                    exc,
                    source=source,
                    operation=operation_name,
                    attempt=attempts,
                )

                last_error = error

                classification = (
                    self.failure_classifier.classify(
                        error
                    )
                )

                # --------------------------------------------------
                # Retry (bounded; single execution per attempt)
                # --------------------------------------------------

                if attempts < retry_budget:

                    retry_decision = (
                        self.retry_policy.evaluate(
                            classification,
                            current_attempt=attempts,
                        )
                    )

                    if retry_decision.should_retry:

                        await self.retry.delay(
                            retry_decision.delay
                        )

                        # Re-run the operation through the outer
                        # loop, which validates and counts it.
                        continue

                # --------------------------------------------------
                # Recovery / Replan
                # --------------------------------------------------

                if (
                    recovery_operation is not None
                    and self._should_recover(
                        classification
                    )
                ):

                    recovery_result = (
                        await self.recovery.execute(
                            error=error,
                            operation=recovery_operation,
                        )
                    )

                    if recovery_result.recovered:

                        return ReliabilityResult(
                            success=True,
                            result=recovery_result.result,
                            attempts=attempts,
                            recovery_attempted=True,
                            reason=(
                                recovery_result.reason
                                or "Recovery succeeded."
                            ),
                        )

                    if recovery_result.error is not None:

                        last_error = (
                            recovery_result.error
                        )

                    return ReliabilityResult(
                        success=False,
                        result=None,
                        error=last_error,
                        attempts=attempts,
                        recovery_attempted=True,
                        reason=(
                            recovery_result.reason
                            or (
                                last_error.message
                                if last_error
                                else "Recovery failed."
                            )
                        ),
                    )

                # --------------------------------------------------
                # Final failure
                # --------------------------------------------------

                return ReliabilityResult(
                    success=False,
                    result=None,
                    error=last_error,
                    attempts=attempts,
                    recovery_attempted=False,
                    reason=(
                        last_error.message
                        if last_error
                        else "Operation failed."
                    ),
                )

    def _should_recover(
        self,
        classification,
    ) -> bool:

        decision = (
            self.recovery_policy.evaluate(
                classification
            )
        )

        return (
            decision.action
            == RecoveryAction.REPLAN
        )