from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.policies.recovery_policy import (
    RecoveryAction,
    RecoveryPolicy,
)
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.retry import Retry

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class ReliabilityResult(Generic[T]):
    success: bool
    result: T | None = None
    error: AgentError | None = None
    attempts: int = 0
    recovery_attempted: bool = False
    recovery_count: int = 0
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
        max_recoveries: int = 2,
        max_total_executions: int | None = None,
    ) -> None:
        if max_recoveries < 0:
            raise ValueError(
                "max_recoveries cannot be negative."
            )

        self.error_handler = error_handler
        self.failure_classifier = failure_classifier
        self.retry_policy = retry_policy
        self.recovery_policy = recovery_policy
        self.retry = retry
        self.recovery = recovery
        self.max_recoveries = max_recoveries

        self.max_total_executions = (
            max_total_executions
            if max_total_executions is not None
            else retry_policy.max_attempts + max_recoveries
        )

        if self.max_total_executions <= 0:
            raise ValueError(
                "max_total_executions must be greater than 0."
            )

    async def _execute_recovery(
        self,
        *,
        error: AgentError | None,
        attempts: int,
        recovery_count: int,
        recovery_operation: Callable[[AgentError], Awaitable[Any]] | None,
        recovery_change_detector: Callable[[Any], bool] | None,
    ) -> tuple[bool, ReliabilityResult[T] | None]:
        """Helper to handle recovery execution and validation.
        Returns (should_continue_loop, error_or_success_result).
        """
        if recovery_operation is None:
            return False, ReliabilityResult(
                success=False,
                error=error,
                attempts=attempts,
                recovery_attempted=recovery_count > 0,
                recovery_count=recovery_count,
                reason="Retry budget exhausted and no recovery operation is available.",
            )

        if error is None:
            return False, ReliabilityResult(
                success=False,
                attempts=attempts,
                recovery_attempted=False,
                recovery_count=recovery_count,
                reason="Recovery requested without a failure.",
            )

        recovery_count += 1

        recovery_result = await self.recovery.execute(
            error=error,
            operation=recovery_operation,
            change_detector=recovery_change_detector,
        )

        if not recovery_result.recovered:
            return False, ReliabilityResult(
                success=False,
                error=recovery_result.error or error,
                attempts=attempts,
                recovery_attempted=True,
                recovery_count=recovery_count,
                reason=recovery_result.reason or "Recovery failed.",
            )

        if recovery_change_detector is None:
            return False, ReliabilityResult(
                success=False,
                error=error,
                attempts=attempts,
                recovery_attempted=True,
                recovery_count=recovery_count,
                reason=(
                    "Recovery succeeded but no change detector was provided. "
                    "Replanning cannot be verified safely."
                ),
            )

        return True, None

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
        recovery_change_detector: (
            Callable[[Any], bool] | None
        ) = None,
        max_attempts: int | None = None,
        max_recoveries: int | None = None,
    ) -> ReliabilityResult[T]:

        retry_budget = (
            max_attempts
            if max_attempts is not None
            else self.retry_policy.max_attempts
        )

        recovery_budget = (
            max_recoveries
            if max_recoveries is not None
            else self.max_recoveries
        )

        if not isinstance(retry_budget, int) or isinstance(
            retry_budget, bool
        ):
            raise TypeError(
                "max_attempts must be an integer."
            )

        if retry_budget <= 0:
            raise ValueError(
                "max_attempts must be greater than 0."
            )

        if not isinstance(recovery_budget, int) or isinstance(
            recovery_budget, bool
        ):
            raise TypeError(
                "max_recoveries must be an integer."
            )

        if recovery_budget < 0:
            raise ValueError(
                "max_recoveries cannot be negative."
            )

        attempts = 0
        attempts_since_recovery = 0
        recovery_count = 0

        last_error: AgentError | None = None

        while attempts < self.max_total_executions:

            if attempts_since_recovery >= retry_budget:
                if recovery_count >= recovery_budget:
                    return ReliabilityResult(
                        success=False,
                        error=last_error,
                        attempts=attempts,
                        recovery_attempted=recovery_count > 0,
                        recovery_count=recovery_count,
                        reason=(
                            "Retry and recovery budgets exhausted."
                        ),
                    )

                classification = (
                    self.failure_classifier.classify(
                        last_error
                    )
                    if last_error is not None
                    else None
                )

                recovery_decision = (
                    self.recovery_policy.evaluate(
                        classification
                    )
                    if classification is not None
                    else None
                )

                if recovery_decision and recovery_decision.action is not RecoveryAction.REPLAN:
                    return ReliabilityResult(
                        success=False,
                        error=last_error,
                        attempts=attempts,
                        recovery_attempted=recovery_count > 0,
                        recovery_count=recovery_count,
                        reason=recovery_decision.reason,
                    )

                if recovery_count >= recovery_budget:
                    return ReliabilityResult(
                        success=False,
                        error=last_error,
                        attempts=attempts,
                        recovery_attempted=recovery_count > 0,
                        recovery_count=recovery_count,
                        reason="Recovery budget exhausted.",
                    )

                success_flag, res = await self._execute_recovery(
                    error=last_error,
                    attempts=attempts,
                    recovery_count=recovery_count,
                    recovery_operation=recovery_operation,
                    recovery_change_detector=recovery_change_detector,
                )
                if not success_flag:
                    return res  

                recovery_count += 1 
                attempts_since_recovery = 0
                last_error = None
                continue

            if attempts >= self.max_total_executions:
                break

            attempts += 1
            attempts_since_recovery += 1

            try:
                result = await operation()

                if validator is not None:
                    try:
                        valid = validator(result)
                    except Exception as exc:
                        raise ValueError(
                            f"{operation_name} validator failed: {exc}"
                        ) from exc

                    if not valid:
                        raise ValueError(
                            f"{operation_name} returned "
                            "an invalid result."
                        )

                return ReliabilityResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    recovery_attempted=recovery_count > 0,
                    recovery_count=recovery_count,
                    reason="Operation succeeded.",
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
                    self.failure_classifier.classify(error)
                )

                if (
                    attempts_since_recovery < retry_budget
                    and attempts < self.max_total_executions
                ):
                    retry_decision = (
                        self.retry_policy.evaluate(
                            classification,
                            current_attempt=attempts_since_recovery,
                        )
                    )

                    if retry_decision.should_retry:
                        await self.retry.delay(
                            retry_decision.delay
                        )
                        continue

                recovery_decision = (
                    self.recovery_policy.evaluate(
                        classification
                    )
                )

                if (
                    recovery_decision.action
                    is RecoveryAction.REPLAN
                    and recovery_operation is not None
                    and recovery_count < recovery_budget
                ):
                    success_flag, res = await self._execute_recovery(
                        error=error,
                        attempts=attempts,
                        recovery_count=recovery_count,
                        recovery_operation=recovery_operation,
                        recovery_change_detector=recovery_change_detector,
                    )
                    if not success_flag:
                        return res  # type: ignore

                    recovery_count += 1
                    attempts_since_recovery = 0
                    last_error = None
                    continue

                return ReliabilityResult(
                    success=False,
                    error=last_error,
                    attempts=attempts,
                    recovery_attempted=recovery_count > 0,
                    recovery_count=recovery_count,
                    reason=(
                        recovery_decision.reason
                        if recovery_decision.action
                        is RecoveryAction.STOP
                        else (
                            last_error.message
                            if last_error is not None
                            else "Operation failed."
                        )
                    ),
                )

        return ReliabilityResult(
            success=False,
            error=last_error,
            attempts=attempts,
            recovery_attempted=recovery_count > 0,
            recovery_count=recovery_count,
            reason=(
                "Global reliability execution budget exhausted."
            ),
        )