from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.failure_classifier import (
    FailureClass,
    FailureClassifier,
)
from gaia_agent.reliability.policies.recovery_policy import (
    RecoveryAction,
    RecoveryPolicy,
)
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.recovery import Recovery, RecoveryResult
from gaia_agent.reliability.retry import Retry


class ReliabilityAction(str, Enum):
    RETRY = "retry"
    REPLAN = "replan"
    STOP = "stop"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ReliabilityResult:
    action: ReliabilityAction
    error: AgentError | None = None
    failure_class: FailureClass | None = None
    attempt: int = 0
    recovery_attempted: bool = False
    recovery_result: RecoveryResult | None = None
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

    async def handle_failure(
        self,
        *,
        error: AgentError,
        attempt: int,
        max_attempts: int,
        recovery_operation: Callable[[], Awaitable[Any]] | None = None,
        change_detector: Callable[[Any], bool] | None = None,
    ) -> ReliabilityResult:
        if not isinstance(error, AgentError):
            raise TypeError(
                "ReliabilityEngine.handle_failure() expects an AgentError."
            )

        if not isinstance(attempt, int) or isinstance(attempt, bool):
            raise TypeError("attempt must be an integer.")

        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
            raise TypeError("max_attempts must be an integer.")

        if attempt <= 0:
            raise ValueError("attempt must be greater than zero.")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero.")

        normalized_error = self.error_handler.handle(error)

        failure_class = self.failure_classifier.classify(
            normalized_error
        )

        if attempt < max_attempts:
            retry_decision = self.retry_policy.evaluate(
                failure_class,
                current_attempt=attempt,
            )

            if retry_decision.should_retry:
                await self.retry.delay(
                    retry_decision.delay
                )

                return ReliabilityResult(
                    action=ReliabilityAction.RETRY,
                    error=normalized_error,
                    failure_class=failure_class,
                    attempt=attempt,
                    reason=retry_decision.reason,
                )

        recovery_decision = self.recovery_policy.evaluate(
            failure_class
        )

        if recovery_decision.action == RecoveryAction.REPLAN:
            return await self._handle_replan(
                error=normalized_error,
                failure_class=failure_class,
                attempt=attempt,
                recovery_operation=recovery_operation,
                change_detector=change_detector,
            )

        return ReliabilityResult(
            action=ReliabilityAction.STOP,
            error=normalized_error,
            failure_class=failure_class,
            attempt=attempt,
            reason=recovery_decision.reason,
        )

    async def _handle_replan(
        self,
        *,
        error: AgentError,
        failure_class: FailureClass,
        attempt: int,
        recovery_operation: Callable[[], Awaitable[Any]] | None,
        change_detector: Callable[[Any], bool] | None,
    ) -> ReliabilityResult:
        if recovery_operation is None:
            return ReliabilityResult(
                action=ReliabilityAction.STOP,
                error=error,
                failure_class=failure_class,
                attempt=attempt,
                reason="No recovery operation was supplied.",
            )

        if change_detector is None:
            return ReliabilityResult(
                action=ReliabilityAction.STOP,
                error=error,
                failure_class=failure_class,
                attempt=attempt,
                reason="No change detector was supplied.",
            )

        try:
            recovery_result = await self.recovery.execute(
                operation=recovery_operation,
                change_detector=change_detector,
            )

        except Exception as exc:
            recovery_error = self.error_handler.handle(
                exc,
                source="reliability",
                operation="recovery",
                attempt=attempt,
            )

            return ReliabilityResult(
                action=ReliabilityAction.STOP,
                error=recovery_error,
                failure_class=FailureClass.UNKNOWN,
                attempt=attempt,
                recovery_attempted=True,
                reason=recovery_error.message,
            )

        if recovery_result.recovered:
            return ReliabilityResult(
                action=ReliabilityAction.REPLAN,
                error=error,
                failure_class=failure_class,
                attempt=attempt,
                recovery_attempted=True,
                recovery_result=recovery_result,
                reason=recovery_result.reason,
            )

        return ReliabilityResult(
            action=ReliabilityAction.STOP,
            error=error,
            failure_class=failure_class,
            attempt=attempt,
            recovery_attempted=True,
            recovery_result=recovery_result,
            reason=recovery_result.reason,
        )