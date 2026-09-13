from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.failure_classifier import (
    FailureClassification,
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
    failure_class: FailureClassification | None = None
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
        recovery_operation: (
            Callable[[AgentError], Awaitable[Any]] | None
        ) = None,
        change_detector: (
            Callable[[Any], bool] | None
        ) = None,
    ) -> ReliabilityResult:
        if not isinstance(error, AgentError):
            raise TypeError(
                "ReliabilityEngine expects AgentError."
            )

        normalized_error = self.error_handler.handle(error)

        classification = self.failure_classifier.classify(
            normalized_error
        )

        # ---------------------------------------------------------
        # 1. Retry has priority while retry budget remains.
        # ---------------------------------------------------------
        if attempt < max_attempts:
            retry_decision = self.retry_policy.evaluate(
                classification,
                current_attempt=attempt,
            )

            if retry_decision.should_retry:
                await self.retry.delay(
                    retry_decision.delay
                )

                return ReliabilityResult(
                    action=ReliabilityAction.RETRY,
                    error=normalized_error,
                    failure_class=classification,
                    attempt=attempt,
                    recovery_attempted=False,
                    reason="Retry policy selected retry.",
                )

        # ---------------------------------------------------------
        # 2. Ask recovery policy what should happen after
        #    retry budget is exhausted / retry is not appropriate.
        # ---------------------------------------------------------
        recovery_decision = self.recovery_policy.evaluate(
            classification
        )

        # ---------------------------------------------------------
        # 3. REPLAN is an orchestration decision.
        #
        #    IMPORTANT:
        #    Do NOT execute Recovery here.
        #
        #    The Orchestrator owns Planner.replan(), because it
        #    owns plan state, context, plan version, and execution
        #    lifecycle.
        # ---------------------------------------------------------
        if recovery_decision.action == RecoveryAction.REPLAN:
            return ReliabilityResult(
                action=ReliabilityAction.REPLAN,
                error=normalized_error,
                failure_class=classification,
                attempt=attempt,
                recovery_attempted=False,
                reason=(
                    "Recovery policy selected replanning. "
                    "Orchestrator must perform the replan."
                ),
            )

        # ---------------------------------------------------------
        # 4. Other recovery actions may use the generic Recovery
        #    mechanism when an operation was explicitly supplied.
        # ---------------------------------------------------------
        if recovery_operation is not None:
            recovery_result = await self._execute_recovery(
                error=normalized_error,
                classification=classification,
                attempt=attempt,
                recovery_operation=recovery_operation,
                change_detector=change_detector,
            )

            if recovery_result.recovered:
                return ReliabilityResult(
                    action=ReliabilityAction.RETRY,
                    error=normalized_error,
                    failure_class=classification,
                    attempt=attempt,
                    recovery_attempted=True,
                    recovery_result=recovery_result,
                    reason=(
                        "Operational recovery succeeded; "
                        "retrying execution."
                    ),
                )

            return ReliabilityResult(
                action=ReliabilityAction.STOP,
                error=normalized_error,
                failure_class=classification,
                attempt=attempt,
                recovery_attempted=True,
                recovery_result=recovery_result,
                reason=recovery_result.reason
                or "Operational recovery failed.",
            )

        # ---------------------------------------------------------
        # 5. No recovery path.
        # ---------------------------------------------------------
        return ReliabilityResult(
            action=ReliabilityAction.STOP,
            error=normalized_error,
            failure_class=classification,
            attempt=attempt,
            recovery_attempted=False,
            reason="No recovery action is available.",
        )

    async def _execute_recovery(
        self,
        *,
        error: AgentError,
        classification: FailureClassification,
        attempt: int,
        recovery_operation: Callable[
            [AgentError],
            Awaitable[Any],
        ],
        change_detector: Callable[[Any], bool] | None,
    ) -> RecoveryResult:
        try:
            recovery_result = await self.recovery.execute(
                error=error,
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

            return RecoveryResult(
                recovered=False,
                result=None,
                changed=False,
                reason="Recovery execution failed.",
                error=recovery_error,
            )

        return recovery_result