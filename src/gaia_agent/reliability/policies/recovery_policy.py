from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gaia_agent.reliability.failure_classifier import (
    FailureClassification,
    FailureType,
)


class RecoveryAction(str, Enum):
    NONE = "none"
    REPLAN = "replan"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str = ""


class RecoveryPolicy:
    def __init__(
        self,
        *,
        allow_replanning: bool = True,
    ) -> None:
        self.allow_replanning = allow_replanning

    def evaluate(
        self,
        classification: FailureClassification,
    ) -> RecoveryDecision:
        if classification.failure_type is FailureType.RECOVERABLE:
            if self.allow_replanning:
                return RecoveryDecision(
                    action=RecoveryAction.REPLAN,
                    reason="Failure is recoverable through replanning.",
                )

            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Replanning is disabled.",
            )

        if classification.failure_type is FailureType.TRANSIENT:
            return RecoveryDecision(
                action=RecoveryAction.NONE,
                reason="Transient failures are handled by RetryPolicy.",
            )

        if classification.failure_type is FailureType.PERMANENT:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Permanent failure cannot be recovered automatically.",
            )

        if classification.failure_type is FailureType.FATAL:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Fatal failure requires termination.",
            )

        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason="Failure classification is unknown.",
        )