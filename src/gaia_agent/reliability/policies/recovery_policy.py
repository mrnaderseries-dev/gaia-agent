from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gaia_agent.reliability.failure_classifier import FailureClass


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
        failure_class: FailureClass,
    ) -> RecoveryDecision:
        if failure_class == FailureClass.RECOVERABLE:
            if self.allow_replanning:
                return RecoveryDecision(
                    action=RecoveryAction.REPLAN,
                    reason="Failure is recoverable through replanning.",
                )

            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Replanning is disabled.",
            )

        if failure_class == FailureClass.TRANSIENT:
            return RecoveryDecision(
                action=RecoveryAction.NONE,
                reason="Transient failures are handled by RetryPolicy.",
            )

        if failure_class == FailureClass.PERMANENT:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Permanent failure cannot be recovered automatically.",
            )

        if failure_class == FailureClass.FATAL:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Fatal failure requires termination.",
            )

        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason="Failure classification is unknown.",
        )