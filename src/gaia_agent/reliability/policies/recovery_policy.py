from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gaia_agent.reliability.failure_classifier import (
    FailureClassification,
    FailureType,
)


class RecoveryAction(str, Enum):
    REPLAN = "replan"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str


class RecoveryPolicy:

    def __init__(
        self,
        *,
        allow_replan: bool = True,
    ) -> None:

        self.allow_replan = allow_replan

    def evaluate(
        self,
        classification: FailureClassification,
    ) -> RecoveryDecision:

        if classification.failure_type == FailureType.FATAL:

            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=(
                    "Fatal failure requires "
                    "execution to stop."
                ),
            )

        if classification.failure_type == FailureType.PERMANENT:

            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=(
                    "Permanent failure cannot be "
                    "recovered automatically."
                ),
            )

        if classification.failure_type == FailureType.TRANSIENT:

            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=(
                    "Transient failure should be "
                    "handled by RetryPolicy."
                ),
            )

        if classification.failure_type == FailureType.RECOVERABLE:

            if self.allow_replan:

                return RecoveryDecision(
                    action=RecoveryAction.REPLAN,
                    reason=(
                        "Failure may be recovered "
                        "through replanning."
                    ),
                )

            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="Replanning is disabled.",
            )

        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason=(
                "Unknown failure cannot be "
                "recovered safely."
            ),
        )