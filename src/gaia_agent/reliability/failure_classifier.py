from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorSeverity,
)


class FailureType(str, Enum):
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    PERMANENT = "permanent"
    FATAL = "fatal"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    failure_type: FailureType
    reason: str


class FailureClassifier:

    def classify(
        self,
        error: AgentError,
    ) -> FailureClassification:

        if error.severity is ErrorSeverity.CRITICAL:
            return FailureClassification(
                failure_type=FailureType.FATAL,
                reason="Critical runtime failure.",
            )

        if error.retryable:
            return FailureClassification(
                failure_type=FailureType.TRANSIENT,
                reason="Failure is marked as retryable.",
            )

        if error.recoverable:
            return FailureClassification(
                failure_type=FailureType.RECOVERABLE,
                reason="Failure may be handled by recovery.",
            )

        if error.severity is ErrorSeverity.HIGH:
            return FailureClassification(
                failure_type=FailureType.PERMANENT,
                reason=(
                    "High-severity failure is neither "
                    "retryable nor recoverable."
                ),
            )

        return FailureClassification(
            failure_type=FailureType.UNKNOWN,
            reason="Failure could not be classified.",
        )