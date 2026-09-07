from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)


class FailureType(str, Enum):
    """
    Reliability-level classification.

    This answers:
        "What should the reliability layer do with this failure?"

    It does NOT describe what actually went wrong.
    That information belongs to AgentError.category.
    """

    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    PERMANENT = "permanent"
    FATAL = "fatal"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    """
    Result of classifying an AgentError.

    failure_type:
        Reliability behavior category.

    reason:
        Human-readable explanation for logs/observability.
    """

    failure_type: FailureType
    reason: str


class FailureClassifier:
    """
    Converts the canonical AgentError into a reliability-level
    FailureClassification.

    Responsibility:
        AgentError -> FailureType

    Not responsible for:
        - catching exceptions
        - creating AgentError
        - retry decisions
        - recovery decisions
        - executing recovery
        - inspecting legacy exception classes
    """

    def classify(
        self,
        error: AgentError,
    ) -> FailureClassification:
        if error.category is ErrorCategory.STATE_TRANSITION_ERROR:
            return FailureClassification(
                failure_type=FailureType.FATAL,
                reason=(
                    "State transition failure indicates a "
                    "control-plane contract violation."
                ),
            )
        if error.severity is ErrorSeverity.CRITICAL:
            return FailureClassification(
                failure_type=FailureType.FATAL,
                reason=(
                    "Critical failure cannot be safely retried "
                    "or recovered."
                ),
            )
        if error.retryable:
            return FailureClassification(
                failure_type=FailureType.TRANSIENT,
                reason=(
                    "Failure is explicitly marked as retryable."
                ),
            )
        if error.recoverable:
            return FailureClassification(
                failure_type=FailureType.RECOVERABLE,
                reason=(
                    "Failure is explicitly marked as recoverable "
                    "through recovery or replanning."
                ),
            )
        if error.severity is ErrorSeverity.HIGH:
            return FailureClassification(
                failure_type=FailureType.PERMANENT,
                reason=(
                    "High-severity failure is neither retryable "
                    "nor recoverable."
                ),
            )
        return FailureClassification(
            failure_type=FailureType.UNKNOWN,
            reason=(
                "Failure does not contain enough reliability "
                "information to classify it safely."
            ),
        )