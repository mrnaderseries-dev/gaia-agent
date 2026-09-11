from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
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


class FailureClassifier:
    def classify(
        self,
        error: AgentError,
    ) -> FailureClassification:
        if not isinstance(error, AgentError):
            return FailureClassification(
                FailureType.UNKNOWN
            )

        category = error.category

        if category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
        }:
            return FailureClassification(
                FailureType.TRANSIENT
            )

        if category in {
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.PLAN_SEMANTIC_ERROR,
            ErrorCategory.PLAN_SCHEMA_ERROR,
            ErrorCategory.PLAN_STRATEGY_ERROR,
            ErrorCategory.PLAN_GENERATION_ERROR,
            ErrorCategory.PLAN_RECOVERY_ERROR,
            ErrorCategory.TOOL_ARGUMENT_ERROR,
            ErrorCategory.TOOL_CONTRACT_ERROR,
            ErrorCategory.TOOL_NOT_FOUND,
            ErrorCategory.PYTHON_SYNTAX_ERROR,
            ErrorCategory.PYTHON_IMPORT_ERROR,
            ErrorCategory.FILE_NOT_FOUND,
            ErrorCategory.ARTIFACT_NOT_FOUND,
            ErrorCategory.EMPTY_RESULT,
            ErrorCategory.INVALID_RESULT,
            ErrorCategory.LLM_OUTPUT_ERROR,
            ErrorCategory.VALIDATION,
        }:
            return FailureClassification(
                FailureType.RECOVERABLE
            )

        if category in {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.APPROVAL_BLOCKED,
            ErrorCategory.TRANSITION_FAILURE,
        }:
            return FailureClassification(
                FailureType.PERMANENT
            )

        if category in {
            ErrorCategory.LOOP_DETECTED,
            ErrorCategory.INTERNAL,
        }:
            return FailureClassification(
                FailureType.FATAL
            )

        if category in {
            ErrorCategory.TOOL_EXECUTION_ERROR,
            ErrorCategory.LLM_FAILURE,
            ErrorCategory.EXECUTION,
        }:
            if error.retryable:
                return FailureClassification(
                    FailureType.TRANSIENT
                )

            if error.recoverable:
                return FailureClassification(
                    FailureType.RECOVERABLE
                )

            return FailureClassification(
                FailureType.PERMANENT
            )

        if error.retryable:
            return FailureClassification(
                FailureType.TRANSIENT
            )

        if error.recoverable:
            return FailureClassification(
                FailureType.RECOVERABLE
            )

        if error.severity is ErrorSeverity.CRITICAL:
            return FailureClassification(
                FailureType.FATAL
            )

        if error.severity is ErrorSeverity.HIGH:
            return FailureClassification(
                FailureType.PERMANENT
            )

        return FailureClassification(
            FailureType.UNKNOWN
        )