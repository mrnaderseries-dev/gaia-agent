from __future__ import annotations

from enum import Enum

from gaia_agent.reliability.errors import AgentError, ErrorCategory


class FailureClass(str, Enum):
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    PERMANENT = "permanent"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class FailureClassifier:
    def classify(self, error: AgentError) -> FailureClass:
        if not isinstance(error, AgentError):
            return FailureClass.UNKNOWN

        category = error.category

        if category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
        }:
            return FailureClass.TRANSIENT

        if category in {
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
            return FailureClass.RECOVERABLE

        if category in {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.APPROVAL_BLOCKED,
            ErrorCategory.TRANSITION_FAILURE,
        }:
            return FailureClass.PERMANENT

        if category in {
            ErrorCategory.LOOP_DETECTED,
            ErrorCategory.INTERNAL,
        }:
            return FailureClass.FATAL

        if category in {
            ErrorCategory.TOOL_EXECUTION_ERROR,
            ErrorCategory.LLM_FAILURE,
            ErrorCategory.EXECUTION,
        }:
            if error.retryable:
                return FailureClass.TRANSIENT

            if error.recoverable:
                return FailureClass.RECOVERABLE

            return FailureClass.PERMANENT

        if category == ErrorCategory.UNKNOWN:
            return FailureClass.UNKNOWN

        if error.retryable:
            return FailureClass.TRANSIENT

        if error.recoverable:
            return FailureClass.RECOVERABLE

        return FailureClass.UNKNOWN