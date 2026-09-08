from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any
from uuid import UUID


class ErrorCategory(str, Enum):
    UNKNOWN = "unknown"
    INTERNAL = "internal"
    VALIDATION = "validation"
    EXECUTION = "execution"

    PLANNER = "planner"
    PLANNER_CONFIGURATION = "planner_configuration"
    PLAN_GENERATION_ERROR = "plan_generation_error"
    PLAN_SCHEMA_ERROR = "plan_schema_error"
    PLAN_SEMANTIC_ERROR = "plan_semantic_error"
    PLAN_STRATEGY_ERROR = "plan_strategy_error"
    PLAN_RECOVERY_ERROR = "plan_recovery_error"

    TOOL = "tool"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_CONTRACT_ERROR = "tool_contract_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"

    LLM = "llm"
    LLM_FAILURE = "llm_failure"
    LLM_OUTPUT_ERROR = "llm_output_error"

    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"

    FILE_NOT_FOUND = "file_not_found"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    EMPTY_RESULT = "empty_result"
    INVALID_RESULT = "invalid_result"

    PYTHON_SYNTAX_ERROR = "python_syntax_error"
    PYTHON_IMPORT_ERROR = "python_import_error"

    APPROVAL_BLOCKED = "approval_blocked"
    TRANSITION_FAILURE = "transition_failure"
    LOOP_DETECTED = "loop_detected"


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AgentError(Exception):
    error_type: str
    message: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retryable: bool = False
    recoverable: bool = False
    source: str | None = None
    operation: str | None = None
    attempt: int = 0
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    original_exception: Exception | None = None
    correlation_id: UUID | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def with_detail(self, key: str, value: Any) -> AgentError:
        return replace(
            self,
            details={
                **self.details,
                key: value,
            },
        )

    def with_details(self, **details: Any) -> AgentError:
        return replace(
            self,
            details={
                **self.details,
                **details,
            },
        )

    def with_attempt(self, attempt: int) -> AgentError:
        return replace(
            self,
            attempt=attempt,
        )

    def with_correlation_id(self, correlation_id: UUID) -> AgentError:
        return replace(
            self,
            correlation_id=correlation_id,
        )

    def __str__(self) -> str:
        return self.message