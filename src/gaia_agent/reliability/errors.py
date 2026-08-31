from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID


class ErrorCategory(str, Enum):
    TOOL = "tool"
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    LLM = "llm"
    LLM_FAILURE = "llm_failure"
    NETWORK = "network"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    EXECUTION = "execution"
    INTERNAL = "internal"
    PYTHON_SYNTAX_ERROR = "python_syntax_error"
    PYTHON_IMPORT_ERROR = "python_import_error"
    FILE_NOT_FOUND = "file_not_found"
    EMPTY_RESULT = "empty_result"
    INVALID_RESULT = "invalid_result"
    APPROVAL_BLOCKED = "approval_blocked"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AgentError:
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

    details: dict[str, Any] = field(
        default_factory=dict
    )

    original_exception: Exception | None = None

    correlation_id: UUID | None = None

    def with_detail(
        self,
        key: str,
        value: Any,
    ) -> "AgentError":

        updated_details = dict(
            self.details
        )

        updated_details[key] = value

        return AgentError(
            error_type=self.error_type,
            message=self.message,
            category=self.category,
            severity=self.severity,
            retryable=self.retryable,
            recoverable=self.recoverable,
            source=self.source,
            operation=self.operation,
            attempt=self.attempt,
            error_code=self.error_code,
            details=updated_details,
            original_exception=self.original_exception,
            correlation_id=self.correlation_id,
        )
    