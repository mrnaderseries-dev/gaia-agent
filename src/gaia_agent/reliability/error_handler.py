from __future__ import annotations

from typing import Any

from .errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)
from .exception import (
    ApprovalBlockedError,
    AuthenticationError,
    AuthorizationError,
    EmptyResultError,
    InvalidResultError,
    LLMFailure,
    LLMOutputError,
    NetworkError,
    PythonImportError,
    PythonSyntaxError,
    RateLimitError,
    ToolArgumentError,
    ToolExecutionError,
    ValidationError,
)


class ErrorHandler:
    def handle(
        self,
        exception: Exception,
        *,
        source: str = "unknown",
        operation: str = "unknown",
        attempt: int = 0,
        correlation_id: Any = None,
    ) -> AgentError:
        if isinstance(exception, AgentError):
            return self._enrich(
                exception,
                source=source,
                operation=operation,
                attempt=attempt,
                correlation_id=correlation_id,
            )

        category = self._category(exception)
        severity = self._severity(category)

        retryable = bool(
            getattr(exception, "retryable", False)
        )

        recoverable = bool(
            getattr(exception, "recoverable", False)
        )

        if category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
        }:
            retryable = True

        if category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.FILE_NOT_FOUND,
            ErrorCategory.TOOL_ARGUMENT_ERROR,
            ErrorCategory.TOOL_EXECUTION_ERROR,
            ErrorCategory.LLM_OUTPUT_ERROR,
        }:
            recoverable = True

        return AgentError(
            error_type=type(exception).__name__,
            message=str(exception) or type(exception).__name__,
            category=category,
            severity=severity,
            retryable=retryable,
            recoverable=recoverable,
            source=source,
            operation=operation,
            attempt=attempt,
            original_exception=exception,
            correlation_id=correlation_id,
        )

    def _enrich(
        self,
        error: AgentError,
        *,
        source: str,
        operation: str,
        attempt: int,
        correlation_id: Any,
    ) -> AgentError:
        if error.source is None or error.source == "unknown":
            error.source = source

        if error.operation is None or error.operation == "unknown":
            error.operation = operation

        if error.attempt == 0:
            error.attempt = attempt

        if error.correlation_id is None:
            error.correlation_id = correlation_id

        return error

    def _category(
        self,
        exception: Exception,
    ) -> ErrorCategory:
        if isinstance(exception, AuthenticationError):
            return ErrorCategory.AUTHENTICATION

        if isinstance(exception, AuthorizationError):
            return ErrorCategory.AUTHORIZATION

        if isinstance(exception, RateLimitError):
            return ErrorCategory.RATE_LIMIT

        if isinstance(exception, NetworkError):
            return ErrorCategory.NETWORK

        if isinstance(exception, ToolArgumentError):
            return ErrorCategory.TOOL_ARGUMENT_ERROR

        if isinstance(exception, ToolExecutionError):
            return ErrorCategory.TOOL_EXECUTION_ERROR

        if isinstance(exception, LLMFailure):
            return ErrorCategory.LLM_FAILURE

        if isinstance(exception, LLMOutputError):
            return ErrorCategory.LLM_OUTPUT_ERROR

        if isinstance(exception, ApprovalBlockedError):
            return ErrorCategory.APPROVAL_BLOCKED

        if isinstance(exception, PythonSyntaxError):
            return ErrorCategory.PYTHON_SYNTAX_ERROR

        if isinstance(exception, PythonImportError):
            return ErrorCategory.PYTHON_IMPORT_ERROR

        if isinstance(exception, EmptyResultError):
            return ErrorCategory.EMPTY_RESULT

        if isinstance(exception, InvalidResultError):
            return ErrorCategory.INVALID_RESULT

        if isinstance(exception, ValidationError):
            return ErrorCategory.VALIDATION

        if isinstance(exception, FileNotFoundError):
            return ErrorCategory.FILE_NOT_FOUND

        if isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT

        name = type(exception).__name__.lower()
        message = str(exception).lower()

        if "timeout" in name or "timeout" in message:
            return ErrorCategory.TIMEOUT

        if any(
            value in name
            for value in (
                "network",
                "connection",
                "connect",
            )
        ):
            return ErrorCategory.NETWORK

        if any(
            value in message
            for value in (
                "connection refused",
                "connection reset",
                "connection aborted",
                "network is unreachable",
            )
        ):
            return ErrorCategory.NETWORK

        if isinstance(exception, TypeError):
            return ErrorCategory.VALIDATION

        if isinstance(exception, ValueError):
            return ErrorCategory.VALIDATION

        return ErrorCategory.UNKNOWN

    def _severity(
        self,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        if category in {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.INTERNAL,
        }:
            return ErrorSeverity.HIGH

        if category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
        }:
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.MEDIUM