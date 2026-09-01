from __future__ import annotations

import builtins
from dataclasses import replace
from typing import Any
from uuid import UUID

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)

from gaia_agent.reliability.exception import (
    AgentRuntimeError,
    ApprovalBlockedError,
    AuthenticationError,
    AuthorizationError,
    ContextCompressionError,
    EmptyResultError,
    InternalAgentError,
    InvalidResultError,
    ModelExecutionError,
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
        error: Exception,
        *,
        source: str | None = None,
        operation: str | None = None,
        attempt: int = 0,
        correlation_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> AgentError:

        if attempt < 0:
            raise ValueError(
                "attempt cannot be negative."
            )

        if isinstance(error, AgentError):

            merged_details = dict(error.details)

            if details:
                merged_details.update(details)

            return replace(
                error,
                attempt=attempt,
                source=source or error.source,
                operation=operation or error.operation,
                correlation_id=(
                    correlation_id or error.correlation_id
                ),
                details=merged_details,
            )

        return AgentError(
            error_type=type(error).__name__,
            message=str(error),
            category=self._get_category(error),
            severity=self._get_severity(error),
            retryable=self._is_retryable(error),
            recoverable=self._is_recoverable(error),
            source=source,
            operation=operation,
            attempt=attempt,
            error_code=self._get_error_code(error),
            details=dict(details or {}),
            original_exception=error,
            correlation_id=correlation_id,
        )

    def _get_category(
        self,
        error: Exception,
    ) -> ErrorCategory:

        if isinstance(error, ApprovalBlockedError):
            return ErrorCategory.APPROVAL_BLOCKED

        if isinstance(error, ToolArgumentError):
            return ErrorCategory.TOOL_ARGUMENT_ERROR

        if isinstance(error, PythonSyntaxError):
            return ErrorCategory.PYTHON_SYNTAX_ERROR

        if isinstance(error, PythonImportError):
            return ErrorCategory.PYTHON_IMPORT_ERROR

        if isinstance(error, EmptyResultError):
            return ErrorCategory.EMPTY_RESULT

        if isinstance(error, InvalidResultError):
            return ErrorCategory.INVALID_RESULT

        if isinstance(error, FileNotFoundError):
            return ErrorCategory.FILE_NOT_FOUND

        if isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT

        if isinstance(error, builtins.NameError):
           
            return ErrorCategory.TOOL_EXECUTION_ERROR

        if isinstance(error, KeyError) and "not registered" in str(error):
          
            return ErrorCategory.TOOL_NOT_FOUND

        if isinstance(error, TypeError):
        
            message = str(error)
            if "keyword argument" in message or "missing" in message:
                return ErrorCategory.TOOL_ARGUMENT_ERROR
            return ErrorCategory.TOOL_EXECUTION_ERROR

        if isinstance(error, AuthenticationError):
            return ErrorCategory.AUTHENTICATION

        if isinstance(error, AuthorizationError):
            return ErrorCategory.AUTHORIZATION

        if isinstance(error, RateLimitError):
            return ErrorCategory.RATE_LIMIT

        if isinstance(error, ToolExecutionError):
            return ErrorCategory.TOOL_EXECUTION_ERROR

        if isinstance(error, ModelExecutionError):
            return ErrorCategory.LLM_FAILURE

        if isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION

        if isinstance(error, ContextCompressionError):
            return ErrorCategory.EXECUTION

        if isinstance(error, NetworkError):
            return ErrorCategory.NETWORK

        if isinstance(error, InternalAgentError):
            return ErrorCategory.INTERNAL

        if isinstance(error, ConnectionError):
            return ErrorCategory.NETWORK

        if isinstance(error, ValueError):
            return ErrorCategory.VALIDATION

        if isinstance(error, builtins.ImportError):
            return ErrorCategory.PYTHON_IMPORT_ERROR

        if isinstance(error, (builtins.SyntaxError, IndentationError)):
            return ErrorCategory.PYTHON_SYNTAX_ERROR

        return ErrorCategory.UNKNOWN

    def _get_error_code(
        self,
        error: Exception,
    ) -> str | None:

        if isinstance(error, ApprovalBlockedError):
            return "APPROVAL_BLOCKED"

        if isinstance(error, ToolArgumentError):
            return "TOOL_ARGUMENT_ERROR"

        if isinstance(error, PythonSyntaxError):
            return "PYTHON_SYNTAX_ERROR"

        if isinstance(error, PythonImportError):
            return "PYTHON_IMPORT_ERROR"

        if isinstance(error, EmptyResultError):
            return "EMPTY_RESULT"

        if isinstance(error, InvalidResultError):
            return "INVALID_RESULT"

        if isinstance(error, FileNotFoundError):
            return "FILE_NOT_FOUND"

        if isinstance(error, TimeoutError):
            return "TIMEOUT"

        if isinstance(error, builtins.NameError):
            return "NAME_ERROR"

        if isinstance(error, KeyError) and "not registered" in str(error):
            return "TOOL_NOT_FOUND"

        if isinstance(error, ToolExecutionError):
            return "TOOL_EXECUTION_ERROR"

        return None

    def _get_severity(
        self,
        error: Exception,
    ) -> ErrorSeverity:

        if isinstance(
            error,
            InternalAgentError,
        ):
            return ErrorSeverity.CRITICAL

        if isinstance(
            error,
            (
                AuthenticationError,
                AuthorizationError,
            ),
        ):
            return ErrorSeverity.HIGH

        if isinstance(
            error,
            (
                RateLimitError,
                NetworkError,
                TimeoutError,
                ConnectionError,
            ),
        ):
            return ErrorSeverity.MEDIUM

        if isinstance(
            error,
            (
                ValidationError,
                ToolArgumentError,
                ToolExecutionError,
                PythonSyntaxError,
                PythonImportError,
                EmptyResultError,
                builtins.NameError,
            ),
        ):
            return ErrorSeverity.LOW

        if isinstance(error, KeyError) and "not registered" in str(error):
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def _is_retryable(
        self,
        error: Exception,
    ) -> bool:

        if isinstance(
            error,
            AgentRuntimeError,
        ):
            return error.retryable

        if isinstance(
            error,
            (
                TimeoutError,
                ConnectionError,
                RateLimitError,
                NetworkError,
            ),
        ):
            return True

        return False

    def _is_recoverable(
        self,
        error: Exception,
    ) -> bool:
  
        if isinstance(
            error,
            (
                ValidationError,
                ToolArgumentError,
                ToolExecutionError,
                ModelExecutionError,
                EmptyResultError,
                InvalidResultError,
                FileNotFoundError,
                PythonSyntaxError,
                PythonImportError,
                builtins.NameError,
            ),
        ):
            return True

  
        if isinstance(
            error,
            (
                RateLimitError,
                NetworkError,
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True

        if isinstance(error, AgentRuntimeError):
            return error.recoverable

        if isinstance(error, KeyError) and "not registered" in str(error):
            return True

        return False