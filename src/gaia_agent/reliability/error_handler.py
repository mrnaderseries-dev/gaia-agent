from __future__ import annotations

from typing import Any

from .errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)


class AgentRuntimeError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.recoverable = recoverable


class AuthenticationError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Authentication failed.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=False,
        )


class AuthorizationError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Authorization failed.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=False,
        )


class RateLimitError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Rate limit exceeded.",
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            recoverable=False,
        )


class NetworkError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Network operation failed.",
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            recoverable=False,
        )


class ToolExecutionError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Tool execution failed.",
        *,
        retryable: bool = False,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            message,
            retryable=retryable,
            recoverable=recoverable,
        )


class ModelExecutionError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Model execution failed.",
        *,
        retryable: bool = False,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            message,
            retryable=retryable,
            recoverable=recoverable,
        )


class LLMFailure(AgentRuntimeError):
    def __init__(
        self,
        message: str = "LLM execution failed.",
        *,
        retryable: bool = False,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            message,
            retryable=retryable,
            recoverable=recoverable,
        )


class LLMOutputError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "LLM returned invalid output.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class ValidationError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Validation failed.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=False,
        )


class InternalAgentError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Internal agent failure.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=False,
        )


class ContextCompressionError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Context compression failed.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class ToolArgumentError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Tool argument validation failed.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class ApprovalBlockedError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Action requires human approval.",
        *,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class PythonSyntaxError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Python code contains a syntax error.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class PythonImportError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Python code imports an unavailable module.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class EmptyResultError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Tool returned an empty result.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
        )


class InvalidResultError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "Operation returned an invalid result.",
        *,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            recoverable=recoverable,
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

        if category == ErrorCategory.LLM_OUTPUT_ERROR:
            recoverable = True

        return AgentError(
            error_type=type(exception).__name__,
            message=str(exception)
            or type(exception).__name__,
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
        if error.source == "unknown":
            error.source = source

        if error.operation == "unknown":
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
        if isinstance(
            exception,
            AuthenticationError,
        ):
            return ErrorCategory.AUTHENTICATION

        if isinstance(
            exception,
            AuthorizationError,
        ):
            return ErrorCategory.AUTHORIZATION

        if isinstance(
            exception,
            RateLimitError,
        ):
            return ErrorCategory.RATE_LIMIT

        if isinstance(
            exception,
            NetworkError,
        ):
            return ErrorCategory.NETWORK

        if isinstance(
            exception,
            ToolArgumentError,
        ):
            return ErrorCategory.TOOL_ARGUMENT_ERROR

        if isinstance(
            exception,
            ToolExecutionError,
        ):
            return ErrorCategory.TOOL_EXECUTION_ERROR

        if isinstance(
            exception,
            ModelExecutionError,
        ):
            return ErrorCategory.LLM_FAILURE

        if isinstance(
            exception,
            LLMFailure,
        ):
            return ErrorCategory.LLM_FAILURE

        if isinstance(
            exception,
            LLMOutputError,
        ):
            return ErrorCategory.LLM_OUTPUT_ERROR

        if isinstance(
            exception,
            ApprovalBlockedError,
        ):
            return ErrorCategory.APPROVAL_BLOCKED

        if isinstance(
            exception,
            PythonSyntaxError,
        ):
            return ErrorCategory.PYTHON_SYNTAX_ERROR

        if isinstance(
            exception,
            PythonImportError,
        ):
            return ErrorCategory.PYTHON_IMPORT_ERROR

        if isinstance(
            exception,
            EmptyResultError,
        ):
            return ErrorCategory.EMPTY_RESULT

        if isinstance(
            exception,
            InvalidResultError,
        ):
            return ErrorCategory.INVALID_RESULT

        if isinstance(
            exception,
            ValidationError,
        ):
            return ErrorCategory.VALIDATION

        if isinstance(
            exception,
            TimeoutError,
        ):
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
            ErrorCategory.UNKNOWN,
        }:
            return ErrorSeverity.HIGH

        if category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
        }:
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.MEDIUM