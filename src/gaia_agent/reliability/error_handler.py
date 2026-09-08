from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)


class ErrorHandler:
    def __init__(self, *, correlation_id: UUID | None = None) -> None:
        self.correlation_id = correlation_id

    def handle(
        self,
        exc: Exception,
        *,
        source: str | None = None,
        operation: str | None = None,
        attempt: int = 0,
        correlation_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> AgentError:
        if isinstance(exc, AgentError):
            return self._enrich_agent_error(
                exc,
                source=source,
                operation=operation,
                attempt=attempt,
                correlation_id=correlation_id,
                details=details,
            )

        category, severity, retryable, recoverable = self._classify_exception(exc)

        error = AgentError(
            error_type=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
            category=category,
            severity=severity,
            retryable=retryable,
            recoverable=recoverable,
            source=source,
            operation=operation,
            attempt=attempt,
            details={
                **(details or {}),
            },
            original_exception=exc,
            correlation_id=(
                correlation_id
                or self.correlation_id
                or uuid4()
            ),
        )

        return error

    def _enrich_agent_error(
        self,
        error: AgentError,
        *,
        source: str | None,
        operation: str | None,
        attempt: int,
        correlation_id: UUID | None,
        details: dict[str, Any] | None,
    ) -> AgentError:
        merged_details = {
            **error.details,
            **(details or {}),
        }

        return AgentError(
            error_type=error.error_type,
            message=error.message,
            category=error.category,
            severity=error.severity,
            retryable=error.retryable,
            recoverable=error.recoverable,
            source=source or error.source,
            operation=operation or error.operation,
            attempt=attempt or error.attempt,
            error_code=error.error_code,
            details=merged_details,
            original_exception=(
                error.original_exception
                or error
            ),
            correlation_id=(
                correlation_id
                or error.correlation_id
                or self.correlation_id
                or uuid4()
            ),
        )

    def _classify_exception(
        self,
        exc: Exception,
    ) -> tuple[
        ErrorCategory,
        ErrorSeverity,
        bool,
        bool,
    ]:
        name = type(exc).__name__
        message = str(exc).lower()

        if name in {
            "SemanticPlanError",
        }:
            return (
                ErrorCategory.PLAN_SEMANTIC_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "PlannerRecoveryRequired",
        }:
            return (
                ErrorCategory.PLAN_RECOVERY_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "PlanGenerationError",
        }:
            return (
                ErrorCategory.PLAN_GENERATION_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "PlanSchemaError",
        }:
            return (
                ErrorCategory.PLAN_SCHEMA_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "StrategySelectionError",
            "PlanStrategyError",
        }:
            return (
                ErrorCategory.PLAN_STRATEGY_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "ToolContractError",
        }:
            return (
                ErrorCategory.TOOL_CONTRACT_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "ToolNotFoundError",
        }:
            return (
                ErrorCategory.TOOL_NOT_FOUND,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "ToolArgumentError",
        }:
            return (
                ErrorCategory.TOOL_ARGUMENT_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "ToolExecutionError",
        }:
            return (
                ErrorCategory.TOOL_EXECUTION_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "AuthenticationError",
        }:
            return (
                ErrorCategory.AUTHENTICATION,
                ErrorSeverity.HIGH,
                False,
                False,
            )

        if name in {
            "AuthorizationError",
        }:
            return (
                ErrorCategory.AUTHORIZATION,
                ErrorSeverity.HIGH,
                False,
                False,
            )

        if name in {
            "RateLimitError",
        }:
            return (
                ErrorCategory.RATE_LIMIT,
                ErrorSeverity.MEDIUM,
                True,
                False,
            )

        if name in {
            "TimeoutError",
            "ReadTimeout",
            "ConnectTimeout",
            "WriteTimeout",
            "PoolTimeout",
        }:
            return (
                ErrorCategory.TIMEOUT,
                ErrorSeverity.MEDIUM,
                True,
                False,
            )

        if name in {
            "NetworkError",
            "ConnectError",
            "ReadError",
            "WriteError",
        }:
            return (
                ErrorCategory.NETWORK,
                ErrorSeverity.MEDIUM,
                True,
                False,
            )

        if name in {
            "PythonSyntaxError",
            "SyntaxError",
        }:
            return (
                ErrorCategory.PYTHON_SYNTAX_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "PythonImportError",
            "ImportError",
            "ModuleNotFoundError",
        }:
            return (
                ErrorCategory.PYTHON_IMPORT_ERROR,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "FileNotFoundError",
        }:
            return (
                ErrorCategory.FILE_NOT_FOUND,
                ErrorSeverity.MEDIUM,
                False,
                True,
            )

        if name in {
            "ArtifactNotFoundError",
        }:
            return (
                ErrorCategory.ARTIFACT_NOT_FOUND,
                ErrorSeverity.MEDIUM,
                False,
                True,
            )

        if name in {
            "EmptyResultError",
        }:
            return (
                ErrorCategory.EMPTY_RESULT,
                ErrorSeverity.MEDIUM,
                False,
                True,
            )

        if name in {
            "InvalidResultError",
        }:
            return (
                ErrorCategory.INVALID_RESULT,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "ApprovalBlockedError",
        }:
            return (
                ErrorCategory.APPROVAL_BLOCKED,
                ErrorSeverity.HIGH,
                False,
                False,
            )

        if name in {
            "TransitionError",
            "InvalidTransitionError",
            "StateTransitionError",
        }:
            return (
                ErrorCategory.TRANSITION_FAILURE,
                ErrorSeverity.HIGH,
                False,
                False,
            )

        if name in {
            "LoopDetected",
            "LoopDetectedError",
            "InfiniteLoopError",
        }:
            return (
                ErrorCategory.LOOP_DETECTED,
                ErrorSeverity.CRITICAL,
                False,
                False,
            )

        if name in {
            "ValidationError",
        }:
            return (
                ErrorCategory.VALIDATION,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        if name in {
            "InternalAgentError",
            "AgentRuntimeError",
        }:
            return (
                ErrorCategory.INTERNAL,
                ErrorSeverity.HIGH,
                False,
                False,
            )

        if name in {
            "ModelExecutionError",
            "LLMFailure",
            "LLMError",
        }:
            return (
                ErrorCategory.LLM_FAILURE,
                ErrorSeverity.HIGH,
                True,
                False,
            )

        if name in {
            "LLMOutputError",
        }:
            return (
                ErrorCategory.LLM_OUTPUT_ERROR,
                ErrorSeverity.MEDIUM,
                False,
                True,
            )

        if "timeout" in message:
            return (
                ErrorCategory.TIMEOUT,
                ErrorSeverity.MEDIUM,
                True,
                False,
            )

        if any(
            token in message
            for token in (
                "rate limit",
                "too many requests",
                "429",
            )
        ):
            return (
                ErrorCategory.RATE_LIMIT,
                ErrorSeverity.MEDIUM,
                True,
                False,
            )

        if any(
            token in message
            for token in (
                "connection",
                "connect error",
                "network",
                "temporarily unavailable",
                "service unavailable",
            )
        ):
            return (
                ErrorCategory.NETWORK,
                ErrorSeverity.MEDIUM,
                True,
                False,
            )

        if isinstance(exc, ValueError):
            return (
                ErrorCategory.VALIDATION,
                ErrorSeverity.MEDIUM,
                False,
                True,
            )

        if isinstance(exc, TypeError):
            return (
                ErrorCategory.VALIDATION,
                ErrorSeverity.HIGH,
                False,
                True,
            )

        return (
            ErrorCategory.UNKNOWN,
            ErrorSeverity.MEDIUM,
            False,
            False,
        )