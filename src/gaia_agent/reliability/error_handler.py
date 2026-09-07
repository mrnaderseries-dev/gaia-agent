from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
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
            raise ValueError("attempt cannot be negative.")

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
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retryable=False,
            recoverable=False,
            source=source,
            operation=operation,
            attempt=attempt,
            details=dict(details or {}),
            original_exception=error,
            correlation_id=correlation_id,
        )
    