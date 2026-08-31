from __future__ import annotations


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


# ----------------------------------------------------------------------
# Phase 5 taxonomy: actionable, machine-actionable error types
# ----------------------------------------------------------------------


class ToolArgumentError(AgentRuntimeError):
    """
    Raised when tool arguments violate the registered tool contract.
    Should never reach the underlying tool implementation.
    Recovery strategy: repair arguments, validate, retry once.
    """

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
    """
    Raised when an action requires human approval that is not available.
    Recovery strategy: do not replan; stop and report the block.
    """

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
    """
    Raised when generated python code cannot be compiled.
    Recovery strategy: do not retry identical code.
    """

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
    """
    Raised when generated python code imports a module that is not
    available inside the controlled sandbox.
    Recovery strategy: generate sandbox-compatible code or choose
    another tool.
    """

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
    """
    Raised when a tool or step produces no usable content.
    Recovery strategy: replan with a different source of evidence.
    """

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
    """
    Raised when an operation returns a result that fails the
    registered validator.
    Recovery strategy: replan with a bounded budget.
    """

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