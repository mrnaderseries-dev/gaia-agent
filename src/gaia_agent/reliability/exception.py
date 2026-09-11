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
        recoverable: bool = True,
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