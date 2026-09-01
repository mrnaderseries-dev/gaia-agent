from __future__ import annotations

import pytest

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import ErrorCategory
from gaia_agent.reliability.exception import (
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    ToolArgumentError,
    ToolExecutionError,
)
from gaia_agent.reliability.failure_classifier import (
    FailureClassifier,
    FailureType,
)


@pytest.fixture
def handler() -> ErrorHandler:
    return ErrorHandler()


@pytest.fixture
def classifier() -> FailureClassifier:
    return FailureClassifier()


def make_error(
    handler: ErrorHandler,
    exception: Exception,
    *,
    operation: str = "test_operation",
):
    return handler.handle(
        exception,
        source="test",
        operation=operation,
        attempt=1,
    )


# ---------------------------------------------------------------------------
# P0.3 - Authorization / 403
# ---------------------------------------------------------------------------


def test_403_authorization_is_not_retryable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    403 / authorization failure must never cause blind retries.

    Expected:
        retryable = False
        recoverable = True
        classification = RECOVERABLE

    The intended recovery path is:
        403 -> replan/change strategy
    """

    error = make_error(
        handler,
        AuthorizationError("403 Forbidden"),
        operation="web_search",
    )

    print("\n403 AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("403 Classification:")
    print(classification)

    assert error.category is ErrorCategory.AUTHORIZATION
    assert error.retryable is False
    assert error.recoverable is True

    assert classification.failure_type is FailureType.RECOVERABLE


# ---------------------------------------------------------------------------
# P0.3 - Authentication / 401
# ---------------------------------------------------------------------------


def test_authentication_failure_is_not_blindly_retried(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Authentication failure is different from a recoverable tool failure.

    A missing/invalid credential should not produce:
        retry -> retry -> retry

    Expected:
        retryable = False
        classification = PERMANENT
    """

    error = make_error(
        handler,
        AuthenticationError("401 Unauthorized"),
        operation="web_search",
    )

    print("\n401 AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("401 Classification:")
    print(classification)

    assert error.category is ErrorCategory.AUTHENTICATION
    assert error.retryable is False

    assert classification.failure_type is FailureType.PERMANENT


# ---------------------------------------------------------------------------
# P0.3 - Missing file
# ---------------------------------------------------------------------------


def test_missing_file_is_recoverable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Missing files are recoverable because the agent can potentially:

        locate another file
        search for the correct path
        ask a different tool
        replan the operation

    It must NOT blindly repeat the same file access.
    """

    error = make_error(
        handler,
        FileNotFoundError("file missing"),
        operation="read_file",
    )

    print("\nMissing-file AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("Missing-file Classification:")
    print(classification)

    assert error.category is ErrorCategory.FILE_NOT_FOUND
    assert error.retryable is False
    assert error.recoverable is True

    assert classification.failure_type is FailureType.RECOVERABLE


# ---------------------------------------------------------------------------
# P0.3 - Invalid tool arguments
# ---------------------------------------------------------------------------


def test_invalid_tool_arguments_are_recoverable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Invalid tool arguments should trigger replanning, not blind retry.

    Expected:

        invalid args
            ↓
        no retry
            ↓
        recoverable
            ↓
        replan with corrected arguments
    """

    error = make_error(
        handler,
        ToolArgumentError("invalid tool arguments"),
        operation="web_search",
    )

    print("\nTool-argument AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("Tool-argument Classification:")
    print(classification)

    assert error.category is ErrorCategory.TOOL_ARGUMENT_ERROR
    assert error.retryable is False
    assert error.recoverable is True

    assert classification.failure_type is FailureType.RECOVERABLE


# ---------------------------------------------------------------------------
# P0.3 - Tool execution failure
# ---------------------------------------------------------------------------


def test_tool_execution_failure_is_recoverable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Generic tool execution failures should be eligible for recovery.

    The recovery layer can ask the planner for a different strategy.
    """

    error = make_error(
        handler,
        ToolExecutionError("tool execution failed"),
        operation="web_search",
    )

    print("\nTool-execution AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("Tool-execution Classification:")
    print(classification)

    assert error.category is ErrorCategory.TOOL_EXECUTION_ERROR
    assert error.retryable is False
    assert error.recoverable is True

    assert classification.failure_type is FailureType.RECOVERABLE


# ---------------------------------------------------------------------------
# P0.3 - Timeout
# ---------------------------------------------------------------------------


def test_timeout_is_transient_and_retryable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Timeout is fundamentally different from authorization/file/argument
    failures.

    Expected:

        timeout
            ↓
        TRANSIENT
            ↓
        retry

    Replanning should happen only after retry exhaustion.
    """

    error = make_error(
        handler,
        TimeoutError("request timed out"),
        operation="web_search",
    )

    print("\nTimeout AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("Timeout Classification:")
    print(classification)

    assert error.category is ErrorCategory.TIMEOUT
    assert error.retryable is True
    assert error.recoverable is True

    assert classification.failure_type is FailureType.TRANSIENT


# ---------------------------------------------------------------------------
# P0.3 - Connection / network failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("connection reset"),
        NetworkError("network failure"),
    ],
)
def test_network_failures_are_transient(
    handler: ErrorHandler,
    classifier: FailureClassifier,
    exception: Exception,
):
    """
    Network failures should normally be treated as transient.

    They should go through RetryPolicy before recovery/replanning.
    """

    error = make_error(
        handler,
        exception,
        operation="web_search",
    )

    print("\nNetwork AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("Network Classification:")
    print(classification)

    assert error.retryable is True
    assert error.recoverable is True

    assert classification.failure_type is FailureType.TRANSIENT


# ---------------------------------------------------------------------------
# P0.3 - Unknown failures
# ---------------------------------------------------------------------------


def test_unknown_failure_is_not_safely_recoverable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Unknown failures must not automatically become recovery candidates.

    Safety invariant:

        unknown
            ↓
        STOP

    rather than blindly retrying/replanning.
    """

    error = make_error(
        handler,
        RuntimeError("completely unknown failure"),
        operation="test_operation",
    )

    print("\nUnknown AgentError:")
    print(error)

    classification = classifier.classify(error)

    print("Unknown Classification:")
    print(classification)

    assert classification.failure_type is FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# P0.3 - Classification priority
# ---------------------------------------------------------------------------


def test_retryable_failure_has_priority_over_recoverable(
    handler: ErrorHandler,
    classifier: FailureClassifier,
):
    """
    Important invariant:

    If an error is both retryable and recoverable, the classifier must
    classify it as TRANSIENT first.

    This guarantees:

        transient -> RetryPolicy

    instead of immediately:

        recoverable -> RecoveryPolicy
    """

    error = make_error(
        handler,
        TimeoutError("temporary timeout"),
        operation="web_search",
    )

    classification = classifier.classify(error)

    assert error.retryable is True
    assert error.recoverable is True

    assert classification.failure_type is FailureType.TRANSIENT


# ---------------------------------------------------------------------------
# P0.3 - Error classification matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception, expected_category, expected_retryable, expected_recoverable, expected_failure_type",
    [
        (
            AuthorizationError("403 Forbidden"),
            ErrorCategory.AUTHORIZATION,
            False,
            True,
            FailureType.RECOVERABLE,
        ),
        (
            FileNotFoundError("file missing"),
            ErrorCategory.FILE_NOT_FOUND,
            False,
            True,
            FailureType.RECOVERABLE,
        ),
        (
            ToolArgumentError("invalid arguments"),
            ErrorCategory.TOOL_ARGUMENT_ERROR,
            False,
            True,
            FailureType.RECOVERABLE,
        ),
        (
            ToolExecutionError("tool failed"),
            ErrorCategory.TOOL_EXECUTION_ERROR,
            False,
            True,
            FailureType.RECOVERABLE,
        ),
        (
            TimeoutError("timeout"),
            ErrorCategory.TIMEOUT,
            True,
            True,
            FailureType.TRANSIENT,
        ),
    ],
)
def test_p0_3_failure_classification_matrix(
    handler: ErrorHandler,
    classifier: FailureClassifier,
    exception: Exception,
    expected_category: ErrorCategory,
    expected_retryable: bool,
    expected_recoverable: bool,
    expected_failure_type: FailureType,
):
    """
    Single regression matrix for the four core P0.3 failure families:

        403 / authorization
        missing file
        invalid arguments
        tool execution failure
        timeout

    This test becomes the regression contract for P0.3.
    """

    error = make_error(
        handler,
        exception,
        operation="test_operation",
    )

    classification = classifier.classify(error)

    print("\nException:", type(exception).__name__)
    print("AgentError:", error)
    print("Classification:", classification)

    assert error.category is expected_category
    assert error.retryable is expected_retryable
    assert error.recoverable is expected_recoverable

    assert classification.failure_type is expected_failure_type