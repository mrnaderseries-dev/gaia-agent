from __future__ import annotations

import pytest

from gaia_agent.reliability.errors import (
    AgentError,
    ErrorSeverity,
)
from gaia_agent.reliability.failure_classifier import (
    FailureClassification,
    FailureClassifier,
    FailureType,
)
from gaia_agent.reliability.policies.retry_policy import (
    RetryPolicy,
)
from gaia_agent.reliability.recovery import (
    Recovery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_error(
    *,
    retryable: bool = False,
    recoverable: bool = False,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    message: str = "test failure",
) -> AgentError:
    return AgentError(
        error_type="TestError",
        message=message,
        severity=severity,
        retryable=retryable,
        recoverable=recoverable,
        source="test",
        operation="test_operation",
        attempt=1,
        original_exception=RuntimeError(message),
    )


def classification_for(
    classifier: FailureClassifier,
    *,
    retryable: bool = False,
    recoverable: bool = False,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
) -> FailureClassification:
    error = make_error(
        retryable=retryable,
        recoverable=recoverable,
        severity=severity,
    )
    return classifier.classify(error)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> FailureClassifier:
    return FailureClassifier()


@pytest.fixture
def retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=3,
        base_delay=0.0,
        max_delay=10.0,
    )


@pytest.fixture
def recovery() -> Recovery:
    return Recovery()


# ---------------------------------------------------------------------------
# RetryPolicy integration with FailureClassifier
# ---------------------------------------------------------------------------


def test_transient_failure_enters_retry_path(
    classifier: FailureClassifier,
    retry_policy: RetryPolicy,
):
    """
    P0.3 contract:

        retryable error
            ->
        TRANSIENT classification
            ->
        RetryPolicy allows retry
    """

    classification = classification_for(
        classifier,
        retryable=True,
        recoverable=True,
    )

    assert classification.failure_type is FailureType.TRANSIENT

    decision = retry_policy.evaluate(
        classification,
        current_attempt=1,
    )

    assert decision.should_retry is True
    assert decision.max_attempts == 3


def test_transient_failure_stops_after_retry_budget(
    classifier: FailureClassifier,
    retry_policy: RetryPolicy,
):
    """
    A transient error must not retry forever.

        TRANSIENT
            ->
        attempt reaches max_attempts
            ->
        STOP
    """

    classification = classification_for(
        classifier,
        retryable=True,
        recoverable=True,
    )

    decision = retry_policy.evaluate(
        classification,
        current_attempt=3,
    )

    assert decision.should_retry is False
    assert decision.reason == "Maximum retry attempts reached."


def test_transient_failure_uses_exponential_backoff(
    classifier: FailureClassifier,
):
    """
    P0.3 regression test for retry delay.

    attempt 1 -> base delay
    attempt 2 -> 2 * base delay
    attempt 3 -> 4 * base delay
    """

    policy = RetryPolicy(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
    )

    classification = classification_for(
        classifier,
        retryable=True,
        recoverable=True,
    )

    first = policy.evaluate(
        classification,
        current_attempt=1,
    )

    second = policy.evaluate(
        classification,
        current_attempt=2,
    )

    third = policy.evaluate(
        classification,
        current_attempt=3,
    )

    assert first.should_retry is True
    assert first.delay == 1.0

    assert second.should_retry is True
    assert second.delay == 2.0

    assert third.should_retry is True
    assert third.delay == 4.0


def test_retry_delay_is_capped(
    classifier: FailureClassifier,
):
    """
    Retry backoff must respect max_delay.
    """

    policy = RetryPolicy(
        max_attempts=10,
        base_delay=5.0,
        max_delay=10.0,
    )

    classification = classification_for(
        classifier,
        retryable=True,
        recoverable=True,
    )

    decision = policy.evaluate(
        classification,
        current_attempt=5,
    )

    assert decision.should_retry is True
    assert decision.delay == 10.0


# ---------------------------------------------------------------------------
# Non-transient failures must NOT enter RetryPolicy retry path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retryable,recoverable,severity,expected_type",
    [
        (
            False,
            True,
            ErrorSeverity.LOW,
            FailureType.RECOVERABLE,
        ),
        (
            False,
            False,
            ErrorSeverity.HIGH,
            FailureType.PERMANENT,
        ),
        (
            False,
            False,
            ErrorSeverity.CRITICAL,
            FailureType.FATAL,
        ),
        (
            False,
            False,
            ErrorSeverity.MEDIUM,
            FailureType.UNKNOWN,
        ),
    ],
)
def test_non_transient_failures_never_retry(
    classifier: FailureClassifier,
    retry_policy: RetryPolicy,
    retryable: bool,
    recoverable: bool,
    severity: ErrorSeverity,
    expected_type: FailureType,
):
    """
    P0.3 safety contract:

    Only TRANSIENT failures may enter automatic retry.

    RECOVERABLE / PERMANENT / FATAL / UNKNOWN
    must not be blindly retried.
    """

    classification = classification_for(
        classifier,
        retryable=retryable,
        recoverable=recoverable,
        severity=severity,
    )

    assert classification.failure_type is expected_type

    decision = retry_policy.evaluate(
        classification,
        current_attempt=1,
    )

    assert decision.should_retry is False
    assert decision.delay == 0.0
    assert decision.reason == "Failure is not transient."


# ---------------------------------------------------------------------------
# Recovery behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recoverable_failure_can_execute_recovery(
    classifier: FailureClassifier,
    recovery: Recovery,
):
    """
    Recoverable failure should NOT be retried blindly.

    Instead:

        RECOVERABLE
            ->
        Recovery
            ->
        meaningful change
            ->
        recovered=True
    """

    error = make_error(
        retryable=False,
        recoverable=True,
    )

    classification = classifier.classify(error)

    assert classification.failure_type is FailureType.RECOVERABLE

    calls = 0

    async def operation(received_error: AgentError):
        nonlocal calls

        calls += 1

        assert received_error is error

        return {
            "strategy": "alternative_tool",
        }

    result = await recovery.execute(
        error=error,
        operation=operation,
    )

    assert calls == 1
    assert result.recovered is True
    assert result.changed is True
    assert result.error is None


@pytest.mark.asyncio
async def test_recovery_without_meaningful_change_is_rejected(
    recovery: Recovery,
):
    """
    Critical P0.3 regression:

    Recovery must not report success when it produces
    the same execution/strategy.
    """

    error = make_error(
        recoverable=True,
    )

    async def operation(received_error: AgentError):
        return {
            "tool": "web_search",
            "arguments": {
                "query": "Malko",
            },
        }

    result = await recovery.execute(
        error=error,
        operation=operation,
        change_detector=lambda value: False,
    )

    assert result.recovered is False
    assert result.changed is False
    assert result.error is None
    assert "no meaningful change" in result.reason


@pytest.mark.asyncio
async def test_recovery_with_meaningful_change_is_accepted(
    recovery: Recovery,
):
    """
    Recovery should succeed when the change detector confirms
    that the strategy actually changed.
    """

    error = make_error(
        recoverable=True,
    )

    async def operation(received_error: AgentError):
        return {
            "tool": "python",
            "arguments": {
                "query": "Malko",
            },
        }

    result = await recovery.execute(
        error=error,
        operation=operation,
        change_detector=lambda value: (
            value["tool"] == "python"
        ),
    )

    assert result.recovered is True
    assert result.changed is True
    assert result.result["tool"] == "python"
    assert result.error is None


@pytest.mark.asyncio
async def test_recovery_operation_failure_is_not_reported_as_success(
    recovery: Recovery,
):
    """
    Recovery itself can fail.
    """

    error = make_error(
        recoverable=True,
    )

    async def operation(received_error: AgentError):
        raise RuntimeError("recovery operation failed")

    result = await recovery.execute(
        error=error,
        operation=operation,
    )

    assert result.recovered is False
    assert result.changed is False
    assert result.error is not None
    assert result.error.source == "recovery"
    assert result.error.operation == "recovery"


@pytest.mark.asyncio
async def test_change_detector_failure_does_not_claim_recovery_success(
    recovery: Recovery,
):
    """
    If the recovery operation succeeds but validation of the
    change fails, Recovery must fail closed.
    """

    error = make_error(
        recoverable=True,
    )

    async def operation(received_error: AgentError):
        return {
            "strategy": "new_strategy",
        }

    def broken_change_detector(value):
        raise RuntimeError("change detector crashed")

    result = await recovery.execute(
        error=error,
        operation=operation,
        change_detector=broken_change_detector,
    )

    assert result.recovered is False
    assert result.changed is False
    assert result.error is not None
    assert result.error.source == "recovery"
    assert result.error.operation == "change_detection"
    assert "change validation failed" in result.reason


# ---------------------------------------------------------------------------
# End-to-end P0.3 decision matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retryable,recoverable,severity,expected_type,expected_retry",
    [
        (
            True,
            True,
            ErrorSeverity.MEDIUM,
            FailureType.TRANSIENT,
            True,
        ),
        (
            False,
            True,
            ErrorSeverity.LOW,
            FailureType.RECOVERABLE,
            False,
        ),
        (
            False,
            False,
            ErrorSeverity.HIGH,
            FailureType.PERMANENT,
            False,
        ),
        (
            False,
            False,
            ErrorSeverity.CRITICAL,
            FailureType.FATAL,
            False,
        ),
        (
            False,
            False,
            ErrorSeverity.MEDIUM,
            FailureType.UNKNOWN,
            False,
        ),
    ],
)
def test_p0_3_failure_to_action_matrix(
    classifier: FailureClassifier,
    retry_policy: RetryPolicy,
    retryable: bool,
    recoverable: bool,
    severity: ErrorSeverity,
    expected_type: FailureType,
    expected_retry: bool,
):
    """
    Final P0.3 behavioral contract.
    """

    classification = classification_for(
        classifier,
        retryable=retryable,
        recoverable=recoverable,
        severity=severity,
    )

    assert classification.failure_type is expected_type

    decision = retry_policy.evaluate(
        classification,
        current_attempt=1,
    )

    assert decision.should_retry is expected_retry