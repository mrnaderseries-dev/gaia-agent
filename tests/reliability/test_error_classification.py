import pytest

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.failure_classifier import (
    FailureClassifier,
    FailureType,
)
from gaia_agent.reliability.exception import (
    ToolArgumentError,
    ToolExecutionError,
)


@pytest.mark.parametrize(
    "exception",
    [
        ToolArgumentError("invalid tool arguments"),
        ToolExecutionError("tool execution failed"),
        TimeoutError("request timed out"),
        FileNotFoundError("file missing"),
    ],
)
def test_recoverable_error_classification(exception):
    handler = ErrorHandler()
    classifier = FailureClassifier()

    error = handler.handle(
        exception,
        source="test",
        operation="test_operation",
        attempt=1,
    )

    print("\nException:", type(exception).__name__)
    print("AgentError:", error)

    classification = classifier.classify(error)

    print("Classification:", classification)

    assert error.recoverable is True
    assert classification.failure_type in {
        FailureType.RECOVERABLE,
        FailureType.TRANSIENT,
    }