from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gaia_agent.core.orchestration.orchestrator import (
    MAX_VERIFICATION_ATTEMPTS,
    Orchestrator,
)


@pytest.mark.asyncio
async def test_verification_budget_keeps_answer_unverified():
    """
    Exhausting the verification budget must never mark the task
    as successfully completed.
    """
    orchestrator = Orchestrator.__new__(Orchestrator)

    orchestrator.state = SimpleNamespace(
        verification_attempts=MAX_VERIFICATION_ATTEMPTS,
        final_answer_ready=True,
        final_answer_verified=False,
        task_completed=False,
        tool_error=None,
    )

    orchestrator.metrics = Mock()

    error = Mock()

    await orchestrator._handle_verification_failure(error)

    assert orchestrator.state.final_answer_ready is True
    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False

    assert (
        orchestrator.state.tool_error
        == "Verification attempts exhausted; "
        "final answer remains unverified."
    )

    orchestrator.metrics.increment.assert_called_once_with(
        "answers_unverified"
    )
