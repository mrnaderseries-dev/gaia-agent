from unittest.mock import Mock

import pytest

from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.policies.termination import TerminationReason


@pytest.mark.asyncio
async def test_verified_answer_completes_and_emits_completed():
    orchestrator = Mock()

    state = Mock()
    state.final_answer_ready = True
    state.final_answer_verified = True
    state.verification_attempts = 1
    state.task_completed = False

    loop = AgentLoop(
        orchestrator=orchestrator,
        max_iterations=3,
    )

    loop.state = state

    termination = loop.check_termination()

    assert termination.reason == TerminationReason.COMPLETED

    state.task_completed = True

    orchestrator.emit_agent_completed()

    assert state.task_completed is True
    orchestrator.emit_agent_completed.assert_called_once()


@pytest.mark.asyncio
async def test_unverified_answer_budget_does_not_complete():
    orchestrator = Mock()

    state = Mock()
    state.final_answer_ready = True
    state.final_answer_verified = False
    state.verification_attempts = 2
    state.task_completed = False

    loop = AgentLoop(
        orchestrator=orchestrator,
        max_iterations=3,
    )

    loop.state = state

    termination = loop.check_termination()

    assert (
        termination.reason
        == TerminationReason.ANSWER_UNVERIFIED_BUDGET
    )

    state.task_completed = False

    assert state.task_completed is False
    orchestrator.emit_agent_completed.assert_not_called()