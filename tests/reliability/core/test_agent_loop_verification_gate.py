from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.policies.termination import (
    TerminationPolicy,
    TerminationReason,
)


def make_state(
    *,
    verified: bool,
    verification_attempts: int,
):
    return SimpleNamespace(
        user_request="test",
        final_answer_ready=True,
        final_answer_verified=verified,
        verification_attempts=verification_attempts,
        iteration=0,
        fatal_error=False,
        human_aborted=False,
        explicit_stop=False,
        timed_out=False,
        task_completed=False,
    )


@pytest.mark.asyncio
async def test_verified_answer_completes_and_emits_completed():
    orchestrator = Mock()

    state = make_state(
        verified=True,
        verification_attempts=1,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=TerminationPolicy(
            max_iterations=3,
            max_verification_attempts=2,
        ),
    )

    result = await loop.run(state)

    assert result.final_answer_verified is True
    assert result.termination_reason == TerminationReason.COMPLETED

    orchestrator.emit_agent_completed.assert_called_once()


@pytest.mark.asyncio
async def test_unverified_answer_budget_does_not_complete():
    orchestrator = Mock()

    state = make_state(
        verified=False,
        verification_attempts=2,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=TerminationPolicy(
            max_iterations=3,
            max_verification_attempts=2,
        ),
    )

    result = await loop.run(state)

    assert result.final_answer_verified is False
    assert (
        result.termination_reason
        == TerminationReason.ANSWER_UNVERIFIED_BUDGET
    )

    assert result.task_completed is False
    orchestrator.emit_agent_completed.assert_not_called()