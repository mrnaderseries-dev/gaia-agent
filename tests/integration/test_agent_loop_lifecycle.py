from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
    TransitionReason,
)
from gaia_agent.core.orchestration.models import (
    OrchestrationAction,
    OrchestrationOutcome,
)
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.core.policies.termination import (
    TerminationPolicy,
    TerminationReason,
)


# ============================================================================
# Helpers
# ============================================================================


def make_state(
    *,
    request: str = "Calculate 2 + 2",
) -> AgentState:
    return AgentState(
        user_request=request,
    )


def make_run(
    *,
    final_answer: str | None = None,
    is_verified: bool = False,
    verification_attempts: int = 0,
):
    return SimpleNamespace(
        final_answer=final_answer,
        is_verified=is_verified,
        verification_attempts=verification_attempts,
    )


def make_outcome(
    action: OrchestrationAction,
    *,
    reason: str | None = None,
    error=None,
):
    return OrchestrationOutcome(
        action=action,
        reason=reason,
        error=error,
    )


def move_to_executing(state: AgentState) -> None:
    """
    Put the state into a legitimate active execution phase.

    IDLE -> PLANNING -> EXECUTING
    """
    state.start()
    state.begin_execution()


def move_to_verifying(state: AgentState) -> None:
    """
    Put the state into a legitimate verification phase.

    IDLE -> PLANNING -> EXECUTING -> VERIFYING
    """
    move_to_executing(state)
    state.begin_verification()


def make_loop(
    *,
    orchestrator,
    policy: TerminationPolicy | None = None,
) -> AgentLoop:
    return AgentLoop(
        orchestrator=orchestrator,
        termination_policy=policy or TerminationPolicy(),
    )


# ============================================================================
# COMPLETE
# ============================================================================


@pytest.mark.asyncio
async def test_complete_lifecycle():
    """
    COMPLETE must:

        - copy final answer from run
        - mark answer ready
        - mark answer verified
        - mark task completed
        - transition to COMPLETED
        - stop the loop
        - never execute another step
        - cleanup loop-owned references
    """
    state = make_state()

    # COMPLETE is only valid from VERIFYING.
    move_to_verifying(state)

    run = make_run(
        final_answer="4",
        is_verified=True,
        verification_attempts=1,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.COMPLETE,
            reason="verification_passed",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state

    assert state.phase is AgentPhase.COMPLETED
    assert state.final_answer == "4"
    assert state.final_answer_ready is True
    assert state.final_answer_verified is True
    assert state.task_completed is True
    assert state.verification_attempts == 1

    orchestrator.start.assert_awaited_once()
    orchestrator.step.assert_awaited_once()

    # AgentLoop must clean its references after the run.
    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# FAIL
# ============================================================================


@pytest.mark.asyncio
async def test_fail_outcome_transitions_active_state_to_failed():
    """
    FAIL from an active lifecycle phase must transition to FAILED.
    """
    state = make_state()

    move_to_executing(state)

    run = make_run()
    
    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.FAIL,
            reason="execution_failed",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.FAILED
    assert state.task_completed is False

    orchestrator.start.assert_awaited_once()
    orchestrator.step.assert_awaited_once()

    assert loop.state is None
    assert loop.run_context is None


@pytest.mark.asyncio
async def test_fail_outcome_preserves_error_message():
    """
    FAIL with an explicit error must project the error into tool_error.
    """
    state = make_state()

    move_to_executing(state)

    run = make_run()

    error = RuntimeError("python execution failed")

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.FAIL,
            error=error,
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.FAILED
    assert state.tool_error == "python execution failed"


# ============================================================================
# WAIT FOR APPROVAL
# ============================================================================


@pytest.mark.asyncio
async def test_wait_for_approval_stops_loop_without_failing():
    """
    WAIT_FOR_APPROVAL is not a failure.

    AgentLoop must:
        - mark blocked
        - mark waiting_for_approval
        - stop current loop
        - preserve lifecycle phase
        - not transition to FAILED
    """
    state = make_state()

    move_to_executing(state)

    original_phase = state.phase

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.WAIT_FOR_APPROVAL,
            reason="human_approval_required",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state

    assert state.phase is original_phase
    assert state.blocked is True
    assert state.waiting_for_approval is True
    assert state.phase is not AgentPhase.FAILED

    orchestrator.step.assert_awaited_once()

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# TERMINATE OUTCOME
# ============================================================================


@pytest.mark.asyncio
async def test_terminate_outcome_stops_loop():
    """
    Explicit TERMINATE outcome must stop the loop.
    """
    state = make_state()

    move_to_executing(state)

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.TERMINATE,
            reason="operator_requested_stop",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    orchestrator.step.assert_awaited_once()

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# TERMINATION POLICY
# ============================================================================


@pytest.mark.asyncio
async def test_policy_explicit_stop_prevents_orchestrator_step():
    """
    explicit_stop must be evaluated before another orchestration step.
    """
    state = make_state()

    move_to_executing(state)

    state.explicit_stop = True

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    orchestrator.start.assert_awaited_once()
    orchestrator.step.assert_not_awaited()

    assert loop.state is None
    assert loop.run_context is None


@pytest.mark.asyncio
async def test_policy_human_abort_prevents_orchestrator_step():
    """
    human_aborted must stop the lifecycle before another step.
    """
    state = make_state()

    move_to_executing(state)

    state.human_aborted = True

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    orchestrator.step.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_timeout_prevents_orchestrator_step():
    """
    timed_out must stop the lifecycle before another step.
    """
    state = make_state()

    move_to_executing(state)

    state.timed_out = True

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    orchestrator.step.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_fatal_error_prevents_orchestrator_step():
    """
    fatal_error must stop the lifecycle before another step.
    """
    state = make_state()

    move_to_executing(state)

    state.fatal_error = True

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    orchestrator.step.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_max_iterations_stops_lifecycle():
    """
    Reaching max_iterations must prevent another orchestration step.
    """
    state = make_state()

    move_to_executing(state)

    policy = TerminationPolicy(
        max_iterations=5,
    )

    state.iteration = 5

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
        policy=policy,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    orchestrator.step.assert_not_awaited()


# ============================================================================
# VERIFIED ANSWER
# ============================================================================


@pytest.mark.asyncio
async def test_verified_answer_can_complete_normally():
    """
    A verified answer should not be rejected by the termination policy.

    The orchestrator is responsible for returning COMPLETE.
    """
    state = make_state()

    move_to_verifying(state)

    run = make_run(
        final_answer="4",
        is_verified=True,
        verification_attempts=1,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.COMPLETE,
            reason="verification_passed",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.COMPLETED
    assert state.final_answer_verified is True
    assert state.task_completed is True


@pytest.mark.asyncio
async def test_unverified_answer_does_not_complete():
    """
    COMPLETE returned with an unverified answer is a contract violation.

    AgentLoop must reject it instead of silently completing the task.
    """
    state = make_state()

    move_to_verifying(state)

    run = make_run(
        final_answer="4",
        is_verified=False,
        verification_attempts=1,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.COMPLETE,
            reason="invalid_completion",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        RuntimeError,
        match="COMPLETE without a verified final answer",
    ):
        await loop.run_agent(state)

    # finally must still clean the AgentLoop.
    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# MULTI-STEP LOOP
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_execute_outcomes_continue_until_complete():
    """
    Non-terminal orchestration outcomes must keep the AgentLoop alive.

    EXECUTE -> EXECUTE -> COMPLETE
    """
    state = make_state()

    move_to_executing(state)

    run = make_run(
        final_answer="4",
        is_verified=True,
        verification_attempts=1,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    execute_1 = make_outcome(
        OrchestrationAction.EXECUTE,
        reason="continue",
    )

    execute_2 = make_outcome(
        OrchestrationAction.EXECUTE,
        reason="continue",
    )

    complete = make_outcome(
        OrchestrationAction.COMPLETE,
        reason="verification_passed",
    )

    outcomes = [
        execute_1,
        execute_2,
        complete,
    ]

    async def step_side_effect(
        state_arg,
        run_arg,
    ):
        outcome = outcomes.pop(0)

        if outcome.action is OrchestrationAction.COMPLETE:
            # COMPLETE is valid only from VERIFYING.
            if state_arg.phase is AgentPhase.EXECUTING:
                state_arg.begin_verification()

        return outcome

    orchestrator.step = AsyncMock(
        side_effect=step_side_effect,
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.COMPLETED

    assert orchestrator.step.await_count == 3

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# TERMINAL STATE IDEMPOTENCY
# ============================================================================


@pytest.mark.asyncio
async def test_already_completed_state_does_not_execute_step():
    """
    A COMPLETED state must not execute another orchestration step.
    """
    state = make_state()

    move_to_verifying(state)

    state.final_answer = "4"
    state.final_answer_ready = True
    state.final_answer_verified = True

    state.complete()

    assert state.phase is AgentPhase.COMPLETED

    run = make_run(
        final_answer="4",
        is_verified=True,
        verification_attempts=1,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.COMPLETED

    orchestrator.start.assert_awaited_once()
    orchestrator.step.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminated_state_is_rejected_before_start():
    """
    A TERMINATED state cannot be reused for another AgentLoop run.
    """
    state = make_state()

    state.terminate()

    assert state.phase is AgentPhase.TERMINATED

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock()
    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        ValueError,
        match="already terminated",
    ):
        await loop.run_agent(state)

    orchestrator.start.assert_not_awaited()
    orchestrator.step.assert_not_awaited()

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# INPUT VALIDATION
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_state_type_is_rejected():
    """
    AgentLoop must reject objects that are not AgentState.
    """
    orchestrator = MagicMock(spec=Orchestrator)

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        TypeError,
        match="expects AgentState",
    ):
        await loop.run_agent(
            object(),
        )

    orchestrator.start.assert_not_awaited()
    orchestrator.step.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_user_request_is_rejected():
    """
    Empty user requests must be rejected before orchestration starts.
    """
    state = make_state(
        request="   ",
    )

    orchestrator = MagicMock(spec=Orchestrator)

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        ValueError,
        match="user_request cannot be empty",
    ):
        await loop.run_agent(state)

    orchestrator.start.assert_not_awaited()
    orchestrator.step.assert_not_awaited()


# ============================================================================
# EXCEPTION PROPAGATION + CLEANUP
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_exception_propagates_and_loop_cleans_up():
    """
    AgentLoop must not swallow unexpected Orchestrator exceptions.

    The exception belongs to the orchestration layer.

    AgentLoop must only guarantee cleanup through finally.
    """
    state = make_state()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        side_effect=RuntimeError(
            "unexpected orchestrator failure"
        )
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected orchestrator failure",
    ):
        await loop.run_agent(state)

    orchestrator.start.assert_awaited_once()
    orchestrator.step.assert_not_awaited()

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# START FAILURE
# ============================================================================


@pytest.mark.asyncio
async def test_start_failure_cleans_up_loop():
    """
    If Orchestrator.start() fails, AgentLoop must still cleanup.
    """
    state = make_state()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        side_effect=ValueError(
            "invalid orchestration state"
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        ValueError,
        match="invalid orchestration state",
    ):
        await loop.run_agent(state)

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# RUN ALIAS
# ============================================================================


@pytest.mark.asyncio
async def test_run_delegates_to_run_agent():
    """
    run() must remain a thin public alias for run_agent().
    """
    state = make_state()

    move_to_verifying(state)

    run = make_run(
        final_answer="4",
        is_verified=True,
        verification_attempts=1,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.COMPLETE,
            reason="verification_passed",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run(state)

    assert result is state
    assert state.phase is AgentPhase.COMPLETED

    orchestrator.start.assert_awaited_once()
    orchestrator.step.assert_awaited_once()

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# TERMINATION POLICY INSPECTION
# ============================================================================


def test_check_termination_requires_bound_state():
    """
    check_termination() cannot be called before AgentLoop binds a state.
    """
    orchestrator = MagicMock(spec=Orchestrator)

    loop = make_loop(
        orchestrator=orchestrator,
    )

    with pytest.raises(
        RuntimeError,
        match="AgentState is not bound",
    ):
        loop.check_termination()


@pytest.mark.asyncio
async def test_check_termination_reports_continue_for_normal_active_state():
    """
    A normal active state with no stop condition must continue.
    """
    state = make_state()

    move_to_executing(state)

    orchestrator = MagicMock(spec=Orchestrator)

    loop = make_loop(
        orchestrator=orchestrator,
    )

    loop._state = state

    decision = loop.check_termination()

    assert decision.should_stop is False
    assert decision.reason is None

    loop._state = None


# ============================================================================
# TERMINATION REASON PROJECTION
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "attribute",
        "expected_reason",
    ),
    [
        (
            "explicit_stop",
            TerminationReason.EXPLICIT_STOP,
        ),
        (
            "human_aborted",
            TerminationReason.HUMAN_ABORTED,
        ),
        (
            "timed_out",
            TerminationReason.TIMED_OUT,
        ),
        (
            "fatal_error",
            TerminationReason.FATAL_ERROR,
        ),
    ],
)
async def test_policy_stop_records_policy_reason(
    attribute,
    expected_reason,
):
    """
    The policy reason should remain available on AgentState.

    This verifies that AgentLoop does not accidentally replace the
    policy-level termination reason with the generic lifecycle
    TransitionReason.TERMINATION.
    """
    state = make_state()

    move_to_executing(state)

    setattr(
        state,
        attribute,
        True,
    )

    orchestrator = MagicMock(spec=Orchestrator)

    run = make_run()

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock()

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    assert state.termination_reason == expected_reason

    orchestrator.step.assert_not_awaited()

    assert loop.state is None
    assert loop.run_context is None


# ============================================================================
# OUTCOME TERMINATION REASON
# ============================================================================


@pytest.mark.asyncio
async def test_orchestration_terminate_reason_is_preserved():
    """
    If Orchestrator returns TERMINATE with a reason, AgentLoop should
    preserve that reason on AgentState.
    """
    state = make_state()

    move_to_executing(state)

    run = make_run()

    orchestrator = MagicMock(spec=Orchestrator)

    orchestrator.start = AsyncMock(
        return_value=run,
    )

    orchestrator.step = AsyncMock(
        return_value=make_outcome(
            OrchestrationAction.TERMINATE,
            reason="loop_limit_reached",
        )
    )

    loop = make_loop(
        orchestrator=orchestrator,
    )

    result = await loop.run_agent(state)

    assert result is state
    assert state.phase is AgentPhase.TERMINATED

    assert state.termination_reason == "loop_limit_reached"

    assert loop.state is None
    assert loop.run_context is None