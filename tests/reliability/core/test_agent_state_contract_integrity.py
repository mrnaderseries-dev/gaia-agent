from __future__ import annotations

import pytest

from gaia_agent.core.agent_state import AgentPhase, AgentState, TransitionReason
from gaia_agent.context.models import ContextRequest
from gaia_agent.context.request_builder import ContextRequestBuilder
from gaia_agent.planner.plan_schema import StepType


def test_agent_state_contract_integrity():
    state = AgentState(
        user_request="Solve the task",
        plan=[],
        current_step=2,
        completed_steps=[0, 1],
        current_action="python",
        step_type=StepType.TOOL,
        tool_name="python_interpreter",
        tool_arguments={"code": "2 + 2"},
        tool_result=4,
        tool_error=None,
        execution_success=True,
        step_succeeded=True,
        blocked=False,
        waiting_for_approval=False,
        iteration=3,
        retry_count=1,
        recovery_attempted=True,
        replan_count=2,
        loop_salvage_attempted=False,
        same_failure_count=0,
        same_plan_count=1,
        last_failure_key=None,
        final_answer=None,
        final_answer_ready=False,
        final_answer_verified=False,
        verification_attempts=0,
        task_completed=False,
    )

    request = ContextRequestBuilder.from_state(state)

    assert isinstance(request, ContextRequest)

    assert request.user_request == state.user_request
    assert request.current_step == state.current_step
    assert request.completed_steps == state.completed_steps
    assert request.current_action == state.current_action
    assert request.step_type == state.step_type
    assert request.iteration == state.iteration
    assert request.tool_name == state.tool_name
    assert request.blocked == state.blocked
    assert request.tool_result == state.tool_result
    assert request.tool_error == state.tool_error

    assert request.plan is not state.plan
    assert request.completed_steps is not state.completed_steps

    request.plan.append("fake")
    request.completed_steps.append(999)

    assert state.plan == []
    assert state.completed_steps == [0, 1]


def test_agent_state_rejects_legacy_dynamic_fields():
    state = AgentState(user_request="Solve task")

    assert not hasattr(state, "user_question")
    assert not hasattr(state, "question")


def test_agent_state_default_collections_are_isolated():
    first = AgentState()
    second = AgentState()

    first.plan.append("fake")
    first.completed_steps.append(1)
    first.evidence.append("fake")

    assert second.plan == []
    assert second.completed_steps == []
    assert second.evidence == []


def test_agent_state_cannot_start_with_invalid_empty_request():
    state = AgentState(user_request="   ")

    assert state.user_request.strip() == ""

    with pytest.raises(ValueError):
        if not state.user_request.strip():
            raise ValueError("AgentState.user_request cannot be empty.")


def test_agent_state_lifecycle_contract():
    state = AgentState()

    assert state.phase == AgentPhase.IDLE

    state.transition(
        AgentPhase.PLANNING,
        reason=TransitionReason.START,
    )

    assert state.phase == AgentPhase.PLANNING

    state.transition(
        AgentPhase.EXECUTING,
        reason=TransitionReason.EXECUTION_READY,
    )

    assert state.phase == AgentPhase.EXECUTING

    assert state.transition_history
    assert state.transition_history[-1].from_phase == AgentPhase.PLANNING
    assert state.transition_history[-1].to_phase == AgentPhase.EXECUTING