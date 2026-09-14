from __future__ import annotations

import ast

import pytest

from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
    TransitionReason,
)
from gaia_agent.core.orchestration.models import PlanRuntime
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.reliability.errors import AgentError, ErrorCategory, ErrorSeverity


def make_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Do work",
                step_type=StepType.TOOL,
                tool_name="python",
                arguments={"code": "1 + 1"},
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="Return final answer",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


def test_completion_requires_verification() -> None:
    state = AgentState(user_request="test")
    state.start()
    state.begin_execution()
    state.final_answer_verified = False

    with pytest.raises(AgentError, match="before the final answer is verified"):
        state.complete()

    assert state.phase is AgentPhase.EXECUTING
    assert state.task_completed is False


def test_verified_completion_is_canonical() -> None:
    state = AgentState(user_request="test")
    state.start()
    state.begin_execution()
    state.begin_verification()
    state.final_answer_verified = True

    state.complete()

    assert state.phase is AgentPhase.COMPLETED
    assert state.task_completed is True
    assert state.transition_history[-1].reason is TransitionReason.VERIFICATION_PASSED


def test_plan_runtime_increments_version_without_losing_history() -> None:
    runtime = PlanRuntime()
    first = make_plan()
    second = make_plan()

    runtime.set_plan(first)
    runtime.mark_completed(0)
    assert runtime.plan_version == 1
    assert runtime.completed_steps == {0}

    runtime.set_plan(second, reset_progress=True)
    assert runtime.plan_version == 2
    assert runtime.current_step == 0
    assert runtime.completed_steps == set()


def test_orchestrator_failure_uses_state_transition_api() -> None:
    orchestrator = object.__new__(Orchestrator)
    state = AgentState(user_request="test")
    state.start()
    state.begin_execution()

    error = AgentError(
        error_type="TestFailure",
        message="failure",
        category=ErrorCategory.EXECUTION,
        severity=ErrorSeverity.HIGH,
        retryable=False,
        recoverable=False,
        source="test",
        operation="test",
    )

    orchestrator._fail(state, error)

    assert state.phase is AgentPhase.FAILED
    assert state.transition_history[-1].to_phase is AgentPhase.FAILED
    assert state.transition_history[-1].reason is TransitionReason.EXECUTION_FAILED
    assert state.tool_error == "failure"


def test_orchestrator_has_no_direct_phase_assignment() -> None:
    source = open(
        "src/gaia_agent/core/orchestration/orchestrator.py",
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert not (
                    isinstance(target, ast.Attribute)
                    and target.attr == "phase"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "state"
                ), "Orchestrator must not mutate state.phase directly"
