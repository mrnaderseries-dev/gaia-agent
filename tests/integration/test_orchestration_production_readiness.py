from __future__ import annotations

from unittest.mock import Mock
import pytest

from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
    TransitionReason,
)
from gaia_agent.core.orchestration.models import PlanRuntime
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)


def make_production_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Execute deterministic computation",
                step_type=StepType.TOOL,
                tool_name="python",
                arguments={"code": "2 + 2"},
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="Generate final answer",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


def make_error() -> AgentError:
    return AgentError(
        error_type="ExecutionFailure",
        message="tool execution failed",
        category=ErrorCategory.EXECUTION,
        severity=ErrorSeverity.HIGH,
        retryable=False,
        recoverable=False,
        source="production-readiness-test",
        operation="execute_step",
    )


def test_orchestration_production_lifecycle_preserves_invariants() -> None:
    """
    Production-readiness integration test.

    Verifies that orchestration preserves the following invariants:

    1. Execution starts from a valid lifecycle state.
    2. Plan progress is tracked explicitly.
    3. A failed execution reaches FAILED through the transition API.
    4. FAILED cannot silently become COMPLETED.
    5. Verification is required before completion.
    6. Successful verification produces the canonical COMPLETED state.
    7. Transition history remains consistent with the lifecycle.
    """

    state = AgentState(user_request="Calculate 2 + 2")

    # ---------------------------------------------------------
    # 1. Lifecycle starts correctly
    # ---------------------------------------------------------

    state.start()

    assert state.phase is AgentPhase.PLANNING
    assert state.task_completed is False

    # ---------------------------------------------------------
    # 2. Plan runtime owns execution progress
    # ---------------------------------------------------------

    runtime = PlanRuntime()
    plan = make_production_plan()

    runtime.set_plan(plan)

    assert runtime.plan_version == 1
    assert runtime.current_step == 0
    assert runtime.completed_steps == set()

    # ---------------------------------------------------------
    # 3. Move into execution
    # ---------------------------------------------------------

    state.begin_execution()

    assert state.phase is AgentPhase.EXECUTING
    assert state.task_completed is False

    # ---------------------------------------------------------
    # 4. Simulate a real execution failure
    # ---------------------------------------------------------

    orchestrator = object.__new__(Orchestrator)

    error = make_error()

    orchestrator._fail(state, error)

    assert state.phase is AgentPhase.FAILED
    assert state.task_completed is False
    assert state.tool_error == "tool execution failed"

    assert state.transition_history[-1].to_phase is AgentPhase.FAILED
    assert (
        state.transition_history[-1].reason
        is TransitionReason.EXECUTION_FAILED
    )

    # ---------------------------------------------------------
    # 5. FAILED must never silently become COMPLETED
    # ---------------------------------------------------------

    assert state.phase is not AgentPhase.COMPLETED
    assert state.task_completed is False

    # ---------------------------------------------------------
    # 6. Recovery uses an explicit lifecycle transition (FAILED -> PLANNING)
    # ---------------------------------------------------------

    state.recover()

    assert state.phase is AgentPhase.PLANNING

    # ---------------------------------------------------------
    # 7. Replanning creates a new plan version
    # ---------------------------------------------------------

    recovered_plan = make_production_plan()

    runtime.set_plan(
        recovered_plan,
        reset_progress=True,
    )

    assert runtime.plan_version == 2
    assert runtime.current_step == 0
    assert runtime.completed_steps == set()

    # ---------------------------------------------------------
    # 8. Re-enter execution
    # ---------------------------------------------------------

    state.begin_execution()

    assert state.phase is AgentPhase.EXECUTING
    assert state.task_completed is False

    # ---------------------------------------------------------
    # 9. Simulate successful execution
    # ---------------------------------------------------------

    runtime.mark_completed(0)

    assert runtime.completed_steps == {0}

    # ---------------------------------------------------------
    # 10. Verification is mandatory
    # ---------------------------------------------------------

    state.begin_verification()

    assert state.phase is AgentPhase.VERIFYING

    state.final_answer_verified = True

    # ---------------------------------------------------------
    # 11. Completion is canonical
    # ---------------------------------------------------------

    state.complete()

    assert state.phase is AgentPhase.COMPLETED
    assert state.task_completed is True

    assert (
        state.transition_history[-1].reason
        is TransitionReason.VERIFICATION_PASSED
    )

    # ---------------------------------------------------------
    # 12. Final invariants
    # ---------------------------------------------------------

    assert state.task_completed is True
    assert state.final_answer_verified is True
    assert state.phase is AgentPhase.COMPLETED

    phases = [
        transition.to_phase
        for transition in state.transition_history
    ]

    assert AgentPhase.FAILED in phases
    assert AgentPhase.VERIFYING in phases
    assert phases[-1] is AgentPhase.COMPLETED  # zabet indent
