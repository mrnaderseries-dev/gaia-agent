from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from gaia_agent.agents.verifier import VerificationStatus
from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
    TransitionReason,
)
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.planner.models import PlanningResult
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.task_classifier import TaskClassifier
from gaia_agent.reliability.engine import ReliabilityAction
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)


# ============================================================================
# PLAN FIXTURES
# ============================================================================


def make_initial_plan() -> PlanSchema:
    """
    Initial plan intentionally contains a failing TOOL step.

    Expected lifecycle:

        PLAN
          ↓
        TOOL execution
          ↓
        FAILURE
          ↓
        RELIABILITY
          ↓
        REPLAN
    """

    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Execute failing tool",
                step_type=StepType.TOOL,
                tool_name="python_interpreter",
                arguments={
                    "code": "raise RuntimeError('boom')",
                },
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


def make_replanned_plan() -> PlanSchema:
    """
    Recovery plan.

    The failing TOOL step is removed and the agent can directly
    produce the final answer.
    """

    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Generate corrected final answer",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


# ============================================================================
# TASK ANALYSIS
# ============================================================================


def make_task_analysis():
    """
    Use the real production TaskClassifier.

    This keeps the integration test aligned with the actual Planner
    contract.
    """

    classifier = TaskClassifier()

    return classifier.classify(
        "Calculate 2 + 2",
        available_files=(),
        available_tools=("python_interpreter",),
    )


def make_planning_result(
    plan: PlanSchema,
    task_analysis,
) -> PlanningResult:
    """
    Production Planner contract:

        PlanningResult(
            plan=PlanSchema,
            task_analysis=TaskAnalysis,
        )
    """

    return PlanningResult(
        plan=plan,
        task_analysis=task_analysis,
    )


# ============================================================================
# ERROR FIXTURE
# ============================================================================


def make_execution_error() -> AgentError:
    return AgentError(
        error_type="ToolExecutionFailure",
        message="Simulated tool execution failure.",
        category=ErrorCategory.TOOL_EXECUTION_ERROR,
        severity=ErrorSeverity.HIGH,
        retryable=False,
        recoverable=True,
        source="production-readiness-test",
        operation="execute_tool",
    )


# ============================================================================
# DIAGNOSTICS
# ============================================================================


def print_diagnostics(
    *,
    state: AgentState,
    run,
    outcome,
    planner,
    context_builder,
    reliability_engine,
    agent_execution,
) -> None:
    print()
    print("=" * 80)
    print("ORCHESTRATOR E2E DIAGNOSTICS")
    print("=" * 80)

    print("\n--- OUTCOME ---")
    print(repr(outcome))

    print("\n--- STATE ---")
    print(repr(state))

    print("\n--- STATE PHASE ---")
    print(state.phase)

    print("\n--- STATE CURRENT STEP ---")
    print(state.current_step)

    print("\n--- STATE COMPLETED STEPS ---")
    print(state.completed_steps)

    print("\n--- TASK COMPLETED ---")
    print(state.task_completed)

    print("\n--- FINAL ANSWER ---")
    print(repr(state.final_answer))

    print("\n--- FINAL ANSWER READY ---")
    print(state.final_answer_ready)

    print("\n--- FINAL ANSWER VERIFIED ---")
    print(state.final_answer_verified)

    print("\n--- TERMINATION REASON ---")
    print(repr(state.termination_reason))

    print("\n--- STATE PLAN ---")
    print(repr(state.plan))

    print("\n--- STATE REPLAN COUNT ---")
    print(state.replan_count)

    print("\n--- RUN ---")
    print(repr(run))

    print("\n--- PLAN RUNTIME ---")
    print(repr(run.plan_runtime))

    print("\n--- PLAN RUNTIME CURRENT STEP ---")
    print(run.plan_runtime.current_step)

    print("\n--- PLAN RUNTIME COMPLETED STEPS ---")
    print(run.plan_runtime.completed_steps)

    print("\n--- PLAN RUNTIME VERSION ---")
    print(run.plan_runtime.plan_version)

    print("\n--- RUN EXECUTION HISTORY ---")
    print(repr(run.execution_history))

    print("\n--- RUN VERIFICATION HISTORY ---")
    print(repr(run.verification_history))

    print("\n--- RUN FINAL ANSWER ---")
    print(repr(run.final_answer))

    print("\n--- RUN ITERATION ---")
    print(run.iteration)

    print("\n--- RUN CURRENT ATTEMPT ---")
    print(run.current_attempt)

    print("\n--- RUN TERMINATION REASON ---")
    print(repr(run.termination_reason))

    print("\n--- TRANSITIONS ---")

    for index, transition in enumerate(
        state.transition_history,
        start=1,
    ):
        print(
            f"{index}. "
            f"{transition.from_phase} "
            f"-> "
            f"{transition.to_phase} "
            f"| reason={transition.reason}"
        )

    print("\n--- PLANNER.generate_plan ---")
    print(
        "await_count:",
        planner.generate_plan.await_count,
    )
    print(
        "call_args:",
        planner.generate_plan.call_args,
    )

    print("\n--- PLANNER.replan ---")
    print(
        "await_count:",
        planner.replan.await_count,
    )
    print(
        "call_args:",
        planner.replan.call_args,
    )

    print("\n--- CONTEXT BUILDER ---")
    print(
        "await_count:",
        context_builder.build.await_count,
    )
    print(
        "call_args:",
        context_builder.build.call_args,
    )

    print("\n--- RELIABILITY ENGINE ---")
    print(
        "await_count:",
        reliability_engine.handle_failure.await_count,
    )
    print(
        "call_args:",
        reliability_engine.handle_failure.call_args,
    )

    print("\n--- AGENT EXECUTION ---")
    print(
        "execute.await_count:",
        agent_execution.execute.await_count,
    )
    print(
        "call_args:",
        agent_execution.execute.call_args,
    )

    print()
    print("=" * 80)
    print("END DIAGNOSTICS")
    print("=" * 80)
    print()


# ============================================================================
# E2E TEST
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_recovers_from_execution_failure_and_completes() -> None:
    """
    End-to-end orchestration recovery test.

    Scenario:

        1. Orchestrator starts.
        2. Planner generates the initial plan.
        3. Orchestrator enters EXECUTING.
        4. The first TOOL execution fails.
        5. Reliability receives the failure.
        6. Reliability returns REPLAN.
        7. Planner generates a replacement plan.
        8. Replacement execution succeeds with "4".
        9. Orchestrator verifies the answer.
        10. Agent reaches COMPLETED.

    This test validates the existing production contracts.
    It does not modify production architecture to satisfy the test.
    """

    # ========================================================================
    # PLANS
    # ========================================================================

    initial_plan = make_initial_plan()

    replanned_plan = make_replanned_plan()

    # ========================================================================
    # REAL TASK ANALYSIS
    # ========================================================================

    task_analysis = make_task_analysis()

    # ========================================================================
    # PLANNER RESULTS
    # ========================================================================

    initial_planning_result = make_planning_result(
        initial_plan,
        task_analysis,
    )

    replanned_planning_result = make_planning_result(
        replanned_plan,
        task_analysis,
    )

    # ========================================================================
    # PLANNER
    # ========================================================================

    planner = MagicMock()

    planner.generate_plan = AsyncMock(
        return_value=initial_planning_result,
    )

    planner.replan = AsyncMock(
        return_value=replanned_planning_result,
    )

    def strategy_family(step):
        if step.step_type is StepType.LLM:
            return "LLM"

        if step.tool_name == "python_interpreter":
            return "PYTHON"

        return "UNKNOWN"

    planner.strategy_family.side_effect = strategy_family

    # ========================================================================
    # CONTEXT BUILDER
    # ========================================================================

    context_builder = MagicMock()

    context_builder.build = AsyncMock(
        return_value=MagicMock(
            name="FinalContext",
        )
    )

    # ========================================================================
    # AGENT EXECUTION
    # ========================================================================

    execution_error = make_execution_error()

    failed_execution = ExecutionResult(
        success=False,
        output=None,
        error=execution_error,
        step_id=0,
        tool_name="python_interpreter",
    )

    successful_execution = ExecutionResult(
        success=True,
        output="4",
        step_id=0,
        tool_name=None,
    )

    agent_execution = MagicMock(
        correlation_id=uuid4(),
    )

    agent_execution.execute = AsyncMock(
        side_effect=[
            failed_execution,
            successful_execution,
        ],
    )

    # ========================================================================
    # RELIABILITY
    # ========================================================================

    reliability_engine = MagicMock()

    # IMPORTANT:
    #
    # Production Orchestrator expects the ReliabilityResult to contain
    # BOTH:
    #
    #     action
    #     error
    #
    # The previous mock only supplied action, which caused:
    #
    #     AttributeError:
    #         'SimpleNamespace' object has no attribute 'error'
    #
    # That AttributeError was inside Orchestrator recovery handling and
    # prevented planner.replan() from ever being reached.

    reliability_engine.handle_failure = AsyncMock(
        return_value=SimpleNamespace(
            action=ReliabilityAction.REPLAN,
            error=execution_error,
        )
    )

    # ========================================================================
    # LOOP DETECTOR
    # ========================================================================

    loop_detector = MagicMock()

    loop_detector.check.return_value = SimpleNamespace(
        detected=False,
        reason=None,
    )

    loop_detector.record = MagicMock()

    # ========================================================================
    # VERIFIER
    # ========================================================================

    verifier = MagicMock()

    verifier.verify = AsyncMock(
        return_value=SimpleNamespace(
            status=VerificationStatus.VERIFIED,
            reason=None,
        )
    )

    # ========================================================================
    # ORCHESTRATOR
    # ========================================================================

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
    )

    # ========================================================================
    # STATE
    # ========================================================================

    state = AgentState(
        user_request="Calculate 2 + 2",
    )

    # ========================================================================
    # START
    # ========================================================================

    run = await orchestrator.start(state)

    assert run is not None

    # ========================================================================
    # STEP 1 — PLANNING
    # ========================================================================

    outcome = await orchestrator.step(
        state,
        run,
    )

    print_diagnostics(
        state=state,
        run=run,
        outcome=outcome,
        planner=planner,
        context_builder=context_builder,
        reliability_engine=reliability_engine,
        agent_execution=agent_execution,
    )

    assert outcome is not None

    assert state.phase is not AgentPhase.FAILED

    planner.generate_plan.assert_awaited_once()

    assert state.plan is not None

    assert len(state.plan) == 2

    assert run.plan_runtime.plan_version == 1

    assert run.plan_runtime.current_step == 0

    assert state.current_step == 0

    assert run.task_analysis is task_analysis

    assert agent_execution.execute.await_count == 0

    assert state.phase is AgentPhase.EXECUTING

    assert outcome.action.value == "execute"

    # ========================================================================
    # STEP 2 — FAILING TOOL + RECOVERY
    # ========================================================================

    outcome = await orchestrator.step(
        state,
        run,
    )

    print_diagnostics(
        state=state,
        run=run,
        outcome=outcome,
        planner=planner,
        context_builder=context_builder,
        reliability_engine=reliability_engine,
        agent_execution=agent_execution,
    )

    assert outcome is not None

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    assert agent_execution.execute.await_count == 1

    first_call = agent_execution.execute.await_args

    assert first_call is not None

    # ------------------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------------------

    reliability_engine.handle_failure.assert_awaited_once()

    reliability_result = (
        reliability_engine.handle_failure.return_value
    )

    assert (
        reliability_result.action
        is ReliabilityAction.REPLAN
    )

    assert reliability_result.error is execution_error

    # ------------------------------------------------------------------------
    # Replanning
    # ------------------------------------------------------------------------

    planner.replan.assert_awaited_once()

    assert state.replan_count == 1

    assert run.plan_runtime.plan_version == 2

    # ------------------------------------------------------------------------
    # Replacement plan
    # ------------------------------------------------------------------------

    assert state.plan is not None

    assert len(state.plan) == 1

    replacement_step = state.plan[0]

    assert replacement_step.is_final_answer is True

    assert replacement_step.step_type is StepType.LLM

    assert replacement_step.tool_name is None

    assert run.plan_runtime.current_step == 0

    assert state.current_step == 0

    assert state.task_completed is False

    assert state.phase is AgentPhase.EXECUTING

    # ========================================================================
    # STEP 3 — SUCCESSFUL REPLANNED EXECUTION
    # ========================================================================

    outcome = await orchestrator.step(
        state,
        run,
    )

    print_diagnostics(
        state=state,
        run=run,
        outcome=outcome,
        planner=planner,
        context_builder=context_builder,
        reliability_engine=reliability_engine,
        agent_execution=agent_execution,
    )

    assert outcome is not None

    assert agent_execution.execute.await_count == 2

    # ========================================================================
    # FINAL STATE
    # ========================================================================

    assert state.phase is AgentPhase.COMPLETED

    assert state.task_completed is True

    assert state.final_answer == "4"

    assert state.final_answer_ready is True

    assert state.final_answer_verified is True

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    verifier.verify.assert_awaited_once()

    assert state.verification_attempts == 1

    assert len(run.verification_history) == 1

    verification = run.verification_history[0]

    assert verification.verified is True

    assert verification.answer == "4"

    # ========================================================================
    # EXECUTION HISTORY
    # ========================================================================

    assert len(run.execution_history) == 2

    first_execution = run.execution_history[0]

    second_execution = run.execution_history[1]

    # Original execution failed.

    assert first_execution.result.success is False

    assert first_execution.result.error is execution_error

    assert first_execution.result.step_id == 0

    assert (
        first_execution.result.tool_name
        == "python_interpreter"
    )

    # Replanned execution succeeded.

    assert second_execution.result.success is True

    assert second_execution.result.output == "4"

    assert second_execution.result.step_id == 0

    assert second_execution.result.tool_name is None

    # ========================================================================
    # LOOP DETECTOR
    # ========================================================================

    assert loop_detector.check.call_count >= 2

    loop_detector.record.assert_called()

    # ========================================================================
    # LIFECYCLE TRANSITIONS
    # ========================================================================

    transitions = state.transition_history

    transition_phases = [
        transition.to_phase
        for transition in transitions
    ]

    transition_reasons = [
        transition.reason
        for transition in transitions
    ]

    assert AgentPhase.PLANNING in transition_phases

    assert AgentPhase.EXECUTING in transition_phases

    assert AgentPhase.VERIFYING in transition_phases

    assert AgentPhase.COMPLETED in transition_phases

    assert (
        TransitionReason.RECOVERY
        in transition_reasons
    )

    assert (
        TransitionReason.EXECUTION_COMPLETED
        in transition_reasons
    )

    assert (
        TransitionReason.VERIFICATION_PASSED
        in transition_reasons
    )

    assert (
        transitions[-1].to_phase
        is AgentPhase.COMPLETED
    )

    assert (
        transitions[-1].reason
        is TransitionReason.VERIFICATION_PASSED
    )

    # ========================================================================
    # FINAL INVARIANTS
    # ========================================================================

    assert state.task_completed is True

    assert state.final_answer_verified is True

    assert state.replan_count == 1

    assert run.plan_runtime.plan_version == 2

    assert run.plan_runtime.current_step == 0

    assert agent_execution.execute.await_count == 2

    assert len(run.execution_history) == 2

    assert len(run.verification_history) == 1