from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_agent.agents.verifier import VerificationStatus
from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
    TransitionReason,
)
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.core.policies.termination import TerminationPolicy
from gaia_agent.planner.models import PlanningResult
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.task_classifier import TaskClassifier
from gaia_agent.reliability.engine import ReliabilityAction


# ============================================================================
# PLAN BUILDERS
# ============================================================================


def make_initial_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Execute calculation",
                step_type=StepType.TOOL,
                tool_name="python_interpreter",
                arguments={
                    "code": "raise RuntimeError('simulated tool failure')",
                },
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="Generate final answer",
                step_type=StepType.LLM,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


def make_replanned_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Generate corrected final answer",
                step_type=StepType.LLM,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


# ============================================================================
# REAL TASK ANALYSIS
# ============================================================================


def make_task_analysis():
    classifier = TaskClassifier()

    return classifier.classify(
        "Calculate 2 + 2",
        available_files=(),
        available_tools=("python_interpreter",),
    )


# ============================================================================
# TEST TOOL
# ============================================================================


class FailingPythonTool:
    async def execute(self, **arguments):
        raise RuntimeError(
            "simulated tool execution failure"
        )


# ============================================================================
# TEST EXECUTION POLICY
# ============================================================================


class AllowExecutionPolicy:
    async def evaluate(self, state):
        return SimpleNamespace(
            allowed=True,
            permitted=True,
            blocked=False,
            requires_approval=False,
            reason=None,
        )


# ============================================================================
# TEST AGENT EXECUTION
# ============================================================================


def build_real_agent_execution():
    tool_registry = MagicMock()
    tool_registry.get.return_value = FailingPythonTool()

    execution_policy = AllowExecutionPolicy()

    llm_executor = MagicMock()
    llm_executor.execute = AsyncMock(
        return_value="4",
    )

    agent_execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=None,
        approval_policy=None,
        llm_executor=llm_executor,
    )

    return (
        agent_execution,
        tool_registry,
        llm_executor,
    )


# ============================================================================
# PLANNER
# ============================================================================


def build_planner(
    initial_result: PlanningResult,
    replanned_result: PlanningResult,
):
    planner = MagicMock()

    planner.generate_plan = AsyncMock(
        return_value=initial_result,
    )

    planner.replan = AsyncMock(
        return_value=replanned_result,
    )

    def strategy_family(step):
        if step.step_type is StepType.LLM:
            return "LLM"

        if step.tool_name == "python_interpreter":
            return "PYTHON"

        return "UNKNOWN"

    planner.strategy_family.side_effect = strategy_family

    return planner


# ============================================================================
# CONTEXT BUILDER
# ============================================================================


def build_context_builder():
    context_builder = MagicMock()

    context_builder.build = AsyncMock(
        return_value=MagicMock(
            name="FullE2EContext",
        )
    )

    return context_builder


# ============================================================================
# RELIABILITY
# ============================================================================


def build_reliability_engine():
    reliability_engine = MagicMock()

    async def handle_failure(
        error,
        *,
        attempt,
        max_attempts,
    ):
        return SimpleNamespace(
            action=ReliabilityAction.REPLAN,
            error=error,
        )

    reliability_engine.handle_failure = AsyncMock(
        side_effect=handle_failure,
    )

    return reliability_engine


# ============================================================================
# LOOP DETECTOR
# ============================================================================


def build_loop_detector():
    loop_detector = MagicMock()

    loop_detector.check.return_value = SimpleNamespace(
        detected=False,
        reason=None,
    )

    loop_detector.record = MagicMock()

    return loop_detector


# ============================================================================
# VERIFIER
# ============================================================================


def build_verifier():
    verifier = MagicMock()

    verifier.verify = AsyncMock(
        return_value=SimpleNamespace(
            status=VerificationStatus.VERIFIED,
            reason=None,
        )
    )

    return verifier


# ============================================================================
# TEST
# ============================================================================


@pytest.mark.asyncio
async def test_full_agent_e2e_recovery_from_tool_failure_to_completion():
    task_analysis = make_task_analysis()

    initial_plan = make_initial_plan()
    replanned_plan = make_replanned_plan()

    initial_result = PlanningResult(
        plan=initial_plan,
        task_analysis=task_analysis,
    )

    replanned_result = PlanningResult(
        plan=replanned_plan,
        task_analysis=task_analysis,
    )

    planner = build_planner(
        initial_result,
        replanned_result,
    )

    context_builder = build_context_builder()

    (
        agent_execution,
        tool_registry,
        llm_executor,
    ) = build_real_agent_execution()

    reliability_engine = build_reliability_engine()
    loop_detector = build_loop_detector()
    verifier = build_verifier()

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
    )

    termination_policy = TerminationPolicy(
        max_iterations=23,
        max_verification_attempts=2,
    )

    agent_loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=termination_policy,
    )

    state = AgentState(
        user_request="Calculate 2 + 2",
    )

    result = await agent_loop.run_agent(state)

    assert result is state
    assert agent_loop.state is None
    assert agent_loop.run_context is None

    assert state.phase is AgentPhase.COMPLETED
    assert state.task_completed is True
    assert state.final_answer == "4"
    assert state.final_answer_ready is True
    assert state.final_answer_verified is True
    assert state.fatal_error is False

    planner.generate_plan.assert_awaited_once()
    planner.replan.assert_awaited_once()

    assert state.replan_count == 1

    assert state.plan is not None
    assert len(state.plan) == 1

    final_step = state.plan[0]
    assert final_step.step_id == 0
    assert final_step.step_type is StepType.LLM
    assert final_step.is_final_answer is True
    assert final_step.tool_name is None
    assert final_step.arguments == {}

    tool_registry.get.assert_called_once_with(
        "python_interpreter",
    )

    assert llm_executor.execute.await_count == 1

    llm_call = llm_executor.execute.await_args
    assert llm_call is not None
    assert state.final_answer == "4"

    reliability_engine.handle_failure.assert_awaited_once()

    reliability_call = (
        reliability_engine.handle_failure.await_args
    )

    assert reliability_call is not None

    # `error` is passed as a keyword argument by the Orchestrator.
    failure_error = reliability_call.kwargs["error"]

    assert failure_error is not None

    assert failure_error.recoverable is True

    assert failure_error.category.value == (
        "tool_execution_error"
    )

    # The failure happened on the first execution attempt.
    assert reliability_call.kwargs["attempt"] == 1

    # The Orchestrator delegated the execution retry/recovery budget
    # to ReliabilityEngine.
    assert reliability_call.kwargs["max_attempts"] == 3

    verifier.verify.assert_awaited_once()
    assert state.verification_attempts == 1
    assert state.final_answer_verified is True

    transitions = state.transition_history

    phases = [
        transition.to_phase
        for transition in transitions
    ]

    reasons = [
        transition.reason
        for transition in transitions
    ]

    assert AgentPhase.PLANNING in phases
    assert AgentPhase.EXECUTING in phases
    assert AgentPhase.VERIFYING in phases
    assert AgentPhase.COMPLETED in phases

    assert (
        TransitionReason.RECOVERY
        in reasons
    )
    assert (
        TransitionReason.EXECUTION_COMPLETED
        in reasons
    )
    assert (
        TransitionReason.VERIFICATION_PASSED
        in reasons
    )

    assert (
        transitions[-1].to_phase
        is AgentPhase.COMPLETED
    )
    assert (
        transitions[-1].reason
        is TransitionReason.VERIFICATION_PASSED
    )

    assert state.phase is AgentPhase.COMPLETED
    assert state.task_completed is True
    assert state.final_answer_ready is True
    assert state.final_answer_verified is True
    assert state.replan_count == 1
    assert state.current_step == 0
    assert state.verification_attempts == 1

    assert agent_loop.state is None
    assert agent_loop.run_context is None