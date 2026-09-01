from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.reliability.errors import AgentError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_orchestrator() -> Orchestrator:
    """
    Build an Orchestrator without bootstrapping the complete runtime.

    These tests target deterministic P0.3 recovery/orchestration
    invariants rather than the complete application runtime.
    """
    orchestrator = object.__new__(Orchestrator)
    orchestrator.metrics = Mock()
    orchestrator.execution_history = set()
    return orchestrator


def make_state() -> AgentState:
    return AgentState(
        user_id="test-user",
        user_request="Find Malko",
    )


def make_step(
    tool_name: str = "web_search",
    query: str = "Malko",
    *,
    is_final_answer: bool = False,
) -> PlanStep:
    return PlanStep(
        step_id=0,
        action=f"Run {tool_name}",
        step_type=StepType.TOOL,
        tool_name=tool_name,
        arguments={"query": query},
        is_final_answer=is_final_answer,
    )


def make_plan(
    plan_id: str,
    tool_name: str = "web_search",
    query: str = "Malko",
) -> PlanSchema:
    return PlanSchema(
        plan_id=plan_id,
        steps=[
            make_step(
                tool_name,
                query,
            )
        ],
    )


def make_recovery_result(
    *,
    success: bool,
    result=None,
    error: AgentError | None = None,
    reason: str = "",
):
    """
    Small test double matching the attributes consumed by
    Orchestrator._handle_execution_recovery().
    """
    return SimpleNamespace(
        success=success,
        result=result,
        error=error,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# P0.3 — Recoverable failure -> alternative execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recoverable_failure_replans_to_different_tool():
    """
    P0.3 critical path:

        web_search("Malko")
              |
              | failure
              v
        recovery
              |
              v
        python("Malko")
              |
              v
        new execution accepted

    The recovery result must actually replace the failed step.
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    alternative = make_step(
        "python",
        "Malko",
    )

    state.plan = [original]
    state.current_step = 0
    state.tool_error = "web_search failed"

    # _handle_execution_recovery() calls these methods after
    # receiving a successful recovery result.
    orchestrator._prepare_step = Mock()

    result = make_recovery_result(
        success=True,
        result=alternative,
        reason="Changed strategy from web_search to python.",
    )

    await orchestrator._handle_execution_recovery(result)

    current = state.plan[state.current_step]

    assert current.tool_name == "python"
    assert current.arguments == {
        "query": "Malko",
    }

    assert current.tool_name != original.tool_name

    orchestrator._prepare_step.assert_called_once_with(
        alternative,
    )

    assert state.tool_error is None


# ---------------------------------------------------------------------------
# P0.3 — Same execution must NEVER be accepted as recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_same_execution_is_rejected_by_execution_identity():
    """
    Critical regression:

        web_search("Malko")
              |
              | failure
              v
        recovery proposes
        web_search("Malko")
              |
              v
        MUST NOT be treated as a new execution.

    This protects against the exact loop that appeared in the GAIA logs.
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    duplicate = make_step(
        "web_search",
        "Malko",
    )

    state.plan = [original]
    state.current_step = 0

    assert orchestrator._same_execution(
        original,
        duplicate,
    ) is True

    if hasattr(orchestrator, "_is_new_execution"):
        assert orchestrator._is_new_execution(
            duplicate,
        ) is False


# ---------------------------------------------------------------------------
# P0.3 — Different arguments are a genuinely new execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_same_tool_but_different_arguments_is_new_execution():
    """
    Same tool does not automatically mean same execution.

        web_search("Malko")
              |
              v
        web_search("Malko Lebanon")

    is a different execution because arguments changed.
    """

    orchestrator = make_test_orchestrator()

    original = make_step(
        "web_search",
        "Malko",
    )

    alternative = make_step(
        "web_search",
        "Malko Lebanon",
    )

    assert orchestrator._same_execution(
        original,
        alternative,
    ) is False

    if hasattr(orchestrator, "_is_new_execution"):
        orchestrator.state = make_state()
        orchestrator.state.plan = [original]
        orchestrator.state.current_step = 0

        assert orchestrator._is_new_execution(
            alternative,
        ) is True


# ---------------------------------------------------------------------------
# P0.3 — Recovery must fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_recovery_does_not_modify_failed_plan():
    """
    If Recovery itself fails:

        failed A
           |
           v
        Recovery FAILS
           |
           v
        original failed step remains

    We must not silently replace the plan with an invalid result.
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    state.plan = [original]
    state.current_step = 0

    state.tool_error = "original execution failed"

    orchestrator._emit_agent_failure = Mock()

    result = make_recovery_result(
        success=False,
        result=None,
        reason="Recovery budget exhausted.",
    )

    await orchestrator._handle_execution_recovery(result)

    current = state.plan[state.current_step]

    assert current.tool_name == "web_search"
    assert current.arguments == {
        "query": "Malko",
    }

    assert state.tool_error == "Recovery budget exhausted."

    orchestrator._emit_agent_failure.assert_called_once()


# ---------------------------------------------------------------------------
# P0.3 — Invalid recovery output must fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_recovery_result_does_not_modify_plan():
    """
    Recovery succeeded mechanically but returned something other
    than PlanStep.

    The orchestrator must reject it rather than executing it.
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    state.plan = [original]
    state.current_step = 0

    orchestrator.error_handler = Mock()
    orchestrator.error_handler.handle.return_value = AgentError(
        error_type="TypeError",
        message="Recovery did not produce a PlanStep.",
        source="orchestrator",
        operation="execution_recovery",
        recoverable=False,
        original_exception=TypeError(
            "Recovery did not produce a PlanStep."
        ),
    )

    orchestrator._emit_agent_failure = Mock()

    result = make_recovery_result(
        success=True,
        result={
            "tool": "python",
            "query": "Malko",
        },
    )

    await orchestrator._handle_execution_recovery(result)

    current = state.plan[state.current_step]

    assert current.tool_name == "web_search"

    assert state.tool_error == (
        "Recovery did not produce a PlanStep."
    )

    orchestrator.metrics.increment.assert_called_with(
        "recovery_failures",
    )

    orchestrator._emit_agent_failure.assert_called_once()


# ---------------------------------------------------------------------------
# P0.3 — Recovery -> replacement -> execution state reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_recovery_resets_execution_error_state():
    """
    A successful replan must clear stale execution state.

        failure
           |
           v
        recovery
           |
           v
        new step
           |
           v
        old error cleared
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    alternative = make_step(
        "python",
        "Malko",
    )

    state.plan = [original]
    state.current_step = 0

    state.tool_error = "403 Forbidden"
    state.tool_result = "stale result"
    state.recovery_attempted = True

    orchestrator._prepare_step = Mock()

    result = make_recovery_result(
        success=True,
        result=alternative,
        reason="Alternative strategy selected.",
    )

    await orchestrator._handle_execution_recovery(result)

    assert state.plan[state.current_step].tool_name == "python"

    assert state.tool_error is None
    assert state.recovery_attempted is False

    orchestrator._prepare_step.assert_called_once_with(
        alternative,
    )


# ---------------------------------------------------------------------------
# P0.3 — Same execution after replacement remains detectable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_execution_after_replacement_still_cannot_repeat():
    """
    Strong regression test:

        A = web_search("Malko")
        B = python("Malko")
        A again = web_search("Malko")

    Once B has replaced the failed step, the original A must
    still be recognized as an already-failed execution if the
    orchestration identity/history says so.

    At minimum, the immediate execution identity contract must
    remain correct:

        B != A
        A == A
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    step_a = make_step(
        "web_search",
        "Malko",
    )

    step_b = make_step(
        "python",
        "Malko",
    )

    state.plan = [step_b]
    state.current_step = 0

    assert orchestrator._same_execution(
        step_b,
        step_a,
    ) is False

    assert orchestrator._same_execution(
        step_a,
        step_a,
    ) is True


# ---------------------------------------------------------------------------
# P0.3 — Full planner replan must reject repeated plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_replan_rejects_repeated_plan():
    """
    Critical P0.3 loop guard:

        failed plan A
            |
            v
        planner.replan()
            |
            v
        same plan A
            |
            v
        reject

    The orchestrator must not blindly apply a repeated plan.
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_plan(
        "plan-1",
        "web_search",
        "Malko",
    )

    state.plan = list(original.steps)
    state.current_step = 0

    orchestrator.context_builder = Mock()
    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(
            items=[],
        )
    )

    orchestrator.planner = Mock()
    orchestrator.planner.replan = AsyncMock(
        return_value=original,
    )

    orchestrator.loop_detector = Mock()

    orchestrator.loop_detector.check_plan.return_value = (
        SimpleNamespace(
            detected=True,
            message="Repeated plan detected.",
        )
    )

    with pytest.raises(ValueError, match="repeated plan"):
        await orchestrator._replan_full_plan(
            AgentError(
                error_type="ToolExecutionError",
                message="web_search failed",
                retryable=False,
                recoverable=True,
                source="test",
                operation="web_search",
                original_exception=RuntimeError(
                    "web_search failed"
                ),
            )
        )

    orchestrator.planner.replan.assert_awaited_once()


# ---------------------------------------------------------------------------
# P0.3 — New plan must actually replace old plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_replan_with_new_strategy_replaces_plan():
    """
    Positive counterpart:

        failed A
           |
           v
        planner
           |
           v
        B
           |
           v
        B becomes active plan
    """

    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_plan(
        "plan-1",
        "web_search",
        "Malko",
    )

    alternative = make_plan(
        "plan-2",
        "python",
        "Malko",
    )

    state.plan = list(original.steps)
    state.current_step = 0

    orchestrator.context_builder = Mock()
    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(
            items=[],
        )
    )

    orchestrator.planner = Mock()
    orchestrator.planner.replan = AsyncMock(
        return_value=alternative,
    )

    orchestrator.loop_detector = Mock()

    orchestrator.loop_detector.check_plan.return_value = (
        SimpleNamespace(
            detected=False,
            message="No repeated plan.",
        )
    )

    orchestrator._validate_plan = Mock()

    orchestrator._apply_full_plan = Mock()

    result = await orchestrator._replan_full_plan(
        AgentError(
            error_type="ToolExecutionError",
            message="web_search failed",
            retryable=False,
            recoverable=True,
            source="test",
            operation="web_search",
            original_exception=RuntimeError(
                "web_search failed"
            ),
        )
    )

    assert result is alternative

    orchestrator.planner.replan.assert_awaited_once()

    orchestrator._apply_full_plan.assert_called_once_with(
        alternative,
    )