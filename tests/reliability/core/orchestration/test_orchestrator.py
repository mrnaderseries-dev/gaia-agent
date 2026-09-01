from __future__ import annotations

import pytest

from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)


def make_plan(
    plan_id: str,
    tool_name: str = "web_search",
    query: str = "Malko",
) -> PlanSchema:
    return PlanSchema(
        plan_id=plan_id,
        steps=[
            PlanStep(
                step_id=0,
                action=f"Run {tool_name}",
                step_type=StepType.TOOL,
                tool_name=tool_name,
                arguments={"query": query},
                is_final_answer=False,
            )
        ],
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


def make_test_orchestrator() -> Orchestrator:
    """
    Create an Orchestrator without bootstrapping the complete runtime.

    These tests target deterministic orchestration invariants.
    """
    return object.__new__(Orchestrator)


def make_state() -> AgentState:
    return AgentState(
        user_id="test-user",
        user_request="Find Malko",
    )


# ---------------------------------------------------------------------------
# Existing execution-identity tests
# ---------------------------------------------------------------------------


def test_same_execution_logic():
    orchestrator = make_test_orchestrator()

    first = make_step(
        "web_search",
        "Malko",
    )

    second = make_step(
        "web_search",
        "Malko",
    )

    assert orchestrator._same_execution(
        first,
        second,
    ) is True


def test_new_replan_is_accepted():
    orchestrator = make_test_orchestrator()

    original = make_step(
        "web_search",
        "Malko",
    )

    replanned = make_step(
        "python",
        "Malko",
    )

    assert orchestrator._same_execution(
        original,
        replanned,
    ) is False


def test_same_replan_is_rejected():
    orchestrator = make_test_orchestrator()

    original = make_step(
        "web_search",
        "Malko",
    )

    replanned = make_step(
        "web_search",
        "Malko",
    )

    assert orchestrator._same_execution(
        original,
        replanned,
    ) is True


def test_same_execution_with_replanned_step_is_rejected():
    orchestrator = make_test_orchestrator()

    original = make_step(
        "web_search",
        "Malko",
    )

    replanned = make_step(
        "web_search",
        "Malko",
    )

    assert orchestrator._same_execution(
        original,
        replanned,
    ) is True


def test_different_tool_is_new_execution():
    orchestrator = make_test_orchestrator()

    original = make_step(
        "web_search",
        "Malko",
    )

    replanned = make_step(
        "python",
        "Malko",
    )

    assert orchestrator._same_execution(
        original,
        replanned,
    ) is False


def test_different_arguments_are_new_execution():
    orchestrator = make_test_orchestrator()

    original = make_step(
        "web_search",
        "Malko",
    )

    replanned = make_step(
        "web_search",
        "Malko Lebanon",
    )

    assert orchestrator._same_execution(
        original,
        replanned,
    ) is False


# ---------------------------------------------------------------------------
# P0.2 Integration & Regression Tests
# ---------------------------------------------------------------------------


def test_replace_failed_step_with_new_strategy():
    """
    P0.2:
        failed A
          ↓
        replan
          ↓
        B
          ↓
        failed step is actually replaced by B
    """
    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    new_step = make_step(
        "python",
        "Malko",
    )

    state.plan = [
        original,
    ]

    state.current_step = 0

    orchestrator._replace_failed_step(
        new_step
    )

    assert len(state.plan) == 1

    replaced = state.plan[0]

    assert replaced.tool_name == "python"
    assert replaced.arguments == {
        "query": "Malko"
    }

    assert replaced.tool_name != original.tool_name


def test_duplicate_replan_does_not_look_different_after_plan_replacement():
    """
    Ensures duplicate steps are accurately recognized as identical
    after considering execution semantics without modifying schemas.
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

    state.plan = [
        original,
    ]

    state.current_step = 0

    assert orchestrator._same_execution(
        state.plan[state.current_step],
        duplicate,
    ) is True


def test_new_strategy_can_replace_failed_execution():
    """
    Explicit A → B transition.

    A = web_search(Malko)
    B = python(Malko)

    B must become the current plan step.
    """
    orchestrator = make_test_orchestrator()

    state = make_state()
    orchestrator.state = state

    original = make_step(
        "web_search",
        "Malko",
    )

    new_step = make_step(
        "python",
        "Malko",
    )

    state.plan = [
        original,
    ]

    state.current_step = 0

    orchestrator._replace_failed_step(
        new_step
    )

    assert state.plan[state.current_step].tool_name == "python"


@pytest.mark.asyncio
async def test_replan_same_execution_must_stop():
    """
    Critical regression test:

    web_search("Malko")
        -> failure
        -> planner returns web_search("Malko")
        -> orchestrator must reject it
    """
    orchestrator = make_test_orchestrator()

    state = AgentState(
        user_id="test-user",
        user_request="Find Malko",
    )
    orchestrator.state = state

    original = make_plan(
        "plan-1",
        "web_search",
        "Malko",
    )

    replanned = make_plan(
        "plan-2",
        "web_search",
        "Malko",
    )

    state.plan = original
    state.current_step = 0

    assert orchestrator._same_execution(
        original.steps[0],
        replanned.steps[0],
    ) is True

    if hasattr(orchestrator, "_is_new_execution"):
        assert orchestrator._is_new_execution(
            replanned.steps[0],
        ) is False


@pytest.mark.asyncio
async def test_replan_different_tool_must_be_executable():
    """
    Regression test:

    web_search("Malko")
        -> failure
        -> planner changes strategy
        -> python(...)
        -> new execution must be accepted
    """
    orchestrator = make_test_orchestrator()

    state = AgentState(
        user_id="test-user",
        user_request="Find Malko",
    )
    orchestrator.state = state

    original = make_plan(
        "plan-1",
        "web_search",
        "Malko",
    )

    replanned = make_plan(
        "plan-2",
        "python",
        "Malko",
    )

    state.plan = original
    state.current_step = 0

    assert orchestrator._same_execution(
        original.steps[0],
        replanned.steps[0],
    ) is False

    if hasattr(orchestrator, "_is_new_execution"):
        assert orchestrator._is_new_execution(
            replanned.steps[0],
        ) is True