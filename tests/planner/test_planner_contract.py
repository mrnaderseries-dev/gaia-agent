from __future__ import annotations

from gaia_agent.planner.plan_schema import PlanStep, StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_step(
    step_id: int,
    tool_name: str = "web_search",
    arguments: dict | None = None,
) -> PlanStep:
    """Create a valid TOOL plan step."""
    return PlanStep(
        step_id=step_id,
        action="tool",
        step_type=StepType.TOOL,
        tool_name=tool_name,
        arguments=arguments or {},
        is_final_answer=False,
    )


def make_final_step(
    step_id: int,
) -> PlanStep:
    """Create a valid final-answer LLM step."""
    return PlanStep(
        step_id=step_id,
        action="llm",
        step_type=StepType.LLM,
        tool_name=None,
        arguments={},
        is_final_answer=True,
    )


def execution_signature(step: PlanStep) -> tuple:
    """
    Return a hashable signature identifying the execution.

    step_id is intentionally excluded because recovery may assign
    a different ID to the same execution.

    arguments is converted from dict to a sorted tuple so the
    complete signature can safely be used inside a set.
    """

    arguments_signature = tuple(
        sorted(
            step.arguments.items(),
            key=lambda item: item[0],
        )
    )

    return (
        step.action,
        step.step_type,
        step.tool_name,
        arguments_signature,
        step.is_final_answer,
    )


def assert_valid_plan(plan: list[PlanStep]) -> None:
    """
    Validate the basic Planner contract.

    A valid plan must:
    - contain at least one step
    - use sequential IDs starting from 0
    - contain exactly one final-answer step
    - have no duplicate IDs
    """
    assert plan, "Plan must not be empty."

    ids = [step.step_id for step in plan]

    assert ids == list(range(len(plan))), (
        "Step IDs must be sequential starting from 0. "
        f"Got: {ids}"
    )

    assert len(ids) == len(set(ids)), (
        "Plan contains duplicate step IDs: "
        f"{ids}"
    )

    final_steps = [
        step
        for step in plan
        if step.is_final_answer
    ]

    assert len(final_steps) == 1, (
        "Plan must contain exactly one final-answer step. "
        f"Found {len(final_steps)}."
    )

    final_step = final_steps[0]

    assert final_step.step_type == StepType.LLM
    assert final_step.action == "llm"
    assert final_step.tool_name is None


# ---------------------------------------------------------------------------
# Basic PlanStep construction
# ---------------------------------------------------------------------------


def test_tool_step_uses_tool_step_type() -> None:
    step = make_tool_step(0)

    assert step.step_type == StepType.TOOL
    assert step.action == "tool"
    assert step.tool_name == "web_search"
    assert step.is_final_answer is False


def test_final_step_uses_llm_step_type() -> None:
    step = make_final_step(0)

    assert step.step_type == StepType.LLM
    assert step.action == "llm"
    assert step.tool_name is None
    assert step.is_final_answer is True


# ---------------------------------------------------------------------------
# Step ID contract
# ---------------------------------------------------------------------------


def test_plan_step_ids_start_from_zero() -> None:
    plan = [
        make_tool_step(0),
        make_final_step(1),
    ]

    assert_valid_plan(plan)


def test_plan_step_ids_must_be_sequential() -> None:
    plan = [
        make_tool_step(0),
        make_tool_step(2),
        make_final_step(3),
    ]

    try:
        assert_valid_plan(plan)
    except AssertionError as exc:
        assert "sequential" in str(exc)
    else:
        raise AssertionError(
            "A plan with non-sequential step IDs was accepted."
        )


def test_plan_step_ids_cannot_start_from_one() -> None:
    plan = [
        make_tool_step(1),
        make_final_step(2),
    ]

    try:
        assert_valid_plan(plan)
    except AssertionError as exc:
        assert "sequential" in str(exc)
    else:
        raise AssertionError(
            "A plan whose IDs start from 1 was accepted."
        )


def test_plan_step_ids_cannot_contain_duplicates() -> None:
    plan = [
        make_tool_step(0),
        make_tool_step(1),
        make_final_step(1),
    ]

    try:
        assert_valid_plan(plan)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "A plan containing duplicate step IDs was accepted."
        )


# ---------------------------------------------------------------------------
# Final-answer contract
# ---------------------------------------------------------------------------


def test_plan_must_have_exactly_one_final_answer_step() -> None:
    plan = [
        make_tool_step(0),
        make_final_step(1),
    ]

    assert_valid_plan(plan)


def test_plan_without_final_answer_is_invalid() -> None:
    plan = [
        make_tool_step(0),
        make_tool_step(
            1,
            tool_name="python_interpreter",
        ),
    ]

    try:
        assert_valid_plan(plan)
    except AssertionError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError(
            "A plan without a final-answer step was accepted."
        )


def test_plan_with_two_final_answers_is_invalid() -> None:
    plan = [
        make_tool_step(0),
        make_final_step(1),
        make_final_step(2),
    ]

    try:
        assert_valid_plan(plan)
    except AssertionError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError(
            "A plan with two final-answer steps was accepted."
        )


def test_final_answer_step_is_last_step() -> None:
    plan = [
        make_tool_step(0),
        make_final_step(1),
    ]

    assert_valid_plan(plan)

    final_indexes = [
        index
        for index, step in enumerate(plan)
        if step.is_final_answer
    ]

    assert final_indexes == [len(plan) - 1]


# ---------------------------------------------------------------------------
# Tool contract
# ---------------------------------------------------------------------------


def test_tool_step_has_tool_name() -> None:
    step = make_tool_step(
        0,
        tool_name="web_search",
    )

    assert step.step_type == StepType.TOOL
    assert step.action == "tool"
    assert step.tool_name


def test_llm_final_step_has_no_tool_name() -> None:
    step = make_final_step(0)

    assert step.step_type == StepType.LLM
    assert step.action == "llm"
    assert step.tool_name is None


# ---------------------------------------------------------------------------
# Tool availability contract
# ---------------------------------------------------------------------------


def test_plan_tool_names_must_exist_in_available_tools() -> None:
    available_tools = {
        "web_search",
        "python_interpreter",
        "file_reader",
        "analyze_image",
    }

    plan = [
        make_tool_step(
            0,
            tool_name="web_search",
        ),
        make_final_step(1),
    ]

    for step in plan:
        if step.step_type == StepType.TOOL:
            assert step.tool_name in available_tools


def test_unavailable_tool_is_rejected() -> None:
    available_tools = {
        "web_search",
        "python_interpreter",
        "file_reader",
        "analyze_image",
    }

    plan = [
        make_tool_step(
            0,
            tool_name="analyze_text",
        ),
        make_final_step(1),
    ]

    for step in plan:
        if step.step_type == StepType.TOOL:
            try:
                assert step.tool_name in available_tools, (
                    f"Unknown or unavailable tool: {step.tool_name}"
                )
            except AssertionError as exc:
                assert "Unknown or unavailable tool" in str(exc)
                return

    raise AssertionError(
        "The unavailable tool analyze_text was incorrectly accepted."
    )


# ---------------------------------------------------------------------------
# Recovery contract
# ---------------------------------------------------------------------------


def test_recovery_plan_must_not_repeat_failed_tool() -> None:
    """
    Valid recovery case.

    The original failed execution uses web_search.
    The recovery uses python_interpreter instead.
    """

    failed_step = make_tool_step(
        0,
        tool_name="web_search",
    )

    recovery_plan = [
        make_tool_step(
            0,
            tool_name="python_interpreter",
        ),
        make_final_step(1),
    ]

    failed_signature = execution_signature(failed_step)

    recovery_signatures = {
        execution_signature(step)
        for step in recovery_plan
        if not step.is_final_answer
    }

    assert failed_signature not in recovery_signatures

    assert_valid_plan(recovery_plan)


def test_recovery_plan_with_different_tool_is_allowed() -> None:
    failed_step = make_tool_step(
        0,
        tool_name="web_search",
    )

    recovery_step = make_tool_step(
        0,
        tool_name="file_reader",
    )

    assert execution_signature(failed_step) != execution_signature(
        recovery_step
    )


def test_regression_repeated_failed_execution_is_rejected() -> None:
    """
    Regression test for:

        PlannerRecoveryRequired:
        Replanned plan repeats the failed execution.

    This test intentionally creates a BROKEN recovery plan.

    Therefore the assertion inside pytest.raises MUST fail.
    """

    failed_step = make_tool_step(
        0,
        tool_name="web_search",
    )

    repeated_plan = [
        make_tool_step(
            0,
            tool_name="web_search",
        ),
        make_final_step(1),
    ]

    failed_signature = execution_signature(failed_step)

    repeated_signatures = {
        execution_signature(step)
        for step in repeated_plan
        if not step.is_final_answer
    }

    # The failed signature IS present.
    # Therefore this assertion MUST raise AssertionError.
    try:
        assert failed_signature not in repeated_signatures
    except AssertionError:
        return

    raise AssertionError(
        "A recovery plan that repeats the failed execution "
        "was incorrectly accepted."
    )


# ---------------------------------------------------------------------------
# Regression tests for previously observed Planner failures
# ---------------------------------------------------------------------------


def test_regression_step_ids_one_and_two_are_invalid() -> None:
    """
    Previous failure:

        Step IDs must be sequential starting from 0.

    Broken output:

        1, 2

    Correct output:

        0, 1
    """

    broken_plan = [
        make_tool_step(1),
        make_final_step(2),
    ]

    try:
        assert_valid_plan(broken_plan)
    except AssertionError as exc:
        assert "sequential" in str(exc)
    else:
        raise AssertionError(
            "Planner accepted step IDs starting from 1."
        )


def test_regression_analyze_text_is_not_available() -> None:
    """
    Previous failure:

        Unknown or unavailable tool: analyze_text
    """

    available_tools = {
        "web_search",
        "python_interpreter",
        "file_reader",
        "analyze_image",
    }

    broken_step = make_tool_step(
        0,
        tool_name="analyze_text",
    )

    assert broken_step.tool_name not in available_tools


def test_regression_missing_final_answer_is_rejected() -> None:
    """
    Previous failure:

        Plan must contain exactly one final-answer step.
    """

    broken_plan = [
        make_tool_step(
            0,
            tool_name="web_search",
        ),
        make_tool_step(
            1,
            tool_name="python_interpreter",
        ),
    ]

    try:
        assert_valid_plan(broken_plan)
    except AssertionError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError(
            "Planner accepted a plan without a final-answer step."
        )


# ---------------------------------------------------------------------------
# Valid plan examples
# ---------------------------------------------------------------------------


def test_simple_tool_then_final_answer_plan_is_valid() -> None:
    plan = [
        make_tool_step(
            0,
            tool_name="web_search",
        ),
        make_final_step(1),
    ]

    assert_valid_plan(plan)


def test_multi_tool_plan_is_valid() -> None:
    plan = [
        make_tool_step(
            0,
            tool_name="web_search",
        ),
        make_tool_step(
            1,
            tool_name="python_interpreter",
        ),
        make_tool_step(
            2,
            tool_name="file_reader",
        ),
        make_final_step(3),
    ]

    assert_valid_plan(plan)


def test_tool_arguments_are_supported() -> None:
    step = make_tool_step(
        0,
        tool_name="web_search",
        arguments={
            "query": "GAIA benchmark",
        },
    )

    assert step.arguments == {
        "query": "GAIA benchmark",
    }


# ---------------------------------------------------------------------------
# Complete Planner contract
# ---------------------------------------------------------------------------


def test_complete_planner_contract() -> None:
    """
    Minimum complete plan contract.

    The expected structure is:

        TOOL
          ↓
        TOOL
          ↓
        LLM FINAL ANSWER
    """

    available_tools = {
        "web_search",
        "python_interpreter",
        "file_reader",
        "analyze_image",
    }

    plan = [
        make_tool_step(
            0,
            tool_name="web_search",
        ),
        make_tool_step(
            1,
            tool_name="python_interpreter",
        ),
        make_final_step(2),
    ]

    assert_valid_plan(plan)

    for step in plan:
        if step.step_type == StepType.TOOL:
            assert step.tool_name in available_tools
            assert step.is_final_answer is False

        elif step.step_type == StepType.LLM:
            assert step.is_final_answer is True
            assert step.tool_name is None

        else:
            raise AssertionError(
                f"Unsupported step type: {step.step_type}"
            )