"""Regression tests: strategy selection must not break plan installation.

Session-8 evidence: an uncommitted "strategy enforcement" block in
``SemanticPlanValidator.validate`` rejected every plan whose deterministic
primary tool (``python_interpreter``) was absent. For ARITHMETIC /
TEXT_TRANSFORMATION questions without a mechanically extractable expression
(GAIA Q3 reversed sentence, GAIA Q6 operation table, arithmetic word
problems) neither the model plan nor the planner's own emergency fallback can
contain ``python_interpreter``, so ``create_plan`` raised ``SemanticPlanError``
and NO plan was installed at all -- a hard planning failure instead of the
existing controlled degradation (LLM-only step, answer refused by the
verifier).

These tests lock the shipped contract (no per-question hacks):

1. ``create_plan`` always returns an installable PlanSchema for these tasks.
2. The deterministic strategy tool is used whenever the architecture can
   actually synthesize it (extractable expression).
3. Tool-vs-intent consistency (the validator's real job) still holds: an
   LLM-only plan is accepted, a forbidden-tool plan is still rejected.
"""
from __future__ import annotations

from typing import Any

import pytest

from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.planner import Planner
from gaia_agent.planner.semantic_validator import (
    SemanticPlanError,
    SemanticPlanValidator,
)
from gaia_agent.planner.strategy_selector import StrategyContext
from gaia_agent.planner.task_classifier import TaskClassifier, TaskIntent
from gaia_agent.planner.tool_spec import ToolCapability, ToolSpec

_Q3_TEXT = (
    '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,'
    "ecnetnes siht dnatsrednu uoy fI"
)

_Q6_TEXT = (
    "Given this table defining * on the set S = {a, b, c, d, e}\n\n"
    "|*|a|b|c|d|e|\n|---|---|---|---|---|\n"
    "|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n"
    "|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\n"
    "provide the subset of S involved in any possible counter-examples that "
    "prove * is not commutative. Provide your answer as a comma separated "
    "list of the elements in the set in alphabetical order."
)

_WORD_PROBLEM = (
    "A bakery sold 120 croissants at 2 dollars each and 45 cakes at 7 "
    "dollars each. What is the total revenue?"
)

_TOOLS = {
    "python_interpreter": ToolSpec(
        name="python_interpreter",
        description="Execute Python code.",
        arguments_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        capability=ToolCapability.COMPUTATION,
    ),
    "web_search": ToolSpec(
        name="web_search",
        description="Search the web.",
        arguments_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        capability=ToolCapability.NETWORK_READ,
    ),
}


class StaticLLM:
    """Returns the plan a small local model actually produced (LLM-only)."""

    def __init__(self, plan: PlanSchema) -> None:
        self.plan = plan

    async def generate(self, messages: Any, *, model: Any, output_schema: Any):
        return self.plan


def llm_only_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Reason about the task and answer",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            )
        ]
    )


def web_search_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Search the web",
                step_type=StepType.TOOL,
                tool_name="web_search",
                arguments={"query": "test"},
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="Answer",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


def make_planner(plan: PlanSchema) -> Planner:
    return Planner(
        client=StaticLLM(plan),
        model="test-model",
        available_tools=_TOOLS,
        available_files=[],
    )


def tool_steps(plan: PlanSchema) -> list[str]:
    return [
        step.tool_name or ""
        for step in plan.steps
        if step.step_type == StepType.TOOL and not step.is_final_answer
    ]


def select_strategy(planner: Planner, question: str):
    analysis = TaskClassifier().classify(
        question,
        available_files=[],
        available_tools=sorted(_TOOLS),
    )
    strategy = planner.strategy_selector.select(
        analysis,
        StrategyContext(
            available_tools=frozenset(_TOOLS),
            available_files=(),
        ),
    )

    return analysis, strategy


@pytest.mark.asyncio
async def test_reversed_text_question_still_installs_a_plan() -> None:
    planner = make_planner(llm_only_plan())

    result = await planner.create_plan(_Q3_TEXT)

    assert isinstance(result.plan, PlanSchema)
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].is_final_answer
    assert "web_search" not in tool_steps(result.plan)


@pytest.mark.asyncio
async def test_operation_table_question_still_installs_a_plan() -> None:
    planner = make_planner(llm_only_plan())

    result = await planner.create_plan(_Q6_TEXT)

    assert isinstance(result.plan, PlanSchema)
    assert result.plan.steps[-1].is_final_answer
    assert "web_search" not in tool_steps(result.plan)


@pytest.mark.asyncio
async def test_arithmetic_word_problem_still_installs_a_plan() -> None:
    planner = make_planner(llm_only_plan())

    result = await planner.create_plan(_WORD_PROBLEM)

    assert isinstance(result.plan, PlanSchema)
    assert result.plan.steps[-1].is_final_answer


@pytest.mark.asyncio
async def test_extractable_expression_uses_the_deterministic_tool() -> None:
    planner = make_planner(llm_only_plan())

    result = await planner.create_plan("Calculate 2 + 2.")

    assert tool_steps(result.plan) == ["python_interpreter"]
    assert result.plan.steps[0].arguments["code"].strip() == "result = 2 + 2"


def test_semantic_validator_accepts_tool_less_plan_for_deterministic_strategy() -> None:
    planner = make_planner(llm_only_plan())
    analysis, strategy = select_strategy(planner, _Q3_TEXT)

    assert analysis.intent is TaskIntent.TEXT_TRANSFORMATION
    assert strategy.deterministic
    assert strategy.primary_tool == "python_interpreter"

    SemanticPlanValidator().validate(
        llm_only_plan(),
        analysis=analysis,
        strategy=strategy,
    )


def test_semantic_validator_still_rejects_a_forbidden_tool() -> None:
    planner = make_planner(llm_only_plan())
    analysis, strategy = select_strategy(planner, _Q3_TEXT)

    with pytest.raises(SemanticPlanError):
        SemanticPlanValidator().validate(
            web_search_plan(),
            analysis=analysis,
            strategy=strategy,
        )
