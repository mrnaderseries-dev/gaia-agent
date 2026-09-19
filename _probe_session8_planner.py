"""Session-8 planner probe: effect of the uncommitted SemanticPlanValidator
strategy-enforcement change on the plans the Planner actually installs.

Real Planner, real TaskClassifier, real StrategySelector, real PlanSchema,
real fallback planners. ONLY the Ollama client is an AsyncMock (no LLM cost),
returning the plan a real model produced for these questions in Session-6
(LLM-only final-answer step, because structured output kept failing).

Case A: enforcement ACTIVE (current working tree).
Case B: enforcement DISABLED (validator stub) -> shows the behaviour that
        existed before the uncommitted change.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType  # noqa: E402
from gaia_agent.planner.planner import Planner  # noqa: E402
from gaia_agent.planner.tool_spec import ToolCapability, ToolSpec  # noqa: E402

TOOLS = {
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

QUESTIONS = {
    "Q3 reversed sentence (TEXT_TRANSFORMATION)": (
        '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
    ),
    "Q6 operation table (ARITHMETIC)": (
        "Given this table defining * on the set S = {a, b, c, d, e}\n\n"
        "|*|a|b|c|d|e|\n|---|---|---|---|---|\n"
        "|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n"
        "|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\n"
        "provide the subset of S involved in any possible counter-examples "
        "that prove * is not commutative. Provide your answer as a comma "
        "separated list of the elements in the set in alphabetical order."
    ),
    "arithmetic word problem (ARITHMETIC, no extractable expression)": (
        "A bakery sold 120 croissants at 2 dollars each and 45 cakes at 7 "
        "dollars each. What is the total revenue?"
    ),
    "default arithmetic (ARITHMETIC, extractable)": "Calculate 2 + 2.",
}


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


class _NoEnforcement:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, *args, **kwargs) -> None:
        self.calls += 1


async def run_case(enforcement: bool, label: str, question: str) -> None:
    client = AsyncMock()
    client.generate.return_value = llm_only_plan()
    planner = Planner(
        client=client,
        model="probe-model",
        available_tools=TOOLS,
        available_files=[],
    )
    if not enforcement:
        planner.semantic_validator = _NoEnforcement()

    analysis = planner._classify(question)
    try:
        result = await planner.create_plan(question)
        tools = [
            s.tool_name
            for s in result.plan.steps
            if s.step_type == StepType.TOOL and not s.is_final_answer
        ]
        print(
            f"[enforce={enforcement}] {label}\n"
            f"    intent={analysis.intent.value} "
            f"installed_tools={tools} "
            f"steps={[s.step_type.value for s in result.plan.steps]}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[enforce={enforcement}] {label}\n"
            f"    intent={analysis.intent.value} "
            f"RAISED {type(exc).__name__}: {exc}"
        )


async def main() -> None:
    for label, question in QUESTIONS.items():
        for enforcement in (True, False):
            await run_case(enforcement, label, question)
        print()


asyncio.run(main())
