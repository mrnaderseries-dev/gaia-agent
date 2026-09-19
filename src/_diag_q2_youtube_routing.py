"""Evidence probe (READ-ONLY, run BEFORE the fix): GAIA Q2/Q7 YouTube routing.

Q2: "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the
highest number of bird species to be on camera simultaneously?"
Q7: "Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec. ...
     What does Teal'c say in response to the question \"Isn't that hot?\""

Both must be answered from the VIDEO. Transcript retrieval now exists
(youtube_transcript, Problem 13), so the planner must route a YouTube URL to
it. No Ollama generation call is made here (the model plan is injected), so
this probe only measures the deterministic routing/validation layers.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.config import settings
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.llm.service import LLMService
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.planner import Planner
from gaia_agent.planner.semantic_validator import (
    SemanticPlanError,
    SemanticPlanValidator,
)
from gaia_agent.planner.strategy_selector import StrategyContext
from gaia_agent.planner.task_classifier import TaskClassifier
from gaia_agent.tools.registry import ToolRegistry

Q2 = (
    "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the "
    "highest number of bird species to be on camera simultaneously?"
)

TEXT_MODEL = LLMModel(
    provider="ollama",
    model="qwen2.5:3b",
    max_tokens=1024,
    temperature=0.2,
)


class StaticLLM:
    """Injects the model plan (the typical ungrounded web_search plan)."""

    def __init__(self, plan: PlanSchema) -> None:
        self.plan = plan
        self.calls = 0

    async def generate(self, messages, *, model, output_schema, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.plan


def web_search_only_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Search for the bird species count in that video",
                step_type=StepType.TOOL,
                tool_name="web_search",
                arguments={"query": "video highest number of bird species"},
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="Synthesize the final answer using only the evidence obtained.",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )


def show(plan: PlanSchema) -> str:
    return " | ".join(
        f"#{step.step_id}:{step.step_type.value}:{step.tool_name or '-'}"
        f"{'(final)' if step.is_final_answer else ''}"
        for step in plan.steps
    )


async def main() -> None:
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=settings.ollama_timeout,
    )
    registry = ToolRegistry(
        base_dir=".",
        llm_service=LLMService(client=client, model=TEXT_MODEL),
    )
    tools = {spec.name: spec for spec in registry.get_tool_specs()}

    print("REGISTERED TOOLS:", sorted(tools), flush=True)

    classifier = TaskClassifier()
    analysis = classifier.classify(
        Q2,
        available_files=[],
        available_tools=sorted(tools),
    )

    print("\n== CLASSIFICATION (Q2) ==", flush=True)
    print("intent:", analysis.intent.value, flush=True)
    print("recommended_first_tool:", analysis.recommended_first_tool, flush=True)
    print("forbidden_tools:", analysis.forbidden_tools, flush=True)
    print("analysis_text:", analysis.analysis_text, flush=True)

    planner_for_strategy = Planner(
        client=StaticLLM(web_search_only_plan()),
        model=TEXT_MODEL,
        available_tools=tools,
        available_files=[],
    )
    strategy = planner_for_strategy.strategy_selector.select(
        analysis,
        StrategyContext(available_tools=frozenset(tools), available_files=()),
    )

    print("\n== STRATEGY ==", flush=True)
    print("family:", strategy.strategy.value, flush=True)
    print("primary_tool:", strategy.primary_tool, flush=True)
    print("deterministic:", strategy.deterministic, flush=True)

    print("\n== SEMANTIC VALIDATION of a web_search-only plan ==", flush=True)
    try:
        SemanticPlanValidator().validate(
            web_search_only_plan(),
            analysis=analysis,
            strategy=strategy,
        )
        print("RESULT: ACCEPTED (video never inspected)", flush=True)
    except SemanticPlanError as exc:
        print("RESULT: REJECTED:", exc, flush=True)

    print("\n== PLANNER: which plan is actually installed? ==", flush=True)
    result = await planner_for_strategy.create_plan(Q2, context=None)
    print("installed:", show(result.plan), flush=True)

    print("\n== DETERMINISTIC EMERGENCY FALLBACK ==", flush=True)
    fallback = planner_for_strategy._emergency_fallback_plan(
        user_question=Q2,
        analysis=analysis,
    )
    print("fallback:", show(fallback), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
