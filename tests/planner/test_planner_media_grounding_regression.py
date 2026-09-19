"""Regression: a video/audio (YouTube) task must be grounded in the media.

Documented GAIA failures (analysis.md): Q2 ``a1e91b78`` (highest number of
bird species on camera in ``watch?v=L1vXCYZAYYM``) and Q7 ``9d191bce``
(Teal'c's reply in ``watch?v=1htKBjuUWec``) failed with "irrelevant web
evidence, no video grounding".

After the transcript capability was added (Problem 13) the planner still
could not use it for these questions. Pre-fix runtime evidence
(``_diag_q2_youtube_routing.py``, Q2):

    intent: audio_video
    recommended_first_tool: None            (classifier: no media guidance)
    strategy: AUDIO_VIDEO / youtube_transcript / deterministic=True
    semantic validation of a web_search-only plan: ACCEPTED
    installed plan:      #0:tool:web_search | #1:llm(final)
    emergency fallback:  #0:tool:web_search | #1:llm(final)

i.e. the selector picked the media tool but nothing enforced it, the
classifier still told the planner to prefer web snippets, and the
deterministic fallback could not ground a video either.

These tests lock the fixed contract:
- the classifier recommends the registered media tool and forbids answering
  from snippets;
- validation rejects a plan that never inspects the media, while still
  allowing extra evidence steps;
- the deterministic fallback and create_plan produce
  ``youtube_transcript(video_url=<exact url>)`` + one final-answer step;
- non-media routing (web search, non-YouTube URLs) is unchanged.
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

Q2 = (
    "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the "
    "highest number of bird species to be on camera simultaneously?"
)
Q1 = (
    "How many studio albums were published by Mercedes Sosa between 2000 "
    "and 2009 (included)? You can use the latest 2022 version of english "
    "wikipedia."
)
ARTICLE_URL = "https://example.org/report"

_TOOLS = {
    "web_search": ToolSpec(
        name="web_search",
        description="Search the web.",
        arguments_schema={"type": "object", "properties": {}},
        capability=ToolCapability.NETWORK_READ,
    ),
    "visit_webpage": ToolSpec(
        name="visit_webpage",
        description="Visit one exact URL.",
        arguments_schema={"type": "object", "properties": {}},
        capability=ToolCapability.NETWORK_READ,
    ),
    "youtube_transcript": ToolSpec(
        name="youtube_transcript",
        description="Fetch a YouTube transcript.",
        arguments_schema={
            "type": "object",
            "properties": {"video_url": {"type": "string"}},
            "required": ["video_url"],
            "additionalProperties": False,
        },
        capability=ToolCapability.NETWORK_READ,
    ),
}

_TOOLS_WITHOUT_MEDIA = {
    name: spec
    for name, spec in _TOOLS.items()
    if name != "youtube_transcript"
}


class StaticLLM:
    """Injects the ungrounded plan a small model tends to produce."""

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
                action="Search for the video's bird species count",
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


def tool_step_plan(*tool_names: str) -> PlanSchema:
    steps = [
        PlanStep(
            step_id=index,
            action=f"Use {name}",
            step_type=StepType.TOOL,
            tool_name=name,
            arguments=(
                {"video_url": "https://www.youtube.com/watch?v=L1vXCYZAYYM"}
                if name == "youtube_transcript"
                else {"query": "x"}
            ),
            is_final_answer=False,
        )
        for index, name in enumerate(tool_names)
    ]
    steps.append(
        PlanStep(
            step_id=len(steps),
            action="Synthesize the final answer using only the evidence obtained.",
            step_type=StepType.LLM,
            tool_name=None,
            arguments={},
            is_final_answer=True,
        )
    )
    return PlanSchema(steps=steps)


def make_planner(tools: dict[str, ToolSpec], plan: PlanSchema) -> Planner:
    return Planner(
        client=StaticLLM(plan),
        model="test-model",
        available_tools=tools,
        available_files=[],
    )


def analyze(planner: Planner, question: str, tools: dict[str, ToolSpec]) -> Any:
    return TaskClassifier().classify(
        question,
        available_files=[],
        available_tools=sorted(tools),
    )


def strategy_for(planner: Planner, analysis: Any, tools: dict[str, ToolSpec]) -> Any:
    return planner.strategy_selector.select(
        analysis,
        StrategyContext(
            available_tools=frozenset(tools),
            available_files=(),
        ),
    )


# ---------------------------------------------------------------------------
# Classifier: the media capability is recommended for a YouTube question
# ---------------------------------------------------------------------------


def test_youtube_question_recommends_the_media_tool() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())
    analysis = analyze(planner, Q2, _TOOLS)
    strategy = strategy_for(planner, analysis, _TOOLS)

    assert analysis.intent is TaskIntent.AUDIO_VIDEO
    assert analysis.recommended_first_tool == "youtube_transcript"
    assert "youtube_transcript" in analysis.analysis_text
    assert "web snippet" in analysis.analysis_text

    assert strategy.primary_tool == "youtube_transcript"
    assert strategy.deterministic is True


def test_youtube_question_without_a_media_tool_keeps_the_web_fallback() -> None:
    planner = make_planner(_TOOLS_WITHOUT_MEDIA, web_search_only_plan())
    analysis = analyze(planner, Q2, _TOOLS_WITHOUT_MEDIA)
    strategy = strategy_for(planner, analysis, _TOOLS_WITHOUT_MEDIA)

    assert analysis.intent is TaskIntent.AUDIO_VIDEO
    assert analysis.recommended_first_tool is None
    assert "web search" in analysis.analysis_text
    assert strategy.primary_tool is None


# ---------------------------------------------------------------------------
# Validation: the media must actually be inspected
# ---------------------------------------------------------------------------


def test_spurious_web_search_plan_is_rejected_for_audio_video() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())
    analysis = analyze(planner, Q2, _TOOLS)
    strategy = strategy_for(planner, analysis, _TOOLS)

    with pytest.raises(SemanticPlanError) as excinfo:
        SemanticPlanValidator().validate(
            web_search_only_plan(),
            analysis=analysis,
            strategy=strategy,
        )

    assert "youtube_transcript" in str(excinfo.value)


def test_plan_that_inspects_the_media_is_accepted() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())
    analysis = analyze(planner, Q2, _TOOLS)
    strategy = strategy_for(planner, analysis, _TOOLS)

    validator = SemanticPlanValidator()

    validator.validate(
        tool_step_plan("youtube_transcript"),
        analysis=analysis,
        strategy=strategy,
    )

    # extra evidence steps remain legal: only the missing media capability
    # is rejected
    validator.validate(
        tool_step_plan("youtube_transcript", "web_search"),
        analysis=analysis,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Planner: deterministic recovery grounds the video
# ---------------------------------------------------------------------------


def test_emergency_fallback_grounds_the_video() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())
    analysis = analyze(planner, Q2, _TOOLS)

    fallback = planner._emergency_fallback_plan(
        user_question=Q2,
        analysis=analysis,
    )

    assert [step.tool_name for step in fallback.steps] == [
        "youtube_transcript",
        None,
    ]
    assert fallback.steps[0].arguments == {
        "video_url": "https://www.youtube.com/watch?v=L1vXCYZAYYM"
    }
    assert fallback.steps[-1].is_final_answer is True
    assert len([s for s in fallback.steps if s.is_final_answer]) == 1


@pytest.mark.asyncio
async def test_create_plan_replaces_an_ungrounded_model_plan() -> None:
    llm = StaticLLM(web_search_only_plan())
    planner = make_planner(_TOOLS, web_search_only_plan())
    planner.client = llm

    result = await planner.create_plan(Q2, context=None)

    assert result.task_analysis.intent is TaskIntent.AUDIO_VIDEO
    assert [step.tool_name for step in result.plan.steps] == [
        "youtube_transcript",
        None,
    ]
    assert result.plan.steps[-1].is_final_answer is True
    assert [step.step_id for step in result.plan.steps] == [0, 1]
    # The audio/video strategy is deterministic (deterministic=True), so
    # create_plan installs the grounded media plan WITHOUT consulting the
    # LLM at all: a zero-call run is the correct contract here. What matters
    # is that the installed plan is grounded in the transcript, which the
    # assertions above lock in.
    assert llm.calls == 0


def test_media_steps_have_their_own_recovery_family() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())

    step = tool_step_plan("youtube_transcript").steps[0]

    assert planner._strategy_family(step) == "AUDIO_VIDEO"


# ---------------------------------------------------------------------------
# Guards: non-media routing is unchanged
# ---------------------------------------------------------------------------


def test_factual_question_still_uses_web_search() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())
    analysis = analyze(planner, Q1, _TOOLS)

    assert analysis.intent is TaskIntent.FACTUAL_SEARCH

    fallback = planner._emergency_fallback_plan(
        user_question=Q1,
        analysis=analysis,
    )

    assert fallback.steps[0].tool_name == "web_search"


def test_non_youtube_url_still_uses_visit_webpage() -> None:
    planner = make_planner(_TOOLS, web_search_only_plan())
    question = f"Summarize the findings in {ARTICLE_URL}"
    analysis = analyze(planner, question, _TOOLS)

    assert analysis.intent is TaskIntent.URL_PAGE
    assert analysis.recommended_first_tool == "visit_webpage"

    fallback = planner._emergency_fallback_plan(
        user_question=question,
        analysis=analysis,
    )

    assert fallback.steps[0].tool_name == "visit_webpage"
    assert fallback.steps[0].arguments == {"url": ARTICLE_URL}
