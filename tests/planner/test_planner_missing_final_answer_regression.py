"""Regression: model structured output with NO final-answer step.

Live official GAIA Q1 evidence (run ``evaluation_runs/official_20260918``,
qwen2.5:3b via Ollama ``format=json_schema``): the planner LLM returned a
schema-valid payload whose steps were ALL tool steps (no final-answer step):

    {"steps": [
      {"step_id": 1, "action": "web_search", "step_type": "tool", ...},
      {"step_id": 2, "action": "visit_webpage", "step_type": "tool", ...},
      {"step_id": 3, "action": "analyze_excel", "step_type": "tool", ...}]}

``OllamaClient._parse_structured_output`` therefore raised
``LLMOutputError: Ollama structured output failed schema validation.`` with
``PlanSchema -> steps -> ValueError: Plan must contain exactly one
final-answer step.``

These tests lock the contract for that exact payload:

1. ``PlanSchema`` still rejects a tool-only plan (the invariant is NOT
   weakened and is not re-implemented here).
2. The raw-payload repair never invents the missing final-answer step.
3. ``create_plan`` still returns a valid, executable plan through the
   existing bounded deterministic fallback: exactly one final-answer step,
   last, LLM, ``tool_name=None``, ``arguments={}``, sequential ids, and no
   duplicate steps.
4. Planning stays bounded: one model call per ``create_plan`` (no hidden
   retry/replanning loop when the model output is semantically invalid).
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from gaia_agent.planner.plan_schema import PlanSchema, StepType
from gaia_agent.planner.planner import Planner
from gaia_agent.planner.task_classifier import TaskClassifier, TaskIntent
from gaia_agent.planner.tool_spec import ToolCapability, ToolSpec
from gaia_agent.reliability.exception import LLMOutputError

_Q1_QUESTION = (
    "How many studio albums were published by Mercedes Sosa between 2000 "
    "and 2009 (included)? You can use the latest 2022 version of english "
    "wikipedia."
)

# Verbatim model output captured from the live official-Q1 reproduction.
LIVE_Q1_TOOL_ONLY_PAYLOAD = json.dumps(
    {
        "steps": [
            {
                "step_id": 1,
                "action": "web_search",
                "step_type": "tool",
                "tool_name": "web_search",
                "arguments": {"query": "Mercedes Sosa albums 2000-2009"},
            },
            {
                "step_id": 2,
                "action": "visit_webpage",
                "step_type": "tool",
                "tool_name": "visit_webpage",
                "arguments": {
                    "url": "https://en.wikipedia.org/wiki/Mercedes_Sosa"
                },
            },
            {
                "step_id": 3,
                "action": "analyze_excel",
                "step_type": "tool",
                "tool_name": "analyze_excel",
                "arguments": {
                    "file_path": "Mercedes_Sosa_albums_2000-2009_data.xlsx",
                    "question": _Q1_QUESTION,
                },
            },
        ]
    },
    indent=2,
)

_TOOLS = {
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
    "visit_webpage": ToolSpec(
        name="visit_webpage",
        description="Visit one exact URL.",
        arguments_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        capability=ToolCapability.NETWORK_READ,
    ),
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
}


class SchemaFailingLLM:
    """Mirrors ``OllamaClient`` for a schema-invalid structured plan.

    The raw payload is parsed and validated exactly like
    ``OllamaClient._parse_structured_output`` does, so a rejected plan
    surfaces as the same ``LLMOutputError`` that carried ``raw_content`` in
    production.
    """

    def __init__(self, raw_content: str) -> None:
        self.raw_content = raw_content
        self.calls = 0

    async def generate(
        self,
        messages: Any,
        *,
        model: Any,
        output_schema: type[PlanSchema],
        **kwargs: Any,
    ) -> PlanSchema:
        self.calls += 1

        try:
            return output_schema.model_validate(json.loads(self.raw_content))
        except ValidationError as exc:
            raise LLMOutputError(
                "Ollama structured output failed schema validation.",
                raw_content=self.raw_content,
            ) from exc


def make_planner(llm: SchemaFailingLLM) -> Planner:
    return Planner(
        client=llm,
        model="test-model",
        available_tools=_TOOLS,
        available_files=[],
    )


def classify(planner: Planner) -> Any:
    return TaskClassifier().classify(
        _Q1_QUESTION,
        available_files=[],
        available_tools=sorted(_TOOLS),
    )
def final_steps(plan: PlanSchema) -> list[Any]:
    return [step for step in plan.steps if step.is_final_answer]


def assert_contract_holds(plan: PlanSchema) -> None:
    """The invariant PlanSchema enforces, asserted on the returned plan."""

    assert isinstance(plan, PlanSchema)
    assert [step.step_id for step in plan.steps] == list(
        range(len(plan.steps))
    )

    finals = final_steps(plan)

    assert len(finals) == 1, "exactly one final-answer step must exist"

    final = finals[0]

    assert final is plan.steps[-1], "the final-answer step must be last"
    assert final.step_type == StepType.LLM
    assert final.tool_name is None
    assert final.arguments == {}
    assert final.action.strip()

    fingerprints = [
        (step.tool_name, json.dumps(step.arguments, sort_keys=True))
        for step in plan.steps
        if step.step_type == StepType.TOOL
    ]
    assert len(fingerprints) == len(set(fingerprints))


def test_plan_schema_still_rejects_the_live_tool_only_payload() -> None:
    """The invariant stays intact: this payload must never be accepted."""

    with pytest.raises(ValidationError) as excinfo:
        PlanSchema.model_validate(json.loads(LIVE_Q1_TOOL_ONLY_PAYLOAD))

    assert "exactly one final-answer step" in str(excinfo.value)


def test_raw_payload_repair_never_invents_the_final_answer_step() -> None:
    """Repair is lossless: it must NOT fabricate a missing step."""

    planner = make_planner(SchemaFailingLLM(LIVE_Q1_TOOL_ONLY_PAYLOAD))

    assert (
        planner._repair_and_validate_plan(
            LIVE_Q1_TOOL_ONLY_PAYLOAD,
            analysis=classify(planner),
        )
        is None
    )


@pytest.mark.asyncio
async def test_create_plan_recovers_with_exactly_one_final_answer_step() -> None:
    """The planner contract: a valid plan is always returned."""

    llm = SchemaFailingLLM(LIVE_Q1_TOOL_ONLY_PAYLOAD)
    planner = make_planner(llm)

    result = await planner.create_plan(_Q1_QUESTION, context=None)

    assert result.task_analysis.intent is TaskIntent.FACTUAL_SEARCH

    assert_contract_holds(result.plan)

    # Deterministic strategy fallback for FACTUAL_SEARCH with web_search
    # available: one evidence step + the single final-answer step.
    tool_steps = [
        step.tool_name
        for step in result.plan.steps
        if step.step_type == StepType.TOOL
    ]
    assert tool_steps == ["web_search"]

    # exactly one model call was made; the plan came from bounded recovery
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_planning_stays_bounded_when_the_model_output_is_invalid() -> None:
    """No hidden retry loop: exactly one model call per create_plan."""

    llm = SchemaFailingLLM(LIVE_Q1_TOOL_ONLY_PAYLOAD)
    planner = make_planner(llm)

    first = await planner.create_plan(_Q1_QUESTION, context=None)
    second = await planner.create_plan(_Q1_QUESTION, context=None)

    assert llm.calls == 2

    assert_contract_holds(first.plan)
    assert_contract_holds(second.plan)