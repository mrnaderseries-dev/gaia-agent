from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from gaia_agent.context.models import FinalContext
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.planner import Planner


@dataclass
class FakeTool:
    name: str


class CapturingLLM:
    def __init__(self, plan: PlanSchema) -> None:
        self.plan = plan
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages,
        *,
        model,
        output_schema,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "output_schema": output_schema,
            }
        )
        return self.plan

    @property
    def last_prompt(self) -> str:
        assert self.calls
        return self.calls[-1]["messages"][1]["content"]


def make_final_plan() -> PlanSchema:
    return PlanSchema(
        plan_id="context-contract-plan",
        steps=[
            PlanStep(
                step_id=0,
                action="Provide the final answer using the supplied evidence",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            )
        ],
    )


def make_planner(client: Any) -> Planner:
    return Planner(
        client=client,
        model="test-model",
        available_tools={},
        available_files=[],
    )


@pytest.mark.asyncio
async def test_planner_consumes_final_context_and_preserves_attachment_evidence():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[
            {
                "source_type": "attachment",
                "filename": "financial_report.txt",
                "content": "CONFIDENTIAL_REVENUE_VALUE_847291",
            },
        ],
        token_count=17,
    )

    plan = await planner.create_plan(
        user_question="What is the revenue reported in the attached file?",
        context=context,
    )

    assert isinstance(plan, PlanSchema)
    assert len(client.calls) == 1

    prompt = client.last_prompt

    assert "financial_report.txt" in prompt
    assert "CONFIDENTIAL_REVENUE_VALUE_847291" in prompt


@pytest.mark.asyncio
async def test_planner_uses_final_context_items_not_final_context_object_repr():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[
            {
                "source_type": "attachment",
                "filename": "evidence.txt",
                "content": "UNIQUE_EVIDENCE_TOKEN_991827",
            },
        ],
        token_count=999,
    )

    await planner.create_plan(
        user_question="Answer using the attached evidence.",
        context=context,
    )

    prompt = client.last_prompt

    assert "UNIQUE_EVIDENCE_TOKEN_991827" in prompt
    assert "FinalContext(" not in prompt


@pytest.mark.asyncio
async def test_planner_does_not_confuse_token_count_with_context_items():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[
            {
                "source_type": "attachment",
                "filename": "report.txt",
                "content": "METRIC_COUNT_TRAP_EVIDENCE_123456",
            },
        ],
        token_count=1,
    )

    await planner.create_plan(
        user_question="Use the attached report.",
        context=context,
    )

    prompt = client.last_prompt

    assert "METRIC_COUNT_TRAP_EVIDENCE_123456" in prompt
    assert "token_count" not in prompt.lower()


@pytest.mark.asyncio
async def test_planner_context_is_structurally_rich_not_flattened_before_formatting():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[
            {
                "source_type": "attachment",
                "filename": "dataset.csv",
                "metadata": {
                    "artifact_id": "artifact-991",
                    "origin": "user_attachment",
                },
                "content": "ROW_VALUE_12345",
            },
        ],
        token_count=42,
    )

    await planner.create_plan(
        user_question="Analyze the attached dataset.",
        context=context,
    )

    prompt = client.last_prompt

    assert "dataset.csv" in prompt
    assert "artifact-991" in prompt
    assert "user_attachment" in prompt
    assert "ROW_VALUE_12345" in prompt


@pytest.mark.asyncio
async def test_planner_empty_final_context_has_explicit_empty_context_behavior():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[],
        token_count=0,
    )

    await planner.create_plan(
        user_question="What information is available?",
        context=context,
    )

    prompt = client.last_prompt

    assert "No additional context." in prompt


@pytest.mark.asyncio
async def test_planner_none_context_and_empty_final_context_are_semantically_equivalent():
    client_none = CapturingLLM(make_final_plan())
    planner_none = make_planner(client_none)

    await planner_none.create_plan(
        user_question="What information is available?",
        context=None,
    )

    client_empty = CapturingLLM(make_final_plan())
    planner_empty = make_planner(client_empty)

    await planner_empty.create_plan(
        user_question="What information is available?",
        context=FinalContext(
            items=[],
            token_count=0,
        ),
    )

    assert "No additional context." in client_none.last_prompt
    assert "No additional context." in client_empty.last_prompt


@pytest.mark.asyncio
async def test_planner_receives_attachment_evidence_even_when_multiple_context_items_exist():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[
            {
                "source_type": "history",
                "content": "old irrelevant information",
            },
            {
                "source_type": "conversation",
                "content": "another irrelevant message",
            },
            {
                "source_type": "attachment",
                "filename": "important.pdf",
                "content": "CRITICAL_ATTACHMENT_FACT_555888",
            },
        ],
        token_count=100,
    )

    await planner.create_plan(
        user_question="What does the attached document say?",
        context=context,
    )

    prompt = client.last_prompt

    assert "CRITICAL_ATTACHMENT_FACT_555888" in prompt
    assert "important.pdf" in prompt


@pytest.mark.asyncio
async def test_planner_context_limit_is_applied_to_items_not_final_context_object():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    planner.MAX_CONTEXT_ITEMS = 2

    context = FinalContext(
        items=[
            {"content": "OLD_CONTEXT_1"},
            {"content": "OLD_CONTEXT_2"},
            {"content": "KEPT_CONTEXT_3"},
            {"content": "KEPT_CONTEXT_4"},
        ],
        token_count=500,
    )

    await planner.create_plan(
        user_question="Use the latest context.",
        context=context,
    )

    prompt = client.last_prompt

    assert "OLD_CONTEXT_1" not in prompt
    assert "OLD_CONTEXT_2" not in prompt
    assert "KEPT_CONTEXT_3" in prompt
    assert "KEPT_CONTEXT_4" in prompt


@pytest.mark.asyncio
async def test_planner_rejects_old_sequence_contract_after_context_migration():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    old_style_context = [
        {
            "source_type": "attachment",
            "filename": "legacy.txt",
            "content": "OLD_SEQUENCE_CONTEXT",
        }
    ]

    with pytest.raises((AttributeError, TypeError)):
        await planner.create_plan(
            user_question="Use the attachment.",
            context=old_style_context,
        )


@pytest.mark.asyncio
async def test_attachment_evidence_changes_the_information_available_to_planner():
    empty_client = CapturingLLM(make_final_plan())
    empty_planner = make_planner(empty_client)

    question = "What is the exact value written in the attached report?"

    await empty_planner.create_plan(
        user_question=question,
        context=FinalContext(
            items=[],
            token_count=0,
        ),
    )

    evidence_client = CapturingLLM(make_final_plan())
    evidence_planner = make_planner(evidence_client)

    await evidence_planner.create_plan(
        user_question=question,
        context=FinalContext(
            items=[
                {
                    "source_type": "attachment",
                    "filename": "report.txt",
                    "content": "EXACT_VALUE_774411",
                }
            ],
            token_count=10,
        ),
    )

    empty_prompt = empty_client.last_prompt
    evidence_prompt = evidence_client.last_prompt

    assert "EXACT_VALUE_774411" not in empty_prompt
    assert "EXACT_VALUE_774411" in evidence_prompt


@pytest.mark.asyncio
async def test_planner_consumes_items_and_does_not_serialize_final_context_metadata():
    client = CapturingLLM(make_final_plan())
    planner = make_planner(client)

    context = FinalContext(
        items=[
            {
                "source_type": "attachment",
                "filename": "report.txt",
                "content": "UNIQUE_EVIDENCE_VALUE_123456",
            },
        ],
        token_count=987654,
    )

    await planner.create_plan(
        user_question="Use the attached report.",
        context=context,
    )

    prompt = client.last_prompt

    assert "UNIQUE_EVIDENCE_VALUE_123456" in prompt
    assert "report.txt" in prompt

    assert "987654" not in prompt
    assert '"token_count": 987654' not in prompt