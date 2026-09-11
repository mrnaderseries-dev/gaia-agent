from __future__ import annotations

import pytest

from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator
from gaia_agent.context.attachments import Attachment
from gaia_agent.context.models import ContextRequest, FinalContext
from gaia_agent.context.request_builder import ContextRequestBuilder
from gaia_agent.context.sources.attachments import AttachmentSource
from gaia_agent.core.agent_state import AgentState
from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.planner.planner import Planner


class DisabledSource:
    async def get(self, request: ContextRequest) -> list[object]:
        return []

    def is_available(self, request: ContextRequest) -> bool:
        return False


class PassthroughCompressor:
    async def compress(self, context: list[object]) -> list[object]:
        return context


class CapturingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, messages, **kwargs) -> str:
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, dict)
            )

        self.prompts.append(prompt)

        return """
        {
            "steps": [
                {
                    "step_id": 0,
                    "action": "Answer using the attachment",
                    "step_type": "llm",
                    "is_final_answer": true
                }
            ]
        }
        """


def make_context_builder() -> ContextBuilder:
    budget = ContextBudget(max_tokens=4096)

    return ContextBuilder(
        policy=ContextPolicy(
            include_attachments=True,
            include_conversation=False,
            include_history=False,
            include_memory=False,
            include_runtime=False,
        ),
        budget=budget,
        validator=ContextValidator(budget),
        compressor=PassthroughCompressor(),
        attachment_source=AttachmentSource(),
        conversation_source=DisabledSource(),
        history_source=DisabledSource(),
        memory_source=DisabledSource(),
        runtime_source=DisabledSource(),
    )


def make_state(*attachments: Attachment) -> AgentState:
    return AgentState(
        user_request="Answer using the attached files.",
        attachments=attachments,
        plan=[
            PlanStep(
                step_id=0,
                action="Answer using the attachment",
                step_type=StepType.LLM,
                is_final_answer=True,
            )
        ],
    )


def make_planner(
    capturing_llm: CapturingLLM,
    *,
    available_files: list[str] | None = None,
) -> Planner:
    return Planner(
        client=capturing_llm,
        model="test-model",
        available_tools={},
        available_files=available_files or [],
    )


@pytest.mark.asyncio
async def test_attachment_flows_from_agent_state_to_planner() -> None:
    attachment = Attachment(
        attachment_id="attachment-847291",
        filename="financial_report.txt",
        path="C:/attachments/financial_report.txt",
    )

    state = make_state(attachment)

    request = ContextRequestBuilder.from_state(state)

    assert request.attachments == (attachment,)
    assert request.attachments[0] is attachment

    builder = make_context_builder()
    final_context = await builder.build(request)

    assert isinstance(final_context, FinalContext)
    assert len(final_context.items) == 1
    assert final_context.items[0] is attachment

    llm = CapturingLLM()
    planner = make_planner(llm)

    await planner.create_plan(
        state.user_request,
        final_context,
    )

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert "attachment-847291" in prompt
    assert "financial_report.txt" in prompt
    assert "C:/attachments/financial_report.txt" in prompt


@pytest.mark.asyncio
async def test_attachment_identity_survives_context_pipeline() -> None:
    attachment = Attachment(
        attachment_id="attachment-identity",
        filename="identity.txt",
        path="C:/attachments/identity.txt",
    )

    state = make_state(attachment)

    request = ContextRequestBuilder.from_state(state)
    final_context = await make_context_builder().build(request)

    assert request.attachments[0] is attachment
    assert final_context.items[0] is attachment


@pytest.mark.asyncio
async def test_multiple_attachments_preserve_order_and_identity() -> None:
    first = Attachment(
        attachment_id="attachment-1",
        filename="first.txt",
        path="C:/attachments/first.txt",
    )

    second = Attachment(
        attachment_id="attachment-2",
        filename="second.txt",
        path="C:/attachments/second.txt",
    )

    third = Attachment(
        attachment_id="attachment-3",
        filename="third.txt",
        path="C:/attachments/third.txt",
    )

    state = make_state(first, second, third)

    request = ContextRequestBuilder.from_state(state)
    final_context = await make_context_builder().build(request)

    assert request.attachments == (first, second, third)

    assert final_context.items == [first, second, third]

    assert final_context.items[0] is first
    assert final_context.items[1] is second
    assert final_context.items[2] is third


@pytest.mark.asyncio
async def test_no_attachment_does_not_create_attachment_context() -> None:
    state = make_state()

    request = ContextRequestBuilder.from_state(state)

    assert request.attachments == ()

    assert not request.attachments


@pytest.mark.asyncio
async def test_planner_receives_attachment_reference_not_agent_state() -> None:
    attachment = Attachment(
        attachment_id="attachment-reference",
        filename="report.txt",
        path="C:/attachments/report.txt",
    )

    state = make_state(attachment)

    request = ContextRequestBuilder.from_state(state)
    final_context = await make_context_builder().build(request)

    llm = CapturingLLM()
    planner = make_planner(llm)

    await planner.create_plan(
        state.user_request,
        final_context,
    )

    prompt = llm.prompts[0]

    assert "report.txt" in prompt
    assert "C:/attachments/report.txt" in prompt
    assert "attachment-reference" in prompt

    assert "AgentState" not in prompt
    assert "ContextRequest" not in prompt


@pytest.mark.asyncio
async def test_planner_receives_final_context_not_raw_attachment_sequence() -> None:
    attachment = Attachment(
        attachment_id="attachment-contract",
        filename="contract.txt",
        path="C:/attachments/contract.txt",
    )

    llm = CapturingLLM()
    planner = make_planner(llm)

    with pytest.raises((AttributeError, TypeError)):
        await planner.create_plan(
            "Answer using the attachment.",
            [attachment],
        )