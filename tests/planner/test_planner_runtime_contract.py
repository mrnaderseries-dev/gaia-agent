from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gaia_agent.context.models import FinalContext
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.planner import Planner
from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolSpec,
)
from gaia_agent.reliability.errors import AgentError


def make_tool_specs() -> dict[str, ToolSpec]:
    return {
        "web_search": ToolSpec(
            name="web_search",
            description="Search the web for factual information.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
        ),
        "visit_webpage": ToolSpec(
            name="visit_webpage",
            description="Visit a webpage URL.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to visit",
                    },
                },
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
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            capability=ToolCapability.COMPUTATION,
        ),
        "file_reader": ToolSpec(
            name="file_reader",
            description="Read a local file.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
        ),
        "analyze_image": ToolSpec(
            name="analyze_image",
            description="Analyze an image.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image",
                    },
                    "question": {
                        "type": "string",
                        "description": "Question about the image",
                    },
                },
                "required": ["image_path", "question"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
        ),
        "analyze_excel": ToolSpec(
            name="analyze_excel",
            description="Analyze an Excel workbook.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the workbook",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
        ),
    }


def make_planner(
    client: AsyncMock,
    tools: dict[str, ToolSpec] | None = None,
) -> Planner:
    return Planner(
        client=client,
        model="test-model",
        available_tools=tools or make_tool_specs(),
    )


def make_tool_step(
    step_id: int,
    *,
    tool_name: str = "web_search",
    arguments: dict | None = None,
) -> PlanStep:
    if arguments is None:
        arguments = {"query": "GAIA benchmark"}

    return PlanStep(
        step_id=step_id,
        action=f"Use {tool_name}",
        step_type=StepType.TOOL,
        tool_name=tool_name,
        arguments=arguments,
        is_final_answer=False,
    )


def make_final_step(step_id: int) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        action="Produce the final answer",
        step_type=StepType.LLM,
        tool_name=None,
        arguments={},
        is_final_answer=True,
    )


@pytest.mark.asyncio
async def test_planner_accepts_valid_llm_plan() -> None:
    client = AsyncMock()

    valid_plan = PlanSchema(
        steps=[
            make_tool_step(
                0,
                tool_name="web_search",
                arguments={
                    "query": "GAIA benchmark official website",
                },
            ),
            make_final_step(1),
        ]
    )

    client.generate.return_value = valid_plan

    planner = make_planner(client)

    result = await planner.create_plan(
        "What is the official website of the GAIA benchmark?"
    )

    assert isinstance(result, PlanSchema)
    assert len(result.steps) == 2

    assert result.steps[0].step_id == 0
    assert result.steps[0].step_type == StepType.TOOL
    assert result.steps[0].tool_name == "web_search"
    assert result.steps[0].arguments == {
        "query": "GAIA benchmark official website",
    }

    assert result.steps[1].step_id == 1
    assert result.steps[1].step_type == StepType.LLM
    assert result.steps[1].is_final_answer is True

    client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_recovers_from_plan_missing_final_answer() -> None:
    client = AsyncMock()

    malformed_plan = PlanSchema.model_construct(
        steps=[
            make_tool_step(
                0,
                tool_name="web_search",
                arguments={
                    "query": "GAIA benchmark",
                },
            )
        ]
    )

    client.generate.return_value = malformed_plan

    planner = make_planner(client)

    result = await planner.create_plan(
        "Find information about the GAIA benchmark."
    )

    assert isinstance(result, PlanSchema)

    finals = [
        step
        for step in result.steps
        if step.is_final_answer
    ]

    assert len(finals) == 1
    assert finals[0] == result.steps[-1]
    assert finals[0].step_type == StepType.LLM
    assert result.steps[-1].is_final_answer is True


@pytest.mark.asyncio
async def test_planner_rejects_unknown_tool_and_uses_fallback() -> None:
    client = AsyncMock()

    malformed_plan = PlanSchema.model_construct(
        steps=[
            PlanStep(
                step_id=0,
                action="Use an unavailable tool",
                step_type=StepType.TOOL,
                tool_name="does_not_exist",
                arguments={},
                is_final_answer=False,
            ),
            make_final_step(1),
        ]
    )

    client.generate.return_value = malformed_plan

    planner = make_planner(client)

    result = await planner.create_plan(
        "Find information about the GAIA benchmark."
    )

    assert isinstance(result, PlanSchema)

    assert all(
        step.tool_name != "does_not_exist"
        for step in result.steps
        if step.step_type == StepType.TOOL
    )

    assert result.steps[-1].is_final_answer is True


@pytest.mark.asyncio
async def test_planner_preserves_valid_tool_arguments() -> None:
    client = AsyncMock()

    valid_plan = PlanSchema(
        steps=[
            make_tool_step(
                0,
                tool_name="web_search",
                arguments={
                    "query": (
                        "highest bird species simultaneously "
                        "on camera"
                    ),
                },
            ),
            make_final_step(1),
        ]
    )

    client.generate.return_value = valid_plan

    planner = make_planner(client)

    result = await planner.create_plan(
        "Find the highest number of bird species simultaneously "
        "on camera."
    )

    assert result.steps[0].tool_name == "web_search"

    assert result.steps[0].arguments == {
        "query": (
            "highest bird species simultaneously "
            "on camera"
        ),
    }

    assert result.steps[-1].is_final_answer is True


@pytest.mark.asyncio
async def test_planner_recovery_produces_different_valid_strategy() -> None:
    client = AsyncMock()

    failed_step = make_tool_step(
        0,
        tool_name="web_search",
        arguments={
            "query": "GAIA benchmark",
        },
    )

    recovery_plan = PlanSchema(
        steps=[
            make_tool_step(
                0,
                tool_name="visit_webpage",
                arguments={
                    "url": "https://example.com/gaia",
                },
            ),
            make_final_step(1),
        ]
    )

    client.generate.return_value = recovery_plan

    planner = make_planner(client)

    failure = AgentError(
        error_type="tool_execution",
        message=(
            "web_search failed to retrieve useful evidence."
        ),
    )

    result = await planner.replan(
        user_question="What is the GAIA benchmark?",
        context=FinalContext(
            items=[],
            token_count=0,
        ),
        failed_step=failed_step,
        failure=failure,
    )

    assert isinstance(result, PlanSchema)

    assert len(result.steps) == 2
    assert [step.step_id for step in result.steps] == [0, 1]

    final_steps = [
        step
        for step in result.steps
        if step.is_final_answer
    ]

    assert len(final_steps) == 1
    assert result.steps[-1].is_final_answer is True

    assert result.steps[0].tool_name == "visit_webpage"

    failed_signature = planner._fingerprint_step(
        failed_step
    )

    recovery_signature = planner._fingerprint_step(
        result.steps[0]
    )

    assert recovery_signature != failed_signature

    client.generate.assert_awaited_once()