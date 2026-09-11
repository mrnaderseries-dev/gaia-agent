from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from gaia_agent.context.models import FinalContext
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.planner import (
    Planner,
    PlannerRecoveryRequired,
)
from gaia_agent.planner.semantic_validator import (
    SemanticPlanError,
    SemanticPlanValidator,
)
from gaia_agent.planner.strategy_selector import (
    StrategyContext,
    StrategyFamily,
    StrategySelector,
)
from gaia_agent.planner.task_classifier import (
    TaskClassifier,
    TaskIntent,
    detect_factorial_ratio,
    detect_simple_operation,
)
from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolErrorCode,
    ToolModality,
    ToolSpec,
)
from gaia_agent.reliability.errors import AgentError


def make_tools() -> dict[str, ToolSpec]:
    return {
        "web_search": ToolSpec(
            name="web_search",
            description="Search the web for factual information.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset({ToolModality.TEXT}),
            error_codes=frozenset(
                {
                    ToolErrorCode.NETWORK_ERROR,
                    ToolErrorCode.TIMEOUT,
                    ToolErrorCode.RATE_LIMITED,
                }
            ),
        ),
        "visit_webpage": ToolSpec(
            name="visit_webpage",
            description="Visit a webpage.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset({ToolModality.TEXT}),
        ),
        "python_interpreter": ToolSpec(
            name="python_interpreter",
            description="Execute deterministic Python code.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            capability=ToolCapability.COMPUTATION,
            modalities=frozenset({ToolModality.TEXT}),
        ),
        "file_reader": ToolSpec(
            name="file_reader",
            description="Read a local file.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
            modalities=frozenset(
                {
                    ToolModality.TEXT,
                    ToolModality.FILE,
                }
            ),
        ),
        "analyze_excel": ToolSpec(
            name="analyze_excel",
            description="Analyze an Excel workbook.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
            modalities=frozenset(
                {
                    ToolModality.FILE,
                    ToolModality.EXCEL,
                }
            ),
        ),
        "analyze_image": ToolSpec(
            name="analyze_image",
            description="Analyze an image.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                    },
                    "question": {
                        "type": "string",
                    },
                },
                "required": ["image_path", "question"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
            modalities=frozenset(
                {
                    ToolModality.IMAGE,
                    ToolModality.VISION,
                }
            ),
        ),
        "youtube_transcript": ToolSpec(
            name="youtube_transcript",
            description="Extract a transcript from a video.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            capability=ToolCapability.READ_ONLY,
            modalities=frozenset(
                {
                    ToolModality.AUDIO,
                    ToolModality.VIDEO,
                }
            ),
        ),
    }


def make_planner(
    client: AsyncMock,
    *,
    tools: dict[str, ToolSpec] | None = None,
    files: list[str] | None = None,
) -> Planner:
    return Planner(
        client=client,
        model="test-model",
        available_tools=tools or make_tools(),
        available_files=files or [],
    )


def tool_step(
    step_id: int,
    *,
    tool_name: str,
    arguments: dict | None = None,
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        action=f"Use {tool_name}",
        step_type=StepType.TOOL,
        tool_name=tool_name,
        arguments=arguments or {},
        is_final_answer=False,
    )


def final_step(step_id: int) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        action="Produce the final answer",
        step_type=StepType.LLM,
        arguments={},
        tool_name=None,
        is_final_answer=True,
    )


def analysis_for(
    question: str,
    *,
    tools: dict[str, ToolSpec] | None = None,
    files: list[str] | None = None,
):
    return TaskClassifier().classify(
        question,
        available_tools=list(
            (tools or make_tools()).keys()
        ),
        available_files=files or [],
    )


class TestTaskClassifierAdversarial:
    def test_arithmetic_forbids_web(self) -> None:
        analysis = analysis_for("Calculate 91 * 37.")

        assert analysis.intent == TaskIntent.ARITHMETIC
        assert analysis.needs_external_info is False
        assert "web_search" in analysis.forbidden_tools
        assert "visit_webpage" in analysis.forbidden_tools

    def test_text_transformation_forbids_web(self) -> None:
        analysis = analysis_for("Reverse the string abcdef.")

        assert analysis.intent == TaskIntent.TEXT_TRANSFORMATION
        assert analysis.needs_external_info is False
        assert "web_search" in analysis.forbidden_tools

    def test_reversed_text_is_detected(self) -> None:
        analysis = analysis_for("esrever this string")

        assert analysis.intent == TaskIntent.TEXT_TRANSFORMATION

    def test_factorial_ratio_detector(self) -> None:
        assert detect_factorial_ratio("What is 10! / 5!?") == (
            10,
            5,
        )

    def test_factorial_ratio_does_not_match_normal_text(self) -> None:
        assert detect_factorial_ratio(
            "What is the official website?"
        ) is None

    def test_simple_operation_detector(self) -> None:
        assert detect_simple_operation(
            "What is 12 x 7?"
        ) == "12 * 7"

    def test_division_by_zero_is_not_deterministic(self) -> None:
        assert detect_simple_operation(
            "What is 10 / 0?"
        ) is None

    def test_file_task_without_file_does_not_invent_path(self) -> None:
        analysis = analysis_for(
            "Read the attached report.",
            files=[],
        )

        assert analysis.intent == TaskIntent.LOCAL_FILE
        assert analysis.recommended_first_tool is None
        assert analysis.needs_external_info is True

    def test_excel_file_prefers_excel_tool(self) -> None:
        analysis = analysis_for(
            "Analyze the spreadsheet.",
            files=["financials.xlsx"],
        )

        assert analysis.intent == TaskIntent.LOCAL_FILE
        assert analysis.recommended_first_tool == "analyze_excel"

    def test_image_task_is_not_web_search(self) -> None:
        analysis = analysis_for(
            "What is shown in this image?"
        )

        assert analysis.intent == TaskIntent.IMAGE
        assert "web_search" not in (
            analysis.recommended_first_tool,
        )

    def test_video_task_is_media_intent(self) -> None:
        analysis = analysis_for(
            "Transcribe this video and extract the spoken content."
        )

        assert analysis.intent == TaskIntent.AUDIO_VIDEO

    def test_unknown_question_is_not_forced_into_web(self) -> None:
        analysis = analysis_for(
            "Explain the concept of abstraction."
        )

        assert analysis.intent in {
            TaskIntent.SELF_CONTAINED,
            TaskIntent.UNKNOWN,
        }


class TestStrategySelectorAdversarial:
    def test_arithmetic_selects_python(self) -> None:
        analysis = analysis_for(
            "Calculate 10 * 50."
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        assert decision.strategy == StrategyFamily.LOCAL_COMPUTATION
        assert decision.primary_tool == "python_interpreter"

    def test_arithmetic_without_python_keeps_family(self) -> None:
        tools = make_tools()
        tools.pop("python_interpreter")

        analysis = analysis_for(
            "Calculate 10 * 50.",
            tools=tools,
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(tools.keys())
            ),
        )

        assert decision.strategy == StrategyFamily.LOCAL_COMPUTATION
        assert decision.primary_tool is None

    def test_url_prefers_direct_webpage(self) -> None:
        analysis = analysis_for(
            "Open https://example.com and tell me the title."
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        assert decision.strategy == StrategyFamily.DIRECT_URL
        assert decision.primary_tool == "visit_webpage"

    def test_image_with_real_image_prefers_vision(self) -> None:
        analysis = analysis_for(
            "Analyze the image.",
            files=["board.png"],
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                ),
                available_files=("board.png",),
            ),
        )

        assert decision.strategy == StrategyFamily.VISION
        assert decision.primary_tool == "analyze_image"

    def test_image_without_image_does_not_fake_resolution(self) -> None:
        analysis = analysis_for(
            "Analyze the image."
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                ),
                available_files=(),
            ),
        )

        assert decision.strategy == StrategyFamily.VISION
        assert decision.primary_tool == "analyze_image"
        assert decision.deterministic is False

    def test_spreadsheet_prefers_excel(self) -> None:
        analysis = analysis_for(
            "Analyze this Excel file.",
            files=["sales.xlsx"],
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                ),
                available_files=("sales.xlsx",),
            ),
        )

        assert decision.strategy == StrategyFamily.FILE_ANALYSIS
        assert decision.primary_tool == "analyze_excel"

    def test_non_spreadsheet_file_prefers_reader(self) -> None:
        analysis = analysis_for(
            "Read this text file.",
            files=["notes.txt"],
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                ),
                available_files=("notes.txt",),
            ),
        )

        assert decision.strategy == StrategyFamily.FILE_READING
        assert decision.primary_tool == "file_reader"

    def test_media_never_falls_back_to_web_search(self) -> None:
        analysis = analysis_for(
            "Watch this YouTube video."
        )

        tools = make_tools()
        tools.pop("youtube_transcript")

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(tools.keys())
            ),
        )

        assert decision.strategy == StrategyFamily.AUDIO_VIDEO
        assert decision.primary_tool is None

    def test_media_uses_real_media_capability(self) -> None:
        analysis = analysis_for(
            "Watch this YouTube video."
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        assert decision.strategy == StrategyFamily.AUDIO_VIDEO
        assert decision.primary_tool == "youtube_transcript"

    @pytest.mark.parametrize(
        "failure_message",
        [
            "capability unavailable",
            "permission denied",
            "loop detected",
            "tool repeated",
            "tool forbidden",
            "tool unavailable",
            "not supported",
        ],
    )
    def test_non_transient_failure_changes_recovery_family(
        self,
        failure_message: str,
    ) -> None:
        analysis = analysis_for(
            "Find the official GAIA website."
        )

        decision = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                ),
                failed_strategy="WEB_SEARCH",
                failure_type=failure_message,
            ),
        )

        assert decision.strategy != StrategyFamily.WEB_RETRIEVAL


class TestToolSpecContracts:
    def test_tool_spec_is_frozen(self) -> None:
        spec = make_tools()["web_search"]

        with pytest.raises(ValidationError):
            spec.name = "changed"

    def test_tool_capability_is_exact(self) -> None:
        spec = make_tools()["python_interpreter"]

        assert spec.supports_capability(
            ToolCapability.COMPUTATION
        )
        assert not spec.supports_capability(
            ToolCapability.NETWORK_READ
        )

    def test_tool_modality_is_exact(self) -> None:
        spec = make_tools()["analyze_image"]

        assert spec.supports_modality(
            ToolModality.IMAGE
        )

        assert spec.supports_modality(
            ToolModality.VISION
        )

        assert not spec.supports_modality(
            ToolModality.AUDIO
        )

    def test_empty_tool_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSpec(
                name="",
                description="bad",
                arguments_schema={},
                capability=ToolCapability.READ_ONLY,
            )

    def test_empty_description_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSpec(
                name="bad",
                description="",
                arguments_schema={},
                capability=ToolCapability.READ_ONLY,
            )

    def test_tool_schema_can_be_strict(self) -> None:
        spec = make_tools()["web_search"]

        assert spec.arguments_schema["type"] == "object"
        assert spec.arguments_schema["required"] == ["query"]
        assert (
            spec.arguments_schema["additionalProperties"]
            is False
        )


class TestPlanSchemaAdversarial:
    def test_valid_multi_step_plan(self) -> None:
        plan = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments={"query": "GAIA"},
                ),
                tool_step(
                    1,
                    tool_name="visit_webpage",
                    arguments={
                        "url": "https://example.com"
                    },
                ),
                final_step(2),
            ]
        )

        assert len(plan.steps) == 3
        assert plan.steps[-1].is_final_answer

    def test_ids_must_be_sequential(self) -> None:
        with pytest.raises(ValidationError):
            PlanSchema(
                steps=[
                    tool_step(
                        0,
                        tool_name="web_search",
                        arguments={"query": "GAIA"},
                    ),
                    tool_step(
                        2,
                        tool_name="visit_webpage",
                        arguments={
                            "url": "https://example.com"
                        },
                    ),
                    final_step(3),
                ]
            )

    def test_final_answer_must_be_last(self) -> None:
        with pytest.raises(ValidationError):
            PlanSchema(
                steps=[
                    final_step(0),
                    tool_step(
                        1,
                        tool_name="web_search",
                        arguments={"query": "GAIA"},
                    ),
                ]
            )

    def test_final_answer_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            PlanSchema(
                steps=[
                    final_step(0),
                    final_step(1),
                ]
            )

    def test_tool_cannot_be_final_answer(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(
                step_id=0,
                action="Search",
                step_type=StepType.TOOL,
                tool_name="web_search",
                arguments={"query": "GAIA"},
                is_final_answer=True,
            )

    def test_tool_requires_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(
                step_id=0,
                action="Search",
                step_type=StepType.TOOL,
                tool_name=None,
                arguments={},
                is_final_answer=False,
            )

    def test_llm_cannot_have_arguments(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(
                step_id=0,
                action="Answer",
                step_type=StepType.LLM,
                arguments={"secret": "unexpected"},
                is_final_answer=True,
            )

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "\t",
            "\n",
        ],
    )
    def test_empty_action_is_rejected(
        self,
        value: str,
    ) -> None:
        with pytest.raises(ValidationError):
            PlanStep(
                step_id=0,
                action=value,
                step_type=StepType.LLM,
                is_final_answer=True,
            )

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            " ",
            "none",
            "null",
            "nil",
        ],
    )
    def test_tool_name_normalization(
        self,
        value: str | None,
    ) -> None:
        step = PlanStep(
            step_id=0,
            action="Answer",
            step_type=StepType.LLM,
            tool_name=value,
            arguments={},
            is_final_answer=True,
        )

        assert step.tool_name is None


class TestSemanticValidatorAdversarial:
    def test_arithmetic_cannot_use_web_search(self) -> None:
        analysis = analysis_for(
            "Calculate 25 * 4."
        )

        strategy = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        plan = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments={
                        "query": "25 * 4"
                    },
                ),
                final_step(1),
            ]
        )

        with pytest.raises(SemanticPlanError):
            SemanticPlanValidator().validate(
                plan,
                analysis=analysis,
                strategy=strategy,
            )

    def test_text_transform_cannot_use_web_search(
        self,
    ) -> None:
        analysis = analysis_for(
            "Reverse the string hello."
        )

        strategy = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        plan = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments={
                        "query": "reverse hello"
                    },
                ),
                final_step(1),
            ]
        )

        with pytest.raises(SemanticPlanError):
            SemanticPlanValidator().validate(
                plan,
                analysis=analysis,
                strategy=strategy,
            )

    def test_strategy_family_mismatch_is_detected(
        self,
    ) -> None:
        analysis = analysis_for(
            "Calculate 7 * 8."
        )

        strategy = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        plan = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments={"query": "7 * 8"},
                ),
                final_step(1),
            ]
        )

        with pytest.raises(SemanticPlanError):
            SemanticPlanValidator().validate(
                plan,
                analysis=analysis,
                strategy=strategy,
                strategy_family_resolver=(
                    lambda step: "WEB_SEARCH"
                ),
            )

    def test_valid_strategy_survives_semantic_validation(
        self,
    ) -> None:
        analysis = analysis_for(
            "Calculate 7 * 8."
        )

        strategy = StrategySelector().select(
            analysis,
            StrategyContext(
                available_tools=frozenset(
                    make_tools().keys()
                )
            ),
        )

        plan = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="python_interpreter",
                    arguments={
                        "code": "print(7 * 8)"
                    },
                ),
                final_step(1),
            ]
        )

        SemanticPlanValidator().validate(
            plan,
            analysis=analysis,
            strategy=strategy,
        )


class TestPlannerSubsystemAdversarial:
    @pytest.mark.asyncio
    async def test_planner_does_not_allow_llm_to_escape_strategy(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.return_value = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments={
                        "query": "7 * 8"
                    },
                ),
                final_step(1),
            ]
        )

        planner = make_planner(client)

        result = await planner.create_plan(
            "Calculate 7 * 8."
        )

        assert isinstance(result, PlanSchema)

        for step in result.steps:
            if step.step_type == StepType.TOOL:
                assert step.tool_name != "web_search"

    @pytest.mark.asyncio
    async def test_planner_does_not_allow_unknown_tool_to_survive(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.return_value = PlanSchema.model_construct(
            steps=[
                PlanStep(
                    step_id=0,
                    action="Use a fake tool",
                    step_type=StepType.TOOL,
                    tool_name="totally_fake_tool",
                    arguments={},
                    is_final_answer=False,
                ),
                final_step(1),
            ]
        )

        planner = make_planner(client)

        result = await planner.create_plan(
            "Find information about GAIA."
        )

        assert all(
            step.tool_name != "totally_fake_tool"
            for step in result.steps
        )

        assert result.steps[-1].is_final_answer

    @pytest.mark.asyncio
    async def test_planner_fallback_still_obeys_final_answer_contract(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.side_effect = RuntimeError(
            "LLM exploded"
        )

        planner = make_planner(client)

        result = await planner.create_plan(
            "Find information about GAIA."
        )

        assert isinstance(result, PlanSchema)
        assert len(result.steps) >= 1

        final_steps = [
            step
            for step in result.steps
            if step.is_final_answer
        ]

        assert len(final_steps) == 1
        assert result.steps[-1].is_final_answer
        assert result.steps[-1].step_type == StepType.LLM

    @pytest.mark.asyncio
    async def test_planner_final_context_is_not_flattened(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.return_value = PlanSchema(
            steps=[
                final_step(0),
            ]
        )

        planner = make_planner(client)

        context = FinalContext(
            items=[
                {
                    "source_type": "attachment",
                    "filename": "critical.txt",
                    "content": "UNIQUE_EVIDENCE_9981",
                },
                {
                    "source_type": "runtime",
                    "value": "runtime-evidence",
                },
            ],
            token_count=1234,
        )

        await planner.create_plan(
            "Answer using the evidence.",
            context,
        )

        messages = client.generate.call_args.args[0]

        prompt = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )

        assert "UNIQUE_EVIDENCE_9981" in prompt
        assert "critical.txt" in prompt
        assert "runtime-evidence" in prompt

        assert "FinalContext(items=" not in prompt
        assert "token_count': 1234" not in prompt
        assert '"token_count": 1234' not in prompt

    @pytest.mark.asyncio
    async def test_planner_rejects_legacy_context_shape(
        self,
    ) -> None:
        client = AsyncMock()

        planner = make_planner(client)

        with pytest.raises(
            (AttributeError, TypeError)
        ):
            await planner.create_plan(
                "Use the attachment.",
                [{"filename": "old.txt"}],
            )

    @pytest.mark.asyncio
    async def test_replan_uses_final_context_contract(
        self,
    ) -> None:
        client = AsyncMock()

        failed_step = tool_step(
            0,
            tool_name="web_search",
            arguments={
                "query": "GAIA"
            },
        )

        client.generate.return_value = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="visit_webpage",
                    arguments={
                        "url": "https://example.com"
                    },
                ),
                final_step(1),
            ]
        )

        planner = make_planner(client)

        result = await planner.replan(
            user_question="What is GAIA?",
            context=FinalContext(
                items=[
                    {
                        "source_type": "observation",
                        "content": "Search failed.",
                    }
                ],
                token_count=20,
            ),
            failed_step=failed_step,
            failure=AgentError(
                error_type="tool_execution",
                message="web_search failed",
            ),
        )

        assert isinstance(result, PlanSchema)
        assert result.steps[0].tool_name == "visit_webpage"

    @pytest.mark.asyncio
    async def test_replan_rejects_unchanged_failed_signature(
        self,
    ) -> None:
        client = AsyncMock()

        failed_step = tool_step(
            0,
            tool_name="web_search",
            arguments={"query": "same query"},
        )

        client.generate.return_value = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments={"query": "same query"},
                ),
                final_step(1),
            ]
        )

        planner = make_planner(client)

        with pytest.raises(PlannerRecoveryRequired):
            await planner.replan(
                user_question="Find GAIA.",
                context=FinalContext(
                    items=[],
                    token_count=0,
                ),
                failed_step=failed_step,
                failure=AgentError(
                    error_type="tool_execution",
                    message="search failed",
                ),
            )

    @pytest.mark.asyncio
    async def test_planner_preserves_tool_arguments(
        self,
    ) -> None:
        client = AsyncMock()

        expected_arguments = {
            "query": "GAIA official website"
        }

        client.generate.return_value = PlanSchema(
            steps=[
                tool_step(
                    0,
                    tool_name="web_search",
                    arguments=expected_arguments,
                ),
                final_step(1),
            ]
        )

        planner = make_planner(client)

        result = await planner.create_plan(
            "What is the official GAIA website?"
        )

        assert result.steps[0].arguments == expected_arguments

    @pytest.mark.asyncio
    async def test_planner_does_not_accept_tool_as_final_answer(
        self,
    ) -> None:
        client = AsyncMock()

        malformed_step = PlanStep.model_construct(
            step_id=0,
            action="Search",
            step_type=StepType.TOOL,
            tool_name="web_search",
            arguments={"query": "GAIA"},
            is_final_answer=True,
        )
        malformed = PlanSchema.model_construct(
            steps=[malformed_step]
        )

        client.generate.return_value = malformed

        planner = make_planner(client)

        result = await planner.create_plan(
            "What is GAIA?"
        )

        assert result.steps[-1].step_type == StepType.LLM
        assert result.steps[-1].is_final_answer

    @pytest.mark.asyncio
    async def test_attachment_evidence_changes_planner_information(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.return_value = PlanSchema(
            steps=[
                final_step(0),
            ]
        )

        planner = make_planner(client)

        context_without = FinalContext(
            items=[],
            token_count=0,
        )

        context_with = FinalContext(
            items=[
                {
                    "source_type": "attachment",
                    "filename": "secret.txt",
                    "content": "ATTACHMENT_FACT_XYZ",
                }
            ],
            token_count=10,
        )

        await planner.create_plan(
            "Answer the question.",
            context_without,
        )

        prompt_without = "\n".join(
            str(m.get("content", ""))
            for m in client.generate.call_args.args[0]
            if isinstance(m, dict)
        )

        client.reset_mock()

        await planner.create_plan(
            "Answer the question.",
            context_with,
        )

        prompt_with = "\n".join(
            str(m.get("content", ""))
            for m in client.generate.call_args.args[0]
            if isinstance(m, dict)
        )

        assert "ATTACHMENT_FACT_XYZ" not in prompt_without
        assert "ATTACHMENT_FACT_XYZ" in prompt_with

    @pytest.mark.asyncio
    async def test_planner_does_not_use_token_count_as_evidence(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.return_value = PlanSchema(
            steps=[
                final_step(0),
            ]
        )

        planner = make_planner(client)

        context = FinalContext(
            items=[],
            token_count=987654321,
        )

        await planner.create_plan(
            "Answer the question.",
            context,
        )

        prompt = "\n".join(
            str(m.get("content", ""))
            for m in client.generate.call_args.args[0]
            if isinstance(m, dict)
        )

        assert "987654321" not in prompt

    @pytest.mark.asyncio
    async def test_llm_cannot_smuggle_arguments_into_final_step(
        self,
    ) -> None:
        client = AsyncMock()

        malformed = PlanSchema.model_construct(
            steps=[
                PlanStep.model_construct(
                    step_id=0,
                    action="Final answer",
                    step_type=StepType.LLM,
                    tool_name="web_search",
                    arguments={
                        "query": "smuggled"
                    },
                    is_final_answer=True,
                )
            ]
        )

        client.generate.return_value = malformed

        planner = make_planner(client)

        result = await planner.create_plan(
            "What is GAIA?"
        )

        final = result.steps[-1]

        assert final.is_final_answer
        assert final.step_type == StepType.LLM
        assert final.tool_name is None
        assert final.arguments == {}

    @pytest.mark.asyncio
    async def test_multiple_context_items_preserve_evidence_order(
        self,
    ) -> None:
        client = AsyncMock()

        client.generate.return_value = PlanSchema(
            steps=[
                final_step(0),
            ]
        )

        planner = make_planner(client)

        context = FinalContext(
            items=[
                {"id": "A", "value": "FIRST_UNIQUE"},
                {"id": "B", "value": "SECOND_UNIQUE"},
                {"id": "C", "value": "THIRD_UNIQUE"},
            ],
            token_count=30,
        )

        await planner.create_plan(
            "Use all evidence.",
            context,
        )

        prompt = "\n".join(
            str(m.get("content", ""))
            for m in client.generate.call_args.args[0]
            if isinstance(m, dict)
        )

        assert "FIRST_UNIQUE" in prompt
        assert "SECOND_UNIQUE" in prompt
        assert "THIRD_UNIQUE" in prompt

        assert prompt.index("FIRST_UNIQUE") < prompt.index(
            "SECOND_UNIQUE"
        )
        assert prompt.index("SECOND_UNIQUE") < prompt.index(
            "THIRD_UNIQUE"
        )