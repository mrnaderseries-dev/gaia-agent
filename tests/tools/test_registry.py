from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolErrorCode,
    ToolModality,
    ToolSpec,
)
from gaia_agent.tools.contract_validator import ToolContractValidator
from gaia_agent.tools.registry import RegisteredTool, ToolRegistry


class FakeLLMService:
    def generate_sync(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    def generate_image_sync(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return "fake vision response"


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry(
        base_dir=".",
        llm_service=FakeLLMService(),
        vision_llm_service=FakeLLMService(),
        stt_backend=None,
    )


def test_registry_contains_expected_tools(
    registry: ToolRegistry,
) -> None:
    names = set(registry.names())

    expected = {
        "file_reader",
        "python_interpreter",
        "web_search",
        "visit_webpage",
        "analyze_image",
        "analyze_excel",
    }

    assert expected.issubset(names)


def test_every_registered_tool_has_exactly_one_spec(
    registry: ToolRegistry,
) -> None:
    tools = registry.get_tools()

    assert tools

    for registered in tools:
        assert isinstance(
            registered,
            RegisteredTool,
        )
        assert isinstance(
            registered.spec,
            ToolSpec,
        )
        assert registered.tool.spec is registered.spec


def test_spec_name_matches_implementation(
    registry: ToolRegistry,
) -> None:
    for registered in registry.get_tools():
        assert registered.name == registered.spec.name
        assert registered.tool.name == registered.spec.name


def test_every_spec_has_callable_function(
    registry: ToolRegistry,
) -> None:
    for spec in registry.get_tool_specs():
        assert callable(spec.function)


def test_specs_use_canonical_argument_schema(
    registry: ToolRegistry,
) -> None:
    for spec in registry.get_tool_specs():
        schema = spec.arguments_schema

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "additionalProperties" in schema

        assert isinstance(
            schema["properties"],
            dict,
        )

        assert isinstance(
            schema["required"],
            list,
        )

        assert isinstance(
            schema["additionalProperties"],
            bool,
        )


def test_registered_tool_exposes_same_spec(
    registry: ToolRegistry,
) -> None:
    for registered in registry.get_tools():
        assert registered.spec is registry.get_spec(
            registered.name
        )


def test_get_and_lookup_return_same_tool(
    registry: ToolRegistry,
) -> None:
    for name in registry.names():
        assert registry.get(name) is registry.lookup(name)


def test_has_returns_correct_value(
    registry: ToolRegistry,
) -> None:
    assert registry.has("python_interpreter")
    assert registry.has("file_reader")
    assert registry.has("web_search")

    assert not registry.has("does_not_exist")
    assert not registry.has("")
    assert not registry.has(None)


def test_unknown_tool_raises(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_empty_tool_name_raises(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.get("")


def test_get_spec_returns_tool_spec(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    assert isinstance(spec, ToolSpec)
    assert spec.name == "python_interpreter"


def test_python_tool_contract(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    assert (
        spec.capability
        == ToolCapability.COMPUTATION
    )

    assert (
        ToolModality.TEXT
        in spec.modalities
    )

    assert (
        ToolModality.FILE
        not in spec.modalities
    )

    assert (
        ToolErrorCode.INVALID_ARGUMENT
        in spec.error_codes
    )

    assert (
        ToolErrorCode.EXECUTION_FAILED
        in spec.error_codes
    )

    assert (
        "math"
        in spec.allowed_imports
    )

    assert (
        "json"
        in spec.allowed_imports
    )


def test_file_tool_contract(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "file_reader"
    )

    assert (
        spec.capability
        == ToolCapability.READ_ONLY
    )

    assert (
        ToolModality.FILE
        in spec.modalities
    )

    assert (
        ToolModality.TEXT
        in spec.modalities
    )

    assert (
        ToolErrorCode.FILE_NOT_FOUND
        in spec.error_codes
    )

    assert (
        ToolErrorCode.INVALID_FILE
        in spec.error_codes
    )

    assert (
        ToolErrorCode.DECODE_ERROR
        in spec.error_codes
    )


def test_web_search_contract(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "web_search"
    )

    assert (
        spec.capability
        == ToolCapability.NETWORK_READ
    )

    assert (
        ToolModality.TEXT
        in spec.modalities
    )

    assert (
        ToolErrorCode.NETWORK_ERROR
        in spec.error_codes
    )

    assert (
        ToolErrorCode.RATE_LIMITED
        in spec.error_codes
    )


def test_visit_webpage_contract(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "visit_webpage"
    )

    assert (
        spec.capability
        == ToolCapability.NETWORK_READ
    )

    assert (
        ToolModality.TEXT
        in spec.modalities
    )


def test_vision_tool_contract(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "analyze_image"
    )

    assert (
        spec.capability
        == ToolCapability.READ_ONLY
    )

    assert (
        ToolModality.VISION
        in spec.modalities
    )


def test_audio_tool_contract(
    registry: ToolRegistry,
) -> None:
    if not registry.has("transcribe_audio"):
        pytest.skip(
            "Audio tool is not registered."
        )

    spec = registry.get_spec(
        "transcribe_audio"
    )

    assert (
        ToolModality.AUDIO
        in spec.modalities
    )

    assert (
        spec.capability
        == ToolCapability.READ_ONLY
    )


def test_excel_tool_contract(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "analyze_excel"
    )

    assert (
        ToolModality.EXCEL
        in spec.modalities
    )

    assert (
        spec.capability
        == ToolCapability.READ_ONLY
    )


def test_youtube_tool_is_video_tool(
    registry: ToolRegistry,
) -> None:
    if not registry.has(
        "youtube_transcript"
    ):
        pytest.skip(
            "YouTube transcript dependency is unavailable."
        )

    spec = registry.get_spec(
        "youtube_transcript"
    )

    assert (
        ToolModality.VIDEO
        in spec.modalities
    )

    assert (
        ToolModality.TEXT
        in spec.modalities
    )

    assert (
        spec.capability
        == ToolCapability.NETWORK_READ
    )


def test_audio_registry_does_not_own_youtube(
    registry: ToolRegistry,
) -> None:
    from gaia_agent.tools.audio import AudioTools

    audio_tools = AudioTools(
        stt_backend=None,
        base_dir=".",
    )

    audio_names = {
        tool.name
        for tool in audio_tools.get_tools()
    }

    assert (
        "youtube_transcript"
        not in audio_names
    )


def test_python_resolves_by_capability(
    registry: ToolRegistry,
) -> None:
    tools = registry.resolve(
        capability=ToolCapability.COMPUTATION
    )

    names = {
        tool.name
        for tool in tools
    }

    assert "python_interpreter" in names


def test_file_resolves_by_capability(
    registry: ToolRegistry,
) -> None:
    tools = registry.resolve(
        capability=ToolCapability.READ_ONLY
    )

    names = {
        tool.name
        for tool in tools
    }

    assert "file_reader" in names


def test_video_resolves_by_modality(
    registry: ToolRegistry,
) -> None:
    if not registry.has(
        "youtube_transcript"
    ):
        pytest.skip(
            "YouTube transcript dependency unavailable."
        )

    tools = registry.resolve(
        modality=ToolModality.VIDEO
    )

    names = {
        tool.name
        for tool in tools
    }

    assert "youtube_transcript" in names


def test_vision_resolves_by_modality(
    registry: ToolRegistry,
) -> None:
    tools = registry.resolve(
        modality=ToolModality.VISION
    )

    names = {
        tool.name
        for tool in tools
    }

    assert "analyze_image" in names


def test_excel_resolves_by_modality(
    registry: ToolRegistry,
) -> None:
    tools = registry.resolve(
        modality=ToolModality.EXCEL
    )

    names = {
        tool.name
        for tool in tools
    }

    assert "analyze_excel" in names


def test_resolve_by_modality_and_capability(
    registry: ToolRegistry,
) -> None:
    tools = registry.resolve(
        modality=ToolModality.TEXT,
        capability=ToolCapability.NETWORK_READ,
    )

    names = {
        tool.name
        for tool in tools
    }

    assert "web_search" in names
    assert "visit_webpage" in names


def test_resolve_without_filters_returns_all(
    registry: ToolRegistry,
) -> None:
    tools = registry.resolve()

    assert {
        tool.name
        for tool in tools
    } == set(registry.names())


def test_registered_tool_validation_uses_spec(
    registry: ToolRegistry,
) -> None:
    tool = registry.get(
        "python_interpreter"
    )

    valid = tool.validate_arguments(
        {
            "code": "result = 2 + 2"
        }
    )

    assert valid == {
        "code": "result = 2 + 2"
    }


def test_registered_tool_rejects_missing_required_argument(
    registry: ToolRegistry,
) -> None:
    tool = registry.get(
        "python_interpreter"
    )

    with pytest.raises(ValueError):
        tool.validate_arguments({})


def test_registered_tool_rejects_unknown_argument(
    registry: ToolRegistry,
) -> None:
    tool = registry.get(
        "python_interpreter"
    )

    with pytest.raises(ValueError):
        tool.validate_arguments(
            {
                "code": "result = 1",
                "unexpected": True,
            }
        )


def test_registered_tool_rejects_wrong_argument_type(
    registry: ToolRegistry,
) -> None:
    tool = registry.get(
        "python_interpreter"
    )

    with pytest.raises(TypeError):
        tool.validate_arguments(
            {
                "code": 123,
            }
        )


@pytest.mark.asyncio
async def test_registered_tool_execution_is_async(
    registry: ToolRegistry,
) -> None:
    tool = registry.get(
        "python_interpreter"
    )

    result = await tool.execute(
        code="result = 2 + 2"
    )

    assert result == "4"


def test_python_tool_allowed_imports_are_declared(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    assert spec.allowed_imports

    assert "math" in spec.allowed_imports
    assert "json" in spec.allowed_imports
    assert "statistics" in spec.allowed_imports


def test_specs_declare_error_codes(
    registry: ToolRegistry,
) -> None:
    for spec in registry.get_tool_specs():
        assert isinstance(
            spec.error_codes,
            frozenset,
        )


def test_tool_specs_are_immutable(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    with pytest.raises(Exception):
        spec.name = "changed"


def test_registry_does_not_build_specs_from_names() -> None:
    import gaia_agent.tools.registry as registry_module

    source = Path(
        registry_module.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "TOOL_CAPABILITIES" not in source
    assert "TOOL_MODALITIES" not in source
    assert "TOOL_ERROR_CODES" not in source
    assert "build_spec" not in source


def test_registry_does_not_own_contract_data() -> None:
    import gaia_agent.tools.registry as registry_module

    source = Path(
        registry_module.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "ToolCapability.COMPUTATION" not in source
    assert "ToolCapability.READ_ONLY" not in source
    assert "ToolModality.FILE" not in source
    assert "ToolModality.VIDEO" not in source


def test_duplicate_registration_is_rejected(
    registry: ToolRegistry,
) -> None:
    original = registry.get(
        "python_interpreter"
    )

    with pytest.raises(RuntimeError):
        registry.register(
            original.tool
        )


def test_missing_spec_is_rejected(
    registry: ToolRegistry,
) -> None:
    class BrokenTool:
        name = "broken"

    with pytest.raises(ValueError):
        registry.register(
            BrokenTool()
        )


def test_mismatched_spec_name_is_rejected() -> None:
    class FakeTool:
        name = "actual_name"

        def forward(self) -> str:
            return "ok"

    spec = ToolSpec(
        name="different_name",
        description="test",
        arguments_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        capability=ToolCapability.READ_ONLY,
        modalities=frozenset(
            {ToolModality.TEXT}
        ),
        result_schema={
            "type": "string"
        },
        error_codes=frozenset(),
        allowed_imports=frozenset(),
        function=FakeTool().forward,
    )

    FakeTool.spec = spec

    with pytest.raises(ValueError):
        RegisteredTool(
            tool=FakeTool(),
            spec=spec,
        )


def test_spec_requires_callable_function() -> None:
    class FakeTool:
        name = "fake"

    with pytest.raises(ValueError):
        RegisteredTool(
            tool=FakeTool(),
            spec=ToolSpec(
                name="fake",
                description="test",
                capability=ToolCapability.READ_ONLY,
                function=None,
            ),
        )


def test_non_callable_spec_function_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(
            name="fake",
            description="test",
            capability=ToolCapability.READ_ONLY,
            function="not_callable",
        )


def test_contract_validator_returns_arguments(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    result = ToolContractValidator.validate_arguments(
        spec=spec,
        arguments={
            "code": "result = 10"
        },
    )

    assert result == {
        "code": "result = 10"
    }


def test_contract_validator_rejects_unknown_arguments(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    with pytest.raises(ValueError):
        ToolContractValidator.validate_arguments(
            spec=spec,
            arguments={
                "code": "result = 10",
                "hack": "value",
            },
        )


def test_contract_validator_rejects_missing_arguments(
    registry: ToolRegistry,
) -> None:
    spec = registry.get_spec(
        "python_interpreter"
    )

    with pytest.raises(ValueError):
        ToolContractValidator.validate_arguments(
            spec=spec,
            arguments={},
        )


def test_plan_step_contract_uses_arguments_field(
    registry: ToolRegistry,
) -> None:
    step = PlanStep(
        step_id=0,
        action="execute python calculation",
        step_type=StepType.TOOL,
        tool_name="python_interpreter",
        arguments={
            "code": "result = 21 * 2"
        },
    )

    validated = registry.validate_step(
        step
    )

    assert validated == {
        "code": "result = 21 * 2"
    }


def test_plan_step_contract_rejects_unknown_tool(
    registry: ToolRegistry,
) -> None:
    step = PlanStep(
        step_id=0,
        action="execute unknown tool",
        step_type=StepType.TOOL,
        tool_name="unknown_tool",
        arguments={},
    )

    with pytest.raises(ValueError):
        registry.validate_step(
            step
        )


def test_plan_step_contract_rejects_invalid_arguments(
    registry: ToolRegistry,
) -> None:
    step = PlanStep(
        step_id=0,
        action="execute python calculation",
        step_type=StepType.TOOL,
        tool_name="python_interpreter",
        arguments={
            "wrong": "value"
        },
    )

    with pytest.raises(ValueError):
        registry.validate_step(
            step
        )


def test_no_legacy_modality_values_exist() -> None:
    assert not hasattr(
        ToolModality,
        "WEB",
    )

    assert not hasattr(
        ToolModality,
        "CODE",
    )


def test_no_legacy_error_codes_exist() -> None:
    assert not hasattr(
        ToolErrorCode,
        "RATE_LIMIT",
    )

    assert not hasattr(
        ToolErrorCode,
        "VIDEO_UNAVAILABLE",
    )

    assert not hasattr(
        ToolErrorCode,
        "SYNTAX_ERROR",
    )

    assert not hasattr(
        ToolErrorCode,
        "IMPORT_ERROR",
    )

    assert not hasattr(
        ToolErrorCode,
        "EXECUTION_ERROR",
    )


def test_registered_execution_function_is_callable(
    registry: ToolRegistry,
) -> None:
    for tool in registry.get_tools():
        assert callable(
            tool.spec.function
        )


def test_registered_execution_functions_have_expected_shape(
    registry: ToolRegistry,
) -> None:
    for tool in registry.get_tools():
        assert callable(
            tool.spec.function
        )

        signature = inspect.signature(
            tool.spec.function
        )

        assert signature is not None