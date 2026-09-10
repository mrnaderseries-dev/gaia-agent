from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolErrorCode,
    ToolModality,
    ToolSpec,
)
from gaia_agent.tools.registry import (
    RegisteredTool,
    ToolRegistry,
)


class DummyLLMService:
    pass


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        base_dir=".",
        llm_service=DummyLLMService(),
    )


def test_every_registered_tool_has_exactly_one_spec() -> None:
    registry = build_registry()

    tools = registry.get_tools()
    specs = registry.get_tool_specs()

    assert tools
    assert len(tools) == len(specs)

    tool_names = {tool.name for tool in tools}
    spec_names = {spec.name for spec in specs}

    assert tool_names == spec_names


def test_every_spec_is_canonical_tool_spec() -> None:
    registry = build_registry()

    for tool in registry.get_tools():
        assert isinstance(tool.spec, ToolSpec)
        assert tool.spec.function is not None
        assert callable(tool.spec.function)


def test_spec_name_matches_implementation_name() -> None:
    registry = build_registry()

    for tool in registry.get_tools():
        assert tool.name == tool.spec.name


def test_spec_function_points_to_tool_function() -> None:
    registry = build_registry()

    for tool in registry.get_tools():
        assert tool.spec.function is not None
        assert callable(tool.spec.function)


def test_registered_tool_exposes_spec() -> None:
    registry = build_registry()

    for tool in registry.get_tools():
        assert isinstance(tool, RegisteredTool)
        assert isinstance(tool.spec, ToolSpec)


def test_get_and_lookup_return_same_tool() -> None:
    registry = build_registry()

    for name in registry.names():
        assert registry.get(name) is registry.lookup(name)


def test_has_returns_true_for_registered_tools() -> None:
    registry = build_registry()

    for name in registry.names():
        assert registry.has(name)


def test_has_returns_false_for_unknown_tool() -> None:
    registry = build_registry()

    assert not registry.has("does_not_exist")


def test_unknown_tool_raises() -> None:
    registry = build_registry()

    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_get_spec_returns_canonical_spec() -> None:
    registry = build_registry()

    for name in registry.names():
        spec = registry.get_spec(name)

        assert isinstance(spec, ToolSpec)
        assert spec.name == name


def test_python_tool_is_computation() -> None:
    registry = build_registry()

    spec = registry.get_spec("python_interpreter")

    assert spec.capability == ToolCapability.COMPUTATION


def test_web_tools_are_network_read() -> None:
    registry = build_registry()

    for name in (
        "web_search",
        "visit_webpage",
        "youtube_transcript",
    ):
        spec = registry.get_spec(name)

        assert spec.capability == ToolCapability.NETWORK_READ


def test_file_tool_is_read_only() -> None:
    registry = build_registry()

    spec = registry.get_spec("file_reader")

    assert spec.capability == ToolCapability.READ_ONLY


def test_vision_tool_has_vision_modality() -> None:
    registry = build_registry()

    spec = registry.get_spec("analyze_image")

    assert spec.supports_modality(ToolModality.VISION)


def test_audio_tool_has_audio_modality() -> None:
    registry = build_registry()

    spec = registry.get_spec("transcribe_audio")

    assert spec.supports_modality(ToolModality.AUDIO)


def test_excel_tool_has_excel_modality() -> None:
    registry = build_registry()

    spec = registry.get_spec("analyze_excel")

    assert spec.supports_modality(ToolModality.EXCEL)


def test_youtube_transcript_is_video_tool() -> None:
    registry = build_registry()

    spec = registry.get_spec("youtube_transcript")

    assert spec.supports_modality(ToolModality.VIDEO)


def test_audio_registry_does_not_own_youtube_tool() -> None:
    registry = build_registry()

    audio_names = {
        tool.name
        for tool in registry.get_tools()
        if tool.spec.supports_modality(ToolModality.AUDIO)
    }

    assert "youtube_transcript" not in audio_names


def test_resolve_by_modality() -> None:
    registry = build_registry()

    tools = registry.resolve(
        modality=ToolModality.VISION,
    )

    names = {tool.name for tool in tools}

    assert "analyze_image" in names


def test_resolve_by_capability() -> None:
    registry = build_registry()

    tools = registry.resolve(
        capability=ToolCapability.COMPUTATION,
    )

    names = {tool.name for tool in tools}

    assert "python_interpreter" in names


def test_resolve_by_modality_and_capability() -> None:
    registry = build_registry()

    tools = registry.resolve(
        modality=ToolModality.VISION,
        capability=ToolCapability.READ_ONLY,
    )

    names = {tool.name for tool in tools}

    assert "analyze_image" in names


def test_resolve_with_no_filters_returns_all_tools() -> None:
    registry = build_registry()

    tools = registry.resolve()

    assert {
        tool.name
        for tool in tools
    } == set(registry.names())


def test_duplicate_registration_is_rejected() -> None:
    registry = build_registry()

    existing = registry.get("file_reader")

    with pytest.raises(ValueError):
        registry.register(existing.tool)


def test_mismatched_spec_name_is_rejected() -> None:
    class FakeTool:
        name = "real_name"

        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="different_name",
                description="fake",
                arguments_schema={},
                capability=ToolCapability.READ_ONLY,
                modalities=frozenset(),
                result_schema={"type": "string"},
                error_codes=frozenset(),
                allowed_imports=frozenset(),
                function=self.forward,
            )

        def forward(self) -> str:
            return "ok"

    registry = build_registry()

    with pytest.raises(ValueError):
        registry.register(FakeTool())


def test_missing_spec_is_rejected() -> None:
    class FakeTool:
        name = "fake_tool"

        def forward(self) -> str:
            return "ok"

    registry = build_registry()

    with pytest.raises(ValueError):
        registry.register(FakeTool())


def test_spec_requires_callable_function() -> None:
    class FakeTool:
        name = "fake_tool"

        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="fake_tool",
                description="fake",
                arguments_schema={},
                capability=ToolCapability.READ_ONLY,
                modalities=frozenset(),
                result_schema={"type": "string"},
                error_codes=frozenset(),
                allowed_imports=frozenset(),
                function=None,
            )

    registry = build_registry()

    with pytest.raises(ValueError):
        registry.register(FakeTool())


def test_registered_tool_validation_uses_spec() -> None:
    registry = build_registry()

    tool = registry.get("analyze_image")

    with pytest.raises(Exception):
        tool.validate_arguments({})


def test_registered_tool_execution_is_async() -> None:
    registry = build_registry()

    tool = registry.get("file_reader")

    assert inspect.iscoroutinefunction(
        tool.execute
    )


def test_python_spec_declares_allowed_imports() -> None:
    registry = build_registry()

    spec = registry.get_spec("python_interpreter")

    assert "math" in spec.allowed_imports
    assert "json" in spec.allowed_imports


def test_tool_specs_have_error_codes() -> None:
    registry = build_registry()

    for spec in registry.get_tool_specs():
        assert isinstance(
            spec.error_codes,
            frozenset,
        )

        for error_code in spec.error_codes:
            assert isinstance(
                error_code,
                ToolErrorCode,
            )


def test_registry_does_not_build_specs_from_tool_names() -> None:
    registry = build_registry()

    for tool in registry.get_tools():
        assert isinstance(tool.spec, ToolSpec)

        assert not hasattr(
            tool,
            "build_spec",
        )


def test_registry_does_not_own_tool_contract_data() -> None:
    registry = build_registry()

    registry_attributes = vars(registry)

    assert "TOOL_CAPABILITIES" not in registry_attributes
    assert "TOOL_MODALITIES" not in registry_attributes
    assert "TOOL_ERROR_CODES" not in registry_attributes


def test_registry_has_no_legacy_contract_mapping() -> None:
    import gaia_agent.tools.registry as registry_module

    source = Path(
        registry_module.__file__
    ).read_text(
        encoding="utf-8",
    )

    assert "TOOL_CAPABILITIES" not in source
    assert "TOOL_MODALITIES" not in source
    assert "TOOL_ERROR_CODES" not in source
    assert "build_spec" not in source