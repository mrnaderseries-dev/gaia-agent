from __future__ import annotations

import inspect
from typing import Any

from gaia_agent.llm.service import LLMService
from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolModality,
    ToolSpec,
)
from gaia_agent.tools.contract_validator import (
    ToolContractValidator,
)

from .audio import AudioTools
from .excel import ExcelTools
from .files import FileTools
from .python import PythonTools
from .vision import VisionTools
from .web import WebTools


class RegisteredTool:
    def __init__(
        self,
        tool: Any,
        spec: ToolSpec,
    ) -> None:
        if tool is None:
            raise ValueError(
                "Cannot register a None tool."
            )

        if not isinstance(spec, ToolSpec):
            raise TypeError(
                "Registered tool spec must be a ToolSpec."
            )

        tool_name = getattr(tool, "name", None)

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise ValueError(
                "Every registered tool must have a "
                "non-empty name."
            )

        if tool_name != spec.name:
            raise ValueError(
                f"Tool/spec name mismatch: "
                f"implementation='{tool_name}', "
                f"spec='{spec.name}'."
            )

        if spec.function is None:
            raise ValueError(
                f"Tool '{spec.name}' must declare "
                "a callable function in its ToolSpec."
            )

        if not callable(spec.function):
            raise TypeError(
                f"ToolSpec function for '{spec.name}' "
                "must be callable."
            )

        self.tool = tool
        self.spec = spec
        self.name = spec.name

    async def execute(
        self,
        **arguments: Any,
    ) -> Any:
        validated_arguments = self.validate_arguments(
            arguments
        )

        result = self.spec.function(
            **validated_arguments
        )

        if inspect.isawaitable(result):
            return await result

        return result

    def validate_arguments(
        self,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return ToolContractValidator.validate_arguments(
            spec=self.spec,
            arguments=arguments,
        )

    def supports_modality(
        self,
        modality: ToolModality,
    ) -> bool:
        return self.spec.supports_modality(
            modality
        )

    def supports_capability(
        self,
        capability: ToolCapability,
    ) -> bool:
        return self.spec.supports_capability(
            capability
        )


class ToolRegistry:
    def __init__(
        self,
        base_dir: str = ".",
        *,
        llm_service: LLMService,
        vision_llm_service: LLMService | None = None,
        stt_backend: Any = None,
        stt_model_size: str = "base",
        stt_device: str = "cpu",
        stt_compute_type: str = "int8",
    ) -> None:
        if llm_service is None:
            raise ValueError(
                "ToolRegistry requires an llm_service."
            )

        self.base_dir = base_dir
        self.llm_service = llm_service

        self.vision_llm_service = (
            vision_llm_service
            if vision_llm_service is not None
            else llm_service
        )

        self.stt_backend = stt_backend
        self.stt_model_size = stt_model_size
        self.stt_device = stt_device
        self.stt_compute_type = stt_compute_type

        self._tools_by_name: dict[
            str,
            RegisteredTool,
        ] = {}

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        file_tools = FileTools(
            base_dir=self.base_dir,
        )

        audio_tools = AudioTools(
            stt_backend=self.stt_backend,
            base_dir=self.base_dir,
            stt_model_size=self.stt_model_size,
            stt_device=self.stt_device,
            stt_compute_type=self.stt_compute_type,
        )

        vision_tools = VisionTools(
            llm_service=self.vision_llm_service,
            base_dir=self.base_dir,
        )

        excel_tools = ExcelTools(
            llm_service=self.llm_service,
            base_dir=self.base_dir,
        )

        python_tools = PythonTools()
        web_tools = WebTools()

        groups = (
            file_tools,
            audio_tools,
            vision_tools,
            excel_tools,
            python_tools,
            web_tools,
        )

        for group in groups:
            for tool in group.get_tools():
                self.register(tool)

    def register(
        self,
        tool: Any,
    ) -> RegisteredTool:
        if tool is None:
            raise ValueError(
                "Cannot register a None tool."
            )

        spec = getattr(tool, "spec", None)

        if spec is None:
            raise ValueError(
                f"Tool '{getattr(tool, 'name', '?')}' "
                "does not expose a ToolSpec."
            )

        if callable(spec):
            spec = spec()

        if not isinstance(spec, ToolSpec):
            raise TypeError(
                f"Tool '{getattr(tool, 'name', '?')}' "
                "must expose a ToolSpec."
            )

        registered = RegisteredTool(
            tool=tool,
            spec=spec,
        )

        if registered.name in self._tools_by_name:
            raise RuntimeError(
                "Duplicate tool name detected: "
                f"'{registered.name}'."
            )

        self._tools_by_name[
            registered.name
        ] = registered

        return registered

    def get(
        self,
        tool_name: str,
    ) -> RegisteredTool:
        self._validate_name(tool_name)

        try:
            return self._tools_by_name[
                tool_name
            ]
        except KeyError as exc:
            available = sorted(
                self._tools_by_name
            )

            raise KeyError(
                f"Tool '{tool_name}' is not registered. "
                f"Available tools: {available}"
            ) from exc

    def lookup(
        self,
        tool_name: str,
    ) -> RegisteredTool:
        return self.get(tool_name)

    def has(
        self,
        tool_name: str,
    ) -> bool:
        if not isinstance(tool_name, str):
            return False

        return (
            tool_name in self._tools_by_name
        )

    def resolve(
        self,
        *,
        modality: ToolModality | None = None,
        capability: ToolCapability | None = None,
    ) -> list[RegisteredTool]:
        if (
            modality is None
            and capability is None
        ):
            return list(
                self._tools_by_name.values()
            )

        resolved: list[RegisteredTool] = []

        for tool in self._tools_by_name.values():
            if (
                modality is not None
                and not tool.supports_modality(
                    modality
                )
            ):
                continue

            if (
                capability is not None
                and not tool.supports_capability(
                    capability
                )
            ):
                continue

            resolved.append(tool)

        return resolved

    def get_spec(
        self,
        tool_name: str,
    ) -> ToolSpec:
        return self.get(tool_name).spec

    def get_tool_specs(self) -> list[ToolSpec]:
        return [
            tool.spec
            for tool in self._tools_by_name.values()
        ]

    def get_tools(self) -> list[RegisteredTool]:
        return list(
            self._tools_by_name.values()
        )

    def validate_step(
        self,
        step: Any,
    ) -> dict[str, Any]:
        specs = {
            name: tool.spec
            for name, tool
            in self._tools_by_name.items()
        }

        return ToolContractValidator.validate_step_contract(
            step=step,
            available_tools=specs,
        )

    def names(self) -> list[str]:
        return sorted(
            self._tools_by_name
        )

    def capabilities(
        self,
    ) -> dict[str, ToolCapability]:
        return {
            name: tool.spec.capability
            for name, tool
            in self._tools_by_name.items()
        }

    def _validate_name(
        self,
        tool_name: str,
    ) -> None:
        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise ValueError(
                "tool_name must be a non-empty string."
            )