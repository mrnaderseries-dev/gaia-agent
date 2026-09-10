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


class _RegisteredTool:
    def __init__(
        self,
        tool: Any,
        spec: ToolSpec,
    ) -> None:
        if tool is None:
            raise ValueError(
                "Cannot register a None tool."
            )

        if not isinstance(
            spec,
            ToolSpec,
        ):
            raise TypeError(
                "Tool must provide a ToolSpec."
            )

        implementation_name = getattr(
            tool,
            "name",
            None,
        )

        if (
            not isinstance(
                implementation_name,
                str,
            )
            or not implementation_name.strip()
        ):
            raise ValueError(
                "Every tool must have a "
                "non-empty name."
            )

        if implementation_name != spec.name:
            raise ValueError(
                f"Tool name '{implementation_name}' "
                f"does not match ToolSpec name "
                f"'{spec.name}'."
            )

        if spec.function is None:
            raise ValueError(
                f"ToolSpec '{spec.name}' must "
                "define its function."
            )

        self._tool = tool
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def inputs(self) -> dict[str, Any]:
        return self.spec.arguments_schema

    @property
    def output_type(self) -> str:
        return str(
            self.spec.result_schema.get(
                "type",
                "string",
            )
        )

    async def execute(
        self,
        **arguments: Any,
    ) -> Any:
        validated = (
            self.validate_arguments(
                arguments
            )
        )

        result = self.spec.function(
            **validated
        )

        if inspect.isawaitable(
            result
        ):
            return await result

        return result

    def validate_arguments(
        self,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return ToolContractValidator.validate_arguments(
            self.spec,
            arguments,
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
            _RegisteredTool,
        ] = {}

        self._register_group(
            FileTools(
                base_dir=self.base_dir
            ).get_tools()
        )

        self._register_group(
            AudioTools(
                stt_backend=self.stt_backend,
                base_dir=self.base_dir,
                stt_model_size=self.stt_model_size,
                stt_device=self.stt_device,
                stt_compute_type=self.stt_compute_type,
            ).get_tools()
        )

        self._register_group(
            VisionTools(
                llm_service=self.vision_llm_service,
                base_dir=self.base_dir,
            ).get_tools()
        )

        self._register_group(
            ExcelTools(
                llm_service=self.llm_service,
                base_dir=self.base_dir,
            ).get_tools()
        )

        self._register_group(
            PythonTools().get_tools()
        )

        self._register_group(
            WebTools().get_tools()
        )

    def _register_group(
        self,
        tools: list[Any],
    ) -> None:
        for tool in tools:
            spec = self._extract_spec(
                tool
            )

            if spec.name in self._tools_by_name:
                raise RuntimeError(
                    "Duplicate tool name detected: "
                    f"'{spec.name}'."
                )

            self._validate_spec(
                tool,
                spec,
            )

            self._tools_by_name[
                spec.name
            ] = _RegisteredTool(
                tool=tool,
                spec=spec,
            )

    @staticmethod
    def _extract_spec(
        tool: Any,
    ) -> ToolSpec:
        spec = getattr(
            tool,
            "spec",
            None,
        )

        if not isinstance(
            spec,
            ToolSpec,
        ):
            raise TypeError(
                f"Tool '{getattr(tool, 'name', '?')}' "
                "must expose a ToolSpec through "
                "the 'spec' attribute."
            )

        return spec

    @staticmethod
    def _validate_spec(
        tool: Any,
        spec: ToolSpec,
    ) -> None:
        implementation_name = getattr(
            tool,
            "name",
            None,
        )

        if implementation_name != spec.name:
            raise ValueError(
                f"Tool implementation name "
                f"'{implementation_name}' does not "
                f"match ToolSpec name '{spec.name}'."
            )

        if spec.function is None:
            raise ValueError(
                f"Tool '{spec.name}' has no "
                "execution function."
            )

        if not spec.modalities:
            raise ValueError(
                f"Tool '{spec.name}' must declare "
                "at least one modality."
            )

    def get_tools(
        self,
    ) -> list[_RegisteredTool]:
        return list(
            self._tools_by_name.values()
        )

    def get(
        self,
        tool_name: str,
    ) -> _RegisteredTool:
        if (
            not isinstance(
                tool_name,
                str,
            )
            or not tool_name.strip()
        ):
            raise ValueError(
                "tool_name must be a "
                "non-empty string."
            )

        try:
            return self._tools_by_name[
                tool_name
            ]
        except KeyError as exc:
            raise KeyError(
                f"Tool '{tool_name}' is not registered. "
                f"Available tools: {self.names()}"
            ) from exc

    def get_spec(
        self,
        tool_name: str,
    ) -> ToolSpec:
        return self.get(
            tool_name
        ).spec

    def get_tool_specs(
        self,
    ) -> list[ToolSpec]:
        return [
            tool.spec
            for tool in self._tools_by_name.values()
        ]

    def validate_step(
        self,
        step: Any,
    ) -> dict[str, Any]:
        specs = {
            spec.name: spec
            for spec in self.get_tool_specs()
        }

        return (
            ToolContractValidator
            .validate_step_contract(
                step=step,
                available_tools=specs,
            )
        )

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self.get(
            tool_name
        ).validate_arguments(
            arguments
        )

    def resolve(
        self,
        *,
        modality: ToolModality | None = None,
        capability: ToolCapability | None = None,
    ) -> list[ToolSpec]:
        specs = self.get_tool_specs()

        if modality is not None:
            specs = [
                spec
                for spec in specs
                if spec.supports_modality(
                    modality
                )
            ]

        if capability is not None:
            specs = [
                spec
                for spec in specs
                if spec.supports_capability(
                    capability
                )
            ]

        return sorted(
            specs,
            key=lambda spec: spec.name,
        )

    def resolve_one(
        self,
        *,
        modality: ToolModality | None = None,
        capability: ToolCapability | None = None,
    ) -> ToolSpec | None:
        matches = self.resolve(
            modality=modality,
            capability=capability,
        )

        if not matches:
            return None

        return matches[0]

    def has(
        self,
        tool_name: str,
    ) -> bool:
        return (
            isinstance(
                tool_name,
                str,
            )
            and tool_name
            in self._tools_by_name
        )

    def names(self) -> list[str]:
        return sorted(
            self._tools_by_name
        )

    def capabilities(
        self,
    ) -> dict[str, ToolCapability]:
        return {
            spec.name: spec.capability
            for spec in self.get_tool_specs()
        }

    def modalities(
        self,
    ) -> dict[
        str,
        frozenset[ToolModality],
    ]:
        return {
            spec.name: spec.modalities
            for spec in self.get_tool_specs()
        }