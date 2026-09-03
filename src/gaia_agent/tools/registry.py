from __future__ import annotations

from typing import Any

from gaia_agent.planner.tool_spec import (
    TOOL_CAPABILITIES,
    ToolCapability,
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
PYTHON_ALLOWED_IMPORTS: list[str] = [
    "math",
    "json",
    "re",
    "datetime",
    "itertools",
    "functools",
    "collections",
    "statistics",
    "string",
    "typing",
]


class _RegisteredTool:
    """
    Adapter between smolagents tools and GAIA's execution contract.

    GAIA execution expects:

        await tool.execute(**arguments)

    while a smolagents Tool is normally invoked as:

        tool(**arguments)

    This adapter keeps that implementation detail isolated from
    the rest of the agent.
    """

    def __init__(self, tool: Any) -> None:
        if tool is None:
            raise ValueError("Cannot register a None tool.")

        name = getattr(tool, "name", None)

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Every registered tool must have a non-empty name."
            )

        description = getattr(tool, "description", "")

        if not isinstance(description, str):
            description = str(description)

        inputs = getattr(tool, "inputs", {}) or {}

        if not isinstance(inputs, dict):
            raise TypeError(
                f"Tool '{name}' has invalid inputs schema: "
                f"{type(inputs).__name__}."
            )

        output_type = getattr(
            tool,
            "output_type",
            "string",
        )

        self._tool = tool
        self.name = name
        self.description = description
        self.inputs = dict(inputs)
        self.output_type = output_type

    async def execute(
        self,
        **arguments: Any,
    ) -> Any:
        """
        Validate arguments before invoking the actual tool.
        """

        validated_arguments = self.validate_arguments(
            arguments
        )

        return self._tool(
            **validated_arguments
        )

    def validate_arguments(
        self,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Validate arguments against this tool's canonical ToolSpec.

        No invalid argument should reach the underlying tool.
        """

        spec = self.build_spec()

        return ToolContractValidator.validate_arguments(
            spec=spec,
            arguments=arguments,
        )

    def build_spec(self) -> ToolSpec:
        """
        Build the canonical ToolSpec for this registered tool.

        Every tool must explicitly declare a capability in
        TOOL_CAPABILITIES. We deliberately do NOT default unknown
        tools to READ_ONLY.
        """

        capability = TOOL_CAPABILITIES.get(
            self.name
        )

        if capability is None:
            raise RuntimeError(
                f"Tool '{self.name}' has no declared ToolCapability. "
                "Add it explicitly to TOOL_CAPABILITIES before "
                "registering the tool."
            )

        allowed_imports: list[str] = []

        if self.name == "python_interpreter":
            allowed_imports = list(
                PYTHON_ALLOWED_IMPORTS
            )

        return ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema=dict(self.inputs),
            capability=capability,
            result_schema={
                "type": self.output_type,
            },
            error_codes=[],
            allowed_imports=allowed_imports,
        )


class ToolRegistry:
    """
    Central registry for every executable GAIA tool.

    Responsibilities
    -----------------
    1. Construct tool instances.
    2. Wrap them with the GAIA execution adapter.
    3. Guarantee unique tool names.
    4. Build canonical ToolSpec contracts.
    5. Expose tools to AgentExecution.
    6. Expose ToolSpec objects to Planner.
    7. Validate tool contracts before execution.

    The registry does NOT:
    - execute LLM inference itself;
    - contain provider-specific LLM logic;
    - implement Vision/Excel reasoning;
    - decide which tool the Planner should use.
    """

    def __init__(
        self,
        base_dir: str = ".",
        model: Any = None,
        stt_backend: Any = None,
        *,
        llm_service: Any = None,
    ) -> None:

        self.base_dir = base_dir
        
        self.model = model
        self.stt_backend = stt_backend
        self.llm_service = llm_service

        self._tools: list[_RegisteredTool] = (
            self._build_tools()
        )

        self._tools_by_name: dict[
            str,
            _RegisteredTool,
        ] = {}

        for tool in self._tools:
            if tool.name in self._tools_by_name:
                raise RuntimeError(
                    "Duplicate tool name detected: "
                    f"'{tool.name}'."
                )

            self._tools_by_name[
                tool.name
            ] = tool

        self._specs_by_name: dict[
            str,
            ToolSpec,
        ] = {}
        self.get_tool_specs()

    def _build_tools(
        self,
    ) -> list[_RegisteredTool]:
        """
        Construct every tool exposed to the agent.

        Tool-specific implementation remains inside the individual
        tool containers. The registry only wires them together.
        """

        file_tools = FileTools(
            base_dir=self.base_dir,
        )

        audio_tools = AudioTools(
            stt_backend=self.stt_backend,
            base_dir=self.base_dir,
            model=self.model,
        )

        vision_tools = VisionTools(
            model=self.model,
            base_dir=self.base_dir,
        )

        excel_tools = ExcelTools(
            model=self.model,
            base_dir=self.base_dir,
        )

        python_tools = PythonTools()

        web_tools = WebTools()

        raw_tools: list[Any] = []

        raw_tools.extend(
            file_tools.get_tools()
        )

        raw_tools.extend(
            audio_tools.get_tools()
        )

        raw_tools.extend(
            vision_tools.get_tools()
        )

        raw_tools.extend(
            excel_tools.get_tools()
        )

        raw_tools.extend(
            python_tools.get_tools()
        )

        raw_tools.extend(
            web_tools.get_tools()
        )

        return [
            _RegisteredTool(tool)
            for tool in raw_tools
        ]

    def get_tools(
        self,
    ) -> list[_RegisteredTool]:
        """
        Return all registered GAIA-compatible tools.
        """
        return list(self._tools)

    def get(
        self,
        tool_name: str,
    ) -> _RegisteredTool:
        """
        Return one registered tool by name.
        """

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError(
                "tool_name must be a non-empty string."
            )

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
        
    def get_spec(
        self,
        tool_name: str,
    ) -> ToolSpec:
       
        if not self._specs_by_name:
            self.get_tool_specs()

        try:
            return self._specs_by_name[
                tool_name
            ]

        except KeyError as exc:
            raise KeyError(
                f"No ToolSpec registered for tool "
                f"'{tool_name}'."
            ) from exc

    def get_tool_specs(
        self,
    ) -> list[ToolSpec]:
        """
        Return canonical contracts for every registered tool.

        Planner consumes these contracts instead of inspecting
        smolagents tools directly.
        """

        specs: list[ToolSpec] = []

        for tool in self._tools:
            spec = tool.build_spec()

            specs.append(spec)

        names = [
            spec.name
            for spec in specs
        ]

        if len(names) != len(set(names)):
            raise RuntimeError(
                "Duplicate ToolSpec names detected."
            )

        self._specs_by_name = {
            spec.name: spec
            for spec in specs
        }

        return list(specs)

    def validate_step(
        self,
        step: Any,
    ) -> dict[str, Any]:
        """
        Validate a TOOL PlanStep against the registered contracts.

        This is a convenience boundary for orchestration code.
        """

        return ToolContractValidator.validate_step_contract(
            step=step,
            available_tools=self._specs_by_name,
        )
    def has(
        self,
        tool_name: str,
    ) -> bool:
        """
        Return True when a tool is registered.
        """

        return tool_name in self._tools_by_name

    def names(self) -> list[str]:
        """
        Return registered tool names in deterministic order.
        """

        return sorted(
            self._tools_by_name
        )

    def capabilities(
        self,
    ) -> dict[str, ToolCapability]:
        """
        Return the explicit capability map for registered tools.
        """

        return {
            name: self.get_spec(name).capability
            for name in self.names()
        }