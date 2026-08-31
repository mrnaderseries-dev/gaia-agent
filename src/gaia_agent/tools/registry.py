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


# Modules declared as available inside the python sandbox.
# Single source of truth shared with the PythonInterpreterTool.
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
    Adapter between GAIA's tool execution contract
    and smolagents tools.

    GAIA expects:

        await tool.execute(**arguments)

    while smolagents tools are normally invoked as:

        tool(**arguments)

    This adapter keeps that difference isolated
    inside the ToolRegistry.
    """

    def __init__(
        self,
        tool: Any,
    ) -> None:

        self._tool = tool

        self.name = tool.name
        self.description = tool.description
        self.inputs = getattr(
            tool,
            "inputs",
            {},
        )

        self.output_type = getattr(
            tool,
            "output_type",
            "string",
        )

    async def execute(
        self,
        **arguments: Any,
    ) -> Any:

        return self._tool(
            **arguments,
        )

    def validate_arguments(
        self,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Validate caller-provided arguments against this tool's
        registered contract. Raises ToolArgumentError before the
        underlying implementation is reached.
        """
        spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema=dict(self.inputs or {}),
        )

        return ToolContractValidator.validate_arguments(
            spec=spec,
            arguments=arguments,
        )


class ToolRegistry:
    """
    Central registry for all GAIA agent tools.

    The registry is responsible for:

    - Creating tool instances.
    - Keeping tools indexed by name.
    - Returning GAIA-compatible tools for execution.
    - Returning ToolSpec objects for the planner.
    - Adapting smolagents tools to the GAIA execution contract.
    """

    def __init__(
        self,
        base_dir: str = ".",
        model=None,
        stt_backend=None,
    ) -> None:

        self.base_dir = base_dir
        self.model = model
        self.stt_backend = stt_backend

        self._tools = self._build_tools()

        self._tools_by_name = {
            tool.name: tool
            for tool in self._tools
        }

        # Populated when get_tool_specs() builds the contracts.
        self._specs_by_name: dict[str, ToolSpec] = {}

    # ==========================================================
    # BUILD TOOLS
    # ==========================================================

    def _build_tools(self) -> list[_RegisteredTool]:
        """
        Create all tools available to the agent
        and wrap them with the GAIA execution adapter.
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
            base_dir=self.base_dir,
            model=self.model,
        )

        excel_tools = ExcelTools(
            base_dir=self.base_dir,
            model=self.model,
        )

        python_tools = PythonTools()

        web_tools = WebTools()

        raw_tools = []

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

    # ==========================================================
    # GET ALL TOOLS
    # ==========================================================

    def get_tools(self) -> list[_RegisteredTool]:
        """
        Return all registered GAIA-compatible tools.
        """

        return list(
            self._tools
        )

    # ==========================================================
    # GET TOOL
    # ==========================================================

    def get(
        self,
        tool_name: str,
    ) -> _RegisteredTool:
        """
        Return a registered tool by name.

        The returned object exposes the GAIA execution
        contract:

            await tool.execute(**arguments)
        """

        if not tool_name:
            raise ValueError(
                "tool_name cannot be empty."
            )

        try:

            return self._tools_by_name[
                tool_name
            ]

        except KeyError:

            raise KeyError(
                f"Tool '{tool_name}' is not registered."
            )

    # ==========================================================
    # TOOL SPECS
    # ==========================================================

    def get_spec(
        self,
        tool_name: str,
    ) -> ToolSpec:
        """
        Return the ToolSpec contract for a registered tool.
        """
        return self._specs_by_name[tool_name]

    def get_tool_specs(
        self,
    ) -> list[ToolSpec]:
        """
        Convert registered tools into ToolSpec objects
        for the planner.
        """
        specs: list[ToolSpec] = []

        for tool in self._tools:

            arguments_schema = getattr(
                tool,
                "inputs",
                None,
            )

            name = tool.name

            capability = TOOL_CAPABILITIES.get(
                name,
                ToolCapability.READ_ONLY,
            )

            allowed_imports: list[str] = []

            if name == "python_interpreter":
                allowed_imports = list(PYTHON_ALLOWED_IMPORTS)

            spec = ToolSpec(
                name=name,
                description=tool.description,
                arguments_schema=dict(arguments_schema or {}),
                capability=capability,
                result_schema={
                    "type": getattr(
                        tool,
                        "output_type",
                        "string",
                    ),
                },
                error_codes=[],
                allowed_imports=allowed_imports,
            )

            specs.append(spec)

        self._specs_by_name = {
            spec.name: spec
            for spec in specs
        }

        return specs