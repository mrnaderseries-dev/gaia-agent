from __future__ import annotations

from typing import Any

from smolagents import (
    DuckDuckGoSearchTool,
    Tool,
    VisitWebpageTool,
)

from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolErrorCode,
    ToolModality,
    ToolSpec,
)


class SafeDuckDuckGoSearch:
    name = "web_search"

    def __init__(self) -> None:
        self._tool = DuckDuckGoSearchTool()

        self.name = self._tool.name
        self.description = self._tool.description
        self.inputs = self._tool.inputs
        self.output_type = self._tool.output_type

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema=dict(self.inputs),
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset({
                ToolModality.WEB,
                ToolModality.TEXT,
            }),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset({
                ToolErrorCode.NETWORK_ERROR,
                ToolErrorCode.RATE_LIMIT,
                ToolErrorCode.TIMEOUT,
            }),
            function=self,
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {
                "query",
                "max_results",
            }
        }

        return self._tool(
            *args,
            **clean_kwargs,
        )


class SafeVisitWebpage:
    name = "visit_webpage"

    def __init__(self) -> None:
        self._tool = VisitWebpageTool()

        self.name = self._tool.name
        self.description = self._tool.description
        self.inputs = self._tool.inputs
        self.output_type = self._tool.output_type

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema=dict(self.inputs),
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset({
                ToolModality.WEB,
                ToolModality.TEXT,
            }),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset({
                ToolErrorCode.NETWORK_ERROR,
                ToolErrorCode.RATE_LIMIT,
                ToolErrorCode.TIMEOUT,
            }),
            function=self,
        )

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        return self._tool(**kwargs)


class SafeYoutubeTranscript:
    name = "youtube_transcript"

    def __init__(self) -> None:
        from smolagents import YoutubeTranscriptTool

        self._tool = YoutubeTranscriptTool()

        self.name = self._tool.name
        self.description = self._tool.description
        self.inputs = self._tool.inputs
        self.output_type = self._tool.output_type

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema=dict(self.inputs),
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset({
                ToolModality.VIDEO,
                ToolModality.TEXT,
            }),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset({
                ToolErrorCode.VIDEO_UNAVAILABLE,
                ToolErrorCode.NETWORK_ERROR,
                ToolErrorCode.RATE_LIMIT,
                ToolErrorCode.TIMEOUT,
            }),
            function=self,
        )

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        return self._tool(**kwargs)


class WebTools:
    def __init__(self) -> None:
        self.search = SafeDuckDuckGoSearch()
        self.visit = SafeVisitWebpage()

        self._tools = [
            self.search,
            self.visit,
        ]

        try:
            self.youtube = SafeYoutubeTranscript()
        except ImportError:
            self.youtube = None

        if self.youtube is not None:
            self._tools.append(
                self.youtube
            )

    def get_tools(self) -> list[Any]:
        return list(self._tools)

    def get_specs(self) -> list[ToolSpec]:
        return [
            tool.spec
            for tool in self._tools
        ]