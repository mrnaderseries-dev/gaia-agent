from __future__ import annotations

from typing import Any

from smolagents import (
    DuckDuckGoSearchTool,
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
            arguments_schema={
                "type": "object",
                "properties": dict(self.inputs),
                "required": ["query"],
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset(
                {
                    ToolModality.TEXT,
                }
            ),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset(
                {
                    ToolErrorCode.INVALID_ARGUMENT,
                    ToolErrorCode.NETWORK_ERROR,
                    ToolErrorCode.RATE_LIMITED,
                    ToolErrorCode.TIMEOUT,
                }
            ),
            allowed_imports=frozenset(),
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
            arguments_schema={
                "type": "object",
                "properties": dict(self.inputs),
                "required": ["url"],
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset(
                {
                    ToolModality.TEXT,
                }
            ),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset(
                {
                    ToolErrorCode.INVALID_ARGUMENT,
                    ToolErrorCode.NETWORK_ERROR,
                    ToolErrorCode.RATE_LIMITED,
                    ToolErrorCode.TIMEOUT,
                    ToolErrorCode.NOT_FOUND,
                }
            ),
            allowed_imports=frozenset(),
            function=self,
        )

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        return self._tool(**kwargs)


class YoutubeTranscriptFetcher:
    name = "youtube_transcript"

    def __init__(self) -> None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:
            raise ImportError(
                "youtube-transcript-api is not installed"
            ) from exc

        self._api = YouTubeTranscriptApi
        self.description = (
            "Fetch the transcript/captions for a YouTube video. "
            "Input is a YouTube watch URL or video ID."
        )
        self.inputs = {
            "video_url": {
                "type": "string",
                "description": "YouTube watch URL or video ID",
            }
        }
        self.output_type = "string"

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema={
                "type": "object",
                "properties": dict(self.inputs),
                "required": ["video_url"],
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset(
                {
                    ToolModality.VIDEO,
                    ToolModality.TEXT,
                }
            ),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset(
                {
                    ToolErrorCode.INVALID_ARGUMENT,
                    ToolErrorCode.NOT_FOUND,
                    ToolErrorCode.NETWORK_ERROR,
                    ToolErrorCode.RATE_LIMITED,
                    ToolErrorCode.TIMEOUT,
                }
            ),
            allowed_imports=frozenset(),
            function=self,
        )

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        video_url = str(kwargs.get("video_url", "")).strip()
        if not video_url:
            raise ValueError("video_url is required.")
        video_id = self._video_id(video_url)
        try:
            api = self._api() if isinstance(self._api, type) else self._api
            if hasattr(api, "get_transcript"):
                chunks = api.get_transcript(video_id)
            else:
                fetched = api.fetch(video_id)
                chunks = (
                    fetched.to_raw_data()
                    if hasattr(fetched, "to_raw_data")
                    else list(fetched)
                )
        except Exception as exc:
            raise RuntimeError(
                f"YouTube transcript unavailable for {video_id}: {exc}"
            ) from exc
        text = " ".join(
            str(item.get("text", "")).replace("\n", " ").strip()
            for item in chunks
            if isinstance(item, dict)
        ).strip()
        if not text:
            raise RuntimeError(
                f"YouTube transcript is empty for {video_id}."
            )
        return text

    @staticmethod
    def _video_id(value: str) -> str:
        text = value.strip()
        if "youtu.be/" in text:
            return text.split("youtu.be/", 1)[1].split("?", 1)[0].split("&", 1)[0]
        if "v=" in text:
            return text.split("v=", 1)[1].split("&", 1)[0].split("?", 1)[0]
        if "/shorts/" in text:
            return text.rsplit("/shorts/", 1)[1].split("?", 1)[0].split("&", 1)[0]
        if "/embed/" in text:
            return text.rsplit("/embed/", 1)[1].split("?", 1)[0].split("&", 1)[0]
        return text


class SafeYoutubeTranscript:
    name = "youtube_transcript"

    def __init__(self) -> None:
        from smolagents import YoutubeTranscriptTool

        self._tool = YoutubeTranscriptTool()

        self.name = self._tool.name
        self.description = self._tool.description
        self.inputs = self._tool.inputs
        self.output_type = self._tool.output_type

        properties = dict(self.inputs)

        required = [
            name
            for name in (
                "video_url",
                "url",
                "youtube_url",
            )
            if name in properties
        ]

        if not required and properties:
            required = [next(iter(properties))]

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            capability=ToolCapability.NETWORK_READ,
            modalities=frozenset(
                {
                    ToolModality.VIDEO,
                    ToolModality.TEXT,
                }
            ),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset(
                {
                    ToolErrorCode.INVALID_ARGUMENT,
                    ToolErrorCode.NOT_FOUND,
                    ToolErrorCode.NETWORK_ERROR,
                    ToolErrorCode.RATE_LIMITED,
                    ToolErrorCode.TIMEOUT,
                }
            ),
            allowed_imports=frozenset(),
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

        self._tools: list[Any] = [
            self.search,
            self.visit,
        ]

        try:
            self.youtube = SafeYoutubeTranscript()
        except ImportError:
            try:
                self.youtube = YoutubeTranscriptFetcher()
            except ImportError:
                self.youtube = None

        if self.youtube is not None:
            self._tools.append(self.youtube)

    def get_tools(self) -> list[Any]:
        return list(self._tools)

    def get_specs(self) -> list[ToolSpec]:
        return [
            tool.spec
            for tool in self._tools
        ]