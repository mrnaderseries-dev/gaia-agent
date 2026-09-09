from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .task_classifier import TaskAnalysis, TaskIntent


class StrategyFamily(str, Enum):
    LOCAL_COMPUTATION = "PYTHON"
    LOCAL_TRANSFORMATION = "PYTHON"
    DIRECT_URL = "DIRECT_WEBPAGE"
    VISION = "VISION"
    FILE_ANALYSIS = "FILE_ANALYSIS"
    FILE_READING = "FILE_READER"
    WEB_RETRIEVAL = "WEB_SEARCH"
    AUDIO_VIDEO = "AUDIO_VIDEO"
    LLM_ONLY = "LLM"


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    strategy: StrategyFamily
    primary_tool: str | None
    reason: str
    deterministic: bool


@dataclass(frozen=True, slots=True)
class StrategyContext:
    available_tools: frozenset[str]
    available_files: tuple[str, ...] = ()
    failed_strategy: str | None = None
    failure_type: str | None = None

    @property
    def has_image(self) -> bool:
        return any(
            Path(name).suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".bmp",
                ".tiff",
                ".tif",
            }
            for name in self.available_files
        )

    @property
    def has_spreadsheet(self) -> bool:
        return any(
            Path(name).suffix.lower()
            in {
                ".xlsx",
                ".xls",
                ".xlsm",
                ".csv",
            }
            for name in self.available_files
        )


class StrategySelector:
    """
    Deterministic task-to-strategy selection.

    This class selects a strategy family, not an arbitrary tool.
    The Planner may still ask the LLM to construct the detailed
    multi-step plan, but the selected family becomes a hard
    semantic constraint.
    """

    _NON_TRANSIENT_FAILURE_MARKERS = (
        "capability",
        "access",
        "blocked",
        "forbidden",
        "permission",
        "loop",
        "repeated",
        "not supported",
        "unavailable",
    )

    _INVALID_ARGUMENT_MARKERS = (
        "argument",
        "validation",
        "schema",
        "contract",
        "keyword",
        "unexpected keyword",
    )

    def select(
        self,
        analysis: TaskAnalysis,
        context: StrategyContext,
    ) -> StrategyDecision:
        tools = context.available_tools

        if analysis.intent == TaskIntent.ARITHMETIC:
            return StrategyDecision(
                StrategyFamily.LOCAL_COMPUTATION,
                "python_interpreter"
                if "python_interpreter" in tools
                else None,
                "Arithmetic is a local deterministic computation.",
                "python_interpreter" in tools,
            )

        if analysis.intent == TaskIntent.TEXT_TRANSFORMATION:
            return StrategyDecision(
                StrategyFamily.LOCAL_TRANSFORMATION,
                "python_interpreter"
                if "python_interpreter" in tools
                else None,
                (
                    "Text transformation is self-contained and "
                    "does not require retrieval."
                ),
                "python_interpreter" in tools,
            )

        if analysis.intent == TaskIntent.URL_PAGE:
            return StrategyDecision(
                StrategyFamily.DIRECT_URL,
                "visit_webpage"
                if "visit_webpage" in tools
                else None,
                "A direct URL should be accessed directly before search.",
                "visit_webpage" in tools,
            )

        if analysis.intent == TaskIntent.IMAGE:
            if context.has_image and "analyze_image" in tools:
                return StrategyDecision(
                    StrategyFamily.VISION,
                    "analyze_image",
                    (
                        "A real local image exists, so vision analysis "
                        "is the correct capability."
                    ),
                    True,
                )

            return StrategyDecision(
                StrategyFamily.VISION,
                "analyze_image"
                if "analyze_image" in tools
                else None,
                (
                    "The task is image-oriented; use vision when a "
                    "real image can be resolved."
                ),
                False,
            )

        if analysis.intent == TaskIntent.LOCAL_FILE:
            if context.has_spreadsheet and "analyze_excel" in tools:
                return StrategyDecision(
                    StrategyFamily.FILE_ANALYSIS,
                    "analyze_excel",
                    (
                        "A spreadsheet artifact exists and should be "
                        "analyzed with the spreadsheet tool."
                    ),
                    True,
                )

            if "file_reader" in tools and context.available_files:
                return StrategyDecision(
                    StrategyFamily.FILE_READING,
                    "file_reader",
                    (
                        "A real local artifact exists and should be "
                        "read directly."
                    ),
                    True,
                )

        if analysis.intent == TaskIntent.AUDIO_VIDEO:
            media_tools = (
                "youtube_transcript",
                "transcribe_audio",
                "video_reader",
                "audio_reader",
                "analyze_video",
            )

            available_media_tool = next(
                (
                    tool
                    for tool in media_tools
                    if tool in tools
                ),
                None,
            )

            if available_media_tool is not None:
                return StrategyDecision(
                    StrategyFamily.AUDIO_VIDEO,
                    available_media_tool,
                    (
                        "A dedicated media capability is available "
                        "for the task."
                    ),
                    True,
                )

            return StrategyDecision(
                StrategyFamily.AUDIO_VIDEO,
                None,
                (
                    "The task requires audio/video capability, "
                    "but no dedicated media tool is registered."
                ),
                False,
            )

        if analysis.intent == TaskIntent.FACTUAL_SEARCH:
            tool = analysis.recommended_first_tool

            if tool in tools:
                family = (
                    StrategyFamily.DIRECT_URL
                    if tool == "visit_webpage"
                    else StrategyFamily.WEB_RETRIEVAL
                )

                return StrategyDecision(
                    family,
                    tool,
                    "External factual evidence is required.",
                    False,
                )

            return StrategyDecision(
                StrategyFamily.WEB_RETRIEVAL,
                "web_search"
                if "web_search" in tools
                else None,
                (
                    "External factual evidence is required when a "
                    "retrieval tool exists."
                ),
                False,
            )

        return StrategyDecision(
            StrategyFamily.LLM_ONLY,
            None,
            "The task is self-contained; no external capability is required.",
            True,
        )

    def select_alternative(
        self,
        analysis: TaskAnalysis,
        context: StrategyContext,
    ) -> StrategyDecision | None:
        failure = (
            context.failure_type
            or ""
        ).lower().replace("-", "_")

        failed = context.failed_strategy

        if any(
            marker in failure
            for marker in self._INVALID_ARGUMENT_MARKERS
        ):
            return self.select(analysis, context)

        if not any(
            marker in failure
            for marker in self._NON_TRANSIENT_FAILURE_MARKERS
        ):
            return None

        candidates: list[StrategyDecision] = []

        primary = self.select(analysis, context)
        candidates.append(primary)

        tools = context.available_tools

        if (
            analysis.intent == TaskIntent.URL_PAGE
            and "visit_webpage" in tools
        ):
            candidates.insert(
                0,
                StrategyDecision(
                    StrategyFamily.DIRECT_URL,
                    "visit_webpage",
                    (
                        "Use the exact URL instead of the failed "
                        "retrieval strategy."
                    ),
                    True,
                ),
            )

        if (
            analysis.intent
            in (
                TaskIntent.ARITHMETIC,
                TaskIntent.TEXT_TRANSFORMATION,
            )
            and "python_interpreter" in tools
        ):
            candidates.insert(
                0,
                StrategyDecision(
                    StrategyFamily.LOCAL_COMPUTATION
                    if analysis.intent == TaskIntent.ARITHMETIC
                    else StrategyFamily.LOCAL_TRANSFORMATION,
                    "python_interpreter",
                    (
                        "Fall back to deterministic local computation/"
                        "transformation."
                    ),
                    True,
                ),
            )

        if (
            analysis.intent == TaskIntent.IMAGE
            and context.has_image
            and "analyze_image" in tools
        ):
            candidates.insert(
                0,
                StrategyDecision(
                    StrategyFamily.VISION,
                    "analyze_image",
                    (
                        "Use the real local image instead of external "
                        "retrieval."
                    ),
                    True,
                ),
            )

        for candidate in candidates:
            if candidate.strategy.value != (failed or ""):
                return candidate

        return None