from __future__ import annotations

from pathlib import Path
from typing import Any

from smolagents import Tool

try:
    from smolagents import YoutubeTranscriptTool
except ImportError:
    YoutubeTranscriptTool = None


class AudioTools:
    """
    Audio/media tools.

    Currently this container exposes YouTube transcript extraction
    when supported by the installed smolagents version.

    Local audio STT is intentionally not faked here.
    """

    def __init__(
        self,
        stt_backend: Any = None,
        base_dir: str = ".",
        model: Any = None,
    ) -> None:
        self.stt_backend = stt_backend
        self.base_dir = Path(base_dir).resolve()
        self.model = model

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = []

        if YoutubeTranscriptTool is not None:
            tools.append(YoutubeTranscriptTool())

        return tools