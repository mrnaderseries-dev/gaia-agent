from __future__ import annotations

from pathlib import Path
from typing import Any, List

from smolagents import Tool

try:
    from smolagents import YoutubeTranscriptTool
except ImportError:
    YoutubeTranscriptTool = None


class AudioTools:
    """
    Audio and YouTube transcription tools container optimized for GAIA benchmark evaluation.
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

    def get_tools(self) -> List[Tool]:
        """
        Create and return all audio and YouTube tools.
        """
        tools = []
        
     
        if YoutubeTranscriptTool is not None:
            tools.append(YoutubeTranscriptTool())
            
        return tools