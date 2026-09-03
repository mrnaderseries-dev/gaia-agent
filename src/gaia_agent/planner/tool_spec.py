from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class ToolCapability(str, Enum):
    READ_ONLY = "read_only"
    COMPUTATION = "computation"
    NETWORK_READ = "network_read"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    "web_search": ToolCapability.NETWORK_READ,
    "visit_webpage": ToolCapability.NETWORK_READ,
    "youtube_transcript": ToolCapability.NETWORK_READ,

    "python_interpreter": ToolCapability.COMPUTATION,

    "file_reader": ToolCapability.READ_ONLY,
    "analyze_image": ToolCapability.READ_ONLY,
    "analyze_excel": ToolCapability.READ_ONLY,
    "transcribe_audio": ToolCapability.READ_ONLY,
}


class ToolSpec(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    name: str = Field(
        ...,
        description="Unique registered tool name",
    )

    description: str = Field(
        ...,
        description="What the tool does",
    )

    arguments_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Exact arguments accepted by the tool",
    )

    capability: ToolCapability = Field(
        ...,
        description="Security/side-effect capability",
    )

    result_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Shape/type of the tool result",
    )

    error_codes: list[str] = Field(
        default_factory=list
    )

    allowed_imports: list[str] = Field(
        default_factory=list
    )

    function: Callable[..., Any] | None = None