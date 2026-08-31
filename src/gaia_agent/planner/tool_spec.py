from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCapability(str, Enum):
    """
    Capability classification used for planning, risk assessment
    and approval policy (READ_ONLY / COMPUTATION / NETWORK_READ /
    EXTERNAL_WRITE / DESTRUCTIVE).
    """
    READ_ONLY = "read_only"
    COMPUTATION = "computation"
    NETWORK_READ = "network_read"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


# Deterministic tool-name -> capability map.
# Single source of truth for planner contracts and risk rules.
TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    "web_search": ToolCapability.NETWORK_READ,
    "visit_webpage": ToolCapability.NETWORK_READ,
    "youtube_transcript": ToolCapability.NETWORK_READ,
    "python_interpreter": ToolCapability.COMPUTATION,
    "file_reader": ToolCapability.READ_ONLY,
    "analyze_image": ToolCapability.READ_ONLY,
    "analyze_excel": ToolCapability.READ_ONLY,
}


class ToolSpec(BaseModel):
    """Formal tool contract (Phase 1 - Tool Contracts)."""
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Detailed description of what the tool does")
    arguments_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema describing the exact arguments the tool accepts"
    )
    capability: ToolCapability = Field(
        default=ToolCapability.READ_ONLY,
        description="Capability class used by planner and risk policy",
    )
    result_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema describing the tool result shape",
    )
    error_codes: List[str] = Field(
        default_factory=list,
        description="Error codes the tool is known to emit",
    )
    allowed_imports: List[str] = Field(
        default_factory=list,
        description="Modules available inside the python sandbox (python_interpreter only)",
    )
    function: Optional[Callable[..., Any]] = Field(
        default=None,
        description="Executable Python callable function",
    )

    class Config:
        arbitrary_types_allowed = True
        