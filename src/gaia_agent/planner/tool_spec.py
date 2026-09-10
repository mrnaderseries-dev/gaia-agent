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


class ToolModality(str, Enum):
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    EXCEL = "excel"


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENT = "invalid_argument"
    FILE_NOT_FOUND = "file_not_found"
    UNSUPPORTED_FORMAT = "unsupported_format"
    EXECUTION_FAILED = "execution_failed"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    INVALID_FILE = "invalid_file"
    DECODE_ERROR = "decode_error"





class ToolSpec(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    arguments_schema: dict[str, Any] = Field(
        default_factory=dict,
    )

    capability: ToolCapability

    modalities: frozenset[ToolModality] = Field(
        default_factory=frozenset,
    )

    result_schema: dict[str, Any] = Field(
        default_factory=dict,
    )

    error_codes: frozenset[ToolErrorCode] = Field(
        default_factory=frozenset,
    )

    allowed_imports: frozenset[str] = Field(
        default_factory=frozenset,
    )

    function: Callable[..., Any] | None = None

    def supports_modality(
        self,
        modality: ToolModality,
    ) -> bool:
        return modality in self.modalities

    def supports_capability(
        self,
        capability: ToolCapability,
    ) -> bool:
        return self.capability == capability