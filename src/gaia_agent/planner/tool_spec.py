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
    WEB = "web"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    SPREADSHEET = "spreadsheet"
    CODE = "code"


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    FILE_NOT_FOUND = "file_not_found"
    INVALID_FILE = "invalid_file"
    DECODE_ERROR = "decode_error"
    INVALID_IMAGE = "invalid_image"
    INVALID_SPREADSHEET = "invalid_spreadsheet"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNSUPPORTED_MODALITY = "unsupported_modality"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VIDEO_UNAVAILABLE = "video_unavailable"
    AUDIO_UNAVAILABLE = "audio_unavailable"
    TRANSCRIPTION_ERROR = "transcription_error"
    VISION_ERROR = "vision_error"
    EXECUTION_ERROR = "execution_error"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    LLM_ERROR = "llm_error"


class ToolSpec(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    name: str = Field(
        ...,
        min_length=1,
    )

    description: str

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

    allowed_imports: tuple[str, ...] = Field(
        default_factory=tuple,
    )

    function: Callable[..., Any] | None = Field(
        default=None,
        exclude=True,
    )

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