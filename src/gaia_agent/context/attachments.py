from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Attachment:
    attachment_id: str
    filename: str
    path: str

    def __post_init__(self) -> None:
        if not self.attachment_id.strip():
            raise ValueError("attachment_id must not be empty")

        if not self.filename.strip():
            raise ValueError("filename must not be empty")

        if not self.path.strip():
            raise ValueError("path must not be empty")