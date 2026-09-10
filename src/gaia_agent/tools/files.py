from __future__ import annotations

from pathlib import Path
from typing import Any

from smolagents import Tool

from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolErrorCode,
    ToolModality,
    ToolSpec,
)
from gaia_agent.tools.path_utils import (
    is_placeholder_path,
    resolve_file,
)


class FileReaderTool(Tool):
    name = "file_reader"

    description = (
        "Read local text, markdown, CSV, JSON, or configuration "
        "files securely and handle multi-format file evaluation tasks."
    )

    inputs = {
        "file_path": {
            "type": "string",
            "description": (
                "Path to the target file relative to "
                "the allowed base directory or filename."
            ),
        }
    }

    output_type = "string"

    def __init__(
        self,
        base_dir: str = ".",
    ) -> None:
        super().__init__()

        self.base_dir = Path(
            base_dir
        ).resolve()

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema=dict(
                self.inputs
            ),
            capability=ToolCapability.READ_ONLY,
            modalities=frozenset({
                ToolModality.FILE,
                ToolModality.TEXT,
            }),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset({
                ToolErrorCode.FILE_NOT_FOUND,
                ToolErrorCode.INVALID_FILE,
                ToolErrorCode.DECODE_ERROR,
            }),
            function=self.forward,
        )

    def forward(
        self,
        file_path: str,
    ) -> str:
        try:
            if is_placeholder_path(file_path):
                return (
                    f"Error: File path '{file_path}' is "
                    "a placeholder or invalid."
                )

            path = resolve_file(
                self.base_dir,
                file_path,
            )

            if path is None:
                return (
                    f"Error: File '{file_path}' not found."
                )

            if not path.is_file():
                return (
                    f"Error: Path '{file_path}' "
                    "is not a valid file."
                )

            content = None

            for encoding in (
                "utf-8",
                "latin-1",
                "cp1252",
            ):
                try:
                    with open(
                        path,
                        "r",
                        encoding=encoding,
                    ) as handle:
                        content = handle.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                return (
                    f"Error: Failed to decode "
                    f"file '{file_path}'."
                )

            return content

        except Exception as exc:
            return (
                f"Error reading file '{file_path}': "
                f"{type(exc).__name__}: {exc}"
            )


class FileTools:
    def __init__(
        self,
        base_dir: str = ".",
    ) -> None:
        self.base_dir = Path(
            base_dir
        ).resolve()

    def get_tools(self) -> list[Tool]:
        return [
            FileReaderTool(
                base_dir=str(
                    self.base_dir
                ),
            )
        ]