from __future__ import annotations

from pathlib import Path
from typing import Any, List

from smolagents import Tool

from gaia_agent.tools.path_utils import (
    is_placeholder_path,
    resolve_file,
)


class FileReaderTool(Tool):
    name = "file_reader"

    description = (
        "Read local text, markdown, CSV, JSON, or configuration files securely "
        "and handle multi-format file evaluation tasks."
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
        self.base_dir = Path(base_dir).resolve()

    def forward(
        self,
        file_path: str,
    ) -> str:
        try:
            if is_placeholder_path(file_path):
                return (
                    f"Error: File path '{file_path}' is a placeholder "
                    "or invalid. You must use a real file path that "
                    "exists in the environment."
                )

            path = resolve_file(
                self.base_dir,
                file_path,
            )

            if path is None:
                return (
                    f"Error: File '{file_path}' not found in base_dir "
                    f"'{self.base_dir}' or any working directory. "
                    "Resolve the actual attachment before retrying; "
                    "do not invent file paths."
                )

            if not path.is_file():
                return f"Error: Path '{file_path}' is not a valid file."

            # قراءة محتوى الملف مع تجنب مشاكل الـ Encoding
            encodings = ["utf-8", "latin-1", "cp1252"]
            content = None

            for enc in encodings:
                try:
                    with open(path, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                return f"Error: Failed to decode file '{file_path}' with supported encodings."

            return content

        except Exception as exc:
            return f"Error reading file '{file_path}': {exc}"


class FileTools:
    """
    File tools container optimized for GAIA benchmark evaluation, robust against path issues and missing files.
    """

    def __init__(
        self,
        base_dir: str = ".",
    ) -> None:
        self.base_dir = Path(base_dir).resolve()

    def get_tools(self) -> List[Tool]:
        """
        Create and return all file handling tools.
        """
        return [
            FileReaderTool(
                base_dir=str(self.base_dir),
            )
        ]