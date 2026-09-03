from __future__ import annotations

from pathlib import Path
from typing import Any

from smolagents import Tool

from gaia_agent.tools.path_utils import (
    is_placeholder_path,
    resolve_file,
)


class AnalyzeImageTool(Tool):

    name = "analyze_image"

    description = (
        "Analyze an image, chart, diagram, or visual "
        "and answer a question using only visible information."
    )

    inputs = {
        "image_path": {
            "type": "string",
            "description": "Image filename or path.",
        },
        "question": {
            "type": "string",
            "description": "Question about the image.",
        },
    }

    output_type = "string"

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
    }

    def __init__(
        self,
        model: Any = None,
        base_dir: str = ".",
    ) -> None:

        super().__init__()

        self.model = model
        self.base_dir = Path(
            base_dir
        ).resolve()

    def forward(
        self,
        image_path: str,
        question: str,
    ) -> str:

        if is_placeholder_path(image_path):
            raise ValueError(
                "Image path is a placeholder or invalid."
            )

        path = resolve_file(
            self.base_dir,
            image_path,
        )

        if path is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Not a file: {image_path}"
            )

        if (
            path.suffix.lower()
            not in self.SUPPORTED_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported image format: "
                f"{path.suffix}"
            )

        if self.model is None:
            raise RuntimeError(
                "Image analysis requires a configured "
                "vision-capable language model."
            )

        prompt = f"""
Analyze the image for a GAIA benchmark task.

Question:
{question}

Rules:
- Use only information visible in the image.
- Do not use outside knowledge.
- Do not invent missing details.
- Return only the answer.
"""

        return str(
            self.model.generate(
                prompt=prompt,
                image=str(path),
            )
        )


class VisionTools:

    def __init__(
        self,
        model: Any = None,
        base_dir: str = ".",
    ) -> None:

        self.model = model
        self.base_dir = Path(
            base_dir
        ).resolve()

    def get_tools(self) -> list[Tool]:

        return [
            AnalyzeImageTool(
                model=self.model,
                base_dir=str(
                    self.base_dir
                ),
            )
        ]