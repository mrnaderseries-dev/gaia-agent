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
        "Analyze an image, chart, chess board, or diagram and answer a question about "
        "the visual information contained in the image."
    )

    inputs = {
        "image_path": {
            "type": "string",
            "description": (
                "Path to the image relative to "
                "the allowed base directory or filename."
            ),
        },
        "question": {
            "type": "string",
            "description": (
                "Question that should be answered "
                "using the image."
            ),
        },
    }

    output_type = "string"

    def __init__(
        self,
        model: Any,
        base_dir: str = ".",
    ) -> None:
        super().__init__()
        self.model = model
        self.base_dir = Path(base_dir).resolve()

    def forward(
        self,
        image_path: str,
        question: str,
    ) -> str:
        try:
            if is_placeholder_path(image_path):
                return f"Image not found: {image_path} (placeholder or invalid path)."

            path = resolve_file(self.base_dir, image_path)

            if path is None or not path.exists():
                filename = Path(image_path).name
                possible_paths = [
                    self.base_dir / filename,
                    Path.cwd() / filename,
                    Path.cwd() / "src" / filename,
                    Path(image_path)
                ]
                
                found = False
                for p in possible_paths:
                    if p.exists() and p.is_file():
                        path = p.resolve()
                        found = True
                        break
                
                if not found:
                    return f"Image not found: {image_path} (searched in base_dir: {self.base_dir})"

            if not path.is_file():
                return f"Not a file: {image_path}"

            valid_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
                ".gif",
            }

            if path.suffix.lower() not in valid_extensions:
                return f"Unsupported image format: {path.suffix}"

            if self.model is None:
                return f"Vision fallback mock analysis for '{path.name}' regarding query: '{question}'."

            prompt = (
                "Analyze the provided image carefully to solve a GAIA benchmark evaluation task.\n\n"
                f"Question: {question}\n\n"
                "Give a precise and accurate answer based only on "
                "the visual information available in the image."
            )

            response = self.model.generate(
                prompt=prompt,
                image=str(path),
            )

            return str(response)

        except Exception as exc:
            return f"Error analyzing image: {exc}"


class VisionTools:
    """
    Vision tools for the GAIA agent, fully optimized to handle visual evaluation tasks.
    """

    def __init__(
        self,
        model: Any,
        base_dir: str = ".",
    ) -> None:
        self.model = model
        self.base_dir = Path(base_dir).resolve()

    def get_tools(self) -> list[Tool]:
        """
        Create and return all vision tools.
        """
        return [
            AnalyzeImageTool(
                model=self.model,
                base_dir=str(self.base_dir),
            )
        ]