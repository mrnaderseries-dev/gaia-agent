from __future__ import annotations

from pathlib import Path

from smolagents import Tool

from gaia_agent.llm.service import LLMService
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


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
    }
)


class AnalyzeImageTool(Tool):
    name = "analyze_image"

    description = (
        "Analyze an image, chart, chess board, or diagram "
        "and answer a question using only the visual "
        "information contained in the image."
    )

    inputs = {
        "image_path": {
            "type": "string",
            "description": (
                "Path to the image relative to the allowed "
                "base directory or filename."
            ),
        },
        "question": {
            "type": "string",
            "description": (
                "Question that should be answered using "
                "the image."
            ),
        },
    }

    output_type = "string"

    def __init__(
        self,
        llm_service: LLMService,
        base_dir: str = ".",
    ) -> None:
        super().__init__()

        if llm_service is None:
            raise ValueError(
                "AnalyzeImageTool requires an LLMService."
            )

        self.llm_service = llm_service
        self.base_dir = Path(base_dir).resolve()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Path to the image relative to "
                            "the allowed base directory or "
                            "filename."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Question that should be answered "
                            "using the image."
                        ),
                    },
                },
                "required": [
                    "image_path",
                    "question",
                ],
            },
            capability=ToolCapability.READ_ONLY,
            modalities=frozenset(
                {ToolModality.VISION}
            ),
            result_schema={
                "type": "string",
            },
            error_codes=frozenset(
                {
                    ToolErrorCode.INVALID_ARGUMENT,
                    ToolErrorCode.FILE_NOT_FOUND,
                    ToolErrorCode.UNSUPPORTED_FORMAT,
                    ToolErrorCode.EXECUTION_FAILED,
                }
            ),
            allowed_imports=frozenset(),
            function=self.forward,
        )

    def forward(
        self,
        image_path: str,
        question: str,
    ) -> str:
        try:
            if (
                not isinstance(image_path, str)
                or not image_path.strip()
            ):
                return (
                    "Error: image_path must be a "
                    "non-empty string."
                )

            if (
                not isinstance(question, str)
                or not question.strip()
            ):
                return (
                    "Error: question must be a "
                    "non-empty string."
                )

            if is_placeholder_path(image_path):
                return (
                    f"Error: Image path '{image_path}' "
                    "is a placeholder or invalid."
                )

            path = resolve_file(
                self.base_dir,
                image_path,
            )

            if path is None:
                return (
                    f"Error: Image '{image_path}' was not "
                    "found in the allowed search locations."
                )

            if not path.exists():
                return (
                    f"Error: Image '{image_path}' does not exist."
                )

            if not path.is_file():
                return (
                    f"Error: '{image_path}' is not a file."
                )

            extension = path.suffix.lower()

            if extension not in SUPPORTED_IMAGE_EXTENSIONS:
                return (
                    f"Error: Unsupported image format "
                    f"'{extension}'. Supported formats: "
                    f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
                )

            prompt = (
                "You are solving a GAIA benchmark task using "
                "a visual input.\n\n"
                "Analyze the provided image carefully.\n"
                "Use ONLY information that is actually visible "
                "in the image.\n"
                "Do not invent missing information.\n"
                "If the question requires reading text, numbers, "
                "labels, a chart, a chess position, or a diagram, "
                "inspect the image carefully before answering.\n\n"
                f"Question:\n{question}\n\n"
                "Return the most precise answer possible."
            )

            response = self.llm_service.generate_image_sync(
                image_path=path,
                question=prompt,
                operation="llm.vision",
            )

            answer = str(response).strip()

            if not answer:
                return (
                    "Error: Vision model returned an empty "
                    "response."
                )

            return answer

        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return (
                "Error analyzing image: "
                f"{type(exc).__name__}: {exc}"
            )


class VisionTools:
    def __init__(
        self,
        llm_service: LLMService,
        base_dir: str = ".",
    ) -> None:
        if llm_service is None:
            raise ValueError(
                "VisionTools requires an LLMService."
            )

        self.llm_service = llm_service
        self.base_dir = Path(base_dir).resolve()

    def get_tools(self) -> list[Tool]:
        return [
            AnalyzeImageTool(
                llm_service=self.llm_service,
                base_dir=str(self.base_dir),
            )
        ]