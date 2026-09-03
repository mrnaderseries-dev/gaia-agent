from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from smolagents import Tool

from gaia_agent.tools.path_utils import (
    is_placeholder_path,
    resolve_file,
)


class AnalyzeExcelTool(Tool):
    name = "analyze_excel"

    description = (
        "Analyze an Excel or CSV spreadsheet and answer "
        "a question using only the spreadsheet contents."
    )

    inputs = {
        "file_path": {
            "type": "string",
            "description": "Spreadsheet filename or path.",
        },
        "question": {
            "type": "string",
            "description": "Question to answer from the spreadsheet.",
        },
    }

    output_type = "string"

    SUPPORTED_EXTENSIONS = {
        ".xlsx",
        ".xlsm",
        ".csv",
        ".tsv",
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

    def _read_workbook(
        self,
        path: Path,
    ) -> str:

        workbook = load_workbook(
            filename=path,
            data_only=True,
            read_only=True,
        )

        try:

            output: list[str] = []

            for sheet in workbook.worksheets:

                output.append(
                    f"Sheet: {sheet.title}"
                )

                for index, row in enumerate(
                    sheet.iter_rows(
                        values_only=True
                    )
                ):

                    if index >= 5000:
                        output.append(
                            "[TRUNCATED]"
                        )
                        break

                    values = [
                        "" if value is None
                        else str(value)
                        for value in row
                    ]

                    if any(values):
                        output.append(
                            " | ".join(values)
                        )

                output.append("")

            return "\n".join(output)

        finally:
            workbook.close()

    def _read_delimited(
        self,
        path: Path,
    ) -> str:

        delimiter = (
            "\t"
            if path.suffix.lower() == ".tsv"
            else ","
        )

        content = path.read_text(
            encoding="utf-8-sig"
        )

        rows = csv.reader(
            io.StringIO(content),
            delimiter=delimiter,
        )

        output: list[str] = []

        for index, row in enumerate(rows):

            if index >= 5000:
                output.append(
                    "[TRUNCATED]"
                )
                break

            if any(cell.strip() for cell in row):
                output.append(
                    " | ".join(row)
                )

        return "\n".join(output)

    def _extract(
        self,
        path: Path,
    ) -> str:

        suffix = path.suffix.lower()

        if suffix in {
            ".csv",
            ".tsv",
        }:
            return self._read_delimited(path)

        if suffix in {
            ".xlsx",
            ".xlsm",
        }:
            return self._read_workbook(path)

        raise ValueError(
            f"Unsupported spreadsheet format: {suffix}"
        )

    def forward(
        self,
        file_path: str,
        question: str,
    ) -> str:

        if is_placeholder_path(file_path):
            raise ValueError(
                "Spreadsheet path is a placeholder "
                "or invalid."
            )

        path = resolve_file(
            self.base_dir,
            file_path,
        )

        if path is None:
            raise FileNotFoundError(
                f"Spreadsheet not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Not a file: {file_path}"
            )

        if (
            path.suffix.lower()
            not in self.SUPPORTED_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported spreadsheet format: "
                f"{path.suffix}"
            )

        data = self._extract(path)

        if not data.strip():
            raise ValueError(
                f"Spreadsheet '{path.name}' "
                "contains no readable data."
            )

        if self.model is None:
            raise RuntimeError(
                "Excel analysis requires a configured "
                "language model."
            )

        prompt = f"""
You are solving a GAIA benchmark task.

Question:
{question}

Spreadsheet contents:
{data}

Rules:
- Answer only from the spreadsheet.
- Do not invent information.
- If the spreadsheet does not contain enough information,
  explicitly state that.
- Return only the answer.
"""

        return str(
            self.model.generate(prompt)
        )


class ExcelTools:

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
            AnalyzeExcelTool(
                model=self.model,
                base_dir=str(
                    self.base_dir
                ),
            )
        ]