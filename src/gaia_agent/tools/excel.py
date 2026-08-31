from __future__ import annotations

from pathlib import Path
from typing import Any, List

from openpyxl import load_workbook
from smolagents import Tool

from gaia_agent.tools.path_utils import (
    is_placeholder_path,
    resolve_file,
)


class AnalyzeExcelTool(Tool):
    name = "analyze_excel"

    description = (
        "Analyze an Excel workbook or spreadsheet according to a user's question. "
        "The workbook data is extracted safely and passed to the configured language model."
    )

    inputs = {
        "file_path": {
            "type": "string",
            "description": (
                "Path to the Excel file relative to "
                "the allowed base directory or filename."
            ),
        },
        "question": {
            "type": "string",
            "description": (
                "Question that should be answered using "
                "the Excel spreadsheet."
            ),
        },
    }

    output_type = "string"

    def __init__(
        self,
        model: Any = None,
        base_dir: str = ".",
    ) -> None:
        super().__init__()
        self.model = model
        self.base_dir = Path(base_dir).resolve()

    def _read_excel(
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
                output.append(f"Sheet: {sheet.title}")

                row_count = 0
                for row in sheet.iter_rows(values_only=True):
                    # تقييد الأسطر الكبيرة جداً لمنع تجاوز الـ Token Limit
                    if row_count > 5000:
                        output.append("... [Note: Spreadsheet truncated due to large size] ...")
                        break

                    values = [
                        "" if value is None else str(value)
                        for value in row
                    ]
                    # تخطي الصفوف الفارغة تماماً
                    if any(values):
                        output.append(" | ".join(values))
                        row_count += 1

                output.append("")

            return "\n".join(output)

        finally:
            workbook.close()

    def forward(
        self,
        file_path: str,
        question: str,
    ) -> str:
        try:
            if is_placeholder_path(file_path):
                return (
                    f"Error: File path '{file_path}' is a placeholder "
                    "or invalid. You must use a real file path that "
                    "exists in the environment."
                )

            path = resolve_file(self.base_dir, file_path)

            # البحث الذكي في حال لم يتم إيجاد الملف بالممسار المباشر (مثل باقي الأدوات)
            if path is None or not path.exists():
                filename = Path(file_path).name
                possible_paths = [
                    self.base_dir / filename,
                    Path.cwd() / filename,
                    Path.cwd() / "src" / filename,
                    Path(file_path)
                ]
                
                found = False
                for p in possible_paths:
                    if p.exists() and p.is_file():
                        path = p.resolve()
                        found = True
                        break
                
                if not found:
                    return f"Error: Excel file not found: {file_path} (searched in base_dir: {self.base_dir})"

            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            valid_extensions = {
                ".xlsx",
                ".xlsm",
                ".xls",
                ".csv"
            }

            if path.suffix.lower() not in valid_extensions:
                return f"Unsupported Excel format: {path.suffix}"

            excel_data = self._read_excel(path)

            if self.model is None:
                return f"Excel fallback mock analysis for '{path.name}' regarding query: '{question}'."

            prompt = f"""
You are analyzing an Excel spreadsheet to solve a GAIA benchmark evaluation task.

User question:
{question}

Excel data:
{excel_data}

Instructions:
- Answer the user's question accurately using the spreadsheet data.
- Do not invent information.
- If the spreadsheet does not contain enough information, say that the required information is unavailable.
- Return only the requested clear and concise answer.
"""

            answer = self.model.generate(prompt)

            return str(answer)

        except Exception as exc:
            return f"Excel analysis error: {exc}"


class ExcelTools:
    """
    Excel tools container optimized for GAIA benchmark evaluation across 20 diverse test cases.
    """

    def __init__(
        self,
        model: Any = None,
        base_dir: str = ".",
    ) -> None:
        self.model = model
        self.base_dir = Path(base_dir).resolve()

    def get_tools(self) -> List[Tool]:
        """
        Create and return all Excel tools.
        """
        return [
            AnalyzeExcelTool(
                model=self.model,
                base_dir=str(self.base_dir),
            )
        ]