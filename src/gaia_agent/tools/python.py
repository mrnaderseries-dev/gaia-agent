from __future__ import annotations

import io
import sys
from typing import Any

from smolagents import Tool

from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolErrorCode,
    ToolModality,
    ToolSpec,
)
from gaia_agent.reliability.exception import (
    PythonImportError,
    PythonSyntaxError,
    ToolExecutionError,
)

ALLOWED_IMPORTS: dict[str, str] = {
    "math": "math",
    "json": "json",
    "re": "re",
    "datetime": "datetime",
    "itertools": "itertools",
    "functools": "functools",
    "collections": "collections",
    "statistics": "statistics",
    "string": "string",
    "typing": "typing",
}


class PythonInterpreterTool(Tool):
    name = "python_interpreter"

    description = (
        "Execute Python code safely to perform complex calculations, "
        "data processing, string manipulation, or data analysis for "
        "GAIA evaluation tasks. Always assign your final answer or "
        "result to a variable named 'result'."
    )

    inputs = {
        "code": {
            "type": "string",
            "description": (
                "Valid Python code to execute. "
                "Assign the final answer to 'result'."
            ),
        }
    }

    output_type = "string"

    def __init__(self) -> None:
        super().__init__()

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            arguments_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python code to execute. "
                            "Assign the final answer to 'result'."
                        ),
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            capability=ToolCapability.COMPUTATION,
            modalities=frozenset(
                {
                    ToolModality.TEXT,
                }
            ),
            result_schema={
                "type": self.output_type,
            },
            error_codes=frozenset(
                {
                    ToolErrorCode.INVALID_ARGUMENT,
                    ToolErrorCode.PERMISSION_DENIED,
                    ToolErrorCode.EXECUTION_FAILED,
                }
            ),
            allowed_imports=frozenset(ALLOWED_IMPORTS),
            function=self.forward,
        )

    @staticmethod
    def _sandbox_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise PythonImportError(
                f"Relative imports are not allowed: {name!r}"
            )

        module_name = name.split(".")[0]

        if module_name not in ALLOWED_IMPORTS:
            raise PythonImportError(
                f"Module '{name}' is not allowed."
            )

        return __import__(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    def forward(
        self,
        code: str,
    ) -> str:
        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                "Python code must be a non-empty string."
            )

        try:
            compile(
                code,
                "<gaia_python_interpreter>",
                "exec",
            )
        except SyntaxError:
            raise PythonSyntaxError(
                "Generated Python code failed to compile."
            ) from None

        local_vars: dict[str, Any] = {}

        global_vars: dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bin": bin,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "pow": pow,
                "range": range,
                "round": round,
                "set": set,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "__import__": PythonInterpreterTool._sandbox_import,
            }
        }

        for alias, module_name in ALLOWED_IMPORTS.items():
            try:
                global_vars[alias] = __import__(module_name)
            except ImportError:
                pass

        old_stdout = sys.stdout
        redirected_output = io.StringIO()

        sys.stdout = redirected_output

        try:
            exec(
                code,
                global_vars,
                local_vars,
            )
        except NameError as exc:
            raise ToolExecutionError(
                f"Undefined name: {exc}. "
                "Use registered tools as separate tool steps."
            ) from None
        finally:
            sys.stdout = old_stdout

        printed_output = redirected_output.getvalue()

        if "result" in local_vars:
            result = local_vars["result"]

            if result is None or (
                isinstance(result, str)
                and not result.strip()
            ):
                raise ValueError(
                    "The 'result' variable must "
                    "hold a non-empty value."
                )

            if printed_output.strip():
                return (
                    f"Output:\n"
                    f"{printed_output.strip()}\n\n"
                    f"Result variable: {result}"
                )

            return str(result)

        if printed_output.strip():
            return printed_output.strip()

        raise ValueError(
            "Code executed successfully, but no "
            "'result' variable was defined and no "
            "output was printed."
        )


class PythonTools:
    def get_tools(self) -> list[Tool]:
        return [
            PythonInterpreterTool()
        ]