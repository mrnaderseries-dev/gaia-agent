from __future__ import annotations

import io
import sys
import traceback
from typing import Any, List

from smolagents import Tool

from gaia_agent.reliability.exception import (
    PythonImportError,
    PythonSyntaxError,
    ToolExecutionError,
)


# Explicitly whitelisted stdlib modules available inside the sandbox.
# This list is the SINGLE source of truth for "what the planner is
# allowed to generate". It is also surfaced to the planner through
# the ToolSpec.allowed_imports contract (Phase 3 alignment).
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
        "Execute Python code safely to perform complex calculations, data processing, "
        "string manipulation, or data analysis for GAIA evaluation tasks. "
        "Always assign your final answer or result to a variable named 'result'. "
        "Available modules: math, json, re, datetime, itertools, functools, "
        "collections, statistics, string, typing."
    )

    inputs = {
        "code": {
            "type": "string",
            "description": (
                "Valid Python code to execute. "
                "Make sure to assign the final answer to a variable named 'result'. "
                "Only standard-library imports from the allowed module set are "
                "permitted (math, json, re, datetime, itertools, functools, "
                "collections, statistics, string, typing)."
            ),
        }
    }

    output_type = "string"

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Whitelisted, guarded __import__
    # ------------------------------------------------------------------

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
                f"Relative imports are not allowed in the sandbox: {name!r}"
            )

        module_name = name.split(".")[0]

        if module_name not in ALLOWED_IMPORTS:
            raise PythonImportError(
                f"Module '{name}' is not allowed in the sandbox. "
                f"Allowed modules: {sorted(ALLOWED_IMPORTS)}."
            )

        return __import__(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    def forward(self, code: str) -> str:

        # --------------------------------------------------------------
        # Phase 3a: pre-compile to catch SyntaxError deterministically.
        # --------------------------------------------------------------

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
                "__import__": self._sandbox_import,
            }
        }

        # Pre-warm the fully-qualified stdlib imports.
        for alias, module_name in ALLOWED_IMPORTS.items():
            try:
                global_vars[alias] = __import__(module_name)
            except ImportError:
                pass

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            exec(code, global_vars, local_vars)
        except NameError as name_error:
            # The LLM commonly writes code like
            #   file_reader(file_path=...)
            # inside the sandbox, producing
            #   NameError: name 'file_reader' is not defined.
            # Turn that into a clear, recoverable tool-validation
            # error that guides the planner to use the real tool.
            raise ToolExecutionError(
                f"Python code referenced an undefined name: "
                f"{name_error}. Tool names such as file_reader, "
                "web_search, analyze_excel or analyze_image cannot be "
                "called from inside Python code. Perform file reads, "
                "searches and data analysis as separate tool steps "
                "instead.",
                recoverable=True,
            ) from None
        finally:
            sys.stdout = old_stdout

        printed_output = redirected_output.getvalue()

        if "result" in local_vars:
            res_val = local_vars["result"]
            if res_val is None or (
                isinstance(res_val, str) and not res_val.strip()
            ):
                raise ValueError(
                    "The 'result' variable must hold a non-empty value."
                )
            if printed_output.strip():
                return f"Output:\n{printed_output.strip()}\n\nResult variable: {res_val}"
            return str(res_val)

        if printed_output.strip():
            return printed_output.strip()

        raise ValueError(
            "Code executed successfully, but no 'result' variable "
            "was defined and no output was printed."
        )


class PythonTools:
    """
    Python interpreter tools container optimized for accurate calculations and data processing in GAIA.
    """

    def __init__(self) -> None:
        pass

    def get_tools(self) -> List[Tool]:
        """
        Create and return all python execution tools.
        """
        return [
            PythonInterpreterTool()
        ]