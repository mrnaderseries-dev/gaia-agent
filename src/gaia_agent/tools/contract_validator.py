from __future__ import annotations

from typing import Any

from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.reliability.exception import ToolArgumentError


class ToolContractError(ValueError):
    """
    Raised when a plan step violates a registered tool contract.

    Contract violations must be repaired before execution; they must
    never be passed to the underlying tool implementation.
    """


class ToolContractValidator:
    """
    Deterministic, schema-driven validation of tool calls.

    Enforces, BEFORE a tool is invoked:

    - the tool exists in the registered contract set
    - only declared arguments are used
    - every required argument is present
    - argument values have a compatible type

    GAIA failure mode addressed:
        DuckDuckGoSearchTool.forward()
        got an unexpected keyword argument 'code'
    (planner selected web_search with {"code": ...} instead of
    {"query": ...}).
    """

    @staticmethod
    def validate_step_contract(
        step: PlanStep,
        available_tools: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate one PlanStep against the registered tool contracts.

        Returns the normalized (validated) arguments.
        """
        if not isinstance(step, PlanStep):
            raise ToolContractError(
                "Step must be a PlanStep instance."
            )

        if step.step_type != StepType.TOOL:
            return {}

        if not step.tool_name:
            raise ToolContractError(
                "Tool step does not specify a tool_name."
            )

        if not isinstance(available_tools, dict):
            raise ToolContractError(
                "available_tools must be a dict of name -> contract."
            )

        spec = available_tools.get(step.tool_name)

        if spec is None:
            raise ToolContractError(
                f"Tool '{step.tool_name}' is not registered. "
                f"Registered tools: {sorted(available_tools)}."
            )

        return ToolContractValidator.validate_arguments(
            spec=spec,
            arguments=step.arguments or {},
        )
    @staticmethod
    def validate_arguments(
        spec: Any,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Validate raw tool arguments against the tool's argument schema.

        Returns a normalized copy of the accepted arguments.
        """
        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"Tool '{spec.name}' received non-dict arguments: "
                f"{type(arguments).__name__}."
            )

        schema = ToolContractValidator._resolve_schema(spec)

        args = dict(arguments)

        unknown = sorted(set(args) - set(schema))

        if unknown:
            raise ToolArgumentError(
                f"Tool '{spec.name}' does not accept argument(s): "
                f"{unknown}. "
                f"Allowed arguments: {sorted(schema)}."
            )

        for name, meta in schema.items():

            required = ToolContractValidator._is_required(name, meta)

            if required and (name not in args or args.get(name) is None):
                raise ToolArgumentError(
                    f"Tool '{spec.name}' requires argument '{name}'."
                )

        for name, value in args.items():
            meta = schema.get(name, {})
            ToolContractValidator._validate_type(
                tool_name=spec.name,
                arg_name=name,
                value=value,
                meta=meta,
            )

        return args

    @staticmethod
    def _resolve_schema(spec: Any) -> dict[str, Any]:
        """
        Accept either a JSON-Schema shaped schema
        ({"properties": {...}, "required": [...]}) or the smolagents
        input dict ({"arg": {"type": "string", ...}}).
        """
        schema = getattr(spec, "arguments_schema", None) or {}

        if not isinstance(schema, dict):
            raise ToolContractError(
                f"Tool '{getattr(spec, 'name', '?')}' has an invalid "
                "arguments_schema."
            )

        if "properties" in schema and isinstance(
            schema["properties"], dict
        ):
            properties = schema["properties"]

            required = set(schema.get("required", []) or [])

            normalized: dict[str, Any] = {}

            for name, meta in properties.items():
                item = dict(meta or {})
                item["_required"] = name in required
                normalized[name] = item

            return normalized

        return dict(schema)

    @staticmethod
    def _is_required(name: str, meta: Any) -> bool:
        if not isinstance(meta, dict):
            return True

        if "_required" in meta:
            return bool(meta["_required"])

        if "optional" in meta:
            return not bool(meta["optional"])

        if "default" in meta:
            return False

        return True

    @staticmethod
    def _validate_type(
        *,
        tool_name: str,
        arg_name: str,
        value: Any,
        meta: Any,
    ) -> None:
        if not isinstance(meta, dict):
            return

        value_type = (meta.get("type") or "string").lower()

        if value_type in {"any", "null"}:
            return

        if value is None:
            return

        expected_types = {
            "string": (str,),
            "str": (str,),
            "integer": (int,),
            "int": (int,),
            "number": (int, float),
            "float": (int, float),
            "boolean": (bool,),
            "bool": (bool,),
            "array": (list, tuple),
            "list": (list, tuple),
            "object": (dict,),
            "dict": (dict,),
        }

        if value_type in expected_types:

            if not isinstance(value, expected_types[value_type]):
                raise ToolArgumentError(
                    f"Tool '{tool_name}' argument '{arg_name}' "
                    f"must be of type {value_type}, "
                    f"got {type(value).__name__}."
                )