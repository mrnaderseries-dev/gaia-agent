from __future__ import annotations

from typing import Any

from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.reliability.exception import ToolArgumentError


class ToolContractError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        argument_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.argument_name = argument_name
        self.details = details or {}


class ToolContractValidator:
    @staticmethod
    def validate_step_contract(
        step: PlanStep,
        available_tools: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(step, PlanStep):
            raise ToolContractError(
                "Step must be a PlanStep instance.",
                details={
                    "received_type": type(step).__name__,
                },
            )

        if step.step_type != StepType.TOOL:
            return {}

        if not step.tool_name:
            raise ToolContractError(
                "Tool step does not specify a tool_name.",
                details={
                    "step_id": step.step_id,
                },
            )

        if not isinstance(available_tools, dict):
            raise ToolContractError(
                "available_tools must be a dict of name -> contract.",
                details={
                    "received_type": type(available_tools).__name__,
                },
            )

        spec = available_tools.get(step.tool_name)

        if spec is None:
            raise ToolContractError(
                f"Tool '{step.tool_name}' is not registered. "
                f"Registered tools: {sorted(available_tools)}.",
                tool_name=step.tool_name,
                details={
                    "available_tools": sorted(available_tools),
                    "step_id": step.step_id,
                },
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
        tool_name = getattr(spec, "name", "?")

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"Tool '{tool_name}' received non-dict arguments: "
                f"{type(arguments).__name__}."
            )

        schema = ToolContractValidator._resolve_schema(spec)
        args = dict(arguments)

        unknown = sorted(set(args) - set(schema))

        if unknown:
            raise ToolArgumentError(
                f"Tool '{tool_name}' does not accept argument(s): "
                f"{unknown}. "
                f"Allowed arguments: {sorted(schema)}."
            )

        for name, meta in schema.items():
            if ToolContractValidator._is_required(name, meta):
                if name not in args or args[name] is None:
                    raise ToolArgumentError(
                        f"Tool '{tool_name}' requires argument '{name}'."
                    )

        for name, value in args.items():
            ToolContractValidator._validate_type(
                tool_name=tool_name,
                arg_name=name,
                value=value,
                meta=schema[name],
            )

        return args

    @staticmethod
    def _resolve_schema(spec: Any) -> dict[str, Any]:
        tool_name = getattr(spec, "name", "?")
        schema = getattr(spec, "arguments_schema", None) or {}

        if not isinstance(schema, dict):
            raise ToolContractError(
                f"Tool '{tool_name}' has an invalid arguments_schema.",
                tool_name=tool_name,
                details={
                    "schema_type": type(schema).__name__,
                },
            )

        if "properties" in schema and isinstance(
            schema["properties"],
            dict,
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
    def _is_required(
        name: str,
        meta: Any,
    ) -> bool:
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

        value_type = str(
            meta.get("type", "string")
        ).lower()

        if value_type in {"any", "null"}:
            return

        if value is None:
            return

        if value_type in {"integer", "int"}:
            valid = isinstance(value, int) and not isinstance(value, bool)

        elif value_type in {"number", "float"}:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        elif value_type in {"string", "str"}:
            valid = isinstance(value, str)

        elif value_type in {"boolean", "bool"}:
            valid = isinstance(value, bool)

        elif value_type in {"array", "list"}:
            valid = isinstance(value, (list, tuple))

        elif value_type in {"object", "dict"}:
            valid = isinstance(value, dict)

        else:
            raise ToolContractError(
                f"Tool '{tool_name}' argument '{arg_name}' "
                f"uses unsupported schema type '{value_type}'.",
                tool_name=tool_name,
                argument_name=arg_name,
                details={
                    "schema_type": value_type,
                },
            )

        if not valid:
            raise ToolArgumentError(
                f"Tool '{tool_name}' argument '{arg_name}' "
                f"must be of type {value_type}, "
                f"got {type(value).__name__}."
            )