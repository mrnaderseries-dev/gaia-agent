from __future__ import annotations

from typing import Any

from gaia_agent.planner.plan_schema import (
    PlanStep,
    StepType,
)
from gaia_agent.planner.tool_spec import (
    ToolCapability,
    ToolModality,
    ToolSpec,
)
from gaia_agent.reliability.exception import (
    ToolArgumentError,
)


class ToolContractError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.details = details or {}


class ToolContractValidator:
    @staticmethod
    def validate_step_contract(
        step: PlanStep,
        available_tools: dict[str, ToolSpec],
    ) -> dict[str, Any]:
        if not isinstance(
            step,
            PlanStep,
        ):
            raise ToolContractError(
                "Step must be a PlanStep instance."
            )

        if step.step_type != StepType.TOOL:
            return {}

        if not step.tool_name:
            raise ToolContractError(
                "Tool step does not specify a tool_name."
            )

        spec = available_tools.get(
            step.tool_name
        )

        if spec is None:
            raise ToolContractError(
                f"Tool '{step.tool_name}' is not registered.",
                tool_name=step.tool_name,
            )

        return ToolContractValidator.validate_arguments(
            spec,
            step.arguments or {},
        )

    @staticmethod
    def validate_arguments(
        spec: ToolSpec,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(
            spec,
            ToolSpec,
        ):
            raise ToolContractError(
                "spec must be a ToolSpec instance."
            )

        arguments = (
            {}
            if arguments is None
            else arguments
        )

        if not isinstance(
            arguments,
            dict,
        ):
            raise ToolArgumentError(
                f"Tool '{spec.name}' received "
                f"non-dict arguments."
            )

        schema = (
            ToolContractValidator
            ._resolve_schema(spec)
        )

        unknown = sorted(
            set(arguments) - set(schema)
        )

        if unknown:
            raise ToolArgumentError(
                f"Tool '{spec.name}' does not accept "
                f"argument(s): {unknown}."
            )

        for name, meta in schema.items():
            if ToolContractValidator._is_required(
                meta
            ):
                if (
                    name not in arguments
                    or arguments[name] is None
                ):
                    raise ToolArgumentError(
                        f"Tool '{spec.name}' requires "
                        f"argument '{name}'."
                    )

        for name, value in arguments.items():
            ToolContractValidator._validate_type(
                tool_name=spec.name,
                arg_name=name,
                value=value,
                meta=schema[name],
            )

        return dict(arguments)

    @staticmethod
    def validate_modality(
        spec: ToolSpec,
        modality: ToolModality,
    ) -> None:
        if not spec.supports_modality(
            modality
        ):
            raise ToolContractError(
                f"Tool '{spec.name}' does not support "
                f"modality '{modality.value}'.",
                tool_name=spec.name,
            )

    @staticmethod
    def validate_capability(
        spec: ToolSpec,
        capability: ToolCapability,
    ) -> None:
        if not spec.supports_capability(
            capability
        ):
            raise ToolContractError(
                f"Tool '{spec.name}' does not provide "
                f"capability '{capability.value}'.",
                tool_name=spec.name,
            )

    @staticmethod
    def _resolve_schema(
        spec: ToolSpec,
    ) -> dict[str, Any]:
        schema = spec.arguments_schema

        if not isinstance(
            schema,
            dict,
        ):
            raise ToolContractError(
                f"Tool '{spec.name}' has invalid "
                "arguments_schema."
            )

        if (
            "properties" in schema
            and isinstance(
                schema["properties"],
                dict,
            )
        ):
            properties = schema["properties"]
            required = set(
                schema.get(
                    "required",
                    [],
                )
                or []
            )

            result: dict[str, Any] = {}

            for name, meta in properties.items():
                item = dict(
                    meta or {}
                )
                item["_required"] = (
                    name in required
                )
                result[name] = item

            return result

        return dict(schema)

    @staticmethod
    def _is_required(
        meta: Any,
    ) -> bool:
        if not isinstance(
            meta,
            dict,
        ):
            return True

        if "_required" in meta:
            return bool(
                meta["_required"]
            )

        if "optional" in meta:
            return not bool(
                meta["optional"]
            )

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
        if not isinstance(
            meta,
            dict,
        ):
            return

        value_type = str(
            meta.get(
                "type",
                "string",
            )
        ).lower()

        if value_type in {
            "any",
            "null",
        }:
            return

        if value is None:
            return

        if value_type in {
            "integer",
            "int",
        }:
            valid = (
                isinstance(
                    value,
                    int,
                )
                and not isinstance(
                    value,
                    bool,
                )
            )

        elif value_type in {
            "number",
            "float",
        }:
            valid = (
                isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(
                    value,
                    bool,
                )
            )

        elif value_type in {
            "string",
            "str",
        }:
            valid = isinstance(
                value,
                str,
            )

        elif value_type in {
            "boolean",
            "bool",
        }:
            valid = isinstance(
                value,
                bool,
            )

        elif value_type in {
            "array",
            "list",
        }:
            valid = isinstance(
                value,
                (list, tuple),
            )

        elif value_type in {
            "object",
            "dict",
        }:
            valid = isinstance(
                value,
                dict,
            )

        else:
            raise ToolContractError(
                f"Tool '{tool_name}' argument "
                f"'{arg_name}' uses unsupported "
                f"type '{value_type}'.",
                tool_name=tool_name,
            )

        if not valid:
            raise ToolArgumentError(
                f"Tool '{tool_name}' argument "
                f"'{arg_name}' must be of type "
                f"{value_type}, got "
                f"{type(value).__name__}."
            )