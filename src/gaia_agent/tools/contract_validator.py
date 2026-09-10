from __future__ import annotations

from typing import Any, Mapping

from gaia_agent.planner.plan_schema import PlanStep
from gaia_agent.planner.tool_spec import ToolSpec


class ToolContractValidator:
    @staticmethod
    def validate_step_contract(
        step: PlanStep,
        available_tools: Mapping[str, ToolSpec],
    ) -> dict[str, Any]:
        if step.tool_name not in available_tools:
            raise ValueError(
                f"Unknown tool: {step.tool_name}"
            )

        spec = available_tools[step.tool_name]

        return ToolContractValidator.validate_arguments(
            spec=spec,
            arguments=step.arguments or {},
        )

    @staticmethod
    def validate_arguments(
        spec: ToolSpec,
        arguments: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(spec, ToolSpec):
            raise TypeError(
                "spec must be a ToolSpec"
            )

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, Mapping):
            raise TypeError(
                "arguments must be a mapping"
            )

        schema = spec.arguments_schema or {}

        if schema.get("type") != "object":
            raise ValueError(
                f"Tool '{spec.name}' must define "
                "an object arguments schema."
            )

        properties = schema.get(
            "properties",
            {},
        )

        required = schema.get(
            "required",
            [],
        )

        if not isinstance(properties, Mapping):
            raise TypeError(
                f"Tool '{spec.name}' properties "
                "must be a mapping."
            )

        unknown_arguments = (
            set(arguments)
            - set(properties)
        )

        if (
            unknown_arguments
            and schema.get(
                "additionalProperties",
                True,
            )
            is False
        ):
            raise ValueError(
                "Unknown arguments for "
                f"{spec.name}: "
                f"{sorted(unknown_arguments)}"
            )

        missing_arguments = [
            name
            for name in required
            if name not in arguments
        ]

        if missing_arguments:
            raise ValueError(
                "Missing required arguments "
                f"for {spec.name}: "
                f"{missing_arguments}"
            )

        for name, value in arguments.items():
            if name not in properties:
                continue

            property_schema = properties[name]

            ToolContractValidator._validate_value(
                tool_name=spec.name,
                argument_name=name,
                value=value,
                schema=property_schema,
            )

        return dict(arguments)

    @staticmethod
    def _validate_value(
        *,
        tool_name: str,
        argument_name: str,
        value: Any,
        schema: Mapping[str, Any],
    ) -> None:
        expected_type = schema.get("type")

        if expected_type is None:
            return

        if expected_type == "string":
            if not isinstance(value, str):
                raise TypeError(
                    f"Argument '{argument_name}' "
                    f"for tool '{tool_name}' "
                    "must be a string"
                )
            return

        if expected_type == "integer":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
            ):
                raise TypeError(
                    f"Argument '{argument_name}' "
                    f"for tool '{tool_name}' "
                    "must be an integer"
                )
            return

        if expected_type == "number":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise TypeError(
                    f"Argument '{argument_name}' "
                    f"for tool '{tool_name}' "
                    "must be a number"
                )
            return

        if expected_type == "boolean":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Argument '{argument_name}' "
                    f"for tool '{tool_name}' "
                    "must be a boolean"
                )
            return

        if expected_type == "array":
            if not isinstance(value, list):
                raise TypeError(
                    f"Argument '{argument_name}' "
                    f"for tool '{tool_name}' "
                    "must be an array"
                )
            return

        if expected_type == "object":
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"Argument '{argument_name}' "
                    f"for tool '{tool_name}' "
                    "must be an object"
                )
            return