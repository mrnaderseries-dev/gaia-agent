from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class StepType(str, Enum):
    TOOL = "tool"
    LLM = "llm"

    @classmethod
    def _missing_(cls, value: object) -> StepType | None:
        if isinstance(value, str):
            value_lower = value.strip().lower()

            for member in cls:
                if member.value == value_lower:
                    return member

        return None


class PlanStep(BaseModel):
    step_id: int
    action: str
    step_type: StepType
    tool_name: str | None = Field(default=None)
    arguments: dict[str, Any] = Field(default_factory=dict)
    is_final_answer: bool = False

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Step action cannot be empty.")

        return value

    @field_validator("tool_name", mode="before")
    @classmethod
    def normalize_tool_name(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(
                "tool_name must be a string or null."
            )

        normalized = value.strip()

        if normalized.lower() in {
            "",
            "none",
            "null",
            "nil",
        }:
            return None

        return normalized

    @model_validator(mode="after")
    def validate_step_contract(self) -> PlanStep:
        if self.step_type == StepType.TOOL:
            if not self.tool_name:
                raise ValueError(
                    "TOOL step must specify a tool_name."
                )

            if self.is_final_answer:
                raise ValueError(
                    "TOOL step cannot be a final-answer step."
                )

            return self

        if self.step_type == StepType.LLM:
            self.tool_name = None

            if self.arguments:
                raise ValueError(
                    "LLM step cannot contain arguments."
                )

            return self

        raise ValueError(
            f"Unsupported step type: {self.step_type}"
        )


class PlanSchema(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_model_output(cls, data: Any) -> Any:
        if isinstance(data, list):
            data = {"steps": data}

        if not isinstance(data, dict):
            return data

        steps = data.get("steps")
        if not isinstance(steps, list):
            return data
        for step in steps:
            if not isinstance(step, dict):
                continue
            if (
                step.get("is_final_answer") is True
                and step.get("step_type") == "tool"
                and step.get("tool_name") in (None, "", "null", "none")
            ):
                step["step_type"] = "llm"
                step["tool_name"] = None
                step["arguments"] = {}

        ids: list[int] = []
        for step in steps:
            if not isinstance(step, dict):
                return data
            raw_id = step.get("step_id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                return data
            ids.append(raw_id)

        if len(set(ids)) != len(ids):
            return data

        if ids != list(range(len(ids))):
            rekeyed = []
            for position, step in enumerate(steps):
                fixed = dict(step)
                fixed["step_id"] = position
                rekeyed.append(fixed)
            data = dict(data)
            data["steps"] = rekeyed

        return data

    @field_validator("steps")
    @classmethod
    def validate_steps(
        cls,
        steps: list[PlanStep],
    ) -> list[PlanStep]:
        for expected_id, step in enumerate(steps):
            if step.step_id != expected_id:
                raise ValueError(
                    "Step IDs must be sequential starting from 0."
                )

        final_steps = [
            step
            for step in steps
            if step.is_final_answer
        ]

        if len(final_steps) != 1:
            raise ValueError(
                "Plan must contain exactly one "
                "final-answer step."
            )

        final_step = final_steps[0]

        if final_step.step_id != len(steps) - 1:
            raise ValueError(
                "Final-answer step must be the last step."
            )

        if final_step.step_type != StepType.LLM:
            raise ValueError(
                "Final-answer step must be an LLM step."
            )

        if final_step.tool_name is not None:
            raise ValueError(
                "Final-answer step cannot contain tool_name."
            )

        if final_step.arguments:
            raise ValueError(
                "Final-answer step cannot contain arguments."
            )

        return steps