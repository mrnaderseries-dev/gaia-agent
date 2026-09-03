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