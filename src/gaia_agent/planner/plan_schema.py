from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


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
    
    # السر هنا: السماح بقبول الـ None افتراضياً لمنع فشل الـ Parsing المباشر من الـ LLM
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
    def normalize_tool_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("tool_name must be a string or null.")
        
        normalized = value.strip()
        if normalized.lower() in {"", "none", "null", "nil"}:
            return None
        return normalized

    @model_validator(mode="after")
    def fix_and_validate_tool_usage(self) -> "PlanStep":
        # A TOOL step with a missing tool_name is INVALID. Silently
        # defaulting to "web_search" caused unnecessary web searches.
        # The planner contract validation rejects the step and falls
        # back to a valid, task-appropriate strategy instead.
        if self.step_type == StepType.TOOL and not self.tool_name:
            raise ValueError(
                "TOOL step must specify a tool_name."
            )

        if self.step_type == StepType.LLM:
            self.tool_name = None

        return self


class PlanSchema(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def validate_step_ids(cls, steps: list[PlanStep]) -> list[PlanStep]:
        for expected_id, step in enumerate(steps):
            if step.step_id != expected_id:
                raise ValueError("Step IDs must be sequential starting from 0.")
        return steps