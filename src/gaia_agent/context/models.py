from __future__ import annotations
from gaia_agent.context.attachments import Attachment
from dataclasses import dataclass, field
from typing import Any

from gaia_agent.conversation.models import Message
from gaia_agent.planner.plan_schema import PlanStep, StepType


@dataclass(frozen=True, slots=True)
class ContextRequest:
    user_request: str = ""

    conversation: list[Message] = field(
        default_factory=list
    )

    plan: list[PlanStep] = field(
        default_factory=list
    )

    current_step: int | None = None

    completed_steps: list[int] = field(
        default_factory=list
    )

    current_action: str | None = None

    step_type: StepType | None = None

    iteration: int = 0

    tool_name: str | None = None

    blocked: bool = False

    tool_result: Any | None = None

    tool_error: str | None = None
    attachments:tuple[Attachment,...]=()


@dataclass(slots=True)
class ConversationContext:
    messages: list[Message]


@dataclass(slots=True)
class HistoryContext:
    plan: list[PlanStep]
    current_step: int | None
    completed_steps: list[int]
    current_action: str | None
    step_type: StepType | None


@dataclass(slots=True)
class RuntimeContext:
    iteration: int
    tool_name: str | None
    blocked: bool
    tool_result: Any | None
    tool_error: str | None


@dataclass(slots=True)
class FinalContext:
    items: list[Any]
    token_count: int