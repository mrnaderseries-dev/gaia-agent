from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from gaia_agent.planner.plan_schema import StepType


class ExecutionReason(str, Enum):
    ACTION = "action"
    TOOL = "tool"
    ARGUMENT = "argument"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ExecutionDecision:

    allowed: bool
    reason: str | None = None
    message: str | None = None

    @classmethod
    def allow(cls) -> "ExecutionDecision":
        return cls(
            allowed=True
        )

    @classmethod
    def deny(
        cls,
        reason: str,
        message: str,
    ) -> "ExecutionDecision":

        return cls(
            allowed=False,
            reason=reason,
            message=message,
        )


@dataclass(frozen=True)
class ExecutionState:

    step_type: StepType
    tool_name: str | None
    action_name: str | None
    arguments: dict[str, Any]
    blocked: bool = False


class ExecutionPolicy:
    """
    Validates whether the current execution step
    is allowed to proceed.
    """

    priority = [
        "Action",
        "Tool",
        "Argument",
        "blocked",
    ]

    def evaluate(
        self,
        state: ExecutionState,
    ) -> ExecutionDecision:

        # ------------------------------------------------------
        # Already blocked
        # ------------------------------------------------------

        if state.blocked:

            return ExecutionDecision.deny(
                reason=ExecutionReason.BLOCKED.value,
                message="Execution is blocked.",
            )

        # ------------------------------------------------------
        # Action
        # ------------------------------------------------------

        if not state.action_name:

            return ExecutionDecision.deny(
                reason=ExecutionReason.ACTION.value,
                message="Action name not found.",
            )

        # ------------------------------------------------------
        # TOOL
        # ------------------------------------------------------

        if state.step_type == StepType.TOOL:

            if not state.tool_name:

                return ExecutionDecision.deny(
                    reason=ExecutionReason.TOOL.value,
                    message=(
                        "Tool name not found "
                        "for TOOL step."
                    ),
                )

            # Empty arguments are VALID.
            #
            # A tool can legitimately require zero arguments.
            #
            # We therefore normalize None -> {} and allow it.

            return ExecutionDecision.allow()

        # ------------------------------------------------------
        # LLM
        # ------------------------------------------------------

        if state.step_type == StepType.LLM:

            if state.tool_name is not None:

                return ExecutionDecision.deny(
                    reason=ExecutionReason.TOOL.value,
                    message=(
                        "LLM step must not "
                        "specify a tool."
                    ),
                )

            return ExecutionDecision.allow()

        # ------------------------------------------------------
        # Unsupported
        # ------------------------------------------------------

        return ExecutionDecision.deny(
            reason=ExecutionReason.ACTION.value,
            message=(
                f"Unsupported step type: "
                f"{state.step_type}."
            ),
        )