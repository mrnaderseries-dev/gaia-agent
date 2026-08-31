from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .models import HumanDecision, HumanResponse


class HumanApprovalHandler(ABC):
    """
    Contract for obtaining a human decision.

    This component does not determine whether approval
    is required. ApprovalPolicy is responsible for that.

    This component only obtains and returns the human's
    decision.
    """

    @abstractmethod
    async def request_approval(
        self,
        *,
        action_name: str,
        tool_name: str | None,
        arguments: dict[str, Any],
        reason: str | None = None,
        message: str | None = None,
    ) -> HumanResponse:
        raise NotImplementedError


class CLIApprovalHandler(HumanApprovalHandler):
   

    async def request_approval(
        self,
        *,
        action_name: str,
        tool_name: str | None,
        arguments: dict[str, Any],
        reason: str | None = None,
        message: str | None = None,
    ) -> HumanResponse:

        self._display_request(
            action_name=action_name,
            tool_name=tool_name,
            arguments=arguments,
            reason=reason,
            message=message,
        )

        while True:

            decision = input(
                "Decision [approve/reject/modify]: "
            ).strip().lower()

            if decision == HumanDecision.APPROVE.value:
                return HumanResponse(
                    decision=HumanDecision.APPROVE,
                    message="Action approved by human.",
                )

            if decision == HumanDecision.REJECT.value:
                return HumanResponse(
                    decision=HumanDecision.REJECT,
                    message="Action rejected by human.",
                )

            if decision == HumanDecision.MODIFY.value:

                modified_arguments = (
                    self._read_modified_arguments(
                        arguments
                    )
                )

                return HumanResponse(
                    decision=HumanDecision.MODIFY,
                    modified_arguments=modified_arguments,
                    message="Action modified by human.",
                )

            print(
                "Invalid decision. "
                "Choose: approve, reject, or modify."
            )

    @staticmethod
    def _display_request(
        *,
        action_name: str,
        tool_name: str | None,
        arguments: dict[str, Any],
        reason: str | None,
        message: str | None,
    ) -> None:

        print("\n=== Human Approval Required ===")
        print(f"Action: {action_name}")
        print(f"Tool: {tool_name}")
        print(f"Arguments: {arguments}")

        if reason:
            print(f"Reason: {reason}")

        if message:
            print(f"Message: {message}")

    @staticmethod
    def _read_modified_arguments(
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        print("\nCurrent arguments:")
        print(arguments)

        print(
            "\nEnter modified arguments as JSON."
        )

        while True:

            raw = input(
                "Modified arguments: "
            ).strip()

            try:
                modified = json.loads(raw)

            except json.JSONDecodeError as exc:
                print(
                    "Invalid JSON.",
                    f"Error: {exc}",
                )
                continue

            if not isinstance(modified, dict):
                print(
                    "Modified arguments must be "
                    "a JSON object."
                )
                continue

            return modified