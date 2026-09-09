
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    recovered: bool
    result: Any | None = None
    reason: str = ""
    changed: bool = False


class Recovery:
    """
    Generic recovery mechanism.

    Responsibilities:
        - Execute one recovery operation.
        - Validate whether it produced a meaningful change.

    Does NOT:
        - Decide whether recovery should happen.
        - Perform planning.
        - Detect loops.
        - Classify errors.
        - Handle retries.
        - Manage budgets.
    """

    async def execute(
        self,
        *,
        operation: Callable[
            [],
            Awaitable[Any],
        ],
        change_detector: Callable[
            [Any],
            bool,
        ],
    ) -> RecoveryResult:
        result = await operation()

        changed = bool(
            change_detector(result)
        )

        if not changed:
            return RecoveryResult(
                recovered=False,
                result=result,
                changed=False,
                reason=(
                    "Recovery produced no "
                    "meaningful change."
                ),
            )

        return RecoveryResult(
            recovered=True,
            result=result,
            changed=True,
            reason=(
                "Recovery produced a meaningful "
                "change."
            ),
        )