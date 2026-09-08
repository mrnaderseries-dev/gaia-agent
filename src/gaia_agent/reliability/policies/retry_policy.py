from __future__ import annotations

from dataclasses import dataclass

from gaia_agent.reliability.failure_classifier import FailureClass


@dataclass(frozen=True, slots=True)
class RetryDecision:
    should_retry: bool
    delay: float = 0.0
    reason: str = ""


class RetryPolicy:
    def __init__(
        self,
        *,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
    ) -> None:
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0.")

        if max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay.")

        self.base_delay = base_delay
        self.max_delay = max_delay

    def evaluate(
        self,
        failure_class: FailureClass,
        *,
        current_attempt: int,
    ) -> RetryDecision:
        if current_attempt < 1:
            return RetryDecision(
                should_retry=False,
                reason="Invalid attempt number.",
            )

        if failure_class != FailureClass.TRANSIENT:
            return RetryDecision(
                should_retry=False,
                reason=f"Failure class {failure_class.value} is not transient.",
            )

        delay = min(
            self.base_delay * (2 ** (current_attempt - 1)),
            self.max_delay,
        )

        return RetryDecision(
            should_retry=True,
            delay=delay,
            reason="Transient failure.",
        )