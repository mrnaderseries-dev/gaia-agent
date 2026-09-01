from __future__ import annotations

from dataclasses import dataclass

from gaia_agent.reliability.failure_classifier import (
    FailureClassification,
    FailureType,
)


@dataclass(frozen=True, slots=True)
class RetryDecision:
    should_retry: bool
    max_attempts: int
    delay: float
    reason: str


class RetryPolicy:
    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be greater than 0."
            )

        if base_delay < 0:
            raise ValueError(
                "base_delay cannot be negative."
            )

        if max_delay < base_delay:
            raise ValueError(
                "max_delay must be greater than or equal to base_delay."
            )

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def evaluate(
        self,
        classification: FailureClassification,
        *,
        current_attempt: int,
    ) -> RetryDecision:

        if current_attempt <= 0:
            raise ValueError(
                "current_attempt must be greater than 0."
            )

        if classification.failure_type is not FailureType.TRANSIENT:
            return RetryDecision(
                should_retry=False,
                max_attempts=self.max_attempts,
                delay=0.0,
                reason="Failure is not transient.",
            )

        if current_attempt >= self.max_attempts:
            return RetryDecision(
                should_retry=False,
                max_attempts=self.max_attempts,
                delay=0.0,
                reason="Maximum retry attempts reached.",
            )

        delay = min(
            self.base_delay * (2 ** (current_attempt - 1)),
            self.max_delay,
        )

        return RetryDecision(
            should_retry=True,
            max_attempts=self.max_attempts,
            delay=delay,
            reason="Failure is transient and can be retried.",
        )