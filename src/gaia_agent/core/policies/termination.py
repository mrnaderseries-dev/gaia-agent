from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerminationReason(str, Enum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ANSWER_UNVERIFIED_BUDGET = "answer_unverified_budget"
    HUMAN_ABORTED = "human_aborted"
    EXPLICIT_STOP = "explicit_stop"
    TIMED_OUT = "timed_out"
    FATAL_ERROR = "fatal_error"


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    should_stop: bool
    reason: TerminationReason | None = None


@dataclass(frozen=True, slots=True)
class TerminationState:
    iteration: int
    final_answer_ready: bool
    final_answer_verified: bool
    fatal_error: bool
    human_aborted: bool
    explicit_stop: bool
    timed_out: bool
    verification_attempts: int = 0


class TerminationPolicy:

    def __init__(
        self,
        *,
        max_iterations: int = 23,
        max_verification_attempts: int = 2,
    ) -> None:

        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than 0."
            )

        self.max_iterations = max_iterations
        self.max_verification_attempts = max_verification_attempts

    def evaluate(
        self,
        state: TerminationState,
    ) -> TerminationDecision:

        if state.human_aborted:
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.HUMAN_ABORTED,
            )

        if state.explicit_stop:
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.EXPLICIT_STOP,
            )

        if state.timed_out:
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.TIMED_OUT,
            )

        if state.fatal_error:
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.FATAL_ERROR,
            )

        if (
            state.final_answer_ready
            and state.final_answer_verified
        ):
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.COMPLETED,
            )

        # Bounded verification: if the answer exists but semantic
        # verification kept failing, deliver it honestly as
        # UNVERIFIED rather than looping regenerate->verify forever.
        if (
            state.final_answer_ready
            and state.verification_attempts
            >= self.max_verification_attempts
        ):
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.ANSWER_UNVERIFIED_BUDGET,
            )

        if state.iteration >= self.max_iterations:
            return TerminationDecision(
                should_stop=True,
                reason=TerminationReason.MAX_ITERATIONS,
            )

        return TerminationDecision(
            should_stop=False,
            reason=None,
        )