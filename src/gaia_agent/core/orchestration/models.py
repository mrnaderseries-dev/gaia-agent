from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.planner.plan_schema import PlanSchema


@dataclass(slots=True)
class PlanRuntime:
    plan: PlanSchema | None = None
    current_step: int | None = None
    completed_steps: set[int] = field(default_factory=set)

    @property
    def is_complete(self) -> bool:
        if self.plan is None:
            return False

        return self.current_step is None or (
            self.current_step >= len(self.plan.steps)
        )

    def current(self):
        if self.plan is None:
            return None

        if self.current_step is None:
            return None

        if not 0 <= self.current_step < len(self.plan.steps):
            return None

        return self.plan.steps[self.current_step]

    def mark_completed(self, step_id: int) -> None:
        self.completed_steps.add(step_id)

    def advance(self) -> None:
        if self.current_step is None:
            return

        self.current_step += 1


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    step_id: int
    result: ExecutionResult
    iteration: int


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    attempt: int
    verified: bool
    result: Any
    answer: str


@dataclass(slots=True)
class OrchestrationContext:
    user_request: str

    plan_runtime: PlanRuntime = field(
        default_factory=PlanRuntime
    )

    execution_history: list[ExecutionRecord] = field(
        default_factory=list
    )

    verification_history: list[VerificationRecord] = field(
        default_factory=list
    )

    final_answer: str | None = None

    iteration: int = 0

    def record_execution(
        self,
        *,
        step_id: int,
        result: ExecutionResult,
    ) -> None:
        self.execution_history.append(
            ExecutionRecord(
                step_id=step_id,
                result=result,
                iteration=self.iteration,
            )
        )

    def record_verification(
        self,
        *,
        verified: bool,
        result: Any,
        answer: str,
    ) -> None:
        self.verification_history.append(
            VerificationRecord(
                attempt=len(self.verification_history) + 1,
                verified=verified,
                result=result,
                answer=answer,
            )
        )

    @property
    def verification_attempts(self) -> int:
        return len(self.verification_history)

    @property
    def last_execution(self) -> ExecutionRecord | None:
        if not self.execution_history:
            return None

        return self.execution_history[-1]

    @property
    def is_verified(self) -> bool:
        if not self.verification_history:
            return False

        return self.verification_history[-1].verified