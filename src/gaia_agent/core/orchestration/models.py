from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gaia_agent.core.agent_execution import (
    ExecutionResult,
)
from gaia_agent.core.evidence import (
    ToolResultRecord,
)
from gaia_agent.planner.plan_schema import (
    PlanSchema,
)


@dataclass(slots=True)
class PlanRuntime:

    plan: PlanSchema | None = None

    current_step_index: int = 0

    completed_steps: set[int] = field(
        default_factory=set
    )

    plan_version: int = 0

    @property
    def current(self):
        if self.plan is None:
            return None

        if not (
            0
            <= self.current_step_index
            < len(self.plan.steps)
        ):
            return None

        return self.plan.steps[
            self.current_step_index
        ]

    @property
    def is_complete(self) -> bool:
        if self.plan is None:
            return False

        return (
            self.current_step_index
            >= len(self.plan.steps)
        )

    def install_plan(
        self,
        plan: PlanSchema,
        *,
        reset_progress: bool = True,
    ) -> None:

        self.plan = plan
        self.plan_version += 1

        if reset_progress:
            self.current_step_index = 0
            self.completed_steps.clear()

    def mark_completed(
        self,
        step_id: int,
    ) -> None:

        self.completed_steps.add(
            step_id
        )

    def advance(self) -> None:

        if self.plan is None:
            return

        if self.current_step_index < len(
            self.plan.steps
        ):
            self.current_step_index += 1


@dataclass(frozen=True, slots=True)
class ExecutionRecord:

    step_id: int
    result: ExecutionResult
    iteration: int

    run_id: str
    plan_version: int

    attempt: int


@dataclass(frozen=True, slots=True)
class VerificationRecord:

    attempt: int
    verified: bool
    result: Any
    answer: str


@dataclass(slots=True)
class OrchestrationContext:

    user_request: str

    run_id: str

    plan_runtime: PlanRuntime = field(
        default_factory=PlanRuntime
    )

    execution_history: list[
        ExecutionRecord
    ] = field(
        default_factory=list
    )

    verification_history: list[
        VerificationRecord
    ] = field(
        default_factory=list
    )

    final_answer: str | None = None

    iteration: int = 0

    terminal_reason: str | None = None

    def record_execution(
        self,
        *,
        step_id: int,
        result: ExecutionResult,
        attempt: int,
    ) -> None:

        self.execution_history.append(
            ExecutionRecord(
                step_id=step_id,
                result=result,
                iteration=self.iteration,
                run_id=self.run_id,
                plan_version=(
                    self.plan_runtime.plan_version
                ),
                attempt=attempt,
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
                attempt=(
                    len(
                        self.verification_history
                    )
                    + 1
                ),
                verified=verified,
                result=result,
                answer=answer,
            )
        )

    @property
    def verification_attempts(self) -> int:

        return len(
            self.verification_history
        )

    @property
    def last_execution(
        self,
    ) -> ExecutionRecord | None:

        if not self.execution_history:
            return None

        return self.execution_history[-1]

    @property
    def is_verified(self) -> bool:

        if not self.verification_history:
            return False

        return (
            self.verification_history[-1]
            .verified
        )
    def verification_evidence(
        self,
    ) -> list[dict[str, Any]]:

        evidence: list[
            dict[str, Any]
        ] = []

        for record in self.execution_history:

            result = record.result

            if not result.success:
                continue

            for item in result.evidence:

                evidence.append(
                    {
                        "tool_name": (
                            item.tool_name
                        ),
                        "result": item.result,
                        "artifact_id": (
                            getattr(
                                item,
                                "artifact_id",
                                None,
                            )
                        ),
                        "source_type": (
                            getattr(
                                item,
                                "evidence_type",
                                None,
                            )
                        ),
                        "succeeded": True,
                        # Provenance
                        "step_id": record.step_id,
                        "execution_id": (
                            self.run_id
                        ),
                        "attempt_id": (
                            f"step-{record.step_id}"
                            f"-attempt-{record.attempt}"
                        ),
                        "run_id": self.run_id,
                        "plan_version": (
                            record.plan_version
                        ),
                        "step_status": "completed",
                        "relevant": True,
                    }
                )

        return evidence