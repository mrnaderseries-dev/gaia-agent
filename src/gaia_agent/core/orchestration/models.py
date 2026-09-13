from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from gaia_agent.agents.verifier import (
    VerificationResult,
    VerificationStatus,
)
from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep
from gaia_agent.planner.task_classifier import TaskAnalysis


class PlanningResultLike(Protocol):
    """
    Structural contract for the Planner output.

    The concrete PlanningResult belongs to the Planner layer.

    Orchestration intentionally depends only on the public contract:
        - plan
        - task_analysis

    This keeps orchestration decoupled from the concrete Planner
    result implementation.
    """

    plan: PlanSchema
    task_analysis: TaskAnalysis


class OrchestrationAction(str, Enum):
    CONTINUE = "continue"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    REPLAN = "replan"
    RETRY = "retry"
    WAIT_FOR_APPROVAL = "wait_for_approval"
    COMPLETE = "complete"
    FAIL = "fail"
    TERMINATE = "terminate"


@dataclass(frozen=True, slots=True)
class ObservabilityContext:
    """
    Immutable observability identity for one orchestration scope.

    A child context preserves correlation/run identity while allowing
    step/attempt/iteration-specific metadata.
    """

    correlation_id: UUID
    run_id: UUID
    agent_id: UUID | None = None
    plan_id: UUID | None = None
    iteration: int = 0
    step_id: int | None = None
    attempt: int | None = None

    def child(
        self,
        *,
        iteration: int | None = None,
        step_id: int | None = None,
        attempt: int | None = None,
        plan_id: UUID | None = None,
    ) -> ObservabilityContext:
        """
        Create a child observability context.

        Values are inherited unless explicitly overridden.
        """

        return ObservabilityContext(
            correlation_id=self.correlation_id,
            run_id=self.run_id,
            agent_id=self.agent_id,
            plan_id=(
                self.plan_id
                if plan_id is None
                else plan_id
            ),
            iteration=(
                self.iteration
                if iteration is None
                else iteration
            ),
            step_id=(
                self.step_id
                if step_id is None
                else step_id
            ),
            attempt=(
                self.attempt
                if attempt is None
                else attempt
            ),
        )


@dataclass(slots=True)
class PlanRuntime:
    """
    Runtime execution state for the currently installed plan.

    PlanRuntime owns:
        - installed plan
        - current execution position
        - completed steps
        - plan version
        - replan count

    AgentState only receives synchronized lifecycle projections.
    """

    plan: PlanSchema | None = None

    current_step: int | None = None

    completed_steps: set[int] = field(
        default_factory=set
    )

    replan_count: int = 0

  
    plan_version: int = 0

    def set_plan(
        self,
        plan: PlanSchema,
        *,
        reset_progress: bool = True,
    ) -> None:
        """
        Install a new executable plan.

        Installing a plan always creates a new plan version.

        When reset_progress=True, execution starts from step zero.
        This is required for both initial planning and replanning.
        """

        if not isinstance(plan, PlanSchema):
            raise TypeError(
                "PlanRuntime.set_plan() expects PlanSchema."
            )

        if not plan.steps:
            raise ValueError(
                "Cannot install an empty plan."
            )

        self.plan = plan

        # A newly installed plan gets a new version.
        self.plan_version += 1

        if reset_progress:
            self.current_step = 0
            self.completed_steps.clear()

        elif self.current_step is None:
            self.current_step = 0

    def current(self) -> PlanStep | None:
        """
        Return the currently executable step.
        """

        if self.plan is None:
            return None

        if self.current_step is None:
            return None

        if not 0 <= self.current_step < len(self.plan.steps):
            return None

        return self.plan.steps[self.current_step]

    def mark_completed(
        self,
        step_id: int,
    ) -> None:
        """
        Mark a step as successfully executed.
        """

        self.completed_steps.add(step_id)

    def advance(self) -> None:
        """
        Advance execution to the next plan position.
        """

        if self.current_step is None:
            return

        self.current_step += 1

    @property
    def complete(self) -> bool:
        """
        Whether the current plan has no remaining executable step.
        """

        if self.plan is None:
            return False

        if self.current_step is None:
            return False

        return self.current_step >= len(self.plan.steps)


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """
    Immutable record of one AgentExecution attempt.
    """

    step_id: int
    result: ExecutionResult
    iteration: int
    attempt: int


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """
    Immutable record of one verification attempt.
    """

    attempt: int
    answer: str
    result: VerificationResult

    @property
    def status(self) -> VerificationStatus:
        return (
            self.result.status
            or VerificationStatus.INSUFFICIENT_EVIDENCE
        )

    @property
    def verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED


@dataclass(slots=True)
class OrchestrationContext:
    """
    Single source of truth for one orchestration run.

    Responsibilities:
        - semantic task analysis
        - current executable plan
        - plan runtime
        - execution position/history
        - verification history
        - final answer
        - observability identity
        - orchestration counters

    AgentState is deliberately NOT the semantic source of truth.
    AgentState receives lifecycle projections from orchestration.
    """

    user_request: str

    run_id: UUID = field(
        default_factory=uuid4
    )

    correlation_id: UUID = field(
        default_factory=uuid4
    )

    agent_id: UUID | None = None

    # Planner semantic output.
    task_analysis: TaskAnalysis | None = None

    # Executable plan runtime.
    plan_runtime: PlanRuntime = field(
        default_factory=PlanRuntime
    )

    # Historical execution records.
    execution_history: list[ExecutionRecord] = field(
        default_factory=list
    )

    # Historical verification records.
    verification_history: list[VerificationRecord] = field(
        default_factory=list
    )

    # Final verified answer.
    final_answer: str | None = None

    # Orchestration counters.
    iteration: int = 0
    current_attempt: int = 0

    termination_reason: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def observability(self) -> ObservabilityContext:
        """
        Return the current run-level observability context.
        """

        return ObservabilityContext(
            correlation_id=self.correlation_id,
            run_id=self.run_id,
            agent_id=self.agent_id,
            plan_id=None,
            iteration=self.iteration,
            step_id=(
                self.plan_runtime.current_step
                if self.plan_runtime
                else None
            ),
            attempt=self.current_attempt,
        )

    @property
    def verification_attempts(self) -> int:
        """
        Number of verification attempts performed in this run.
        """

        return len(self.verification_history)

    @property
    def is_verified(self) -> bool:
        """
        Whether the most recent verification passed.
        """

        if not self.verification_history:
            return False

        return self.verification_history[-1].verified

    @property
    def current_step(self) -> PlanStep | None:
        """
        Return the currently executable plan step.
        """

        return self.plan_runtime.current()

    def install_planning_result(
        self,
        result: PlanningResultLike,
    ) -> None:
        """
        Atomically install the semantic Planner result.

        Both parts must be installed together:

            PlanningResult
                ├── task_analysis
                └── plan

        This prevents a new plan from being installed while the
        semantic task analysis still belongs to the previous plan.
        """

        plan = getattr(
            result,
            "plan",
            None,
        )

        task_analysis = getattr(
            result,
            "task_analysis",
            None,
        )

        if not isinstance(plan, PlanSchema):
            raise TypeError(
                "Planning result must contain a PlanSchema."
            )

        if not isinstance(task_analysis, TaskAnalysis):
            raise TypeError(
                "Planning result must contain TaskAnalysis."
            )

        self.task_analysis = task_analysis

        self.plan_runtime.set_plan(
            plan,
            reset_progress=True,
        )

    def record_execution(
        self,
        step_id: int,
        result: ExecutionResult,
    ) -> None:
        """
        Record one execution attempt.
        """

        self.execution_history.append(
            ExecutionRecord(
                step_id=step_id,
                result=result,
                iteration=self.iteration,
                attempt=self.current_attempt,
            )
        )

    def record_verification(
        self,
        answer: str,
        result: VerificationResult,
    ) -> None:
        """
        Record one verification attempt.
        """

        self.verification_history.append(
            VerificationRecord(
                attempt=(
                    len(self.verification_history) + 1
                ),
                answer=answer,
                result=result,
            )
        )

    @property
    def last_execution(self) -> ExecutionRecord | None:
        """
        Return the most recent execution record.
        """

        if not self.execution_history:
            return None

        return self.execution_history[-1]

    def verification_evidence(self) -> list[Any]:
        """
        Derive verification evidence from successful execution history.

        Execution history is the source of truth.

        This avoids maintaining multiple mutable evidence stores such as:
            - state.evidence
            - context.evidence
            - verifier.evidence

        Only successful executions contribute evidence.
        """

        evidence: list[Any] = []

        for record in self.execution_history:
            result = record.result

            if not result.success:
                continue

            evidence.extend(result.evidence)

        return evidence


@dataclass(frozen=True, slots=True)
class OrchestrationOutcome:
    """
    Result of one Orchestrator.step() call.
    """

    action: OrchestrationAction

    result: ExecutionResult | None = None

    error: Exception | None = None

    reason: str | None = None

    verification: VerificationResult | None = None

    @property
    def terminal(self) -> bool:
        """
        Whether this outcome ends orchestration.
        """

        return self.action in {
            OrchestrationAction.COMPLETE,
            OrchestrationAction.FAIL,
            OrchestrationAction.TERMINATE,
        }