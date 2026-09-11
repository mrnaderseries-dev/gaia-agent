from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gaia_agent.core.evidence import ArtifactInfo, ToolResultRecord
from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)


class AgentPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class TransitionReason(str, Enum):
    START = "start"
    PLAN_READY = "plan_ready"
    EXECUTION_READY = "execution_ready"
    EXECUTION_COMPLETED = "execution_completed"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    EXECUTION_FAILED = "execution_failed"
    RECOVERY = "recovery"
    RETRY = "retry"
    USER_ABORT = "user_abort"
    TERMINATION = "termination"


@dataclass(frozen=True, slots=True)
class StateTransition:
    from_phase: AgentPhase
    to_phase: AgentPhase
    reason: TransitionReason


@dataclass
class AgentState:
    user_request: str = ""

    phase: AgentPhase = AgentPhase.IDLE

    metadata: dict[str, Any] = field(
        default_factory=dict
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

    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(
        default_factory=dict
    )
    tool_result: Any = None
    tool_error: str | None = None

    execution_success: bool = False
    step_succeeded: bool = False
    blocked: bool = False
    waiting_for_approval: bool = False

    iteration: int = 0

    retry_count: int = 0
    recovery_attempted: bool = False
    replan_count: int = 0

    loop_salvage_attempted: bool = False
    same_failure_count: int = 0
    same_plan_count: int = 0
    last_failure_key: str | None = None

    final_answer: str | None = None
    final_answer_ready: bool = False
    final_answer_verified: bool = False
    verification_attempts: int = 0

    task_completed: bool = False

    fatal_error: bool = False
    human_aborted: bool = False
    explicit_stop: bool = False
    timed_out: bool = False

    termination_reason: Any = None

    evidence: list[ToolResultRecord] = field(
        default_factory=list
    )
    artifacts: list[ArtifactInfo] = field(
        default_factory=list
    )
    execution_results: list[Any] = field(
        default_factory=list
    )
    messages: list[Any] = field(
        default_factory=list
    )

    transition_history: list[StateTransition] = field(
        default_factory=list
    )

    _ALLOWED_TRANSITIONS: dict[
        AgentPhase,
        frozenset[AgentPhase],
    ] = field(
        default_factory=lambda: {
            AgentPhase.IDLE: frozenset(
                {
                    AgentPhase.PLANNING,
                    AgentPhase.TERMINATED,
                }
            ),
            AgentPhase.PLANNING: frozenset(
                {
                    AgentPhase.EXECUTING,
                    AgentPhase.FAILED,
                    AgentPhase.TERMINATED,
                }
            ),
            AgentPhase.EXECUTING: frozenset(
                {
                    AgentPhase.VERIFYING,
                    AgentPhase.FAILED,
                    AgentPhase.TERMINATED,
                }
            ),
            AgentPhase.VERIFYING: frozenset(
                {
                    AgentPhase.COMPLETED,
                    AgentPhase.PLANNING,
                    AgentPhase.FAILED,
                    AgentPhase.TERMINATED,
                }
            ),
            AgentPhase.COMPLETED: frozenset(
                {
                    AgentPhase.TERMINATED,
                }
            ),
            AgentPhase.FAILED: frozenset(
                {
                    AgentPhase.PLANNING,
                    AgentPhase.TERMINATED,
                }
            ),
            AgentPhase.TERMINATED: frozenset(),
        },
        init=False,
        repr=False,
    )

    def transition(
        self,
        new_phase: AgentPhase,
        *,
        reason: TransitionReason,
    ) -> StateTransition:
        if not isinstance(new_phase, AgentPhase):
            raise AgentError(
                error_type="InvalidPhase",
                message=(
                    f"Invalid phase type: "
                    f"{type(new_phase).__name__}"
                ),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentState",
                operation="transition",
                details={
                    "current_phase": self.phase.value,
                    "target_phase": repr(new_phase),
                },
            )

        if not isinstance(reason, TransitionReason):
            raise AgentError(
                error_type="InvalidTransitionReason",
                message=(
                    f"Invalid transition reason: "
                    f"{reason!r}"
                ),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentState",
                operation="transition",
                details={
                    "current_phase": self.phase.value,
                    "target_phase": new_phase.value,
                    "reason": repr(reason),
                },
            )

        current_phase = self.phase

        allowed_targets = self._ALLOWED_TRANSITIONS.get(
            current_phase,
            frozenset(),
        )

        if new_phase not in allowed_targets:
            raise AgentError(
                error_type="InvalidStateTransition",
                message=(
                    f"Invalid state transition: "
                    f"{current_phase.value} -> "
                    f"{new_phase.value}"
                ),
                category=ErrorCategory.STATE_TRANSITION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                retryable=False,
                recoverable=False,
                source="AgentState",
                operation="transition",
                details={
                    "current_phase": current_phase.value,
                    "target_phase": new_phase.value,
                    "reason": reason.value,
                    "allowed_transitions": [
                        phase.value
                        for phase in allowed_targets
                    ],
                },
            )

        transition = StateTransition(
            from_phase=current_phase,
            to_phase=new_phase,
            reason=reason,
        )

        self.phase = new_phase
        self.transition_history.append(transition)

        return transition

    def start(self) -> StateTransition:
        return self.transition(
            AgentPhase.PLANNING,
            reason=TransitionReason.START,
        )

    def begin_execution(self) -> StateTransition:
        return self.transition(
            AgentPhase.EXECUTING,
            reason=TransitionReason.EXECUTION_READY,
        )

    def begin_verification(self) -> StateTransition:
        return self.transition(
            AgentPhase.VERIFYING,
            reason=TransitionReason.EXECUTION_COMPLETED,
        )

    def complete(self) -> StateTransition:
        return self.transition(
            AgentPhase.COMPLETED,
            reason=TransitionReason.VERIFICATION_PASSED,
        )

    def fail(
        self,
        *,
        reason: TransitionReason = (
            TransitionReason.EXECUTION_FAILED
        ),
    ) -> StateTransition:
        return self.transition(
            AgentPhase.FAILED,
            reason=reason,
        )

    def recover(self) -> StateTransition:
        return self.transition(
            AgentPhase.PLANNING,
            reason=TransitionReason.RECOVERY,
        )

    def retry(self) -> StateTransition:
        return self.transition(
            AgentPhase.PLANNING,
            reason=TransitionReason.RETRY,
        )

    def terminate(self) -> StateTransition:
        return self.transition(
            AgentPhase.TERMINATED,
            reason=TransitionReason.TERMINATION,
        )

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.TERMINATED,
        }

    @property
    def is_failed(self) -> bool:
        return self.phase is AgentPhase.FAILED

    @property
    def is_active(self) -> bool:
        return self.phase in {
            AgentPhase.PLANNING,
            AgentPhase.EXECUTING,
            AgentPhase.VERIFYING,
        }

    def can_transition_to(
        self,
        new_phase: AgentPhase,
    ) -> bool:
        if not isinstance(new_phase, AgentPhase):
            return False

        return new_phase in self._ALLOWED_TRANSITIONS.get(
            self.phase,
            frozenset(),
        )

    def last_transition(
        self,
    ) -> StateTransition | None:
        if not self.transition_history:
            return None

        return self.transition_history[-1]