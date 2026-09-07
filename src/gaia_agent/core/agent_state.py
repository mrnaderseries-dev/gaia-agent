from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentPhase(str, Enum):
    """
    Lifecycle phase of the agent.
    """

    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVIDENCE_AVAILABLE = "evidence_available"
    ANSWER_GENERATED = "answer_generated"
    VERIFYING = "verifying"
    COMPLETED = "completed"

    FAILED = "failed"
    RECOVERING = "recovering"


class TransitionReason(str, Enum):
    """
    Why the agent is moving from one phase to another.
    """

    INITIAL_PLAN = "initial_plan"

    RETRY = "retry"
    RECOVERY = "recovery"

    EVIDENCE = "evidence"
    ANSWER = "answer"
    VERIFICATION = "verification"

    FAILURE = "failure"
    COMPLETION = "completion"


class TerminationReason(str, Enum):
    """
    Why the agent execution ended.
    """

    VERIFIED = "verified"
    USER_ABORTED = "user_aborted"
    UNSUPPORTED = "unsupported"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    FATAL_ERROR = "fatal_error"
    TIMEOUT = "timeout"


class InvalidStateTransition(Exception):
    """
    Raised when an illegal AgentPhase transition is attempted.
    """

    def __init__(
        self,
        current: AgentPhase,
        target: AgentPhase,
        reason: TransitionReason | None = None,
    ) -> None:
        message = (
            f"Invalid agent state transition: "
            f"{current.value} -> {target.value}"
        )

        if reason is not None:
            message += f" (reason={reason.value})"

        super().__init__(message)

        self.current = current
        self.target = target
        self.reason = reason


# ---------------------------------------------------------------------------
# Legal state transitions
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[AgentPhase, dict[AgentPhase, set[TransitionReason]]] = {
    AgentPhase.CREATED: {
        AgentPhase.PLANNING: {
            TransitionReason.INITIAL_PLAN,
        },
    },

    AgentPhase.PLANNING: {
        AgentPhase.EXECUTING: {
            TransitionReason.INITIAL_PLAN,
            TransitionReason.RECOVERY,
        },
        AgentPhase.FAILED: {
            TransitionReason.FAILURE,
        },
    },

    AgentPhase.EXECUTING: {
        AgentPhase.EVIDENCE_AVAILABLE: {
            TransitionReason.EVIDENCE,
        },

        AgentPhase.FAILED: {
            TransitionReason.FAILURE,
        },
    },

    AgentPhase.EVIDENCE_AVAILABLE: {
        AgentPhase.ANSWER_GENERATED: {
            TransitionReason.ANSWER,
        },

        AgentPhase.FAILED: {
            TransitionReason.FAILURE,
        },
    },

    AgentPhase.ANSWER_GENERATED: {
        AgentPhase.VERIFYING: {
            TransitionReason.VERIFICATION,
        },

        AgentPhase.FAILED: {
            TransitionReason.FAILURE,
        },
    },

    AgentPhase.VERIFYING: {
        AgentPhase.COMPLETED: {
            TransitionReason.COMPLETION,
        },

        AgentPhase.FAILED: {
            TransitionReason.FAILURE,
        },

        AgentPhase.RECOVERING: {
            TransitionReason.RECOVERY,
        },
    },

    # -----------------------------------------------------------------------
    # Failure / retry / recovery
    # -----------------------------------------------------------------------

    AgentPhase.FAILED: {
        AgentPhase.EXECUTING: {
            TransitionReason.RETRY,
        },

        AgentPhase.RECOVERING: {
            TransitionReason.RECOVERY,
        },
    },

    AgentPhase.RECOVERING: {
        AgentPhase.EXECUTING: {
            TransitionReason.RECOVERY,
        },
        AgentPhase.FAILED: {
            TransitionReason.FAILURE,
        },
    },

    # COMPLETED intentionally has no outgoing transitions.
    AgentPhase.COMPLETED: {},
}


@dataclass(slots=True)
class AgentState:
    """
    Canonical runtime state of the agent.

    This object stores state.

    It does NOT decide:
      - whether to retry
      - whether to recover
      - whether to replan
      - whether a failure is transient
      - whether a tool should be selected

    Those decisions belong to policies / orchestration layers.
    """

    task_id: str
    user_request: str
    phase: AgentPhase = AgentPhase.CREATED
    transition_history: list[AgentPhase] = field(default_factory=list)
    current_plan: Any | None = None
    current_step: Any | None = None
    executed_steps: list[Any] = field(default_factory=list)
    artifacts: list[Any] = field(default_factory=list)
    last_tool_result: Any | None = None
    last_failure: Any | None = None
    answer: str | None = None
    verification: Any | None = None
    termination_reason: TerminationReason | None = None


    def transition(
        self,
        new_phase: AgentPhase,
        *,
        reason: TransitionReason | None = None,
    ) -> None:
        """
        Transition the agent to a new phase if allowed by the transition rules.
        """
        allowed_reasons = _ALLOWED_TRANSITIONS.get(self.phase, {}).get(new_phase)

        if allowed_reasons is None or (reason is not None and reason not in allowed_reasons):
            raise InvalidStateTransition(self.phase, new_phase, reason)

        self.transition_history.append(self.phase)
        self.phase = new_phase