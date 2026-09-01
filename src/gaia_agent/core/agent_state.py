from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gaia_agent.conversation.models import Message
from gaia_agent.planner.plan_schema import PlanStep, StepType

from .policies.approval import ApprovalDecision
from .policies.execution import ExecutionDecision
from .policies.termination import TerminationReason
from .risk.models import RiskAssessment


@dataclass(slots=True)
class AgentState:
    user_request: str
    user_id: int

    messages: list[Message] = field(default_factory=list)

    plan: list[PlanStep] = field(default_factory=list)

    step_type: StepType | None = None

    current_step: int = 0

    completed_steps: list[int] = field(
        default_factory=list
    )

    current_action: str | None = None

    blocked: bool = False
    waiting_for_approval: bool = False

    execution_results: list[Any] = field(
        default_factory=list
    )

    tool_name: str | None = None

    tool_arguments: dict[str, Any] = field(
        default_factory=dict
    )

    tool_result: Any | None = None

    tool_error: str | None = None

    risk_assessment: RiskAssessment | None = None

    approval_decision: ApprovalDecision | None = None

    execution_decision: ExecutionDecision | None = None

    iteration: int = 0

    final_answer: str | None = None

    final_answer_ready: bool = False

    final_answer_verified: bool = False

    termination_reason: TerminationReason | None = None

    fatal_error: bool = False

    human_aborted: bool = False

    explicit_stop: bool = False

    timed_out: bool = False

    retry_count: int = 0

    recovery_attempted: bool = False

   
    execution_success: bool = False

    step_succeeded: bool = False

    task_completed: bool = False

   
    evidence: list[Any] = field(
        default_factory=list
    )

    replan_count: int = 0
    loop_salvage_attempted: bool = False

    same_failure_count: int = 0

    same_plan_count: int = 0

    last_failure_key: str | None = None
    executed_step_fingerprints: set[str] = field(
    default_factory=set
)

    
    verification_attempts: int = 0