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

    # ==========================================================
    # Phase 2: separated execution states
    #
    # execution_success -> the execution machinery ran cleanly
    # step_succeeded    -> the current step produced a usable result
    # task_completed    -> the whole task finished (answer delivered)
    # final_answer_ready / final_answer_verified -> answer status
    #
    # A blocked action must NEVER be recorded as execution_success.
    # ==========================================================

    execution_success: bool = False

    step_succeeded: bool = False

    task_completed: bool = False

    # ==========================================================
    # Phase 4: evidence / artifact registry (see core.evidence)
    # ==========================================================

    evidence: list[Any] = field(
        default_factory=list
    )

    # ==========================================================
    # Phase 6: bounded recovery counters
    # ==========================================================

    replan_count: int = 0
    loop_salvage_attempted: bool = False

    same_failure_count: int = 0

    same_plan_count: int = 0

    last_failure_key: str | None = None

    # ==========================================================
    # Phase 7: bounded semantic verification.
    # After this many failed verification attempts the answer is
    # delivered honestly as UNVERIFIED instead of looping.
    # ==========================================================

    verification_attempts: int = 0