from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from gaia_agent.agents.verifier import (
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
)
from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.models import (
    ContextRequest,
    FinalContext,
)
from gaia_agent.core.agent_execution import (
    AgentExecution,
    ExecutionRequest,
    ExecutionResult,
)
from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
    TransitionReason,
)
from gaia_agent.observability.events import EventType
from gaia_agent.observability.facade import Observability
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.planner import (
    Planner,
    PlannerRecoveryRequired,
)
from gaia_agent.planner.task_classifier import (
    TaskAnalysis,
)
from gaia_agent.reliability.engine import (
    ReliabilityAction,
    ReliabilityEngine,
)
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)
from gaia_agent.reliability.loop_detector import (
    LoopDetector,
)

from .models import (
    OrchestrationAction,
    OrchestrationContext,
    OrchestrationOutcome,
)


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Configuration settings for the Orchestrator limits."""
    max_step_attempts: int = 3
    max_verification_attempts: int = 2
    max_replans: int = 3


class Orchestrator:
    """
    Central orchestration control plane.

    Responsibilities:
        - coordinate ContextBuilder
        - request planning
        - install PlanningResult
        - execute PlanStep
        - delegate failures to ReliabilityEngine
        - perform replanning when Reliability requests it
        - coordinate verification
        - synchronize AgentState projections
        - emit orchestration observability

    Not responsible for:
        - task classification
        - strategy selection
        - tool execution internals
        - retry policy
        - recovery policy
        - verification semantics
    """

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        planner: Planner,
        agent_execution: AgentExecution,
        reliability_engine: ReliabilityEngine,
        loop_detector: LoopDetector,
        verifier: VerifierAgent,
        observability: Observability | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        """Initialize the orchestrator with required dependencies and configuration."""
        self.context_builder = context_builder
        self.planner = planner
        self.agent_execution = agent_execution
        self.reliability = reliability_engine
        self.loop_detector = loop_detector
        self.verifier = verifier
        self.observability = observability
        self.config = (
            config
            or OrchestratorConfig()
        )

    async def start(
        self,
        state: AgentState,
        *,
        run: OrchestrationContext | None = None,
    ) -> OrchestrationContext:
        """Start the orchestration run and initialize states."""
        if not state.user_request.strip():
            raise ValueError(
                "user_request cannot be empty"
            )

        run = (
            run
            or OrchestrationContext(
                user_request=state.user_request
            )
        )

        if (
            run.user_request
            != state.user_request
        ):
            raise ValueError(
                "OrchestrationContext.user_request "
                "must match AgentState.user_request."
            )

        if state.phase is AgentPhase.IDLE:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.START,
            )

        self._emit(
            EventType.AGENT_STARTED,
            run,
        )

        return run

    async def step(
        self,
        state: AgentState,
        run: OrchestrationContext,
    ) -> OrchestrationOutcome:
        """
        Execute exactly one orchestration cycle.

        AgentLoop owns repetition.
        Orchestrator owns the decision for this cycle.
        """
        run.iteration += 1
        state.iteration = run.iteration

        if state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.FAILED,
            AgentPhase.TERMINATED,
        }:
            return OrchestrationOutcome(
                OrchestrationAction.TERMINATE,
                reason=state.phase.value,
            )

        try:
            with self._span(
                "agent.step",
                run.observability,
            ):
                context = await self._build_context(
                    state,
                    run,
                )

                if (
                    run.plan_runtime.plan
                    is None
                ):
                    return await self._plan(
                        state,
                        run,
                        context,
                    )

             
                step = (
                    run.plan_runtime.current()
                )

                if step is None:
                    if (
                        run.final_answer
                        is not None
                    ):
                        return OrchestrationOutcome(
                            OrchestrationAction.COMPLETE,
                            reason="plan_complete",
                        )

                    # Defensive recovery: a plan existed but no
                    # executable current step remains and no answer exists.
                    return await self._plan(
                        state,
                        run,
                        context,
                    )
                return await self._execute(
                    state,
                    run,
                    step,
                    context,
                )

        except AgentError as error:
            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.AGENT_FAILED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        except Exception as exc:
            error = AgentError(
                error_type=type(exc).name,
                message=(
                    str(exc)
                    or type(exc).name
                ),
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="Orchestrator",
                operation="step",
                original_exception=exc,
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.AGENT_FAILED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )
    async def _build_context(
        self,
        state: AgentState,
        run: OrchestrationContext,
    ) -> FinalContext:
        """Build context required for execution and planning."""
        request = ContextRequest(
            user_request=run.user_request,
            attachments=tuple(
                state.attachments
            ),
            plan=(
                list(
                    run.plan_runtime.plan.steps
                )
                if run.plan_runtime.plan
                else []
            ),
            current_step=(
                run.plan_runtime.current_step
            ),
            completed_steps=sorted(
                run.plan_runtime.completed_steps
            ),
            iteration=run.iteration,
        )

        return await self.context_builder.build(
            request
        )
    async def _plan(
        self,
        state: AgentState,
        run: OrchestrationContext,
        context: FinalContext,
    ) -> OrchestrationOutcome:
        """Generate and install a new execution plan."""
        self._emit(
            EventType.PLANNING_STARTED,
            run,
        )

        try:
            planning_result = (
                await self.planner.generate_plan(
                    run.user_request,
                    context,
                )
            )

            self._install_planning_result(
                state,
                run,
                planning_result,
            )

        except PlannerRecoveryRequired as exc:
            error = self._planner_error(
                exc,
                operation="generate_plan",
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.PLAN_REJECTED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        self._emit(
            EventType.PLAN_GENERATED,
            run,
            metadata={
                "steps": len(
                    run.plan_runtime.plan.steps
                ),
                "task_intent": (
                    run.task_analysis.intent.value
                    if run.task_analysis
                    else None
                ),
            },
        )

        return OrchestrationOutcome(
            OrchestrationAction.EXECUTE,
            reason="plan_ready",
        )

    def _install_planning_result(
        self,
        state: AgentState,
        run: OrchestrationContext,
        planning_result: Any,
    ) -> None:
        """
        Install Planner output as one atomic orchestration contract.

        PlanningResult:
            plan
            task_analysis
        """
        run.install_planning_result(
            planning_result
        )

        plan = run.plan_runtime.plan

        if plan is None:
            raise ValueError(
                "Planner returned no plan."
            )

        self._validate_plan(plan)

        # AgentState is a projection, not the SSOT.
        state.plan = list(
            plan.steps
        )
        state.current_step = (
            run.plan_runtime.current_step
        )
        state.completed_steps.clear()
        state.replan_count = (
            run.plan_runtime.replan_count
        )

        # Keep semantic metadata visible for external state/debugging.
        state.metadata[
            "task_intent"
        ] = run.task_analysis.intent.value

        state.metadata[
            "task_analysis"
        ] = run.task_analysis.analysis_text

        if state.phase is AgentPhase.PLANNING:
            state.transition(
                AgentPhase.EXECUTING,
                reason=TransitionReason.PLAN_READY,
            )

    async def _execute(
        self,
        state: AgentState,
        run: OrchestrationContext,
        step: PlanStep,
        context: FinalContext,
    ) -> OrchestrationOutcome:
        """Execute a single plan step and handle outcomes."""
        step_id = step.step_id

        ctx = run.observability.child(
            step_id=step_id,
        )

        strategy_family = (
            self.planner.strategy_family(
                step
            )
        )

        loop = self.loop_detector.check(
            step,
            strategy_family=strategy_family,
        )

        if loop.detected:
            self._emit(
                EventType.LOOP_DETECTED,
                run,
                metadata={
                    "step_id": step_id,
                    "reason": loop.reason,
                },
            )

            error = AgentError(
                error_type="ExecutionLoopDetected",
                message=loop.reason,
                category=ErrorCategory.LOOP_DETECTED,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="Orchestrator",
                operation="execute",
            )

            self._fail(
                state,
                error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                error=error,
            )

        run.current_attempt += 1

        request = ExecutionRequest(
            step_id=step_id,
            step_type=step.step_type,
            action=step.action,
            tool_name=step.tool_name,
            arguments=dict(
                step.arguments
            ),
            user_request=run.user_request,
            context=context,
            iteration=run.iteration,
            correlation_id=run.correlation_id,
            metadata={
                "run_id": str(
                    run.run_id
                ),
                "attempt": (
                    run.current_attempt
                ),
                "plan_version": (
                    run.plan_runtime.replan_count
                ),
            },
        )

        self._emit(
            EventType.STEP_STARTED,
            run,
        )

        result = (
            await self.agent_execution.execute(
                request
            )
        )

        run.record_execution(
            step_id,
            result,
        )

        state.execution_results.append(
            result
        )

        if result.blocked:
            state.blocked = True
            state.waiting_for_approval = True

            self._emit(
                EventType.APPROVAL_REQUIRED,
                run,
            )

            return OrchestrationOutcome(
                OrchestrationAction.WAIT_FOR_APPROVAL,
                result=result,
                reason="approval_required",
            )

        if not result.success:
            self._emit(
                EventType.STEP_FAILED,
                run,
                error=result.error,
            )

            return await self._handle_failure(
                state=state,
                run=run,
                step=step,
                result=result,
                context=context,
            )
        state.blocked = False
        state.waiting_for_approval = False
        state.execution_success = True
        state.step_succeeded = True
        state.tool_result = result.output
        state.tool_error = None

        self.loop_detector.record(
            step,
            strategy_family=strategy_family,
        )

        run.plan_runtime.mark_completed(
            step_id
        )

        state.completed_steps = sorted(
            run.plan_runtime.completed_steps
        )

        self._emit(
            EventType.STEP_COMPLETED,
            run,
        )

        # Final-answer step is verified before allowing orchestration to complete.
        if step.is_final_answer:
            return await self._verify(
                state,
                run,
                step,
                result,
                context,
            )

        run.plan_runtime.advance()

        state.current_step = (
            run.plan_runtime.current_step
        )

        run.current_attempt = 0

        return OrchestrationOutcome(
            OrchestrationAction.EXECUTE,
            result=result,
            reason="step_completed",
        )

    async def _handle_failure(
        self,
        *,
        state: AgentState,
        run: OrchestrationContext,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> OrchestrationOutcome:
        """Handle execution failures through the reliability engine."""
        if result.error is None:
            error = AgentError(
                error_type="ExecutionFailed",
                message=(
                    "Execution failed without "
                    "an AgentError."
                ),
                category=ErrorCategory.EXECUTION_ERROR,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="execute",
            )

            self._fail(
                state,
                error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                result=result,
                error=error,
            )

        decision = await self.reliability.handle_failure(
            error=result.error,
            attempt=max(
                1,
                run.current_attempt,
            ),
            max_attempts=(
                self.config.max_step_attempts
            ),
        )

        if (
            decision.action
            is ReliabilityAction.RETRY
        ):
            self._emit(
                EventType.RETRY_STARTED,
                run,
                metadata={
                    "attempt": run.current_attempt,
                    "reason": decision.reason,
                },
            )

            return OrchestrationOutcome(
                OrchestrationAction.RETRY,
                result=result,
                error=decision.error,
                reason=decision.reason,
            )

        if (
            decision.action
            is ReliabilityAction.REPLAN
        ):
            return await self._replan(
                state=state,
                run=run,
                failed_step=step,
                failure=(
                    decision.error
                    or result.error
                ),
                context=context,
                reason=decision.reason,
            )

        failure = (
            decision.error
            or result.error
        )

        self._fail(
            state,
            failure,
        )

        return OrchestrationOutcome(
            OrchestrationAction.FAIL,
            result=result,
            error=failure,
            reason=decision.reason,
        )

    async def _replan(
        self,
        *,
        state: AgentState,
        run: OrchestrationContext,
        failed_step: PlanStep,
        failure: AgentError,
        context: FinalContext,
        reason: str,
    ) -> OrchestrationOutcome:
        """Trigger a replanning cycle due to errors or failed verifications."""
        if (
            run.plan_runtime.replan_count
            >= self.config.max_replans
        ):
            self._fail(
                state,
                failure,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                error=failure,
                reason="replan_budget_exhausted",
            )

        self._emit(
            EventType.RECOVERY_STARTED,
            run,
            metadata={
                "reason": reason,
                "replan_count": (
                    run.plan_runtime.replan_count
                ),
            },
        )

        if state.phase is AgentPhase.EXECUTING:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.RECOVERY,
            )

        elif state.phase is AgentPhase.VERIFYING:
            state.transition(
                AgentPhase.PLANNING,
                reason=(
                    TransitionReason.VERIFICATION_FAILED
                ),
            )

        elif state.phase is AgentPhase.FAILED:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.RECOVERY,
            )

        try:
            # IMPORTANT: Planner owns replan construction.
            planning_result = (
                await self.planner.replan(
                    user_question=run.user_request,
                    context=context,
                    failed_step=failed_step,
                    failure=failure,
                )
            )

            run.plan_runtime.replan_count += 1

            self._install_planning_result(
                state,
                run,
                planning_result,
            )

            state.replan_count = (
                run.plan_runtime.replan_count
            )

            # New plan = new execution attempt.
            run.current_attempt = 0

        except PlannerRecoveryRequired as exc:
            error = self._planner_error(
                exc,
                operation="replan",
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.PLAN_REJECTED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        self._emit(
            EventType.PLAN_REPLANNED,
            run,
            metadata={
                "replan_count": (
                    run.plan_runtime.replan_count
                ),
                "steps": len(
                    run.plan_runtime.plan.steps
                ),
                "task_intent": (
                    run.task_analysis.intent.value
                    if run.task_analysis
                    else None
                ),
            },
        )

        self._emit(
            EventType.RECOVERY_COMPLETED,
            run,
        )

        return OrchestrationOutcome(
            OrchestrationAction.REPLAN,
            reason=reason,
        )
    async def _verify(
        self,
        state: AgentState,
        run: OrchestrationContext,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> OrchestrationOutcome:
        """Verify the final answer generated by the agent."""
        answer = self._extract_answer(
            result.output
        )

        if not answer:
            verification = VerificationResult(
                status=VerificationStatus.INVALID,
                reason="Final answer is empty.",
            )
        else:
            state.final_answer = answer
            state.final_answer_ready = True

            if state.phase is AgentPhase.EXECUTING:
                state.transition(
                    AgentPhase.VERIFYING,
                    reason=(
                        TransitionReason.EXECUTION_COMPLETED
                    ),
                )

            self._emit(
                EventType.VERIFICATION_STARTED,
                run,
            )

            task_analysis = (
                run.task_analysis
            )

            if task_analysis is None:
                error = AgentError(
                    error_type="MissingTaskAnalysis",
                    message=(
                        "Cannot verify final answer "
                        "without Planner TaskAnalysis."
                    ),
                    category=(
                        ErrorCategory.STATE_TRANSITION_ERROR
                    ),
                    severity=ErrorSeverity.CRITICAL,
                    retryable=False,
                    recoverable=False,
                    source="Orchestrator",
                    operation="verify",
                )

                self._fail(
                    state,
                    error,
                )

                return OrchestrationOutcome(
                    OrchestrationAction.FAIL,
                    result=result,
                    error=error,
                    reason=error.message,
                )

            verification = (
                await self.verifier.verify(
                    VerificationInput(
                        question=run.user_request,
                        candidate_answer=answer,
                        raw_data=list(
                            result.evidence
                        ),
                        task_type=(
                            task_analysis.intent.value
                        ),
                    )
                )
            )

        run.record_verification(
            answer,
            verification,
        )

        state.verification_attempts = (
            run.verification_attempts
        )

        state.final_answer_verified = (
            verification.status
            is VerificationStatus.VERIFIED
        )

        self._emit(
            EventType.VERIFICATION_COMPLETED,
            run,
            metadata={
                "status": (
                    verification.status.value
                ),
                "task_intent": (
                    run.task_analysis.intent.value
                    if run.task_analysis
                    else None
                ),
            },
        )

        if state.final_answer_verified:
            run.final_answer = answer
            state.final_answer = answer
            state.final_answer_ready = True
            state.task_completed = True

            if state.phase is AgentPhase.VERIFYING:
                state.transition(
                    AgentPhase.COMPLETED,
                    reason=(
                        TransitionReason.VERIFICATION_PASSED
                    ),
                )

            self._emit(
                EventType.AGENT_COMPLETED,
                run,
            )

            return OrchestrationOutcome(
                OrchestrationAction.COMPLETE,
                result=result,
                verification=verification,
                reason="answer_verified",
            )

        if (
            run.verification_attempts
            >= self.config.max_verification_attempts
        ):
            error = AgentError(
                error_type="VerificationBudgetExceeded",
                message=(
                    "Final answer verification "
                    "budget exhausted."
                ),
                category=ErrorCategory.INVALID_RESULT,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="Verifier",
                operation="verify",
            )

            self._fail(
                state,
                error,
            )

            return OrchestrationOutcome(
                OrchestrationAction.FAIL,
                result=result,
                error=error,
                verification=verification,
                reason=(
                    "verification_budget_exhausted"
                ),
            )

        failure = AgentError(
            error_type="AnswerVerificationFailed",
            message=(
                verification.reason
                or "Answer verification failed."
            ),
            category=ErrorCategory.INVALID_RESULT,
            severity=ErrorSeverity.MEDIUM,
            retryable=False,
            recoverable=True,
            source="Verifier",
            operation="verify",
        )

        return await self._replan(
            state=state,
            run=run,
            failed_step=step,
            failure=failure,
            context=context,
            reason=(
                verification.status.value
                if verification.status
                else "verification_failed"
            ),
        )

    @staticmethod
    def _validate_plan(
        plan: PlanSchema,
    ) -> None:
        """Validate structural rules of the generated plan."""
        if not plan.steps:
            raise ValueError(
                "Planner returned an empty plan."
            )

        final_steps = [
            step
            for step in plan.steps
            if step.is_final_answer
        ]

        if len(final_steps) != 1:
            raise ValueError(
                "Plan must contain exactly one "
                "final-answer step."
            )

        final_step = final_steps[0]

        if (
            final_step.step_id
            != len(plan.steps) - 1
        ):
            raise ValueError(
                "Final-answer step must be "
                "the last step."
            )

        if (
            final_step.step_type
            is not StepType.LLM
        ):
            raise ValueError(
                "Final-answer step must be an LLM step."
            )


    @staticmethod
    def _extract_answer(
        output: Any,
    ) -> str:
        """Extract string answer from tool or step output."""
        if output is None:
            return ""

        if isinstance(output, str):
            return output.strip()

        return str(output).strip()

    @staticmethod
    def _planner_error(
        error: Exception,
        *,
        operation: str,
    ) -> AgentError:
        """Format planner exception into a standard AgentError."""
        return AgentError(
            error_type="PlannerRecoveryRequired",
            message=(
                str(error)
                or "Planner recovery required."
            ),
            category=ErrorCategory.PLAN_RECOVERY_ERROR,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            recoverable=True,
            source="Planner",
            operation=operation,
            original_exception=error,
        )

    @staticmethod
    def _fail(
        state: AgentState,
        error: AgentError | None,
    ) -> None:
        """Update agent state to reflect failure status."""
        state.fatal_error = bool(
            error
            and error.severity
            is ErrorSeverity.CRITICAL
        )

        state.tool_error = (
            error.message
            if error
            else "Execution failed."
        )

        if state.phase not in {
            AgentPhase.COMPLETED,
            AgentPhase.TERMINATED,
        }:
            state.phase = AgentPhase.FAILED

    def _emit(
        self,
        event: EventType,
        run: OrchestrationContext,
        *,
        metadata: dict[str, Any] | None = None,
        error: AgentError | None = None,
    ) -> None:
        """Emit observability events if facade is configured."""
        if self.observability:
            self.observability.emit(
                event,
                run.observability,
                metadata=metadata,
                error=(
                    error.message
                    if error
                    else None
                ),
            )

    def _span(
        self,
        operation: str,
        context: Any,
    ):
        """Create an observability tracing span."""
        if self.observability:
            return self.observability.span(
                operation,
                context,
            )

        return nullcontext()