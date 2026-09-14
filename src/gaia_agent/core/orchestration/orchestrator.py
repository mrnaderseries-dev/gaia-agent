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
from gaia_agent.context.models import ContextRequest, FinalContext
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
from gaia_agent.observability.facade import ObservabilityFacade
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.planner import (
    Planner,
    PlannerRecoveryRequired,
)
from gaia_agent.planner.task_classifier import TaskAnalysis
from gaia_agent.reliability.engine import (
    ReliabilityAction,
    ReliabilityEngine,
)
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)
from gaia_agent.reliability.loop_detector import LoopDetector

from .models import (
    OrchestrationAction,
    OrchestrationContext,
    OrchestrationOutcome,
)


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    max_step_attempts: int = 3
    max_verification_attempts: int = 2
    max_replans: int = 3


class Orchestrator:
    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        planner: Planner,
        agent_execution: AgentExecution,
        reliability_engine: ReliabilityEngine,
        loop_detector: LoopDetector,
        verifier: VerifierAgent,
        observability: ObservabilityFacade | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.planner = planner
        self.agent_execution = agent_execution
        self.reliability = reliability_engine
        self.loop_detector = loop_detector
        self.verifier = verifier
        self.observability = observability
        self.config = config or OrchestratorConfig()

    async def start(
        self,
        state: AgentState,
        *,
        run: OrchestrationContext | None = None,
    ) -> OrchestrationContext:
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

        if run.user_request != state.user_request:
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
        run.iteration += 1
        state.iteration = run.iteration

        if state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.FAILED,
            AgentPhase.TERMINATED,
        }:
            return OrchestrationOutcome(
                action=OrchestrationAction.TERMINATE,
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

                if run.plan_runtime.plan is None:
                    return await self._plan(
                        state,
                        run,
                        context,
                    )

                step = run.plan_runtime.current()

                if step is None:
                    if run.final_answer is not None:
                        return OrchestrationOutcome(
                            action=OrchestrationAction.COMPLETE,
                            reason="plan_complete",
                        )

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
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        except Exception as exc:
            error_name = type(exc).__name__

            error = AgentError(
                error_type=error_name,
                message=str(exc) or error_name,
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
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

    async def _build_context(
        self,
        state: AgentState,
        run: OrchestrationContext,
    ) -> FinalContext:
        request = ContextRequest(
            user_request=run.user_request,
            attachments=tuple(state.attachments),
            plan=(
                list(run.plan_runtime.plan.steps)
                if run.plan_runtime.plan
                else []
            ),
            current_step=run.plan_runtime.current_step,
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
        self._emit(
            EventType.PLANNING_STARTED,
            run,
        )

        try:
            planning_result = await self.planner.generate_plan(
                run.user_request,
                context,
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
                action=OrchestrationAction.FAIL,
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
                "plan_version": (
                    run.plan_runtime.plan_version
                ),
            },
        )

        return OrchestrationOutcome(
            action=OrchestrationAction.EXECUTE,
            reason="plan_ready",
        )

    def _install_planning_result(
        self,
        state: AgentState,
        run: OrchestrationContext,
        planning_result: Any,
    ) -> None:
        run.install_planning_result(
            planning_result
        )

        plan = run.plan_runtime.plan

        if plan is None:
            raise ValueError(
                "Planner returned no plan."
            )

        self._validate_plan(plan)

        if state.phase is not AgentPhase.PLANNING:
            raise AgentError(
                error_type="PlanInstalledOutsidePlanning",
                message=(
                    "A plan may only be installed while the agent is "
                    "in PLANNING phase; current phase is "
                    f"{state.phase.value}."
                ),
                category=ErrorCategory.STATE_TRANSITION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                retryable=False,
                recoverable=False,
                source="Orchestrator",
                operation="install_plan",
            )

        state.plan = list(plan.steps)
        state.current_step = (
            run.plan_runtime.current_step
        )

        state.completed_steps.clear()

        state.replan_count = (
            run.plan_runtime.replan_count
        )

        if run.task_analysis is not None:
            state.metadata["task_intent"] = (
                run.task_analysis.intent.value
            )

            state.metadata["task_analysis"] = (
                run.task_analysis.analysis_text
            )

        state.metadata["plan_version"] = (
            run.plan_runtime.plan_version
        )

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
        step_id = step.step_id

        step_observability = (
            run.observability.child(
                step_id=step_id,
                attempt=run.current_attempt + 1,
            )
        )

        strategy_family = (
            self.planner.strategy_family(step)
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
                    "strategy_family": strategy_family,
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
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        run.current_attempt += 1

        # Per-attempt state is a projection, not historical truth.
        # Clear stale success/error flags before every new attempt so a
        # retry cannot inherit the previous attempt's outcome.
        state.execution_success = False
        state.step_succeeded = False
        state.tool_error = None
        state.blocked = False
        state.waiting_for_approval = False

        request = ExecutionRequest(
            step_id=step_id,
            step_type=step.step_type,
            action=step.action,
            tool_name=step.tool_name,
            arguments=step.arguments,
            user_request=run.user_request,
            context=context,
            iteration=run.iteration,
            correlation_id=run.correlation_id,
            metadata={
                "run_id": str(run.run_id),
                "attempt": run.current_attempt,
                "plan_version": (
                    run.plan_runtime.plan_version
                ),
                "replan_count": (
                    run.plan_runtime.replan_count
                ),
                "strategy_family": strategy_family,
            },
        )

        self._emit(
            EventType.STEP_STARTED,
            run,
            metadata={
                "step_id": step_id,
                "step_type": step.step_type.value,
                "tool_name": step.tool_name,
                "plan_version": (
                    run.plan_runtime.plan_version
                ),
                "attempt": run.current_attempt,
            },
        )

        result = await self.agent_execution.execute(
            request
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
                metadata={
                    "step_id": step_id,
                },
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.WAIT_FOR_APPROVAL,
                result=result,
                reason="execution_blocked",
            )

        if not result.success:
            self._emit(
                EventType.STEP_FAILED,
                run,
                metadata={
                    "step_id": step_id,
                    "attempt": run.current_attempt,
                },
                error=result.error,
            )

            return await self._handle_failure(
                state,
                run,
                step,
                result,
                context,
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
            metadata={
                "step_id": step_id,
                "attempt": run.current_attempt,
                "plan_version": (
                    run.plan_runtime.plan_version
                ),
            },
        )

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
            action=OrchestrationAction.EXECUTE,
            result=result,
            reason="step_completed",
        )

    async def _handle_failure(
        self,
        state: AgentState,
        run: OrchestrationContext,
        failed_step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> OrchestrationOutcome:
        if result.error is None:
            error = AgentError(
                error_type="ExecutionFailed",
                message="Agent execution failed without an error.",
                category=ErrorCategory.EXECUTION,
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
                action=OrchestrationAction.FAIL,
                result=result,
                error=error,
                reason=error.message,
            )

        decision = await self.reliability.handle_failure(
            error=result.error,
            attempt=max(
                1,
                run.current_attempt,
            ),
            max_attempts=self.config.max_step_attempts,
        )

        if decision.action is ReliabilityAction.RETRY:
            self._emit(
                EventType.RETRY_STARTED,
                run,
                metadata={
                    "step_id": failed_step.step_id,
                    "attempt": run.current_attempt,
                },
                error=result.error,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.RETRY,
                result=result,
                error=result.error,
                reason="reliability_requested_retry",
            )

        if decision.action is ReliabilityAction.REPLAN:
            return await self._replan(
                state,
                run,
                failed_step=failed_step,
                failure=(
                    decision.error
                    or result.error
                ),
                context=context,
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
            action=OrchestrationAction.FAIL,
            result=result,
            error=failure,
            reason=failure.message,
        )

    async def _replan(
        self,
        state: AgentState,
        run: OrchestrationContext,
        *,
        failed_step: PlanStep,
        failure: AgentError,
        context: FinalContext,
    ) -> OrchestrationOutcome:
        if (
            run.plan_runtime.replan_count
            >= self.config.max_replans
        ):
            self._fail(
                state,
                failure,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.FAIL,
                error=failure,
                reason="replan_budget_exhausted",
            )

        self._emit(
            EventType.RECOVERY_STARTED,
            run,
            metadata={
                "failed_step": failed_step.step_id,
                "replan_count": (
                    run.plan_runtime.replan_count
                ),
            },
            error=failure,
        )

        if state.phase is AgentPhase.EXECUTING:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.RECOVERY,
            )

        elif state.phase is AgentPhase.VERIFYING:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.VERIFICATION_FAILED,
            )

        elif state.phase is AgentPhase.FAILED:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.RECOVERY,
            )

        else:
            error = AgentError(
                error_type="InvalidReplanLifecycle",
                message=(
                    "Replanning is only valid from EXECUTING, "
                    "VERIFYING, or FAILED; current phase is "
                    f"{state.phase.value}."
                ),
                category=ErrorCategory.STATE_TRANSITION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                retryable=False,
                recoverable=False,
                source="Orchestrator",
                operation="replan",
            )
            self._fail(state, error)
            return OrchestrationOutcome(
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        try:
            planning_result = await self.planner.replan(
                user_question=run.user_request,
                context=context,
                failed_step=failed_step,
                failure=failure,
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
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        self._emit(
            EventType.PLAN_REPLANNED,
            run,
            metadata={
                "failed_step": failed_step.step_id,
                "plan_version": (
                    run.plan_runtime.plan_version
                ),
                "replan_count": (
                    run.plan_runtime.replan_count
                ),
            },
        )

        self._emit(
            EventType.RECOVERY_COMPLETED,
            run,
            metadata={
                "plan_version": (
                    run.plan_runtime.plan_version
                ),
                "replan_count": (
                    run.plan_runtime.replan_count
                ),
            },
        )

        return OrchestrationOutcome(
            action=OrchestrationAction.REPLAN,
            reason="replacement_plan_installed",
        )

    async def _verify(
        self,
        state: AgentState,
        run: OrchestrationContext,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> OrchestrationOutcome:
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
                    reason=TransitionReason.EXECUTION_COMPLETED,
                )

            self._emit(
                EventType.VERIFICATION_STARTED,
                run,
                metadata={
                    "step_id": step.step_id,
                    "attempt": (
                        run.verification_attempts + 1
                    ),
                },
            )

            task_analysis = run.task_analysis

            if task_analysis is None:
                error = AgentError(
                    error_type="MissingTaskAnalysis",
                    message=(
                        "Cannot verify final answer "
                        "without TaskAnalysis."
                    ),
                    category=ErrorCategory.INTERNAL,
                    severity=ErrorSeverity.HIGH,
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
                    action=OrchestrationAction.FAIL,
                    error=error,
                    reason=error.message,
                )

            verification = await self.verifier.verify(
                VerificationInput(
                    question=run.user_request,
                    candidate_answer=answer,
                    raw_data=run.verification_evidence(),
                    task_type=task_analysis.intent.value,
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
                "status": verification.status.value,
                "attempt": run.verification_attempts,
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
                    reason=TransitionReason.VERIFICATION_PASSED,
                )

            self._emit(
                EventType.AGENT_COMPLETED,
                run,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.COMPLETE,
                result=result,
                verification=verification,
                reason="verification_passed",
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
                action=OrchestrationAction.FAIL,
                result=result,
                error=error,
                verification=verification,
                reason=error.message,
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
            state,
            run,
            failed_step=step,
            failure=failure,
            context=context,
        )

    @staticmethod
    def _validate_plan(
        plan: PlanSchema,
    ) -> None:
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
                "Plan must contain exactly one final-answer step."
            )

        final_step = final_steps[0]

        if final_step.step_id != len(plan.steps) - 1:
            raise ValueError(
                "Final-answer step must be the last plan step."
            )

        if final_step.step_type is not StepType.LLM:
            raise ValueError(
                "Final-answer step must use StepType.LLM."
            )

    @staticmethod
    def _extract_answer(
        output: Any,
    ) -> str:
        if output is None:
            return ""

        if isinstance(output, str):
            return output.strip()

        return str(output).strip()

    @staticmethod
    def _planner_error(
        exc: PlannerRecoveryRequired,
        *,
        operation: str,
    ) -> AgentError:
        return AgentError(
            error_type="PlannerRecoveryRequired",
            message=str(exc) or "Planner recovery required.",
            category=ErrorCategory.PLAN_RECOVERY_ERROR,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            recoverable=True,
            source="Planner",
            operation=operation,
            original_exception=exc,
        )

    @staticmethod
    def _fail(
        state: AgentState,
        error: AgentError | None,
    ) -> None:
        if error is None:
            state.fatal_error = True
            state.tool_error = "Execution failed."

        else:
            state.fatal_error = (
                error.severity is ErrorSeverity.CRITICAL
                or (
                    not error.retryable
                    and not error.recoverable
                )
            )

            state.tool_error = error.message

        if state.phase not in {
            AgentPhase.COMPLETED,
            AgentPhase.TERMINATED,
            AgentPhase.FAILED,
        }:
            # AgentState is the sole owner of lifecycle transitions.
            # Never mutate state.phase directly from orchestration.
            state.fail(
                reason=TransitionReason.EXECUTION_FAILED,
                fatal=state.fatal_error,
            )

    def _emit(
        self,
        event: EventType,
        run: OrchestrationContext,
        *,
        metadata: dict[str, Any] | None = None,
        error: AgentError | None = None,
    ) -> None:
        if self.observability is None:
            return

        context = run.observability

        self.observability.emit(
            event,
            correlation_id=context.correlation_id,
            metadata=metadata,
            agent_id=context.agent_id,
            run_id=context.run_id,
            iteration=context.iteration,
            step_id=context.step_id,
            attempt=context.attempt,
            phase=(
                context.phase
                if hasattr(context, "phase")
                else None
            ),
            error=(
                error.message
                if error is not None
                else None
            ),
        )

    def _span(
        self,
        operation: str,
        context: Any,
    ):
        if self.observability is None:
            return nullcontext()

        return self.observability.span(
            operation,
            correlation_id=context.correlation_id,
        )