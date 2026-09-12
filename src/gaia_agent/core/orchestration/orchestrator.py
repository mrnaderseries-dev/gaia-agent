from __future__ import annotations

from time import perf_counter
from typing import Any

from gaia_agent.agents.verifier import (
    VerificationInput,
    VerifierAgent,
    VerificationStatus,
)
from gaia_agent.context.ContextBuilder import (
    ContextBuilder,
)
from gaia_agent.context.models import FinalContext
from gaia_agent.context.request_builder import (
    ContextRequestBuilder,
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
from gaia_agent.core.orchestration.models import (
    OrchestrationContext,
)
from gaia_agent.core.policies.termination import (
    TerminationReason,
)
from gaia_agent.observability.events import (
    EventType,
    create_event,
)
from gaia_agent.observability.logger import (
    EventLogger,
)
from gaia_agent.observability.metrics import (
    Metrics,
)
from gaia_agent.observability.tracer import (
    Tracer,
)
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.planner import (
    Planner,
    PlannerRecoveryRequired,
)
from gaia_agent.reliability.engine import (
    ReliabilityAction,
    ReliabilityEngine,
)
from gaia_agent.reliability.loop_detector import (
    LoopDetector,
)


class Orchestrator:

    def init(
        self,
        *,
        context_builder: ContextBuilder,
        planner: Planner,
        agent_execution: AgentExecution,
        reliability_engine: ReliabilityEngine,
        loop_detector: LoopDetector,
        verifier: VerifierAgent,
        event_logger: EventLogger | None = None,
        metrics: Metrics | None = None,
        tracer: Tracer | None = None,
        max_execution_attempts: int = 3,
        max_verification_attempts: int = 2,
    ) -> None:

        self._context_builder = context_builder
        self._planner = planner
        self._agent_execution = agent_execution
        self._reliability = reliability_engine
        self._loop_detector = loop_detector
        self._verifier = verifier

        self._event_logger = event_logger
        self._metrics = metrics
        self._tracer = tracer

        self._max_execution_attempts = (
            max_execution_attempts
        )

        self._max_verification_attempts = (
            max_verification_attempts
        )

        self._state: AgentState | None = None
        self._context: OrchestrationContext | None = None

    def bind_state(
        self,
        state: AgentState,
    ) -> None:

        if self._state is not None:
            raise RuntimeError(
                "Orchestrator is already bound."
            )

        if not state.user_request.strip():
            raise ValueError(
                "AgentState.user_request cannot be empty."
            )

        self._state = state

        run_id = (
            state.metadata.get("run_id")
            if state.metadata
            else None
        )

        if not run_id:
            import uuid

            run_id = str(uuid.uuid4())

            state.metadata["run_id"] = run_id

        self._context = (
            OrchestrationContext(
                user_request=(
                    state.user_request
                ),
                run_id=run_id,
            )
        )

    def unbind(self) -> None:
        self._state = None
        self._context = None

    async def generate_initial_plan(
        self,
    ) -> PlanSchema:

        state = self._require_state()
        runtime = self._require_context()

        if state.phase == AgentPhase.IDLE:
            state.transition(
                AgentPhase.PLANNING,
                reason=TransitionReason.START,
            )

        context = await self._build_context()

        span = self._start_span(
            "orchestration.planning"
        )

        try:

            plan = await self._planner.generate_plan(
                runtime.user_request,
                context,
            )

            self._validate_plan(plan)

            runtime.plan_runtime.install_plan(
                plan
            )

            state.plan = list(
                plan.steps
            )

            state.current_step = 0

            state.transition(
                AgentPhase.EXECUTING,
                reason=(
                    TransitionReason.PLAN_READY
                ),
            )

            self._increment(
                "plans_created"
            )

            return plan

        except PlannerRecoveryRequired:
            state.transition(
                AgentPhase.FAILED,
                reason=(
                    TransitionReason.EXECUTION_FAILED
                ),
            )
            raise

        finally:
            self._end_span(span)

    async def run_iteration(
        self,
    ) -> ExecutionResult | None:

        state = self._require_state()
        runtime = self._require_context()

        if state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.FAILED,
            AgentPhase.TERMINATED,
        }:
            return None

        runtime.iteration += 1
        state.iteration = runtime.iteration

        if runtime.plan_runtime.plan is None:
            await self.generate_initial_plan()
            return None

        step = runtime.plan_runtime.current()

        if step is None:
            return self._finish_if_possible()

        state.current_step = (
            step.step_id
        )

        state.current_action = (
            step.action
        )

        state.step_type = (
            step.step_type
        )

        state.tool_name = (
            step.tool_name
        )

        state.tool_arguments = dict(
            step.arguments
        )

        context = await self._build_context()

        return await self._execute_step(
            step=step,
            context=context,
        )

    async def _execute_step(
        self,
        *,
        step: PlanStep,
        context: FinalContext,
    ) -> ExecutionResult:

        runtime = self._require_context()

        strategy = self._strategy_key(
            step
        )

        if self._loop_detector.check(
            step,
            strategy,
        ):
            return await self._handle_loop(
                step
            )

        attempt = self._step_attempt_count(
            step.step_id
        ) + 1

        request = ExecutionRequest(
            step_id=step.step_id,
            step_type=step.step_type,
            action=step.action,
            tool_name=step.tool_name,
            arguments=dict(
                step.arguments
            ),
            user_request=(
                runtime.user_request
            ),
            context=context,
            iteration=runtime.iteration,
            correlation_id=(
                self._agent_execution
                .correlation_id
            ),
            metadata={
                "run_id": runtime.run_id,
                "plan_version": (
                    runtime.plan_runtime
                    .plan_version
                ),
                "attempt": attempt,
            },
        )

        result = await self._agent_execution.execute(
            request
        )

        runtime.record_execution(
            step_id=step.step_id,
            result=result,
            attempt=attempt,
        )

        self._sync_state_after_execution(
            step,
            result,
        )

        if result.success:

            runtime.plan_runtime.mark_completed(
                step.step_id
            )

            if step.is_final_answer:
                return await self._verify_final_answer(
                    step=step,
                    result=result,
                    context=context,
                )

            runtime.plan_runtime.advance()

            return result

        return await self._handle_execution_failure(
            step=step,
            result=result,
            context=context,
        )

    async def _handle_execution_failure(
        self,
        *,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> ExecutionResult:

        if result.error is None:
            return result

        runtime = self._require_context()

        attempt = self._step_attempt_count(
            step.step_id
        )

        reliability = (
            await self._reliability.handle_failure(
                error=result.error,
                attempt=attempt,
                max_attempts=(
                    self._max_execution_attempts
                ),
            )
        )

        self._increment(
            f"reliability.{reliability.action.value}"
        )

        if (
            reliability.action
            == ReliabilityAction.RETRY
        ):

            return await self._execute_step(
                step=step,
                context=context,
            )

        if (
            reliability.action
            == ReliabilityAction.REPLAN
        ):

            return await self._replan_after_failure(
                step=step,
                failure=result.error,
            )

        self._mark_failed(
            result.error.message
        )

        return result

    async def _replan_after_failure(
        self,
        *,
        step: PlanStep,
        failure: Any,
    ) -> ExecutionResult:

        runtime = self._require_context()
        state = self._require_state()

        state.transition(
            AgentPhase.PLANNING,
            reason=TransitionReason.RECOVERY,
        )

        context = await self._build_context()

        try:

            plan = await self._planner.replan(
                user_question=(
                    runtime.user_request
                ),
                context=context,
                failed_step=step,
                failure=str(failure),
            )

            self._validate_plan(plan)

            runtime.plan_runtime.install_plan(
                plan
            )

            state.plan = list(
                plan.steps
            )

            state.replan_count += 1

            state.current_step = 0

            state.transition(
                AgentPhase.EXECUTING,
                reason=TransitionReason.RECOVERY,
            )

            return ExecutionResult(
                success=False,
                step_id=step.step_id,
                metadata={
                    "replanned": True,
                    "plan_version": (
                        runtime.plan_runtime
                        .plan_version
                    ),
                },
            )

        except Exception as exc:

            self._mark_failed(
                str(exc)
            )

            return ExecutionResult(
                success=False,
                step_id=step.step_id,
                error=exc,
            )

    async def _verify_final_answer(
        self,
        *,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> ExecutionResult:

        state = self._require_state()
        runtime = self._require_context()

        answer = self._extract_answer(
            result
        )

        if not answer:

            return await self._handle_verification_failure(
                step=step,
                answer="",
                reason="final_answer_missing",
            )

        state.transition(
            AgentPhase.VERIFYING,
            reason=(
                TransitionReason
                .EXECUTION_COMPLETED
            ),
        )

        evidence = (
            runtime.verification_evidence()
        )

        verification_input = VerificationInput(
            question=runtime.user_request,
            candidate_answer=answer,
            raw_data=evidence,
            task_type=self._infer_task_type(
                state
            ),
        )

        verification = (
            await self._verifier.verify(
                verification_input
            )
        )

        runtime.record_verification(
            verified=(
                verification.status
                == VerificationStatus.VERIFIED
            ),
            result=verification,
            answer=answer,
        )

        state.verification_attempts = (
            runtime.verification_attempts
        )

        if (
            verification.status
            == VerificationStatus.VERIFIED
        ):

            runtime.final_answer = answer

            state.final_answer = answer
            state.final_answer_ready = True
            state.final_answer_verified = True

            state.transition(
                AgentPhase.COMPLETED,
                reason=(
                    TransitionReason
                    .VERIFICATION_PASSED
                ),
            )

            self._increment(
                "verification_passes"
            )

            return result

        self._increment(
            "verification_failures"
        )

        return await self._handle_verification_failure(
            step=step,
            answer=answer,
            reason=(
                verification.reason
                or verification.status.value
            ),
        )

    async def _handle_verification_failure(
        self,
        *,
        step: PlanStep,
        answer: str,
        reason: str,
    ) -> ExecutionResult:

        state = self._require_state()
        runtime = self._require_context()

        if (
            runtime.verification_attempts
            >= self._max_verification_attempts
        ):

            state.final_answer = (
                answer or None
            )

            state.final_answer_ready = bool(
                answer
            )

            state.final_answer_verified = False

            self._mark_failed(
                "Answer verification budget exhausted."
            )

            return ExecutionResult(
                success=False,
                output=answer,
                step_id=step.step_id,
                metadata={
                    "verification_failed": True,
                    "verification_budget_exhausted": True,
                },
            )

        state.transition(
            AgentPhase.PLANNING,
            reason=(
                TransitionReason
                .VERIFICATION_FAILED
            ),
        )

        context = await self._build_context()

        try:

            plan = await self._planner.replan(
                user_question=(
                    runtime.user_request
                ),
                context=context,
                failed_step=step,
                failure=reason,
            )

            self._validate_plan(plan)

            runtime.plan_runtime.install_plan(
                plan
            )

            state.replan_count += 1
            state.plan = list(
                plan.steps
            )
            state.current_step = 0

            state.transition(
                AgentPhase.EXECUTING,
                reason=(
                    TransitionReason
                    .RECOVERY
                ),
            )

            return ExecutionResult(
                success=False,
                output=answer,
                step_id=step.step_id,
                metadata={
                    "replanned": True,
                    "verification_replan": True,
                },
            )

        except Exception as exc:

            self._mark_failed(
                str(exc)
            )

            return ExecutionResult(
                success=False,
                output=answer,
                step_id=step.step_id,
                error=exc,
            )

    async def _build_context(
        self,
    ) -> FinalContext:

        state = self._require_state()

        request = (
            ContextRequestBuilder.from_state(
                state
            )
        )

        return await self._context_builder.build(
            request
        )

    @staticmethod
    def _validate_plan(
        plan: PlanSchema,
    ) -> None:

        if not isinstance(
            plan,
            PlanSchema,
        ):
            raise TypeError(
                "Planner must return PlanSchema."
            )

        if not plan.steps:
            raise ValueError(
                "Planner returned an empty plan."
            )

        final_step = plan.steps[-1]

        if (
            not final_step.is_final_answer
            or final_step.step_type
            != StepType.LLM
        ):
            raise ValueError(
                "Plan must end with a final LLM answer step."
            )

    def _finish_if_possible(
        self,
    ) -> ExecutionResult | None:

        runtime = self._require_context()

        if runtime.final_answer is not None:
            return ExecutionResult(
                success=True,
                output=runtime.final_answer,
            )

        return None

    def _mark_failed(
        self,
        reason: str,
    ) -> None:

        state = self._require_state()

        state.fatal_error = True

        if state.phase not in {
            AgentPhase.FAILED,
            AgentPhase.COMPLETED,
            AgentPhase.TERMINATED,
        }:

            state.transition(
                AgentPhase.FAILED,
                reason=(
                    TransitionReason
                    .EXECUTION_FAILED
                ),
            )

        runtime = self._require_context()

        runtime.terminal_reason = reason

    def _step_attempt_count(
        self,
        step_id: int,
    ) -> int:

        runtime = self._require_context()

        return sum(
            1
            for record
            in runtime.execution_history
            if record.step_id == step_id
        )

    @staticmethod
    def _strategy_key(
        step: PlanStep,
    ) -> str:

        if step.tool_name:
            return (
                f"tool:{step.tool_name}"
            )

        return "strategy:llm"

    @staticmethod
    def _extract_answer(
        result: ExecutionResult,
    ) -> str | None:

        if result.output is None:
            return None

        if isinstance(
            result.output,
            str,
        ):

            value = (
                result.output.strip()
            )

            return value or None

        return str(
            result.output
        )

    @staticmethod
    def _infer_task_type(
        state: AgentState,
    ) -> str | None:

        value = state.metadata.get(
            "task_type"
        )

        if value is None:
            return None

        return str(value)

    def _sync_state_after_execution(
        self,
        step: PlanStep,
        result: ExecutionResult,
    ) -> None:

        state = self._require_state()

        state.execution_success = (
            result.success
        )

        state.step_succeeded = (
            result.success
        )

        state.blocked = (
            result.blocked
        )

        state.tool_result = (
            result.output
        )

        state.tool_error = (
            result.error.message
            if result.error is not None
            else None
        )

        if result.evidence:
            state.evidence.extend(
                result.evidence
            )

        if result.artifacts:
            state.artifacts.extend(
                result.artifacts
            )

    def _start_span(
        self,
        operation: str,
    ):

        if self._tracer is None:
            return None

        return self._tracer.start_span(
            operation=operation,
            correlation_id=(
                self._agent_execution
                .correlation_id
            ),
        )

    def _end_span(
        self,
        span,
    ) -> None:

        if (
            span is not None
            and self._tracer is not None
        ):
            self._tracer.end_span(
                span
            )

    def _increment(
        self,
        name: str,
    ) -> None:

        if self._metrics is not None:
            self._metrics.increment(
                name
            )

    def _require_state(
        self,
    ) -> AgentState:

        if self._state is None:
            raise RuntimeError(
                "Orchestrator is not bound."
            )

        return self._state

    def _require_context(
        self,
    ) -> OrchestrationContext:

        if self._context is None:
            raise RuntimeError(
                "Orchestrator is not bound."
            )

        return self._context

    def emit_agent_started(self) -> None:

        if self._event_logger is None:
            return

        runtime = self._require_context()

        self._event_logger.log(
            create_event(
                event_type=(
                    EventType.AGENT_STARTED
                ),
                correlation_id=(
                    self._agent_execution
                    .correlation_id
                ),
                metadata={
                    "run_id": runtime.run_id
                },
            )
        )

    def emit_agent_completed(self) -> None:

        if self._event_logger is None:
            return

        runtime = self._require_context()

        self._event_logger.log(
            create_event(
                event_type=(
                    EventType.AGENT_COMPLETED
                ),
                correlation_id=(
                    self._agent_execution
                    .correlation_id
                ),
                metadata={
                    "run_id": runtime.run_id,
                    "final_answer": (
                        runtime.final_answer
                    ),
                },
            )
        )

    def _handle_loop(
        self,
        step: PlanStep,
    ) -> ExecutionResult:

        self._mark_failed(
            f"Execution loop detected at step "
            f"{step.step_id}."
        )

        return ExecutionResult(
            success=False,
            step_id=step.step_id,
            error=RuntimeError(
                f"Execution loop detected at step "
                f"{step.step_id}."
            ),
        )