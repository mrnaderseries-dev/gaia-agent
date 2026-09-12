from __future__ import annotations

from typing import Any

from gaia_agent.agents.verifier import VerifierAgent
from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.models import FinalContext
from gaia_agent.context.request_builder import ContextRequestBuilder
from gaia_agent.core.agent_execution import (
    AgentExecution,
    ExecutionRequest,
    ExecutionResult,
)
from gaia_agent.core.agent_state import AgentPhase, AgentState
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.planner import Planner, PlannerRecoveryRequired
from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.loop_detector import LoopDetector

from .models import (
    ExecutionRecord,
    OrchestrationContext,
    VerificationRecord,
)


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
    ) -> None:
        self._context_builder = context_builder
        self._planner = planner
        self._agent_execution = agent_execution
        self._reliability = reliability_engine
        self._loop_detector = loop_detector
        self._verifier = verifier

        self._state: AgentState | None = None
        self._context: OrchestrationContext | None = None

    def bind(
        self,
        *,
        state: AgentState,
        context: OrchestrationContext,
    ) -> None:
        if self._state is not None:
            raise RuntimeError("Orchestrator is already bound.")

        if not context.user_request.strip():
            raise ValueError("user_request cannot be empty.")

        self._state = state
        self._context = context

    def run_iteration(self) -> ExecutionResult | None:
        state = self._require_state()
        runtime = self._require_context()

        runtime.iteration += 1

        if state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.FAILED,
            AgentPhase.TERMINATED,
        }:
            return None

        if state.phase == AgentPhase.IDLE:
            state.transition(
                AgentPhase.PLANNING,
                reason="orchestration_started",
            )

        final_context = self._build_context()

        if runtime.plan_runtime.plan is None:
            return self._create_plan(final_context)

        step = runtime.plan_runtime.current()

        if step is None:
            return self._complete_plan()

        return self._execute_step(step, final_context)

    def _build_context(self) -> FinalContext:
        state = self._require_state()

        request = ContextRequestBuilder.from_state(state)

        return self._context_builder.build(request)

    def _create_plan(
        self,
        context: FinalContext,
    ) -> ExecutionResult | None:
        runtime = self._require_context()

        try:
            plan = self._planner.generate_plan(
                runtime.user_request,
                context,
            )
        except PlannerRecoveryRequired as exc:
            return self._handle_planner_failure(exc)

        self._validate_plan(plan)

        runtime.plan_runtime.plan = plan
        runtime.plan_runtime.current_step = 0

        if self._state_is_planning():
            self._require_state().transition(
                AgentPhase.EXECUTING,
                reason="plan_created",
            )

        return None

    def _execute_step(
        self,
        step: PlanStep,
        context: FinalContext,
    ) -> ExecutionResult:
        runtime = self._require_context()

        strategy_family = self._strategy_family(step)

        if self._loop_detector.check(
            step,
            strategy_family,
        ):
            return self._handle_loop(step)

        request = ExecutionRequest(
            step_id=step.id,
            step_type=step.step_type,
            action=step.action,
            tool_name=step.tool_name,
            arguments=step.arguments,
            user_request=runtime.user_request,
            context=context,
            iteration=runtime.iteration,
        )

        result = self._agent_execution.execute(request)

        runtime.record_execution(
            ExecutionRecord(
                step_id=step.id,
                result=result,
                iteration=runtime.iteration,
            )
        )

        if not result.success:
            return self._handle_execution_failure(
                step=step,
                result=result,
                context=context,
            )

        runtime.plan_runtime.mark_completed(step.id)

        if self._is_final_step(step):
            return self._verify_final_answer(
                step=step,
                result=result,
                context=context,
            )

        runtime.plan_runtime.advance()

        return result

    def _handle_execution_failure(
        self,
        *,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> ExecutionResult:
        if result.error is None:
            return result

        runtime = self._require_context()

        reliability_result = self._reliability.handle_failure(
            error=result.error,
            attempt=runtime.iteration,
        )

        if reliability_result.recovered:
            return self._execute_recovered_step(
                step=step,
                context=context,
            )

        if reliability_result.result is not None:
            return reliability_result.result

        return result

    def _execute_recovered_step(
        self,
        *,
        step: PlanStep,
        context: FinalContext,
    ) -> ExecutionResult:
        runtime = self._require_context()

        request = ExecutionRequest(
            step_id=step.id,
            step_type=step.step_type,
            action=step.action,
            tool_name=step.tool_name,
            arguments=step.arguments,
            user_request=runtime.user_request,
            context=context,
            iteration=runtime.iteration,
            metadata={"recovered": True},
        )

        result = self._agent_execution.execute(request)

        runtime.record_execution(
            ExecutionRecord(
                step_id=step.id,
                result=result,
                iteration=runtime.iteration,
            )
        )

        if result.success:
            runtime.plan_runtime.mark_completed(step.id)

            if not self._is_final_step(step):
                runtime.plan_runtime.advance()

        return result

    def _verify_final_answer(
        self,
        *,
        step: PlanStep,
        result: ExecutionResult,
        context: FinalContext,
    ) -> ExecutionResult:
        runtime = self._require_context()

        answer = self._extract_answer(result)

        if answer is None:
            return self._verification_failure(
                step=step,
                answer="",
                reason="final_answer_missing",
            )

        if self._state_is_executing():
            self._require_state().transition(
                AgentPhase.VERIFYING,
                reason="final_answer_generated",
            )

        verification = self._verifier.verify(
            user_question=runtime.user_request,
            candidate_answer=answer,
            context=context,
            evidence=result.evidence,
        )

        record = VerificationRecord(
            attempt=runtime.verification_attempts + 1,
            verified=verification.verified,
            result=verification,
            answer=answer,
        )

        runtime.record_verification(record)

        if verification.verified:
            runtime.final_answer = answer

            self._require_state().transition(
                AgentPhase.COMPLETED,
                reason="answer_verified",
            )

            return result

        return self._verification_failure(
            step=step,
            answer=answer,
            reason="answer_verification_failed",
        )

    def _verification_failure(
        self,
        *,
        step: PlanStep,
        answer: str,
        reason: str,
    ) -> ExecutionResult:
        runtime = self._require_context()

        if runtime.verification_attempts >= 2:
            self._require_state().transition(
                AgentPhase.FAILED,
                reason="verification_budget_exhausted",
            )

            return ExecutionResult(
                success=False,
                output=answer,
                step_id=step.id,
                error=RuntimeError(
                    "Final answer verification failed."
                ),
            )

        self._require_state().transition(
            AgentPhase.PLANNING,
            reason=reason,
        )

        try:
            context = self._build_context()

            plan = self._planner.replan(
                user_question=runtime.user_request,
                context=context,
                failed_step=step,
                failure=reason,
            )

            self._validate_plan(plan)

            runtime.plan_runtime.plan = plan
            runtime.plan_runtime.current_step = 0

            self._require_state().transition(
                AgentPhase.EXECUTING,
                reason="verification_replan_created",
            )

        except PlannerRecoveryRequired as exc:
            self._require_state().transition(
                AgentPhase.FAILED,
                reason="verification_replan_failed",
            )

            return ExecutionResult(
                success=False,
                output=answer,
                step_id=step.id,
                error=exc,
            )

        return ExecutionResult(
            success=False,
            output=answer,
            step_id=step.id,
            metadata={"replanned": True},
        )

    def _complete_plan(self) -> ExecutionResult | None:
        runtime = self._require_context()

        if runtime.final_answer is not None:
            return ExecutionResult(
                success=True,
                output=runtime.final_answer,
            )

        return None

    def _handle_planner_failure(
        self,
        error: Exception,
    ) -> ExecutionResult:
        self._require_state().transition(
            AgentPhase.FAILED,
            reason="planner_failure",
        )

        return ExecutionResult(
            success=False,
            error=error,
        )

    def _handle_loop(
        self,
        step: PlanStep,
    ) -> ExecutionResult:
        self._require_state().transition(
            AgentPhase.FAILED,
            reason="execution_loop_detected",
        )

        return ExecutionResult(
            success=False,
            step_id=step.id,
            error=RuntimeError(
                f"Execution loop detected at step {step.id}."
            ),
        )

    @staticmethod
    def _validate_plan(plan: PlanSchema) -> None:
        if not plan.steps:
            raise ValueError("Planner returned an empty plan.")

        final_steps = [
            step
            for step in plan.steps
            if step.step_type == StepType.LLM
            and step.id == len(plan.steps) - 1
        ]

        if len(final_steps) != 1:
            raise ValueError(
                "Plan must contain exactly one final LLM step."
            )

    @staticmethod
    def _is_final_step(step: PlanStep) -> bool:
        return step.step_type == StepType.LLM

    @staticmethod
    def _strategy_family(step: PlanStep) -> str:
        if step.tool_name:
            return step.tool_name

        return "llm"

    @staticmethod
    def _extract_answer(
        result: ExecutionResult,
    ) -> str | None:
        if result.output is None:
            return None

        if isinstance(result.output, str):
            answer = result.output.strip()
            return answer or None

        return str(result.output)

    def _state_is_planning(self) -> bool:
        return self._require_state().phase == AgentPhase.PLANNING

    def _state_is_executing(self) -> bool:
        return self._require_state().phase == AgentPhase.EXECUTING

    def _require_state(self) -> AgentState:
        if self._state is None:
            raise RuntimeError("Orchestrator is not bound.")

        return self._state

    def _require_context(self) -> OrchestrationContext:
        if self._context is None:
            raise RuntimeError("Orchestrator is not bound.")

        return self._context