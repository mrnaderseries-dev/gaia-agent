from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID, uuid4

from gaia_agent.agents.answer_sanitizer import AnswerSanitizer
from gaia_agent.agents.verifier import (
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
    deterministic_verification,
    evidence_supports_candidate,
)
from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_state import AgentState
from gaia_agent.planner.planner import Planner
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.loop_detector import LoopDetector
from gaia_agent.observability.events import (
    EventType,
    create_event,
)
from gaia_agent.observability.logger import EventLogger
from gaia_agent.observability.metrics import Metrics
from gaia_agent.observability.tracer import Tracer


# Bounded recovery limits (Phase 6).
#
# A previous attempt that produced no new information must NOT be
# repeated. These limits make the replan budget explicit.
MAX_REPLANS = 2
MAX_SAME_FAILURE = 1
MAX_SAME_PLAN = 1

# Bounded semantic verification (Phase 7): after this many failed
# verification attempts the answer is delivered honestly as
# UNVERIFIED instead of looping regenerate -> verify -> replan.
MAX_VERIFICATION_ATTEMPTS = 2

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        planner: Planner,
        agent_execution: AgentExecution,
        error_handler: ErrorHandler,
        reliability_engine: ReliabilityEngine,
        loop_detector: LoopDetector,
        verifier: VerifierAgent,
        answer_sanitizer: AnswerSanitizer,
        event_logger: EventLogger,
        metrics: Metrics,
        tracer: Tracer,
    ) -> None:
        self.context_builder = context_builder
        self.planner = planner
        self.agent_execution = agent_execution
        self.error_handler = error_handler
        self.reliability_engine = reliability_engine
        self.loop_detector = loop_detector
        self.verifier = verifier
        self.answer_sanitizer = answer_sanitizer
        self.event_logger = event_logger
        self.metrics = metrics
        self.tracer = tracer
        self.state: AgentState | None = None
        self.correlation_id: UUID = uuid4()

    def bind_state(
        self,
        state: AgentState,
    ) -> None:
        self.state = state
        self.agent_execution.bind_state(state)

    def _require_state(
        self,
    ) -> AgentState:
        if self.state is None:
            raise RuntimeError(
                "AgentState is not bound. "
                "Call bind_state() before execution."
            )
        return self.state

    # ==========================================================
    # Phase 6: bounded recovery / replanning
    # ==========================================================

    def _check_recovery_budget(
        self,
        error: AgentError,
    ) -> bool:
        """
        Enforce MAX_REPLANS / MAX_SAME_FAILURE / MAX_SAME_PLAN.

        Returns True when replanning is still allowed. When the
        budget is exhausted the run is marked FATAL, because
        repeating the same strategy without new information is
        precisely the infinite-loop failure mode we are removing.
        """
        state = self._require_state()

        state.replan_count += 1

        if state.replan_count > MAX_REPLANS:
            state.fatal_error = True
            state.tool_error = (
                "Recovery budget exceeded "
                f"(max replans: {MAX_REPLANS}). "
                "Execution stopped to prevent an infinite loop."
            )
            state.execution_success = False
            return False

        failure_key = (
            f"{error.category.value}:"
            f"{error.error_code or error.error_type}"
        )

        if (
            state.last_failure_key == failure_key
            and state.same_failure_count > 0
        ):
            state.same_failure_count += 1
        else:
            state.same_failure_count = 1
            state.last_failure_key = failure_key

        if state.same_failure_count > MAX_SAME_FAILURE:
            state.fatal_error = True
            state.tool_error = (
                "The same failure recurred "
                f"(max: {MAX_SAME_FAILURE}). Executing the same "
                "strategy again would loop without new information."
            )
            state.execution_success = False
            return False

        return True

    async def run_iteration(
        self,
    ) -> None:
        state = self._require_state()

        span = self.tracer.start_span(
            operation="orchestrator.iteration",
            correlation_id=self.correlation_id,
        )

        try:
            context = await self.context_builder.build(state)

            self.metrics.increment("context_builds")

            if not state.plan:
                await self._create_plan(context.items)

                if state.tool_error is not None:
                    return

            if state.current_step >= len(state.plan):
                await self._handle_plan_completion()
                return

            step = state.plan[state.current_step]

            loop_result = self.loop_detector.check(
                action=step.action,
                tool_name=step.tool_name,
                arguments=step.arguments,
            )

            if loop_result.detected:
                await self._handle_loop(loop_result.message)
                return

            self._prepare_step(step)

            result = await self.reliability_engine.execute(
                operation=self.agent_execution.execute,
                operation_name="agent_execution",
                source="orchestrator",
                validator=self._validate_execution_result,
                recovery_operation=self._recover_execution,
            )

            print("\n=== EXECUTION RESULT ===")
            # ROOT-CAUSE FIX (misleading success semantics): keep
            # execution success (did the step run cleanly), recovery
            # replans and task completion strictly separate. A replan
            # used to print "success: True" even though the tool call
            # had failed and only the PLAN was replaced.
            print("step_success:", result.success)
            print("recovery_attempted:", result.recovery_attempted)
            print("step_succeeded:", state.step_succeeded)
            print("task_completed:", state.task_completed)
            print("result:", result.result)
            print("error:", result.error)
            print("reason:", result.reason)
            print("current_step:", state.current_step)
            print("tool_result:", state.tool_result)
            print("tool_error:", state.tool_error)
            print("final_answer:", state.final_answer)

            if result.recovery_attempted:
                await self._handle_execution_recovery(result)
                return

            if not result.success:
                state.tool_error = result.reason

                self.metrics.increment("agent_execution_failures")

                self._emit_agent_failure(
                    result.reason,
                    error=result.error,
                )

                return

            if state.blocked:
                self.metrics.increment("execution_blocked")
                return

            self._capture_step_result(step)

            self._mark_step_completed(step)

        except Exception as exc:
            error = self.error_handler.handle(
                exc,
                source="orchestrator",
                operation="run_iteration",
            )

            state.tool_error = error.message

            self.metrics.increment("agent_errors")

            self._emit_agent_failure(
                error.message,
                error=error,
            )

        finally:
            self.tracer.end_span(
                span,
                error=(
                    state.tool_error
                    if state.tool_error
                    else None
                ),
            )

    async def _create_plan(
        self,
        context: list[Any],
    ) -> None:
        state = self._require_state()

        result = await self.reliability_engine.execute(
            operation=lambda: self._generate_plan(
                user_question=state.user_request,
                context=context,
            ),
            operation_name="generate_plan",
            source="planner",
            validator=self._validate_plan_result,
            recovery_operation=self._replan_full_plan,
        )

        if result.recovery_attempted:
            if not result.success:
                state.tool_error = result.reason
                state.fatal_error = True

                self.metrics.increment("planning_failures")

                self._emit_agent_failure(
                    result.reason,
                    error=result.error,
                )

                return

            if not isinstance(
                result.result,
                PlanSchema,
            ):
                error = self.error_handler.handle(
                    TypeError(
                        "Recovery produced an invalid "
                        "plan type."
                    ),
                    source="planner",
                    operation="replan",
                )

                state.tool_error = error.message

                self.metrics.increment("planning_failures")

                self._emit_agent_failure(
                    error.message,
                    error=error,
                )

                return

            self._apply_full_plan(result.result)

            return

        if not result.success:
            state.tool_error = result.reason

            self.metrics.increment("planning_failures")

            self._emit_agent_failure(
                result.reason,
                error=result.error,
            )

            return

        plan = result.result

        if not isinstance(
            plan,
            PlanSchema,
        ):
            error = self.error_handler.handle(
                TypeError(
                    "Planner returned an invalid "
                    "result type."
                ),
                source="planner",
                operation="generate_plan",
            )

            state.tool_error = error.message

            self.metrics.increment("planning_failures")

            self._emit_agent_failure(
                error.message,
                error=error,
            )

            return

        self._apply_full_plan(plan)

    async def _generate_plan(
        self,
        *,
        user_question: str,
        context: list[Any],
    ) -> PlanSchema:
        return await self.planner.generate_plan(
            user_question=user_question,
            context=context,
        )

    def _apply_full_plan(
        self,
        plan: PlanSchema,
    ) -> None:
        state = self._require_state()

        state.plan = list(plan.steps)
        state.current_step = 0
        state.completed_steps.clear()

        state.tool_result = None
        state.tool_error = None

        state.final_answer = None
        state.final_answer_ready = False
        state.final_answer_verified = False

        state.blocked = False

        state.execution_decision = None
        state.approval_decision = None
        state.risk_assessment = None

        self.metrics.increment("plans_created")

    async def _replan_full_plan(
        self,
        error: AgentError,
    ) -> PlanSchema:
        state = self._require_state()

        if not self._check_recovery_budget(error):
            raise ValueError(
                state.tool_error or "Recovery budget exceeded."
            )

        context = await self.context_builder.build(state)

        planner_items = context.items + list(
            getattr(state, "evidence", []) or []
        )

        new_plan = await self.planner.replan(
            user_question=state.user_request,
            context=planner_items,
            failed_step=(
                state.plan[state.current_step]
                if state.plan
                else None
            ),
            failure=error,
        )

        self._validate_plan(new_plan)

        loop_result = self.loop_detector.check_plan(
            new_plan.steps
        )

        if loop_result.detected:
            raise ValueError(
                "Planner produced a repeated plan."
            )

        self._apply_full_plan(new_plan)

        self.metrics.increment("replans")

        return new_plan

    def _validate_plan_result(
        self,
        plan: Any,
    ) -> bool:
        if not isinstance(
            plan,
            PlanSchema,
        ):
            return False

        try:
            self._validate_plan(plan)
        except (ValueError, TypeError):
            return False

        loop_result = self.loop_detector.check_plan(
            plan.steps
        )

        return not loop_result.detected

    def _validate_plan(
        self,
        plan: PlanSchema,
    ) -> None:
        if not plan.steps:
            raise ValueError(
                "Plan must contain at least one step."
            )

        expected_step_id = 0

        for step in plan.steps:
            if step.step_id != expected_step_id:
                raise ValueError(
                    "Plan step IDs must be sequential "
                    "starting from 0."
                )

            self._validate_step(step)

            expected_step_id += 1

    def _validate_step(
        self,
        step: PlanStep,
    ) -> None:
        if not step.action.strip():
            raise ValueError(
                f"Step {step.step_id} "
                "has an empty action."
            )

        if step.step_type == StepType.TOOL:
            if not step.tool_name:
                raise ValueError(
                    f"Tool step {step.step_id} "
                    "must specify tool_name."
                )

        elif step.step_type == StepType.LLM:
            if step.tool_name is not None:
                raise ValueError(
                    f"LLM step {step.step_id} "
                    "must not specify tool_name."
                )

        else:
            raise ValueError(
                f"Unsupported step type: "
                f"{step.step_type}"
            )

    async def _recover_execution(
        self,
        error: AgentError,
    ) -> PlanStep:
        state = self._require_state()

        if state.current_step >= len(
            state.plan
        ):
            raise ValueError(
                "Cannot replan because the current "
                "step does not exist."
            )

        failed_step = state.plan[
            state.current_step
        ]

        budget_ok = self._check_recovery_budget(error)

        if not budget_ok:
            # ----------------------------------------------------------
            # ROOT-CAUSE FIX (recovery repeated the same strategy):
            # when the same failure pattern repeats or the replan
            # budget is exhausted, the old code simply terminated with
            # "Recovery budget exceeded" / "The same failure recurred"
            # (web_search -> same web_search -> same query -> budget).
            # Recovery MUST instead choose a meaningfully different
            # strategy; termination is only allowed when no different
            # strategy exists.
            # ----------------------------------------------------------
            alternative = await self._force_different_strategy(
                failed_step=failed_step,
                error=error,
            )

            if alternative is None:
                raise ValueError(
                    state.tool_error
                    or "Recovery budget exceeded."
                )

            state.fatal_error = False
            state.tool_error = None

            logger.info(
                "Same failure repeated, switching to a different "
                "strategy: %s (%s)",
                alternative.action,
                alternative.tool_name or "LLM",
            )

            return alternative

        planner_items = await self._planner_context_items(
            state
        )

        new_step = await self.planner.replan_step(
            user_question=state.user_request,
            context=planner_items,
            failed_step=failed_step,
            failure=error,
        )

        self._validate_step(
            new_step
        )

        # --------------------------------------------------------------
        # ROOT-CAUSE FIX: if the LLM replan produced the SAME
        # execution (same tool + same arguments) as the step that just
        # failed, do not run it again. Force a meaningfully different
        # strategy instead of looping.
        # --------------------------------------------------------------
        if self._same_execution(new_step, failed_step):
            alternative = await self._force_different_strategy(
                failed_step=failed_step,
                error=error,
            )

            if alternative is not None:
                logger.info(
                    "Replan repeated the failed execution; switching "
                    "to a different strategy: %s (%s)",
                    alternative.action,
                    alternative.tool_name or "LLM",
                )
                return alternative

        return new_step

    def _same_execution(
        self,
        step_a: PlanStep | None,
        step_b: PlanStep | None,
    ) -> bool:
        """
        True when two steps represent the SAME execution (same tool
        with equivalent arguments, or the same LLM action). Wording
        differences in the action text do not make a tool call
        meaningfully different.
        """
        if step_a is None or step_b is None:
            return False

        if step_a.step_type != step_b.step_type:
            return False

        if (step_a.tool_name or None) != (
            step_b.tool_name or None
        ):
            return False

        if step_a.step_type == StepType.TOOL:
            return (
                self._arguments_fingerprint(
                    step_a.arguments or {}
                )
                == self._arguments_fingerprint(
                    step_b.arguments or {}
                )
            )

        return (step_a.action or "").strip().lower() == (
            step_b.action or ""
        ).strip().lower()

    @staticmethod
    def _arguments_fingerprint(
        arguments: dict[str, Any],
    ) -> str:
        try:
            return json.dumps(
                arguments,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return repr(arguments)

    async def _force_different_strategy(
        self,
        *,
        failed_step: PlanStep,
        error: AgentError,
    ) -> PlanStep | None:
        """
        Deterministically pick a MEANINGFULLY DIFFERENT strategy after
        a repeated failure:

        1. an alternative tool/plan from the planner (visit_webpage
           failure -> web_search, web_search failure -> visit the URL,
           file reader failure -> the other reader, ...),
        2. otherwise: reason over the evidence that was ALREADY
           gathered (tool result -> inspect evidence -> reason), which
           is always a different strategy than repeating the call.

        Returns None when even this is impossible; the caller then
        terminates the run honestly.
        """
        state = self._require_state()

        alternative = self.planner.get_alternative_strategy(
            user_question=state.user_request,
            failed_step=failed_step,
        )

        if alternative is not None:
            try:
                self._validate_step(alternative)
            except Exception as exc:
                logger.warning(
                    "Forced alternative strategy was invalid: %s",
                    exc,
                )
                alternative = None

        if alternative is not None:
            return alternative

        # No alternative tool strategy: answer from the evidence that
        # was already gathered instead of repeating the same call.
        has_evidence = any(
            getattr(record, "succeeded", False)
            and getattr(record, "result", None)
            for record in (state.evidence or [])
        )

        if has_evidence:
            return PlanStep(
                step_id=state.current_step,
                action=(
                    "Answer the task strictly from the evidence "
                    "already gathered"
                ),
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            )

        return None

    async def _handle_execution_recovery(
        self,
        result: Any,
    ) -> None:
        state = self._require_state()

        if not result.success:
            state.tool_error = result.reason

            self.metrics.increment(
                "recovery_failures"
            )

            self._emit_agent_failure(
                result.reason,
                error=result.error,
            )

            return

        new_step = result.result

        if not isinstance(
            new_step,
            PlanStep,
        ):
            error = self.error_handler.handle(
                TypeError(
                    "Recovery did not produce "
                    "a PlanStep."
                ),
                source="orchestrator",
                operation="execution_recovery",
            )

            state.tool_error = error.message

            self.metrics.increment(
                "recovery_failures"
            )

            self._emit_agent_failure(
                error.message,
                error=error,
            )

            return

        self._replace_failed_step(
            new_step
        )

        self._prepare_step(
            new_step
        )

        state.tool_error = None
        state.recovery_attempted = False

        self.metrics.increment(
            "step_replans"
        )

    def _replace_failed_step(
        self,
        new_step: PlanStep,
    ) -> None:
        state = self._require_state()

        failed_index = state.current_step

        replacement = PlanStep(
            step_id=failed_index,
            action=new_step.action,
            step_type=new_step.step_type,
            tool_name=new_step.tool_name,
            arguments=dict(
                new_step.arguments or {}
            ),
            is_final_answer=getattr(
                new_step,
                "is_final_answer",
                False,
            ),
        )

        state.plan[
            failed_index
        ] = replacement

        # --------------------------------------------------------------
        # ROOT-CAUSE FIX: when a failed tool step is replaced by a
        # final-answer step (e.g. "answer from the evidence already
        # gathered"), the plan must END there. Keeping the old trailing
        # final step produced plans with two final-answer steps and let
        # stale intermediate steps run after the answer existed.
        # --------------------------------------------------------------
        if replacement.is_final_answer:
            state.plan = state.plan[
                : failed_index + 1
            ]

        state.tool_result = None
        state.tool_error = None

        state.blocked = False

        state.execution_decision = None
        state.approval_decision = None
        state.risk_assessment = None

    def _prepare_step(
        self,
        step: PlanStep,
    ) -> None:
        state = self._require_state()

        state.current_action = step.action

        state.step_type = step.step_type

        state.tool_name = step.tool_name

        state.tool_arguments = dict(
            step.arguments or {}
        )

        state.tool_result = None
        state.tool_error = None

        state.blocked = False

        state.execution_decision = None
        state.approval_decision = None
        state.risk_assessment = None

    @staticmethod
    def _validate_execution_result(
        state: AgentState,
    ) -> bool:
        # A blocked action is NEVER success (Phase 2 / analysis item 9).
        if state.blocked:
            return False

        if state.tool_error is not None:
            return False

        # Require an explicit success signal rather than assuming
        # the absence of an error means the step succeeded.
        return bool(
            getattr(state, "step_succeeded", False)
            or getattr(state, "execution_success", False)
        )

    def _capture_step_result(
        self,
        step: PlanStep,
    ) -> None:
        state = self._require_state()

        if not getattr(
            step,
            "is_final_answer",
            False,
        ):
            return

        if state.tool_result is None:
            raise ValueError(
                "Final answer generation "
                "returned no result."
            )

        candidate_answer = (
            self.answer_sanitizer.sanitize(
                str(state.tool_result)
            )
        )

        if not candidate_answer.strip():
            raise ValueError(
                "Final answer generation "
                "returned an empty result."
            )

        state.final_answer = candidate_answer
        state.final_answer_ready = True

    def _mark_step_completed(
        self,
        step: PlanStep,
    ) -> None:
        state = self._require_state()

        if state.current_step not in (
            state.completed_steps
        ):
            state.completed_steps.append(
                state.current_step
            )

        state.current_step += 1

        state.retry_count = 0
        state.recovery_attempted = False

        self.metrics.increment(
            "steps_completed"
        )

    @staticmethod
    def _build_verification_evidence(
        *,
        state: AgentState,
        context_items: list[Any],
    ) -> list[Any]:
        """
        Semantic verification (Phase 7) must judge the candidate
        answer against the ACTUAL evidence gathered during execution,
        not just the context window.

        Evidence records are prepended so they are the primary source
        the verifier evaluates.
        """
        evidence = list(
            getattr(state, "evidence", []) or []
        )

        if not evidence:
            return list(context_items)

        return evidence + list(context_items)

    @staticmethod
    def _evidence_echoes_candidate(
        item: Any,
        candidate_text: str,
    ) -> bool:
        """Remove the final-answer generation echo from evidence."""
        if not candidate_text:
            return False
        if getattr(item, "tool_name", None) == "llm":
            return True
        for attr in ("tool_result", "result", "content"):
            value = getattr(item, attr, None)
            if value is None:
                continue
            text = str(value).strip()
            if text and text == candidate_text:
                return True
        return False

    async def _planner_context_items(
        self,
        state: AgentState,
    ) -> list[Any]:
        """Build planner context that also includes gathered evidence."""
        context = await self.context_builder.build(
            state
        )
        items = list(context.items)
        evidence = list(
            getattr(state, "evidence", []) or []
        )
        if evidence:
            items = items + evidence
        return items

    async def _handle_plan_completion(
        self,
    ) -> None:
        state = self._require_state()

        if not state.final_answer_ready:
            self._create_final_answer_step()
            return

        if not state.final_answer_verified:
            await self._verify_final_answer()
            return

    def _create_final_answer_step(
        self,
    ) -> None:
        state = self._require_state()

        step_id = len(
            state.plan
        )

        state.plan.append(
            PlanStep(
                step_id=step_id,
                action=(
                    "Generate final answer "
                    "from gathered results."
                ),
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            )
        )

        self.metrics.increment(
            "final_answer_generation_steps"
        )

    async def _verify_final_answer(
        self,
    ) -> None:
        state = self._require_state()

        state.verification_attempts = (
            getattr(state, "verification_attempts", 0) + 1
        )

        if state.final_answer is None:
            error = self.error_handler.handle(
                ValueError(
                    "Final answer is missing."
                ),
                source="orchestrator",
                operation="verify_answer",
            )

            state.tool_error = error.message

            self.metrics.increment(
                "verification_failures"
            )

            self._emit_agent_failure(
                error.message,
                error=error,
            )

            return

        context = await self.context_builder.build(
            state
        )

        self.metrics.increment(
            "verification_context_builds"
        )

        full_evidence = self._build_verification_evidence(
            state=state,
            context_items=context.items,
        )

        evidence_for_check = [
            item
            for item in full_evidence
            if not self._evidence_echoes_candidate(
                item,
                state.final_answer,
            )
        ]

        status, gate_reason = deterministic_verification(
            candidate_answer=state.final_answer,
            raw_data=evidence_for_check,
        )

        if status == VerificationStatus.PASS:
            state.final_answer_ready = True
            state.final_answer_verified = True
            state.task_completed = True
            state.tool_error = None
            self.metrics.increment("answers_verified")
            logger.info(
                "Deterministic verification PASS: %s",
                gate_reason,
            )
            return

        if status == VerificationStatus.FAIL:
            state.tool_error = gate_reason
            self.metrics.increment("verification_failures")
            self._emit_agent_failure(gate_reason)
            error = AgentError(
                error_type="AnswerVerificationError",
                message=gate_reason,
                source="verifier",
                operation="verify_answer",
                recoverable=True,
            )
            await self._handle_verification_failure(
                error
            )
            return

        verification_input = VerificationInput(
            question=state.user_request,
            candidate_answer=state.final_answer,
            raw_data=full_evidence,
        )

        result = await self.reliability_engine.execute(
            operation=lambda: self.verifier.verify(
                verification_input
            ),
            operation_name="verify_answer",
            source="verifier",
            validator=self._validate_verification_result,
        )

        if not result.success:
            state.tool_error = result.reason

            self.metrics.increment(
                "verification_failures"
            )

            self._emit_agent_failure(
                result.reason,
                error=result.error,
            )

            return

        verification = result.result

        if verification is None:
            state.tool_error = (
                "Verifier returned no result."
            )

            self.metrics.increment(
                "verification_failures"
            )

            return

        if not verification.verified:
            error = AgentError(
                error_type="AnswerVerificationError",
                message=verification.reason,
                source="verifier",
                operation="verify_answer",
                recoverable=True,
            )

            await self._handle_verification_failure(
                error
            )

            return

        # --------------------------------------------------------------
        # ROOT-CAUSE FIX (false verification): the LLM judge must never
        # simply declare its own answer verified. When strong tool
        # evidence exists but does NOT contain the candidate answer
        # (e.g. evidence pointed at answer 3, agent answered 4, or the
        # agent invented "N2023-001" from thin air), the LLM verdict is
        # rejected and the bounded verification failure path runs.
        # --------------------------------------------------------------
        support = evidence_supports_candidate(
            candidate_answer=state.final_answer,
            raw_data=evidence_for_check,
        )

        if support is False:
            reason = (
                "LLM verification accepted the answer, but the "
                "gathered tool evidence does not contain or support "
                f"it ('{state.final_answer}')."
            )

            state.tool_error = reason

            self.metrics.increment(
                "verification_failures"
            )

            self._emit_agent_failure(reason)

            await self._handle_verification_failure(
                AgentError(
                    error_type="AnswerVerificationError",
                    message=reason,
                    source="verifier",
                    operation="verify_answer",
                    recoverable=True,
                )
            )

            return

        state.final_answer_ready = True
        state.final_answer_verified = True
        state.task_completed = True
        state.tool_error = None

        self.metrics.increment(
            "answers_verified"
        )

    async def _handle_verification_failure(
        self,
        error: AgentError,
    ) -> None:
        state = self._require_state()

        # Bounded verification budget: do NOT keep regenerating and
        # re-verifying forever. Deliver the best candidate answer
        # honestly marked as UNVERIFIED; the termination policy stops
        # the loop with ANSWER_UNVERIFIED_BUDGET.
        if (
            getattr(state, "verification_attempts", 0)
            >= MAX_VERIFICATION_ATTEMPTS
        ):
            state.final_answer_ready = True
            state.task_completed = True
            state.tool_error = None

            self.metrics.increment(
                "answers_accepted_unverified"
            )

            logger.warning(
                "Verification attempts exhausted (%d); "
                "accepting final answer unverified.",
                state.verification_attempts,
            )

            return

        if not self._check_recovery_budget(error):
            self.metrics.increment(
                "verification_failures"
            )
            self._emit_agent_failure(
                state.tool_error or "Recovery budget exceeded.",
                error=error,
            )
            return

        classification = (
            self.reliability_engine.failure_classifier.classify(
                error
            )
        )

        recovery_decision = (
            self.reliability_engine.recovery_policy.evaluate(
                classification
            )
        )

        if recovery_decision.action.name != "REPLAN":
            state.tool_error = (
                recovery_decision.reason
            )

            self.metrics.increment(
                "verification_failures"
            )

            self._emit_agent_failure(
                recovery_decision.reason,
                error=error,
            )

            return

        failed_step = self._find_final_answer_step()

        if failed_step is None:
            state.tool_error = (
                "Final answer step could not be found."
            )

            self.metrics.increment(
                "verification_failures"
            )

            self._emit_agent_failure(
                state.tool_error,
                error=error,
            )

            return

        planner_items = await self._planner_context_items(
            state
        )

        try:
            new_step = await self.planner.replan_step(
                user_question=state.user_request,
                context=planner_items,
                failed_step=failed_step,
                failure=error,
            )

            self._validate_step(
                new_step
            )

        except Exception as exc:
            recovery_error = self.error_handler.handle(
                exc,
                source="planner",
                operation="replan_step",
            )

            state.tool_error = (
                recovery_error.message
            )

            self.metrics.increment(
                "verification_recovery_failures"
            )

            self._emit_agent_failure(
                recovery_error.message,
                error=recovery_error,
            )

            return

        self._replace_failed_step(
            new_step
        )

        self._prepare_step(
            new_step
        )

        state.final_answer = None
        state.final_answer_ready = False
        state.final_answer_verified = False

        self.metrics.increment(
            "answer_replans"
        )

    def _find_final_answer_step(
        self,
    ) -> PlanStep | None:
        state = self._require_state()

        for step in reversed(
            state.plan
        ):
            if getattr(
                step,
                "is_final_answer",
                False,
            ):
                return step

        return None

    @staticmethod
    def _validate_verification_result(
        result: Any,
    ) -> bool:
        return isinstance(
            result,
            VerificationResult,
        )

    async def _handle_loop(
        self,
        reason: str,
    ) -> None:
        state = self._require_state()

        if not self._check_recovery_budget(
            AgentError(
                error_type="LoopDetected",
                message=reason,
                source="loop_detector",
                operation="execution",
            )
        ):
            # ------------------------------------------------------
            # Last-resort salvage (Phase 6): the recovery budget is
            # exhausted, but if real evidence was already gathered and
            # no answer exists yet, make ONE bounded attempt to answer
            # strictly from that evidence instead of terminating with a
            # bare failure. The attempt is flagged so it can happen at
            # most once per task; the state remains unverified unless
            # the verifier confirms the answer.
            # ------------------------------------------------------
            if (
                not state.loop_salvage_attempted
                and state.evidence
                and not state.final_answer_ready
            ):
                final_step = self._find_final_answer_step()

                if final_step is not None:
                    state.loop_salvage_attempted = True

                    for idx, candidate in enumerate(
                        state.plan
                    ):
                        if candidate is final_step:
                            state.current_step = idx
                            break

                    state.fatal_error = False
                    state.tool_error = None

                    self._prepare_step(final_step)

                    self.metrics.increment(
                        "loop_salvages"
                    )

                    return

            self.metrics.increment(
                "loop_recovery_failures"
            )
            self._emit_agent_failure(
                state.tool_error or "Recovery budget exceeded."
            )
            return

        error = AgentError(
            error_type="LoopDetected",
            message=reason,
            source="loop_detector",
            operation="execution",
            recoverable=True,
        )

        classification = (
            self.reliability_engine.failure_classifier.classify(
                error
            )
        )

        recovery_decision = (
            self.reliability_engine.recovery_policy.evaluate(
                classification
            )
        )

        if recovery_decision.action.name != "REPLAN":
            state.tool_error = (
                recovery_decision.reason
            )

            self.metrics.increment(
                "loop_recovery_failures"
            )

            self._emit_agent_failure(
                recovery_decision.reason,
                error=error,
            )

            return

        if state.current_step >= len(
            state.plan
        ):
            state.tool_error = (
                "Loop recovery has no current step."
            )

            self.metrics.increment(
                "loop_recovery_failures"
            )

            self._emit_agent_failure(
                state.tool_error,
                error=error,
            )

            return

        failed_step = state.plan[
            state.current_step
        ]

        planner_items = await self._planner_context_items(
            state
        )

        try:
            new_step = await self.planner.replan_step(
                user_question=state.user_request,
                context=planner_items,
                failed_step=failed_step,
                failure=error,
            )

            self._validate_step(
                new_step
            )

        except Exception as exc:
            recovery_error = self.error_handler.handle(
                exc,
                source="planner",
                operation="replan_step",
            )

            state.tool_error = (
                recovery_error.message
            )

            self.metrics.increment(
                "loop_recovery_failures"
            )

            self._emit_agent_failure(
                recovery_error.message,
                error=recovery_error,
            )

            return

        self._replace_failed_step(
            new_step
        )

        self._prepare_step(
            new_step
        )

        self.metrics.increment(
            "loop_recoveries"
        )

    def emit_agent_started(
        self,
    ) -> None:
        state = self._require_state()

        self.event_logger.log(
            create_event(
                event_type=EventType.AGENT_STARTED,
                correlation_id=self.correlation_id,
                iteration=state.iteration,
            )
        )

        self.metrics.increment(
            "agents_started"
        )

    def emit_agent_completed(
        self,
    ) -> None:
        state = self._require_state()

        self.event_logger.log(
            create_event(
                event_type=EventType.AGENT_COMPLETED,
                correlation_id=self.correlation_id,
                iteration=state.iteration,
                metadata={
                    "final_answer_verified":
                        state.final_answer_verified,
                    "completed_steps":
                        len(
                            state.completed_steps
                        ),
                },
            )
        )

        self.metrics.increment(
            "agents_completed"
        )

    def _emit_agent_failure(
        self,
        reason: str,
        error: AgentError | None = None,
    ) -> None:
        state = self._require_state()

        self.event_logger.log(
            create_event(
                event_type=EventType.AGENT_FAILED,
                correlation_id=self.correlation_id,
                iteration=state.iteration,
                error=reason,
                metadata={
                    "error_type": (
                        error.error_type
                        if error
                        else None
                    ),
                    "source": (
                        error.source
                        if error
                        else None
                    ),
                    "operation": (
                        error.operation
                        if error
                        else None
                    ),
                },
            )
        )