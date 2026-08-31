from __future__ import annotations

import inspect
from time import perf_counter
from typing import Any
from uuid import UUID

from gaia_agent.core.agent_state import AgentState
from gaia_agent.planner.plan_schema import StepType

from gaia_agent.core.policies.execution import (
    ExecutionDecision,
    ExecutionPolicy,
    ExecutionState,
)

from gaia_agent.core.policies.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalState,
)

from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.models import RiskContext

from gaia_agent.core.llm_executor import LLMExecutor
from gaia_agent.tools.registry import ToolRegistry

from gaia_agent.reliability.exception import (
    ApprovalBlockedError,
    EmptyResultError,
    ToolExecutionError,
)

from gaia_agent.core.evidence import ToolResultRecord


# Error-condition signals that file/image/excel/web tools may return
# as plain strings (e.g. "Error: File not found..."). Such results
# MUST be treated as tool failures and trigger replanning instead of
# being recorded as successful evidence.
_STRONG_TOOL_ERROR_MARKERS = (
    "error:",
    "traceback",
    "exception",
    "http 403",
    "http 404",
    "http 500",
    "forbidden",
    "timeout",
    "rate limit",
)

_WEAK_TOOL_ERROR_MARKERS = (
    "file not found",
    "image not found",
    "excel file not found",
    "not found in base_dir",
    "error fetching the webpage",
    "error reading file",
    "error analyzing image",
    "excel analysis error",
    "is a placeholder",
    "unsupported excel format",
    "unsupported image format",
    "not a file:",
    "not defined",
    "is not registered",
    "requires argument",
    "does not accept argument",
)


def is_tool_error_result(result: Any) -> str | None:
    """
    Return the offending text when a tool's result string signals a
    real failure that must trigger error handling / replanning, or
    None when the result is a normal (possibly informational) value.
    """
    if not isinstance(result, str):
        return None

    text = result.strip()

    if not text:
        return None

    head = text[:300]

    lowered = head.lower()

    for marker in _STRONG_TOOL_ERROR_MARKERS:
        if marker in lowered:
            return head

    for marker in _WEAK_TOOL_ERROR_MARKERS:
        if marker in lowered:
            return head

    return None

from gaia_agent.observability.events import (
    EventType,
    create_event,
)
from gaia_agent.observability.logger import EventLogger
from gaia_agent.observability.metrics import Metrics
from gaia_agent.observability.tracer import Tracer
from gaia_agent.observability.token_tracker import TokenTracker


class AgentExecution:

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        execution_policy: ExecutionPolicy,
        risk_assessor: RiskAssessor,
        approval_policy: ApprovalPolicy,
        llm_executor: LLMExecutor,
        event_logger: EventLogger,
        metrics: Metrics,
        tracer: Tracer,
        token_tracker: TokenTracker,
        correlation_id: UUID | None = None,
    ) -> None:

        self.tool_registry = tool_registry
        self.execution_policy = execution_policy
        self.risk_assessor = risk_assessor
        self.approval_policy = approval_policy
        self.llm_executor = llm_executor

        self.event_logger = event_logger
        self.metrics = metrics
        self.tracer = tracer
        self.token_tracker = token_tracker

        self.correlation_id = correlation_id

        self.state: AgentState | None = None

    def bind_state(
        self,
        state: AgentState,
    ) -> None:

        self.state = state

    def _require_state(self) -> AgentState:

        if self.state is None:
            raise RuntimeError(
                "AgentState is not bound. "
                "Call bind_state() before execution."
            )

        return self.state

    async def execute(self) -> AgentState:

        state = self._require_state()

        execution_decision = self.check_execution()

        state.execution_decision = execution_decision

        if not execution_decision.allowed:

            state.blocked = True
            state.waiting_for_approval = False
            state.execution_success = False
            state.step_succeeded = False

            state.tool_error = (
                execution_decision.message
                or execution_decision.reason
                or "Execution was denied."
            )

            self.metrics.increment(
                "execution_blocked"
            )

            raise ApprovalBlockedError(
                state.tool_error
            )

        state.blocked = False

        await self.check_risk()

        approval_decision = self.check_approval()

        state.approval_decision = approval_decision

        if approval_decision.approval_required:

            state.waiting_for_approval = True
            state.blocked = True
            state.execution_success = False
            state.step_succeeded = False

            state.tool_error = (
                approval_decision.message
                or "Human approval is required."
            )

            self.metrics.increment(
                "approval_required"
            )

            raise ApprovalBlockedError(
                state.tool_error
            )

        state.waiting_for_approval = False
        state.blocked = False

        if state.step_type == StepType.TOOL:
            state.step_succeeded = False
            return await self.execute_tool()

        if state.step_type == StepType.LLM:
            state.step_succeeded = False
            return await self.execute_llm()

        raise ValueError(
            f"Unsupported step type: {state.step_type!r}"
        )

    def check_execution(
        self,
    ) -> ExecutionDecision:

        state = self._require_state()

        execution_state = ExecutionState(
            step_type=state.step_type,
            tool_name=state.tool_name,
            action_name=state.current_action,
            arguments=state.tool_arguments or {},
            blocked=state.blocked,
        )

        return self.execution_policy.evaluate(
            execution_state
        )

    async def check_risk(
        self,
    ) -> None:

        state = self._require_state()

        risk_context = RiskContext(
            action=state.current_action or "",
            tool_name=state.tool_name,
            arguments=state.tool_arguments or {},
        )

        state.risk_assessment = (
            await self.risk_assessor.assess(
                risk_context
            )
        )

    def check_approval(
        self,
    ) -> ApprovalDecision:

        state = self._require_state()

        if state.risk_assessment is None:
            raise RuntimeError(
                "Risk assessment must exist before approval."
            )

        approval_state = ApprovalState(
            action_name=state.current_action or "",
            tool_name=state.tool_name,
            risk_assessment=state.risk_assessment,
        )

        return self.approval_policy.evaluate(
            approval_state
        )

    async def execute_tool(
        self,
    ) -> AgentState:

        state = self._require_state()

        if not state.tool_name:
            raise ValueError(
                "Tool step requires tool_name."
            )

        tool = None

        try:
            tool = self.tool_registry.get(
                state.tool_name
            )
        except (KeyError, ValueError) as exc:
            # STEP 1: an unavailable tool must raise a proper
            # tool-validation error (recoverable), never a bare
            # KeyError that the failure classifier cannot map.
            raise ToolExecutionError(
                f"Tool '{state.tool_name}' is not registered or "
                f"unavailable. Registered tools: "
                f"{sorted(self.tool_registry.get_tools(), key=lambda t: t.name)}. "
                f"Details: {exc}",
                recoverable=True,
            ) from exc

        if tool is None:
            raise ToolExecutionError(
                f"Tool not found: {state.tool_name}",
                recoverable=True,
            )

        if not hasattr(tool, "execute"):
            raise TypeError(
                f"Tool '{state.tool_name}' must expose "
                "an execute() method."
            )

        # ----------------------------------------------------------
        # Phase 1: strict argument validation BEFORE the tool runs.
        #
        # GAIA failure mode addressed:
        #   DuckDuckGoSearchTool.forward()
        #   got an unexpected keyword argument 'code'
        # ----------------------------------------------------------

        validated_arguments = tool.validate_arguments(
            state.tool_arguments
        )

        state.tool_arguments = validated_arguments

        span = self.tracer.start_span(
            operation=f"tool.{state.tool_name}",
            correlation_id=self.correlation_id,
        )

        start = perf_counter()

        self.event_logger.log(
            create_event(
                event_type=EventType.TOOL_STARTED,
                correlation_id=self.correlation_id,
                iteration=state.iteration,
                metadata={
                    "tool_name": state.tool_name,
                    "arguments": state.tool_arguments,
                },
            )
        )

        self.metrics.increment(
            "tool_requests"
        )

        try:

            result = tool.execute(
                **(state.tool_arguments or {})
            )

            if inspect.isawaitable(result):
                result = await result

            error_signal = is_tool_error_result(
                result
            )

            if error_signal is not None:
                # A tool that returns an error-string did NOT succeed.
                # Treat it as a recoverable tool failure so the
                # reliability engine can classify it and replan with a
                # different strategy (never re-running the same call).
                raise ToolExecutionError(
                    f"Tool '{state.tool_name}' reported a failure. "
                    f"{error_signal}",
                    recoverable=True,
                )

            if result is None or (
                isinstance(result, str) and not result.strip()
            ):
                raise EmptyResultError(
                    f"Tool '{state.tool_name}' returned an "
                    "empty result."
                )

            state.tool_result = result
            state.tool_error = None
            state.step_succeeded = True
            state.execution_success = True

            # ------------------------------------------------------
            # Phase 4: record every tool execution as evidence.
            # ------------------------------------------------------

            state.evidence.append(
                ToolResultRecord(
                    step_id=getattr(
                        state,
                        "current_step",
                        None,
                    ),
                    tool_name=state.tool_name,
                    arguments=dict(state.tool_arguments or {}),
                    result=result,
                    succeeded=True,
                )
            )

            latency = perf_counter() - start

            self.metrics.record_duration(
                "tool_latency",
                latency,
            )

            self.event_logger.log(
                create_event(
                    event_type=EventType.TOOL_COMPLETED,
                    correlation_id=self.correlation_id,
                    iteration=state.iteration,
                    latency=latency,
                    metadata={
                        "tool_name": state.tool_name,
                    },
                )
            )

            self.tracer.end_span(
                span
            )

            return state

        except Exception as exc:

            latency = perf_counter() - start

            state.tool_error = str(exc)

            self.metrics.increment(
                "tool_failures"
            )

            self.event_logger.log(
                create_event(
                    event_type=EventType.TOOL_FAILED,
                    correlation_id=self.correlation_id,
                    iteration=state.iteration,
                    latency=latency,
                    error=str(exc),
                    metadata={
                        "tool_name": state.tool_name,
                    },
                )
            )

            self.tracer.end_span(
                span,
                error=str(exc),
            )

            raise

    async def execute_llm(
        self,
    ) -> AgentState:

        state = self._require_state()

        span = self.tracer.start_span(
            operation="llm.request",
            correlation_id=self.correlation_id,
        )

        start = perf_counter()

        self.event_logger.log(
            create_event(
                event_type=EventType.LLM_REQUEST_STARTED,
                correlation_id=self.correlation_id,
                iteration=state.iteration,
                metadata={
                    "action": state.current_action,
                },
            )
        )

        self.metrics.increment(
            "llm_requests"
        )

        try:

            result = await self.llm_executor.execute(
                state
            )

            if not isinstance(result, str):
                raise TypeError(
                    "LLMExecutor must return str."
                )

            result = result.strip()

            if not result:
                raise ValueError(
                    "LLM returned an empty result."
                )

            state.tool_result = result
            state.tool_error = None
            state.step_succeeded = True
            state.execution_success = True

            state.evidence.append(
                ToolResultRecord(
                    step_id=getattr(
                        state,
                        "current_step",
                        None,
                    ),
                    tool_name="llm",
                    arguments={},
                    result=result,
                    succeeded=True,
                )
            )

            latency = perf_counter() - start

            self.metrics.record_duration(
                "llm_latency",
                latency,
            )

            self._track_llm_tokens(
                result
            )

            self.event_logger.log(
                create_event(
                    event_type=EventType.LLM_REQUEST_COMPLETED,
                    correlation_id=self.correlation_id,
                    iteration=state.iteration,
                    latency=latency,
                    metadata={
                        "action": state.current_action,
                    },
                )
            )

            self.tracer.end_span(
                span
            )

            return state

        except Exception as exc:

            latency = perf_counter() - start

            state.tool_error = str(exc)

            self.metrics.increment(
                "llm_failures"
            )

            self.event_logger.log(
                create_event(
                    event_type=EventType.LLM_REQUEST_FAILED,
                    correlation_id=self.correlation_id,
                    iteration=state.iteration,
                    latency=latency,
                    error=str(exc),
                    metadata={
                        "action": state.current_action,
                    },
                )
            )

            self.tracer.end_span(
                span,
                error=str(exc),
            )

            raise

    def _track_llm_tokens(
        self,
        result: Any,
    ) -> None:

        usage = getattr(
            result,
            "usage",
            None,
        )

        if usage is None:
            return

        input_tokens = getattr(
            usage,
            "input_tokens",
            0,
        )

        output_tokens = getattr(
            usage,
            "output_tokens",
            0,
        )

        self.token_tracker.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )