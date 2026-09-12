from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from gaia_agent.core.evidence import (
    ArtifactInfo,
    ToolResultRecord,
)
from gaia_agent.core.llm_executor import (
    LLMExecutionRequest,
    LLMExecutor,
)
from gaia_agent.core.policies.approval import (
    ApprovalPolicy,
    ApprovalState,
)
from gaia_agent.core.policies.execution import (
    ExecutionPolicy,
    ExecutionState,
)
from gaia_agent.core.risk.assessor import (
    RiskAssessor,
    RiskContext,
)
from gaia_agent.observability.events import (
    EventType,
    create_event,
)
from gaia_agent.observability.logger import EventLogger
from gaia_agent.observability.metrics import Metrics
from gaia_agent.observability.token_tracker import TokenTracker
from gaia_agent.observability.tracer import Tracer
from gaia_agent.planner.plan_schema import StepType
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)
from gaia_agent.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    step_id: int
    step_type: StepType
    action: str

    tool_name: str | None = None
    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    user_request: str = ""
    context: Any = None

    iteration: int = 0

    correlation_id: UUID | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    success: bool

    output: Any = None

    evidence: tuple[ToolResultRecord, ...] = ()
    artifacts: tuple[ArtifactInfo, ...] = ()

    error: AgentError | None = None

    blocked: bool = False

    step_id: int | None = None
    tool_name: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class AgentExecution:

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        execution_policy: ExecutionPolicy,
        risk_assessor: RiskAssessor | None,
        approval_policy: ApprovalPolicy | None,
        llm_executor: LLMExecutor,
        event_logger: EventLogger | None = None,
        metrics: Metrics | None = None,
        tracer: Tracer | None = None,
        token_tracker: TokenTracker | None = None,
        error_handler: ErrorHandler | None = None,
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

        self.error_handler = (
            error_handler
            or ErrorHandler()
        )

        self.correlation_id = (
            correlation_id
            or uuid4()
        )

    async def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:

        self._validate_request(request)

        correlation_id = (
            request.correlation_id
            or self.correlation_id
        )

        span = None

        if self.tracer is not None:
            span = self.tracer.start_span(
                operation="agent.execution",
                correlation_id=correlation_id,
            )

        started = perf_counter()

        self._emit(
            EventType.EXECUTION_STARTED,
            request,
        )

        self._increment(
            "execution_requests"
        )

        try:

            await self._check_execution_policy(
                request
            )
            
            risk_assessment = await self._check_risk(
                request
            )

            await self._check_approval(
                request,
                risk_assessment,
            )

            if self._is_tool_step(request):

                result = await self._execute_tool(
                    request
                )

            elif self._is_llm_step(request):

                result = await self._execute_llm(
                    request
                )

            else:

                raise AgentError(
                    error_type="UnsupportedStepType",
                    message=(
                        f"Unsupported step type: "
                        f"{request.step_type}"
                    ),
                    category=(
                        ErrorCategory.VALIDATION
                    ),
                    severity=(
                        ErrorSeverity.HIGH
                    ),
                    retryable=False,
                    recoverable=False,
                    source="AgentExecution",
                    operation="execute",
                    details={
                        "step_id": request.step_id,
                        "step_type": str(
                            request.step_type
                        ),
                    },
                )

            self._increment(
                "execution_successes"
            )

            self._record_duration(
                "execution_latency",
                perf_counter() - started,
            )

            self._emit(
                EventType.EXECUTION_COMPLETED,
                request,
            )

            if span is not None:
                self.tracer.end_span(span)

            return result

        except AgentError as error:

            self._increment(
                "execution_failures"
            )

            self._record_duration(
                "execution_latency",
                perf_counter() - started,
            )

            if span is not None:
                self.tracer.end_span(
                    span,
                    error=error.message,
                )

            return self._failure_result(
                request,
                error,
            )

        except Exception as exc:

            error = self.error_handler.handle(
                exc,
                source="AgentExecution",
                operation="execute",
            )

            if not isinstance(
                error,
                AgentError,
            ):
                error = AgentError(
                    error_type=(
                        type(exc).__name__
                    ),
                    message=(
                        str(exc)
                        or type(exc).__name__
                    ),
                    category=(
                        ErrorCategory.INTERNAL
                    ),
                    severity=(
                        ErrorSeverity.HIGH
                    ),
                    retryable=False,
                    recoverable=False,
                    source="AgentExecution",
                    operation="execute",
                    original_exception=exc,
                )

            self._increment(
                "execution_failures"
            )

            if span is not None:
                self.tracer.end_span(
                    span,
                    error=error.message,
                )

            return self._failure_result(
                request,
                error,
            )


    async def _check_execution_policy(
        self,
        request: ExecutionRequest,
    ) -> None:

        execution_state = ExecutionState(
            step_type=request.step_type,
            tool_name=request.tool_name,
            action_name=request.action,
            arguments=dict(
                request.arguments
            ),
            blocked=False,
        )

        try:

            decision = (
                self.execution_policy.evaluate(
                    execution_state
                )
            )

            if inspect.isawaitable(
                decision
            ):
                decision = await decision

        except AgentError:
            raise

        except Exception as exc:

            raise AgentError(
                error_type="ExecutionPolicyFailure",
                message=(
                    f"Execution policy failed: "
                    f"{exc}"
                ),
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_execution_policy",
                original_exception=exc,
            ) from exc

        if not getattr(
            decision,
            "allowed",
            False,
        ):
            raise AgentError(
                error_type="ExecutionBlocked",
                message=(
                    getattr(
                        decision,
                        "message",
                        None,
                    )
                    or "Execution policy blocked "
                    "the requested operation."
                ),
                category=(
                    ErrorCategory.APPROVAL_BLOCKED
                ),
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_execution_policy",
                details={
                    "step_id": request.step_id,
                    "tool_name": request.tool_name,
                    "action": request.action,
                    "reason": getattr(
                        decision,
                        "reason",
                        None,
                    ),
                },
            )

    async def _check_risk(
        self,
        request: ExecutionRequest,
    ) -> Any:

        if self.risk_assessor is None:
            return None

        context = RiskContext(
            action=request.action,
            tool_name=request.tool_name,
            arguments=dict(
                request.arguments
            ),
        )

        try:

            result = (
                await self.risk_assessor.assess(
                    context
                )
            )

            return result

        except AgentError:
            raise

        except Exception as exc:

            raise AgentError(
                error_type="RiskAssessmentFailure",
                message=(
                    f"Risk assessment failed: "
                    f"{exc}"
                ),
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_risk",
                original_exception=exc,
            ) from exc

    async def _check_approval(
        self,
        request: ExecutionRequest,
        risk_assessment: Any,
    ) -> None:

        if (
            self.approval_policy is None
            or risk_assessment is None
        ):
            return

        approval_state = ApprovalState(
            action_name=request.action,
            tool_name=request.tool_name,
            risk_assessment=risk_assessment,
        )

        try:

            decision = (
                self.approval_policy.evaluate(
                    approval_state
                )
            )

            if inspect.isawaitable(
                decision
            ):
                decision = await decision

        except AgentError:
            raise

        except Exception as exc:

            raise AgentError(
                error_type="ApprovalPolicyFailure",
                message=(
                    f"Approval policy failed: "
                    f"{exc}"
                ),
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_approval",
                original_exception=exc,
            ) from exc

        if getattr(
            decision,
            "approval_required",
            False,
        ):
            raise AgentError(
                error_type="ApprovalBlocked",
                message=(
                    getattr(
                        decision,
                        "message",
                        None,
                    )
                    or "Human approval is required."
                ),
                category=(
                    ErrorCategory.APPROVAL_BLOCKED
                ),
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_approval",
                details={
                    "step_id": request.step_id,
                    "action": request.action,
                    "tool_name": request.tool_name,
                    "reason": getattr(
                        decision,
                        "reason",
                        None,
                    ),
                },
            )

 
    async def _execute_tool(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:

        if not request.tool_name:
            raise AgentError(
                error_type="MissingToolName",
                message=(
                    "Tool step does not contain "
                    "a tool name."
                ),
                category=(
                    ErrorCategory.TOOL_ARGUMENT_ERROR
                ),
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="execute_tool",
            )

        tool_name = request.tool_name

        span = None

        if self.tracer is not None:
            span = self.tracer.start_span(
                operation=f"tool.{tool_name}",
                correlation_id=(
                    request.correlation_id
                    or self.correlation_id
                ),
            )

        self._emit(
            EventType.TOOL_STARTED,
            request,
            metadata={
                "tool_name": tool_name
            },
        )

        self._increment(
            "tool_requests"
        )

        started = perf_counter()

        try:

            try:
                tool = self.tool_registry.get(
                    tool_name
                )
            except KeyError as exc:

                raise AgentError(
                    error_type="ToolNotFound",
                    message=str(exc),
                    category=(
                        ErrorCategory.TOOL_NOT_FOUND
                    ),
                    severity=ErrorSeverity.MEDIUM,
                    retryable=False,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_tool",
                    original_exception=exc,
                ) from exc

            # IMPORTANT:
            # ToolRegistry RegisteredTool.execute()
            # expects keyword arguments.
            result = tool.execute(
                **dict(request.arguments)
            )

            if inspect.isawaitable(result):
                result = await result

            if self._is_empty_result(result):
                raise AgentError(
                    error_type="EmptyResult",
                    message=(
                        f"Tool '{tool_name}' "
                        "returned an empty result."
                    ),
                    category=(
                        ErrorCategory.EMPTY_RESULT
                    ),
                    severity=ErrorSeverity.MEDIUM,
                    retryable=True,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_tool",
                    details={
                        "tool_name": tool_name,
                        "step_id": request.step_id,
                    },
                )

            execution_result = (
                self._build_tool_result(
                    request,
                    result,
                )
            )

            self._increment(
                "tool_successes"
            )

            self._record_duration(
                f"tool.{tool_name}.latency",
                perf_counter() - started,
            )

            self._emit(
                EventType.TOOL_COMPLETED,
                request,
                metadata={
                    "tool_name": tool_name
                },
            )

            if span is not None:
                self.tracer.end_span(span)

            return execution_result

        except AgentError as error:

            self._increment(
                "tool_failures"
            )

            self._emit(
                EventType.TOOL_FAILED,
                request,
                error=error,
            )

            if span is not None:
                self.tracer.end_span(
                    span,
                    error=error.message,
                )

            raise

        except Exception as exc:

            error = AgentError(
                error_type=type(exc).__name__,
                message=(
                    f"Tool '{tool_name}' "
                    f"execution failed: {exc}"
                ),
                category=(
                    ErrorCategory.TOOL_EXECUTION_ERROR
                ),
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                recoverable=True,
                source="AgentExecution",
                operation="execute_tool",
                details={
                    "tool_name": tool_name,
                    "step_id": request.step_id,
                },
                original_exception=exc,
            )

            self._increment(
                "tool_failures"
            )

            self._emit(
                EventType.TOOL_FAILED,
                request,
                error=error,
            )

            if span is not None:
                self.tracer.end_span(
                    span,
                    error=error.message,
                )

            raise error from exc

    async def _execute_llm(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:

        span = None

        if self.tracer is not None:
            span = self.tracer.start_span(
                operation="llm.request",
                correlation_id=(
                    request.correlation_id
                    or self.correlation_id
                ),
            )

        self._emit(
            EventType.LLM_REQUEST_STARTED,
            request,
        )

        self._increment(
            "llm_requests"
        )

        started = perf_counter()

        try:

            llm_request = LLMExecutionRequest(
                user_request=request.user_request,
                action=request.action,
                context=request.context,
                metadata=dict(
                    request.metadata
                ),
            )

            output = await self.llm_executor.execute(
                llm_request
            )

            if not isinstance(
                output,
                str,
            ):
                raise AgentError(
                    error_type="InvalidLLMOutput",
                    message=(
                        "LLM executor returned "
                        "a non-string result."
                    ),
                    category=(
                        ErrorCategory.LLM_OUTPUT_ERROR
                    ),
                    severity=ErrorSeverity.MEDIUM,
                    retryable=True,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_llm",
                )

            if not output.strip():
                raise AgentError(
                    error_type="EmptyResult",
                    message=(
                        "LLM returned an empty result."
                    ),
                    category=(
                        ErrorCategory.EMPTY_RESULT
                    ),
                    severity=ErrorSeverity.MEDIUM,
                    retryable=True,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_llm",
                )

            self._increment(
                "llm_successes"
            )

            self._record_duration(
                "llm.latency",
                perf_counter() - started,
            )

            self._emit(
                EventType.LLM_REQUEST_COMPLETED,
                request,
            )

            if span is not None:
                self.tracer.end_span(span)

            return ExecutionResult(
                success=True,
                output=output,
                step_id=request.step_id,
                metadata={
                    "execution_type": "llm"
                },
            )

        except AgentError as error:

            self._increment(
                "llm_failures"
            )

            self._emit(
                EventType.LLM_REQUEST_FAILED,
                request,
                error=error,
            )

            if span is not None:
                self.tracer.end_span(
                    span,
                    error=error.message,
                )

            raise

        except Exception as exc:

            error = AgentError(
                error_type=type(exc).__name__,
                message=(
                    f"LLM execution failed: "
                    f"{exc}"
                ),
                category=(
                    ErrorCategory.LLM_FAILURE
                ),
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                recoverable=True,
                source="AgentExecution",
                operation="execute_llm",
                original_exception=exc,
            )

            self._increment(
                "llm_failures"
            )

            self._emit(
                EventType.LLM_REQUEST_FAILED,
                request,
                error=error,
            )

            if span is not None:
                self.tracer.end_span(
                    span,
                    error=error.message,
                )

            raise error from exc
    def _build_tool_result(
        self,
        request: ExecutionRequest,
        result: Any,
    ) -> ExecutionResult:

        evidence: tuple[
            ToolResultRecord, ...
        ] = ()

        artifacts: tuple[
            ArtifactInfo, ...
        ] = ()

        output = result
        metadata: dict[str, Any] = {}

        if isinstance(result, dict):

            output = result.get(
                "output",
                result.get(
                    "result",
                    result,
                ),
            )

            raw_evidence = result.get(
                "evidence",
                (),
            )

            raw_artifacts = result.get(
                "artifacts",
                (),
            )

            if raw_evidence:

                evidence = tuple(
                    raw_evidence
                    if isinstance(
                        raw_evidence,
                        (list, tuple),
                    )
                    else [raw_evidence]
                )

            if raw_artifacts:

                artifacts = tuple(
                    raw_artifacts
                    if isinstance(
                        raw_artifacts,
                        (list, tuple),
                    )
                    else [raw_artifacts]
                )

            metadata = {
                key: value
                for key, value in result.items()
                if key not in {
                    "output",
                    "result",
                    "evidence",
                    "artifacts",
                }
            }
        if not evidence:

            evidence = (
                ToolResultRecord(
                    step_id=request.step_id,
                    tool_name=request.tool_name
                    or "unknown",
                    arguments=dict(
                        request.arguments
                    ),
                    result=output,
                    succeeded=True,
                    evidence_type="tool_output",
                    source=request.tool_name,
                ),
            )

        return ExecutionResult(
            success=True,
            output=output,
            evidence=evidence,
            artifacts=artifacts,
            step_id=request.step_id,
            tool_name=request.tool_name,
            metadata=metadata,
        )

    def _failure_result(
        self,
        request: ExecutionRequest,
        error: AgentError,
    ) -> ExecutionResult:

        self._emit(
            EventType.EXECUTION_FAILED,
            request,
            error=error,
        )

        return ExecutionResult(
            success=False,
            output=None,
            error=error,
            blocked=(
                error.category
                == ErrorCategory.APPROVAL_BLOCKED
            ),
            step_id=request.step_id,
            tool_name=request.tool_name,
            metadata={
                "execution_type": (
                    "tool"
                    if self._is_tool_step(request)
                    else "llm"
                    if self._is_llm_step(request)
                    else "unknown"
                )
            },
        )

    def _emit(
        self,
        event_type: EventType,
        request: ExecutionRequest | None,
        *,
        error: AgentError | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        if self.event_logger is None:
            return

        event_metadata: dict[str, Any] = {}

        if request is not None:

            event_metadata.update(
                {
                    "step_id": request.step_id,
                    "action": request.action,
                    "tool_name": request.tool_name,
                }
            )

        if metadata:
            event_metadata.update(
                metadata
            )

        if error is not None:

            event_metadata.update(
                {
                    "error_type": error.error_type,
                    "error_category": (
                        error.category.value
                    ),
                    "error_message": error.message,
                }
            )

        event = create_event(
            event_type=event_type,
            correlation_id=(
                request.correlation_id
                if request is not None
                and request.correlation_id is not None
                else self.correlation_id
            ),
            iteration=(
                request.iteration
                if request is not None
                else None
            ),
            metadata=event_metadata,
            error=(
                error.message
                if error is not None
                else None
            ),
        )

        self.event_logger.log(event)

    def _increment(
        self,
        name: str,
    ) -> None:

        if self.metrics is not None:
            self.metrics.increment(name)

    def _record_duration(
        self,
        name: str,
        duration: float,
    ) -> None:

        if self.metrics is not None:
            self.metrics.record_duration(
                name,
                duration,
            )

    @staticmethod
    def _decision_allowed(
        decision: Any,
    ) -> bool:

        return bool(
            getattr(
                decision,
                "allowed",
                False,
            )
        )

    @staticmethod
    def _approval_required(
        decision: Any,
    ) -> bool:

        return bool(
            getattr(
                decision,
                "approval_required",
                False,
            )
        )

    @staticmethod
    def _is_empty_result(
        result: Any,
    ) -> bool:

        if result is None:
            return True

        if isinstance(
            result,
            str,
        ):
            return not result.strip()

        return False

    @staticmethod
    def _is_tool_step(
        request: ExecutionRequest | None,
    ) -> bool:

        return (
            request is not None
            and request.step_type
            == StepType.TOOL
        )

    @staticmethod
    def _is_llm_step(
        request: ExecutionRequest | None,
    ) -> bool:

        return (
            request is not None
            and request.step_type
            == StepType.LLM
        )

    @staticmethod
    def _validate_request(
        request: ExecutionRequest,
    ) -> None:

        if not isinstance(
            request,
            ExecutionRequest,
        ):
            raise TypeError(
                "request must be an ExecutionRequest."
            )

        if request.step_id < 0:
            raise ValueError(
                "step_id must be >= 0."
            )

        if not request.action.strip():
            raise ValueError(
                "action cannot be empty."
            )

        if request.step_type == StepType.TOOL:
            if not request.tool_name:
                raise ValueError(
                    "TOOL execution requires tool_name."
                )

        elif request.step_type == StepType.LLM:
            if request.tool_name is not None:
                raise ValueError(
                    "LLM execution cannot specify tool_name."
                )

        else:
            raise ValueError(
                f"Unsupported step_type: "
                f"{request.step_type}"
            )