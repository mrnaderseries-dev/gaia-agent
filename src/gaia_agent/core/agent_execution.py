from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from gaia_agent.core.evidence import ArtifactInfo, ToolResultRecord
from gaia_agent.core.llm_executor import LLMExecutionRequest, LLMExecutor
from gaia_agent.core.policies.approval import ApprovalPolicy, ApprovalState
from gaia_agent.core.policies.execution import ExecutionPolicy, ExecutionState
from gaia_agent.core.risk.assessor import RiskAssessor, RiskContext
from gaia_agent.observability.events import EventLogger, EventType, create_event
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
    arguments: dict[str, Any] = field(default_factory=dict)
    user_request: str = ""
    context: Any = None
    iteration: int = 0
    correlation_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)


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
        error_handler: ErrorHandler,
        correlation_id: UUID,
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
        self.error_handler = error_handler
        self.correlation_id = correlation_id

    async def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        try:
            self._validate_request(request)

            await self._check_execution_policy(request)

            risk_assessment = await self._check_risk(request)

            await self._check_approval(
                request,
                risk_assessment,
            )

            if self._is_tool_step(request):
                return await self._execute_tool(request)

            if self._is_llm_step(request):
                return await self._execute_llm(request)

            raise AgentError(
                error_type="UnsupportedStepType",
                message=f"Unsupported step type: {request.step_type}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="execute",
                details={
                    "step_id": request.step_id,
                    "step_type": str(request.step_type),
                },
            )

        except AgentError as error:
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

            if not isinstance(error, AgentError):
                error = AgentError(
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                    category=ErrorCategory.INTERNAL,
                    severity=ErrorSeverity.HIGH,
                    retryable=False,
                    recoverable=False,
                    source="AgentExecution",
                    operation="execute",
                    original_exception=exc,
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
            arguments=request.arguments,
            blocked=False,
        )

        try:
            decision = self.execution_policy.evaluate(
                execution_state
            )

            if inspect.isawaitable(decision):
                decision = await decision

        except AgentError:
            raise

        except Exception as exc:
            raise AgentError(
                error_type="ExecutionPolicyFailure",
                message=f"Execution policy failed: {exc}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_execution_policy",
                original_exception=exc,
            ) from exc

        if not self._decision_allowed(decision):
            raise AgentError(
                error_type="ExecutionBlocked",
                message=(
                    "Execution policy blocked "
                    "the requested operation."
                ),
                category=ErrorCategory.APPROVAL_BLOCKED,
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_execution_policy",
                details={
                    "step_id": request.step_id,
                    "tool_name": request.tool_name,
                    "action": request.action,
                    "decision": repr(decision),
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
            arguments=dict(request.arguments),
        )

        try:
            result = self.risk_assessor.assess(
                context
            )

            if inspect.isawaitable(result):
                result = await result

            return result

        except AgentError:
            raise

        except Exception as exc:
            raise AgentError(
                error_type="RiskAssessmentFailure",
                message=f"Risk assessment failed: {exc}",
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
        if self.approval_policy is None:
            return

        if risk_assessment is None:
            return

        approval_state = ApprovalState(
            action_name=request.action,
            tool_name=request.tool_name,
            risk_assessment=risk_assessment,
        )

        try:
            decision = self.approval_policy.evaluate(
                approval_state
            )

            if inspect.isawaitable(decision):
                decision = await decision

            required = self._approval_required(
                decision
            )

        except AgentError:
            raise

        except Exception as exc:
            raise AgentError(
                error_type="ApprovalPolicyFailure",
                message=f"Approval policy failed: {exc}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_approval",
                original_exception=exc,
            ) from exc

        if required:
            raise AgentError(
                error_type="ApprovalBlocked",
                message=(
                    "Approval policy requires "
                    "human approval."
                ),
                category=ErrorCategory.APPROVAL_BLOCKED,
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_approval",
                details={
                    "step_id": request.step_id,
                    "action": request.action,
                    "tool_name": request.tool_name,
                    "risk": repr(risk_assessment),
                    "decision": repr(decision),
                },
            )

    async def _execute_tool(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        tool_name = request.tool_name

        if not tool_name:
            raise AgentError(
                error_type="MissingToolName",
                message=(
                    "Tool step does not contain "
                    "a tool name."
                ),
                category=ErrorCategory.TOOL_ARGUMENT_ERROR,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="execute_tool",
            )

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
                "tool_name": tool_name,
            },
        )

        self._increment("tool_requests")

        try:
            try:
                tool = self.tool_registry.get(
                    tool_name
                )

            except KeyError as exc:
                raise AgentError(
                    error_type="ToolNotFound",
                    message=(
                        f"Tool '{tool_name}' "
                        "is not registered."
                    ),
                    category=ErrorCategory.TOOL_NOT_FOUND,
                    severity=ErrorSeverity.MEDIUM,
                    retryable=False,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_tool",
                    details={
                        "tool_name": tool_name,
                    },
                    original_exception=exc,
                ) from exc

            result = tool.execute(
                dict(request.arguments)
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
                    category=ErrorCategory.EMPTY_RESULT,
                    severity=ErrorSeverity.MEDIUM,
                    retryable=False,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_tool",
                    details={
                        "tool_name": tool_name,
                        "step_id": request.step_id,
                    },
                )

            execution_result = self._build_tool_result(
                request,
                result,
            )

            self._increment("tool_successes")

            self._emit(
                EventType.TOOL_COMPLETED,
                request,
                metadata={
                    "tool_name": tool_name,
                },
            )

            self.tracer.end_span(span)

            return execution_result

        except AgentError as error:
            self._increment("tool_failures")

            self._emit(
                EventType.TOOL_FAILED,
                request,
                error=error,
                metadata={
                    "tool_name": tool_name,
                },
            )

            self.tracer.end_span(
                span,
                error=error,
            )

            raise

        except Exception as exc:
            error = AgentError(
                error_type=type(exc).__name__,
                message=(
                    f"Tool '{tool_name}' "
                    f"execution failed: {exc}"
                ),
                category=ErrorCategory.TOOL_EXECUTION_ERROR,
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="execute_tool",
                details={
                    "tool_name": tool_name,
                    "step_id": request.step_id,
                },
                original_exception=exc,
            )

            self._increment("tool_failures")

            self._emit(
                EventType.TOOL_FAILED,
                request,
                error=error,
                metadata={
                    "tool_name": tool_name,
                },
            )

            self.tracer.end_span(
                span,
                error=error,
            )

            raise error from exc

    async def _execute_llm(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
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

        self._increment("llm_requests")

        try:
            llm_request = LLMExecutionRequest(
                user_request=request.user_request,
                action=request.action,
                context=request.context,
                metadata=dict(request.metadata),
            )

            output = await self.llm_executor.execute(
                llm_request
            )

            if not isinstance(output, str):
                raise AgentError(
                    error_type="InvalidLLMOutput",
                    message=(
                        "LLM executor returned "
                        "a non-string result."
                    ),
                    category=ErrorCategory.LLM_OUTPUT_ERROR,
                    severity=ErrorSeverity.MEDIUM,
                    retryable=True,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_llm",
                    details={
                        "step_id": request.step_id,
                        "output_type": type(output).__name__,
                    },
                )

            if not output.strip():
                raise AgentError(
                    error_type="EmptyResult",
                    message="LLM returned an empty result.",
                    category=ErrorCategory.EMPTY_RESULT,
                    severity=ErrorSeverity.MEDIUM,
                    retryable=True,
                    recoverable=True,
                    source="AgentExecution",
                    operation="execute_llm",
                    details={
                        "step_id": request.step_id,
                    },
                )

            self._increment("llm_successes")

            self._emit(
                EventType.LLM_REQUEST_COMPLETED,
                request,
            )

            self.tracer.end_span(span)

            return ExecutionResult(
                success=True,
                output=output,
                step_id=request.step_id,
                metadata={
                    "execution_type": "llm",
                },
            )

        except AgentError as error:
            self._increment("llm_failures")

            self._emit(
                EventType.LLM_REQUEST_FAILED,
                request,
                error=error,
            )

            self.tracer.end_span(
                span,
                error=error,
            )

            raise

        except Exception as exc:
            error = AgentError(
                error_type=type(exc).__name__,
                message=(
                    f"LLM execution failed: {exc}"
                ),
                category=ErrorCategory.LLM_FAILURE,
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                recoverable=False,
                source="AgentExecution",
                operation="execute_llm",
                original_exception=exc,
            )

            self._increment("llm_failures")

            self._emit(
                EventType.LLM_REQUEST_FAILED,
                request,
                error=error,
            )

            self.tracer.end_span(
                span,
                error=error,
            )

            raise error from exc

    def _build_tool_result(
        self,
        request: ExecutionRequest,
        result: Any,
    ) -> ExecutionResult:
        evidence: tuple[ToolResultRecord, ...] = ()
        artifacts: tuple[ArtifactInfo, ...] = ()
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
        request: ExecutionRequest | None,
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
                is ErrorCategory.APPROVAL_BLOCKED
            ),
            step_id=request.step_id if isinstance(request, ExecutionRequest) else None,
            tool_name=request.tool_name if isinstance(request, ExecutionRequest) else None,
            metadata={
                "execution_type": (
                    "tool"
                    if self._is_tool_step(request)
                    else "llm"
                    if self._is_llm_step(request)
                    else "unknown"
                ),
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

        if isinstance(request, ExecutionRequest):
            event_metadata.update(
                {
                    "step_id": request.step_id,
                    "action": request.action,
                }
            )

        if metadata:
            event_metadata.update(metadata)

        if error is not None:
            event_metadata["error_type"] = error.error_type
            event_metadata["error_category"] = error.category.value
            event_metadata["error_message"] = error.message

        event = create_event(
            event_type=event_type,
            correlation_id=(
                request.correlation_id
                if isinstance(request, ExecutionRequest) and request.correlation_id is not None
                else self.correlation_id
            ),
            iteration=(
                request.iteration
                if isinstance(request, ExecutionRequest)
                else 0
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
        if self.metrics is None:
            return

        increment = getattr(
            self.metrics,
            "increment",
            None,
        )

        if increment is not None:
            increment(name)

    @staticmethod
    def _decision_allowed(
        decision: Any,
    ) -> bool:
        if decision is None:
            return True

        if isinstance(decision, bool):
            return decision

        allowed = getattr(
            decision,
            "allowed",
            None,
        )

        if isinstance(allowed, bool):
            return allowed

        if allowed is None:
            return True

        return False

    @staticmethod
    def _approval_required(
        decision: Any,
    ) -> bool:
        if decision is None:
            return False

        if isinstance(decision, bool):
            return decision

        approval_required = getattr(
            decision,
            "approval_required",
            None,
        )

        if approval_required is None:
            return False

        if not isinstance(
            approval_required,
            bool,
        ):
            raise AgentError(
                error_type="InvalidApprovalDecision",
                message=(
                    "Approval decision must expose "
                    "a boolean approval_required value."
                ),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="check_approval",
                details={
                    "decision_type": (
                        type(decision).__name__
                    ),
                    "approval_required_type": (
                        type(approval_required).__name__
                    ),
                },
            )

        return approval_required

    @staticmethod
    def _is_empty_result(
        result: Any,
    ) -> bool:
        if result is None:
            return True

        if isinstance(result, str):
            return not result.strip()

        return False

    @staticmethod
    def _is_tool_step(
        request: ExecutionRequest | None,
    ) -> bool:
        if not isinstance(request, ExecutionRequest):
            return False

        value = getattr(
            request.step_type,
            "value",
            request.step_type,
        )

        return str(value).lower() == "tool"

    @staticmethod
    def _is_llm_step(
        request: ExecutionRequest | None,
    ) -> bool:
        if not isinstance(request, ExecutionRequest):
            return False

        value = getattr(
            request.step_type,
            "value",
            request.step_type,
        )

        return str(value).lower() == "llm"

    @staticmethod
    def _validate_request(
        request: ExecutionRequest,
    ) -> None:
        if not isinstance(
            request,
            ExecutionRequest,
        ):
            raise AgentError(
                error_type="InvalidExecutionRequest",
                message="Expected ExecutionRequest.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="validate_request",
            )

        if not isinstance(
            request.step_id,
            int,
        ):
            raise AgentError(
                error_type="InvalidStepId",
                message="step_id must be an integer.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="validate_request",
            )

        if request.step_id < 0:
            raise AgentError(
                error_type="InvalidStepId",
                message="step_id cannot be negative.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="validate_request",
            )

        if not isinstance(
            request.step_type,
            StepType,
        ):
            raise AgentError(
                error_type="InvalidStepType",
                message="step_type must be a StepType.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="AgentExecution",
                operation="validate_request",
            )

        if not isinstance(
            request.action,
            str,
        ):
            raise AgentError(
                error_type="InvalidAction",
                message="Execution action must be a string.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="validate_request",
            )

        if not request.action.strip():
            raise AgentError(
                error_type="EmptyAction",
                message="Execution action cannot be empty.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="validate_request",
            )

        if not isinstance(
            request.arguments,
            dict,
        ):
            raise AgentError(
                error_type="InvalidArguments",
                message=(
                    "Execution arguments must "
                    "be a dictionary."
                ),
                category=ErrorCategory.TOOL_ARGUMENT_ERROR,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="validate_request",
            )

        if not isinstance(
            request.metadata,
            dict,
        ):
            raise AgentError(
                error_type="InvalidMetadata",
                message="Execution metadata must be a dictionary.",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=True,
                source="AgentExecution",
                operation="validate_request",
            )

        if request.step_type is StepType.TOOL:
            if not isinstance(
                request.tool_name,
                str,
            ):
                raise AgentError(
                    error_type="MissingToolName",
                    message="Tool step requires a tool name.",
                    category=ErrorCategory.TOOL_ARGUMENT_ERROR,
                    severity=ErrorSeverity.HIGH,
                    retryable=False,
                    recoverable=True,
                    source="AgentExecution",
                    operation="validate_request",
                )

            if not request.tool_name.strip():
                raise AgentError(
                    error_type="EmptyToolName",
                    message="Tool name cannot be empty.",
                    category=ErrorCategory.TOOL_ARGUMENT_ERROR,
                    severity=ErrorSeverity.HIGH,
                    retryable=False,
                    recoverable=True,
                    source="AgentExecution",
                    operation="validate_request",
                )

        if request.step_type is StepType.LLM:
            if request.tool_name is not None:
                raise AgentError(
                    error_type="InvalidLLMToolBinding",
                    message=(
                        "LLM execution cannot specify "
                        "a tool name."
                    ),
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    retryable=False,
                    recoverable=False,
                    source="AgentExecution",
                    operation="validate_request",
                )