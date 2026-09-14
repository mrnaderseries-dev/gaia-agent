from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_agent.core.agent_execution import (
    AgentExecution,
    ExecutionRequest,
)
from gaia_agent.core.llm_executor import LLMExecutor
from gaia_agent.core.policies.approval import ApprovalPolicy
from gaia_agent.core.policies.execution import ExecutionPolicy
from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.planner.plan_schema import StepType
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
)
from gaia_agent.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry():
    registry = MagicMock(spec=ToolRegistry)
    return registry


@pytest.fixture
def execution_policy():
    policy = MagicMock(spec=ExecutionPolicy)
    policy.evaluate = AsyncMock(
        return_value=SimpleNamespace(
            allowed=True,
            message=None,
            reason=None,
        )
    )
    return policy


@pytest.fixture
def llm_executor():
    executor = MagicMock(spec=LLMExecutor)
    executor.execute = AsyncMock(return_value="Final answer")
    return executor


@pytest.fixture
def execution(
    tool_registry,
    execution_policy,
    llm_executor,
):
    return AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=None,
        approval_policy=None,
        llm_executor=llm_executor,
    )


def make_tool_request(
    *,
    step_id: int = 0,
    tool_name: str = "python_interpreter",
    arguments: dict | None = None,
    action: str = "Calculate the requested value",
    iteration: int = 1,
    metadata: dict | None = None,
):
    return ExecutionRequest(
        step_id=step_id,
        step_type=StepType.TOOL,
        action=action,
        tool_name=tool_name,
        arguments=arguments or {"expression": "2 + 2"},
        user_request="Calculate 2 + 2",
        iteration=iteration,
        metadata=metadata or {},
    )


def make_llm_request(
    *,
    step_id: int = 1,
    action: str = "Generate the final answer",
    iteration: int = 2,
    metadata: dict | None = None,
):
    return ExecutionRequest(
        step_id=step_id,
        step_type=StepType.LLM,
        action=action,
        tool_name=None,
        arguments={},
        user_request="What is 2 + 2?",
        context={"previous_result": "4"},
        iteration=iteration,
        metadata=metadata or {},
    )


def allowed_decision():
    return SimpleNamespace(
        allowed=True,
        message=None,
        reason=None,
    )


def blocked_decision():
    return SimpleNamespace(
        allowed=False,
        message="Execution is not allowed.",
        reason="policy_denied",
    )


def approval_not_required():
    return SimpleNamespace(
        approval_required=False,
        message=None,
        reason=None,
    )


def approval_required():
    return SimpleNamespace(
        approval_required=True,
        message="Human approval is required.",
        reason="high_risk_operation",
    )


@pytest.mark.asyncio
async def test_tool_execution_success(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute = MagicMock(return_value=4)

    tool_registry.get.return_value = tool

    request = make_tool_request()

    result = await execution.execute(request)

    assert result.success is True
    assert result.output == 4
    assert result.error is None
    assert result.blocked is False
    assert result.step_id == 0
    assert result.tool_name == "python_interpreter"

    tool_registry.get.assert_called_once_with(
        "python_interpreter"
    )
    tool.execute.assert_called_once_with(
        expression="2 + 2"
    )


@pytest.mark.asyncio
async def test_tool_arguments_are_preserved_exactly(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute = MagicMock(return_value="ok")

    tool_registry.get.return_value = tool

    arguments = {
        "expression": "sqrt(144)",
        "timeout": 10,
        "metadata": {
            "source": "planner",
        },
    }

    request = make_tool_request(
        arguments=arguments,
    )

    result = await execution.execute(request)

    assert result.success is True
    assert result.output == "ok"

    tool.execute.assert_called_once_with(
        expression="sqrt(144)",
        timeout=10,
        metadata={
            "source": "planner",
        },
    )


@pytest.mark.asyncio
async def test_tool_execution_creates_evidence(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute = MagicMock(return_value=4)

    tool_registry.get.return_value = tool

    request = make_tool_request(
        metadata={
            "run_id": "run-123",
            "attempt": 2,
            "plan_version": 3,
        }
    )

    result = await execution.execute(request)

    assert result.success is True
    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert evidence.step_id == request.step_id
    assert evidence.tool_name == request.tool_name
    assert evidence.result == 4
    assert evidence.succeeded is True
    assert evidence.run_id == "run-123"
    assert evidence.attempt_id == "2"
    assert evidence.plan_version == 3


@pytest.mark.asyncio
async def test_tool_result_can_provide_custom_evidence_and_artifacts(
    execution,
    tool_registry,
):
    tool = MagicMock()

    custom_evidence = object()
    custom_artifact = object()

    tool.execute = MagicMock(
        return_value={
            "output": "processed",
            "evidence": [custom_evidence],
            "artifacts": [custom_artifact],
            "source": "test-tool",
        }
    )

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is True
    assert result.output == "processed"

    assert result.evidence == (
        custom_evidence,
    )

    assert result.artifacts == (
        custom_artifact,
    )

    assert result.metadata == {
        "source": "test-tool",
    }


@pytest.mark.asyncio
async def test_async_tool_execution_is_supported(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute = AsyncMock(return_value=42)

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is True
    assert result.output == 42

    tool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_failure(
    execution,
    tool_registry,
):
    tool_registry.get.side_effect = KeyError(
        "unknown_tool"
    )

    request = make_tool_request(
        tool_name="unknown_tool"
    )

    result = await execution.execute(request)

    assert result.success is False
    assert result.output is None
    assert result.error is not None

    assert result.error.error_type == "ToolNotFound"
    assert (
        result.error.category
        == ErrorCategory.TOOL_NOT_FOUND
    )

    assert result.error.recoverable is True
    assert result.blocked is False


@pytest.mark.asyncio
async def test_tool_failure_is_converted_to_agent_error(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.side_effect = RuntimeError(
        "calculator crashed"
    )

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.error is not None

    assert result.error.error_type == "RuntimeError"
    assert (
        result.error.category
        == ErrorCategory.TOOL_EXECUTION_ERROR
    )
    assert result.error.retryable is True
    assert result.error.recoverable is True

    assert "calculator crashed" in (
        result.error.message
    )


@pytest.mark.asyncio
async def test_empty_tool_result_is_failure(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.return_value = ""

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.error is not None

    assert result.error.error_type == "EmptyResult"
    assert (
        result.error.category
        == ErrorCategory.EMPTY_RESULT
    )

    assert result.error.retryable is True
    assert result.error.recoverable is True


@pytest.mark.asyncio
async def test_none_tool_result_is_failure(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.return_value = None

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "EmptyResult"


@pytest.mark.asyncio
async def test_missing_tool_name_is_rejected(
    execution,
):
    request = ExecutionRequest(
        step_id=0,
        step_type=StepType.TOOL,
        action="Execute tool",
        tool_name=None,
        arguments={},
    )

    with pytest.raises(ValueError, match="tool_name"):
        await execution.execute(request)


@pytest.mark.asyncio
async def test_llm_execution_success(
    execution,
    llm_executor,
):
    request = make_llm_request()

    result = await execution.execute(request)

    assert result.success is True
    assert result.output == "Final answer"
    assert result.error is None
    assert result.step_id == 1
    assert result.metadata["execution_type"] == "llm"

    llm_executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_request_receives_correct_information(
    execution,
    llm_executor,
):
    request = make_llm_request(
        action="Answer the user's question",
        metadata={
            "run_id": "run-55",
            "plan_version": 2,
        },
    )

    await execution.execute(request)

    llm_request = (
        llm_executor.execute.await_args.args[0]
    )

    assert (
        llm_request.user_request
        == request.user_request
    )

    assert (
        llm_request.action
        == request.action
    )

    assert (
        llm_request.context
        == request.context
    )

    assert llm_request.metadata == {
        "run_id": "run-55",
        "plan_version": 2,
    }


@pytest.mark.asyncio
async def test_llm_empty_output_is_failure(
    execution,
    llm_executor,
):
    llm_executor.execute.return_value = "   "

    result = await execution.execute(
        make_llm_request()
    )

    assert result.success is False
    assert result.error is not None

    assert result.error.error_type == "EmptyResult"
    assert (
        result.error.category
        == ErrorCategory.EMPTY_RESULT
    )

    assert result.error.retryable is True
    assert result.error.recoverable is True


@pytest.mark.asyncio
async def test_llm_non_string_output_is_failure(
    execution,
    llm_executor,
):
    llm_executor.execute.return_value = {
        "answer": "4"
    }

    result = await execution.execute(
        make_llm_request()
    )

    assert result.success is False
    assert result.error is not None

    assert (
        result.error.error_type
        == "InvalidLLMOutput"
    )

    assert (
        result.error.category
        == ErrorCategory.LLM_OUTPUT_ERROR
    )


@pytest.mark.asyncio
async def test_llm_executor_exception_becomes_structured_failure(
    execution,
    llm_executor,
):
    llm_executor.execute.side_effect = RuntimeError(
        "LLM unavailable"
    )
    result = await execution.execute(
        make_llm_request()
    )

    assert result.success is False
    assert result.error is not None

    assert (
        result.error.error_type
        == "RuntimeError"
    )

    assert (
        result.error.category
        == ErrorCategory.LLM_FAILURE
    )

    assert result.error.retryable is True
    assert result.error.recoverable is True


@pytest.mark.asyncio
async def test_llm_request_cannot_have_tool_name(
    execution,
):
    request = ExecutionRequest(
        step_id=1,
        step_type=StepType.LLM,
        action="Generate answer",
        tool_name="python_interpreter",
    )

    with pytest.raises(
        ValueError,
        match="cannot specify tool_name",
    ):
        await execution.execute(request)


@pytest.mark.asyncio
async def test_execution_policy_is_called_before_tool_execution(
    execution,
    execution_policy,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    await execution.execute(
        make_tool_request()
    )

    execution_policy.evaluate.assert_awaited_once()

    policy_state = (
        execution_policy.evaluate.await_args.args[0]
    )

    assert (
        policy_state.tool_name
        == "python_interpreter"
    )

    assert (
        policy_state.action_name
        == "Calculate the requested value"
    )

    assert policy_state.arguments == {
        "expression": "2 + 2"
    }


@pytest.mark.asyncio
async def test_execution_policy_block_prevents_tool_execution(
    execution,
    execution_policy,
    tool_registry,
):
    execution_policy.evaluate.return_value = (
        blocked_decision()
    )

    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.blocked is True
    assert result.error is not None

    assert (
        result.error.error_type
        == "ExecutionBlocked"
    )

    assert (
        result.error.category
        == ErrorCategory.APPROVAL_BLOCKED
    )

    tool_registry.get.assert_not_called()
    tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_execution_policy_exception_becomes_structured_failure(
    execution,
    execution_policy,
    tool_registry,
):
    execution_policy.evaluate.side_effect = RuntimeError(
        "policy engine crashed"
    )

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.error is not None

    assert (
        result.error.error_type
        == "ExecutionPolicyFailure"
    )

    assert (
        result.error.category
        == ErrorCategory.INTERNAL
    )

    tool_registry.get.assert_not_called()


@pytest.mark.asyncio
async def test_risk_assessment_is_called_before_execution(
    tool_registry,
    execution_policy,
    llm_executor,
):
    risk_assessor = MagicMock(
        spec=RiskAssessor
    )

    risk_assessor.assess = AsyncMock(
        return_value=SimpleNamespace(
            risk_level="low"
        )
    )

    execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=None,
        llm_executor=llm_executor,
    )

    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    await execution.execute(
        make_tool_request()
    )

    risk_assessor.assess.assert_awaited_once()

    context = (
        risk_assessor.assess.await_args.args[0]
    )
    assert (
        context.action
        == "Calculate the requested value"
    )

    assert (
        context.tool_name
        == "python_interpreter"
    )

    assert context.arguments == {
        "expression": "2 + 2"
    }


@pytest.mark.asyncio
async def test_risk_assessment_failure_stops_execution(
    tool_registry,
    execution_policy,
    llm_executor,
):
    risk_assessor = MagicMock(
        spec=RiskAssessor
    )

    risk_assessor.assess = AsyncMock(
        side_effect=RuntimeError(
            "risk service unavailable"
        )
    )

    execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=None,
        llm_executor=llm_executor,
    )

    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.error is not None

    assert (
        result.error.error_type
        == "RiskAssessmentFailure"
    )

    tool_registry.get.assert_not_called()


@pytest.mark.asyncio
async def test_approval_policy_can_block_execution(
    tool_registry,
    execution_policy,
    llm_executor,
):
    risk_assessor = MagicMock(
        spec=RiskAssessor
    )

    risk_assessor.assess = AsyncMock(
        return_value=SimpleNamespace(
            risk_level="high"
        )
    )

    approval_policy = MagicMock(
        spec=ApprovalPolicy
    )

    approval_policy.evaluate = AsyncMock(
        return_value=approval_required()
    )

    execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=approval_policy,
        llm_executor=llm_executor,
    )

    tool = MagicMock()
    tool.execute.return_value = "dangerous"

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request(
            tool_name="dangerous_tool"
        )
    )

    assert result.success is False
    assert result.blocked is True
    assert result.error is not None

    assert (
        result.error.error_type
        == "ApprovalBlocked"
    )

    assert (
        result.error.category
        == ErrorCategory.APPROVAL_BLOCKED
    )

    tool_registry.get.assert_not_called()
    tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_approval_policy_allows_execution(
    tool_registry,
    execution_policy,
    llm_executor,
):
    risk_assessor = MagicMock(
        spec=RiskAssessor
    )

    risk_assessor.assess = AsyncMock(
        return_value=SimpleNamespace(
            risk_level="low"
        )
    )

    approval_policy = MagicMock(
        spec=ApprovalPolicy
    )

    approval_policy.evaluate = AsyncMock(
        return_value=approval_not_required()
    )

    execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=approval_policy,
        llm_executor=llm_executor,
    )

    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is True
    assert result.output == 4

    approval_policy.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_request_type_is_rejected(
    execution,
):
    with pytest.raises(
        TypeError,
        match="ExecutionRequest",
    ):
        await execution.execute(
            object()
        )


@pytest.mark.asyncio
async def test_negative_step_id_is_rejected(
    execution,
):
    request = make_tool_request(
        step_id=-1
    )

    with pytest.raises(
        ValueError,
        match="step_id",
    ):
        await execution.execute(request)


@pytest.mark.asyncio
async def test_empty_action_is_rejected(
    execution,
):
    request = make_tool_request(
        action="   "
    )

    with pytest.raises(
        ValueError,
        match="action",
    ):
        await execution.execute(request)


@pytest.mark.asyncio
async def test_unsupported_step_type_is_rejected(
    execution,
):
    request = ExecutionRequest(
        step_id=0,
        step_type="unsupported",
        action="Do something",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported step_type",
    ):
        await execution.execute(request)


@pytest.mark.asyncio
async def test_failure_result_contains_execution_type(
    execution,
    tool_registry,
):
    tool_registry.get.side_effect = KeyError(
        "missing"
    )

    result = await execution.execute(
        make_tool_request(
            tool_name="missing"
        )
    )

    assert result.success is False
    assert (
        result.metadata["execution_type"]
        == "tool"
    )


@pytest.mark.asyncio
async def test_llm_failure_result_contains_execution_type(
    execution,
    llm_executor,
):
    llm_executor.execute.return_value = ""

    result = await execution.execute(
        make_llm_request()
    )

    assert result.success is False
    assert (
        result.metadata["execution_type"]
        == "llm"
    )


@pytest.mark.asyncio
async def test_request_correlation_id_is_accepted(
    execution,
):
    from uuid import uuid4

    correlation_id = uuid4()

    request = ExecutionRequest(
        step_id=1,
        step_type=StepType.LLM,
        action="Generate answer",
        correlation_id=correlation_id,
    )

    result = await execution.execute(request)

    assert result.success is True


@pytest.mark.asyncio
async def test_execution_metadata_does_not_mutate_request(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    metadata = {
        "run_id": "abc",
        "attempt": 1,
    }

    request = make_tool_request(
        metadata=metadata
    )

    original_metadata = dict(
        request.metadata
    )

    await execution.execute(request)

    assert request.metadata == original_metadata


@pytest.mark.asyncio
async def test_tool_then_llm_execution_sequence(
    execution,
    tool_registry,
    llm_executor,
):
    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    tool_result = await execution.execute(
        make_tool_request(
            step_id=0,
            metadata={
                "run_id": "run-1",
                "attempt": 1,
                "plan_version": 1,
            },
        )
    )

    assert tool_result.success is True
    assert tool_result.output == 4

    llm_executor.execute.return_value = (
        "The answer is 4."
    )
    llm_request = make_llm_request(
        step_id=1,
        metadata={
            "run_id": "run-1",
            "attempt": 1,
            "plan_version": 1,
        },
    )

    llm_request = ExecutionRequest(
        step_id=llm_request.step_id,
        step_type=llm_request.step_type,
        action=llm_request.action,
        tool_name=None,
        arguments={},
        user_request=llm_request.user_request,
        context={
            "tool_output": tool_result.output,
            "evidence": tool_result.evidence,
        },
        iteration=llm_request.iteration,
        metadata=llm_request.metadata,
    )

    llm_result = await execution.execute(
        llm_request
    )

    assert llm_result.success is True
    assert (
        llm_result.output
        == "The answer is 4."
    )

    llm_call = (
        llm_executor.execute.await_args.args[0]
    )

    assert llm_call.context["tool_output"] == 4


@pytest.mark.asyncio
async def test_agent_execution_does_not_mutate_execution_request(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.return_value = 4

    tool_registry.get.return_value = tool

    request = make_tool_request()

    original = {
        "step_id": request.step_id,
        "step_type": request.step_type,
        "action": request.action,
        "tool_name": request.tool_name,
        "arguments": dict(request.arguments),
        "user_request": request.user_request,
        "iteration": request.iteration,
        "metadata": dict(request.metadata),
    }

    await execution.execute(request)

    assert request.step_id == original["step_id"]
    assert request.step_type == original["step_type"]
    assert request.action == original["action"]
    assert request.tool_name == original["tool_name"]
    assert request.arguments == original["arguments"]
    assert request.user_request == original["user_request"]
    assert request.iteration == original["iteration"]
    assert request.metadata == original["metadata"]


@pytest.mark.asyncio
async def test_execution_failure_never_returns_success(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.side_effect = RuntimeError(
        "failure"
    )

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is False
    assert result.error is not None
    assert result.output is None


@pytest.mark.asyncio
async def test_successful_execution_has_no_error(
    execution,
    tool_registry,
):
    tool = MagicMock()
    tool.execute.return_value = "ok"

    tool_registry.get.return_value = tool

    result = await execution.execute(
        make_tool_request()
    )

    assert result.success is True
    assert result.error is None