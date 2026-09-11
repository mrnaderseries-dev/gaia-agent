from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_agent.core.agent_execution import AgentExecution, ExecutionRequest
from gaia_agent.core.llm_executor import LLMExecutionRequest
from gaia_agent.core.policies.execution import ExecutionDecision
from gaia_agent.planner.plan_schema import StepType


def build_execution(
    *,
    llm_executor=None,
    tool_registry=None,
    risk_assessor=None,
):
    execution_policy = MagicMock()
    execution_policy.evaluate.return_value = ExecutionDecision.allow()

    approval_policy = MagicMock()
    approval_policy.evaluate.return_value = False

    event_logger = MagicMock()
    metrics = MagicMock()
    tracer = MagicMock()
    token_tracker = MagicMock()
    error_handler = MagicMock()

    if llm_executor is None:
        llm_executor = AsyncMock()

    if tool_registry is None:
        tool_registry = MagicMock()

    if risk_assessor is None:
        risk_assessor = MagicMock()

    risk_assessor.assess = AsyncMock(return_value=None)

    return AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=approval_policy,
        llm_executor=llm_executor,
        event_logger=event_logger,
        metrics=metrics,
        tracer=tracer,
        token_tracker=token_tracker,
        error_handler=error_handler,
        correlation_id="architectural-test",
    )


@pytest.mark.asyncio
async def test_agent_execution_preserves_cross_layer_architectural_contract():
    llm_executor = AsyncMock()
    llm_executor.execute.return_value = "final answer"

    tool_registry = MagicMock()
    risk_assessor = MagicMock()
    risk_assessor.assess = AsyncMock(return_value=None)

    execution = build_execution(
        llm_executor=llm_executor,
        tool_registry=tool_registry,
        risk_assessor=risk_assessor,
    )

    context = {
        "messages": ["question"],
        "evidence": ["evidence-1"],
    }

    metadata = {
        "trace_id": "trace-123",
        "request_id": "request-123",
    }

    request = ExecutionRequest(
        step_id=7,
        step_type=StepType.LLM,
        action="Generate the final answer",
        user_request="Explain how RAG works.",
        context=context,
        iteration=3,
        metadata=metadata,
    )

    result = await execution.execute(request)

    assert result.success is True
    assert result.output == "final answer"

    llm_executor.execute.assert_awaited_once()
    tool_registry.execute.assert_not_called()

    forwarded = llm_executor.execute.await_args.args[0]

    assert isinstance(forwarded, LLMExecutionRequest)
    assert forwarded.user_request == request.user_request
    assert forwarded.action == request.action
    assert forwarded.context is request.context
    assert forwarded.metadata == request.metadata

    assert not hasattr(forwarded, "phase")
    assert not hasattr(forwarded, "current_step")
    assert not hasattr(forwarded, "completed_steps")
    assert not hasattr(forwarded, "tool_result")
    assert not hasattr(forwarded, "final_answer")
    assert not hasattr(forwarded, "retry_count")
    assert not hasattr(forwarded, "recovery_attempted")
    assert not hasattr(forwarded, "verification_attempts")

    assert risk_assessor.assess.await_count == 1

    risk_context = risk_assessor.assess.await_args.args[0]

    assert risk_context.action == request.action
    assert risk_context.tool_name == request.tool_name
    assert risk_context.arguments == request.arguments

    assert not hasattr(risk_context, "phase")
    assert not hasattr(risk_context, "current_step")
    assert not hasattr(risk_context, "completed_steps")
    assert not hasattr(risk_context, "final_answer")

    assert execution.correlation_id == "architectural-test"


@pytest.mark.asyncio
async def test_agent_execution_rejects_invalid_execution_contract_without_side_effects():
    llm_executor = AsyncMock()
    tool_registry = MagicMock()
    risk_assessor = MagicMock()
    risk_assessor.assess = AsyncMock(return_value=None)

    execution = build_execution(
        llm_executor=llm_executor,
        tool_registry=tool_registry,
        risk_assessor=risk_assessor,
    )

    invalid_requests = [
        None,
        object(),
        {},
        {"step_type": StepType.LLM},
    ]

    for invalid_request in invalid_requests:
        result = await execution.execute(invalid_request)

        assert result.success is False
        assert result.output is None
        assert result.error is not None

    llm_executor.execute.assert_not_awaited()
    tool_registry.execute.assert_not_called()
    risk_assessor.assess.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_execution_does_not_retry_or_recover_inside_execution_boundary():
    llm_executor = AsyncMock()
    llm_executor.execute.side_effect = RuntimeError("LLM failed")

    execution = build_execution(
        llm_executor=llm_executor,
    )

    request = ExecutionRequest(
        step_id=8,
        step_type=StepType.LLM,
        action="Answer",
        user_request="What is RAG?",
        context={},
        metadata={},
    )

    result = await execution.execute(request)

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert llm_executor.execute.await_count == 1

    assert not hasattr(execution, "retry")
    assert not hasattr(execution, "recover")
    assert not hasattr(execution, "replan")

    assert not hasattr(request, "phase")
    assert not hasattr(request, "retry_count")
    assert not hasattr(request, "recovery_attempted")


@pytest.mark.asyncio
async def test_agent_execution_keeps_concurrent_requests_isolated():
    llm_executor = AsyncMock()

    async def generate(request):
        return f"answer:{request.action}"

    llm_executor.execute.side_effect = generate

    execution = build_execution(
        llm_executor=llm_executor,
    )

    request_a = ExecutionRequest(
        step_id=1,
        step_type=StepType.LLM,
        action="Answer A",
        user_request="Question A",
        context={"request": "A"},
        metadata={"id": "A"},
    )

    request_b = ExecutionRequest(
        step_id=2,
        step_type=StepType.LLM,
        action="Answer B",
        user_request="Question B",
        context={"request": "B"},
        metadata={"id": "B"},
    )

    result_a, result_b = await __import__("asyncio").gather(
        execution.execute(request_a),
        execution.execute(request_b),
    )

    assert result_a.success is True
    assert result_b.success is True

    assert result_a.output == "answer:Answer A"
    assert result_b.output == "answer:Answer B"

    assert llm_executor.execute.await_count == 2

    forwarded_a = llm_executor.execute.await_args_list[0].args[0]
    forwarded_b = llm_executor.execute.await_args_list[1].args[0]

    assert forwarded_a.context is request_a.context
    assert forwarded_b.context is request_b.context

    assert forwarded_a.metadata is not request_a.metadata
    assert forwarded_b.metadata is not request_b.metadata

    assert forwarded_a.user_request != forwarded_b.user_request
    assert forwarded_a.action != forwarded_b.action


@pytest.mark.asyncio
async def test_agent_execution_failure_does_not_fake_success():
    llm_executor = AsyncMock()
    llm_executor.execute.side_effect = RuntimeError("model unavailable")

    execution = build_execution(
        llm_executor=llm_executor,
    )

    request = ExecutionRequest(
        step_id=11,
        step_type=StepType.LLM,
        action="Generate answer",
        user_request="Explain agents.",
        context={},
        metadata={},
    )

    result = await execution.execute(request)

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert result.success is not True
    assert llm_executor.execute.await_count == 1