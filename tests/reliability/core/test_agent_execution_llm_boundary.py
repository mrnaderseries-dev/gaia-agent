from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from gaia_agent.core.agent_execution import (
    AgentExecution,
    ExecutionRequest,
)
from gaia_agent.core.llm_executor import LLMExecutionRequest
from gaia_agent.planner.plan_schema import StepType
from gaia_agent.reliability.error_handler import ErrorHandler


def build_execution(
    llm_executor: AsyncMock,
) -> AgentExecution:
    execution_policy = Mock()
    execution_policy.evaluate.return_value = Mock(
        allowed=True
    )

    risk_assessor = None
    approval_policy = None

    event_logger = Mock()
    metrics = Mock()
    tracer = Mock()
    token_tracker = Mock()
    error_handler = ErrorHandler()

    tracer.start_span.return_value = Mock()

    return AgentExecution(
        tool_registry=Mock(),
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=approval_policy,
        llm_executor=llm_executor,
        event_logger=event_logger,
        metrics=metrics,
        tracer=tracer,
        token_tracker=token_tracker,
        error_handler=error_handler,
        correlation_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_agent_execution_llm_boundary_never_leaks_agent_state():
    llm_executor = AsyncMock()
    llm_executor.execute.return_value = "final answer"

    execution = build_execution(llm_executor)

    request = ExecutionRequest(
        step_id=7,
        step_type=StepType.LLM,
        action="Generate the final answer",
        user_request="What is 2 + 2?",
        context={"important": "context"},
        iteration=3,
        metadata={"correlation": "test"},
    )

    result = await execution.execute(request)

    assert result.success is True
    assert result.output == "final answer"

    llm_executor.execute.assert_awaited_once()

    forwarded_request = (
        llm_executor.execute.await_args.args[0]
    )

    assert isinstance(
        forwarded_request,
        LLMExecutionRequest,
    )

    assert (
        forwarded_request.user_request
        == request.user_request
    )

    assert (
        forwarded_request.action
        == request.action
    )

    assert (
        forwarded_request.context
        == request.context
    )

    assert (
        forwarded_request.metadata
        == request.metadata
    )

    assert not hasattr(
        forwarded_request,
        "phase",
    )

    assert not hasattr(
        forwarded_request,
        "current_step",
    )

    assert not hasattr(
        forwarded_request,
        "completed_steps",
    )

    assert not hasattr(
        forwarded_request,
        "tool_result",
    )

    assert not hasattr(
        forwarded_request,
        "final_answer",
    )


@pytest.mark.asyncio
async def test_agent_execution_llm_boundary_preserves_request_identity_without_aliasing():
    llm_executor = AsyncMock()
    llm_executor.execute.return_value = "ok"

    execution = build_execution(llm_executor)

    context = {
        "messages": ["hello"],
        "retrieved": ["evidence"],
    }

    metadata = {
        "trace_id": "abc",
        "iteration": 4,
    }

    request = ExecutionRequest(
        step_id=10,
        step_type=StepType.LLM,
        action="Answer",
        user_request="Explain RAG",
        context=context,
        metadata=metadata,
    )

    await execution.execute(request)

    forwarded = (
        llm_executor.execute.await_args.args[0]
    )

    assert forwarded.context is request.context
    assert forwarded.metadata == request.metadata
    assert forwarded.metadata is not request.metadata

    assert forwarded.user_request == "Explain RAG"
    assert forwarded.action == "Answer"


@pytest.mark.asyncio
async def test_agent_execution_rejects_non_execution_request():
    llm_executor = AsyncMock()

    execution = build_execution(llm_executor)

    fake_state = object()

    result = await execution.execute(fake_state)

    assert result.success is False
    assert result.error is not None

    llm_executor.execute.assert_not_awaited()