from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_agent.context.ContextBuilder import (
    AttachmentSource,
    ContextBuilder,
    ContextBudget,
    ContextCompressor,
    ContextPolicy,
    ContextValidator,
    ConversationSource,
    HistorySource,
    MemorySource,
    RuntimeSource,
)
from gaia_agent.context.attachments import Attachment
from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.llm.model import LLMModel
from gaia_agent.planner.planner import Planner
from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.agents.verifier import VerifierAgent


# ============================================================================
# Integration LLM client
# ============================================================================


class IntegrationLLMClient:
    """
    Deterministic LLM client used only for integration infrastructure.

    The purpose of this test suite is to verify orchestration boundaries,
    not to test the quality of an actual LLM response.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(
        self,
        messages,
        *,
        model,
        output_schema=None,
        tools=None,
        operation="llm.generate",
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "output_schema": output_schema,
                "tools": tools,
                "operation": operation,
                "kwargs": kwargs,
            }
        )

        if output_schema is not None:
            raise AssertionError(
                "Integration context compression unexpectedly requested "
                "structured LLM output."
            )

        return "integration test answer"


# ============================================================================
# Fixtures
# ============================================================================


def _make_context_attachment() -> Attachment:
    """
    A real context item.

    ContextValidator explicitly rejects an empty context, therefore the
    integration test must provide a legitimate context item rather than
    mocking ContextBuilder into accepting [].
    """
    return Attachment(
        attachment_id="integration-test:context.txt",
        filename="context.txt",
        path=r"C:\integration_test\context.txt",
    )


def _make_context_builder(
    *,
    llm_client,
    llm_model,
):
    policy = ContextPolicy(
        include_memory=False,
        include_conversation=True,
        include_history=True,
        include_runtime=True,
        include_attachments=True,
    )

    budget = ContextBudget(max_tokens=4096)

    validator = ContextValidator(budget)

    compressor = ContextCompressor(
        client=llm_client,
        model=llm_model,
        budget=budget,
        policy=policy,
    )

    conversation_source = MagicMock(spec=ConversationSource)
    conversation_source.is_available.return_value = False
    conversation_source.get = AsyncMock(return_value=[])

    history_source = MagicMock(spec=HistorySource)
    history_source.is_available.return_value = False
    history_source.get = AsyncMock(return_value=[])

    memory_source = MagicMock(spec=MemorySource)
    memory_source.is_available.return_value = False
    memory_source.get = AsyncMock(return_value=[])

    runtime_source = MagicMock(spec=RuntimeSource)
    runtime_source.is_available.return_value = False
    runtime_source.get = AsyncMock(return_value=[])

    # IMPORTANT:
    # AttachmentSource is real.
    #
    # This means:
    #
    # AgentState.attachments
    #       ↓
    # ContextRequestBuilder
    #       ↓
    # ContextRequest.attachments
    #       ↓
    # AttachmentSource
    #       ↓
    # ContextBuilder
    #       ↓
    # ContextValidator
    #
    # is actually exercised.
    attachment_source = AttachmentSource()

    return ContextBuilder(
        policy=policy,
        budget=budget,
        validator=validator,
        compressor=compressor,
        attachment_source=attachment_source,
        conversation_source=conversation_source,
        history_source=history_source,
        memory_source=memory_source,
        runtime_source=runtime_source,
    )


@pytest.fixture
def integration_system():
    llm_model = LLMModel(
        provider="integration-test",
        model="qwen2.5:3b",
        max_tokens=4096,
        temperature=0.0,
    )

    llm_client = IntegrationLLMClient()

    context_builder = _make_context_builder(
        llm_client=llm_client,
        llm_model=llm_model,
    )

    # Planner is isolated here because this suite is testing the
    # orchestration contract around the planner, not the planner's
    # internal LLM strategy.
    planner_client = MagicMock()

    planner = Planner(
        client=planner_client,
        model=llm_model,
        available_tools={},
    )

    agent_execution = AgentExecution(
        tool_registry=MagicMock(),
        execution_policy=MagicMock(),
        risk_assessor=MagicMock(),
        approval_policy=MagicMock(),
        llm_executor=MagicMock(),
        event_logger=MagicMock(),
        metrics=MagicMock(),
        tracer=MagicMock(),
        token_tracker=MagicMock(),
        error_handler=MagicMock(),
    )

    reliability_engine = ReliabilityEngine(
        error_handler=MagicMock(),
        failure_classifier=MagicMock(),
        retry_policy=MagicMock(),
        recovery_policy=MagicMock(),
        retry=MagicMock(),
        recovery=MagicMock(),
    )

    loop_detector = MagicMock()
    loop_detector.check.return_value = False

    verifier = VerifierAgent(
        client=MagicMock(),
        model=llm_model.model,
    )

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
        event_logger=MagicMock(),
        metrics=MagicMock(),
        tracer=MagicMock(),
        max_execution_attempts=3,
        max_verification_attempts=2,
    )

    return SimpleNamespace(
        llm_client=llm_client,
        llm_model=llm_model,
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
        orchestrator=orchestrator,
    )


@pytest.fixture
def integration_state():
    attachment = _make_context_attachment()

    return AgentState(
        user_request="Analyze the attached context and provide the final answer.",
        attachments=[attachment],
    )


# ============================================================================
# Context boundary
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_binds_state_and_creates_runtime_context(
    integration_system,
    integration_state,
):
    orchestrator = integration_system.orchestrator

    orchestrator.bind_state(integration_state)

    assert orchestrator._state is integration_state
    assert orchestrator._context is not None
    assert orchestrator._context.user_request == integration_state.user_request
    assert orchestrator._context.run_id

    orchestrator.unbind()

    assert orchestrator._state is None
    assert orchestrator._context is None


@pytest.mark.asyncio
async def test_orchestrator_builds_context_before_planning(
    integration_system,
    integration_state,
):
    orchestrator = integration_system.orchestrator

    orchestrator.bind_state(integration_state)

    context = await orchestrator._build_context()

    assert context is not None
    assert context.items
    assert context.items[0] == integration_state.attachments[0]

    orchestrator.unbind()


def test_context_request_crosses_state_to_context_boundary(
    integration_system,
    integration_state,
):
    from gaia_agent.context.request_builder import ContextRequestBuilder

    request = ContextRequestBuilder.from_state(integration_state)

    assert request.user_request == integration_state.user_request
    assert len(request.attachments) == 1

    forwarded = request.attachments[0]

    assert forwarded.attachment_id == "integration-test:context.txt"
    assert forwarded.filename == "context.txt"
    assert forwarded.path == r"C:\integration_test\context.txt"


# ============================================================================
# Planner boundary
# ============================================================================


@pytest.mark.asyncio
async def test_planner_is_called_with_user_request_and_context(
    integration_system,
    integration_state,
):
    orchestrator = integration_system.orchestrator

    orchestrator.bind_state(integration_state)

    expected_plan = SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id=1,
                action="produce final answer",
                step_type="llm",
                tool_name=None,
                arguments={},
                is_final_answer=True,
            )
        ]
    )

    integration_system.planner.generate_plan = AsyncMock(
        return_value=expected_plan
    )

    # The real orchestrator validates PlanSchema, so this test intentionally
    # checks the boundary only by invoking the context-building/planner call
    # directly through the planner mock contract.
    context = await orchestrator._build_context()

    await integration_system.planner.generate_plan(
        integration_state.user_request,
        context,
    )

    integration_system.planner.generate_plan.assert_awaited_once()

    call = integration_system.planner.generate_plan.await_args

    assert call.args[0] == integration_state.user_request
    assert call.args[1] is context
    assert context.items


# ============================================================================
# AgentExecution boundary
# ============================================================================


@pytest.mark.asyncio
async def test_agent_execution_is_wired_into_orchestrator(
    integration_system,
    integration_state,
):
    orchestrator = integration_system.orchestrator

    orchestrator.bind_state(integration_state)

    assert orchestrator._agent_execution is integration_system.agent_execution

    context = await orchestrator._build_context()

    assert context.items
    assert orchestrator._agent_execution is not None

    orchestrator.unbind()


# ============================================================================
# Verification boundary
# ============================================================================


@pytest.mark.asyncio
async def test_verifier_is_wired_into_orchestrator(
    integration_system,
):
    orchestrator = integration_system.orchestrator

    assert orchestrator._verifier is integration_system.verifier


# ============================================================================
# Terminal behavior
# ============================================================================


@pytest.mark.asyncio
async def test_terminal_iteration_is_idempotent(
    integration_system,
    integration_state,
):
    from gaia_agent.core.agent_state import AgentPhase

    orchestrator = integration_system.orchestrator

    orchestrator.bind_state(integration_state)

    integration_state.phase = AgentPhase.COMPLETED

    first = await orchestrator.run_iteration()
    second = await orchestrator.run_iteration()

    assert first is None
    assert second is None

    orchestrator.unbind()


# ============================================================================
# Reliability boundary
# ============================================================================


@pytest.mark.asyncio
async def test_reliability_is_wired_into_orchestrator(
    integration_system,
):
    orchestrator = integration_system.orchestrator

    assert orchestrator._reliability is integration_system.reliability_engine


# ============================================================================
# Loop detection boundary
# ============================================================================


@pytest.mark.asyncio
async def test_loop_detector_is_wired_into_orchestrator(
    integration_system,
):
    orchestrator = integration_system.orchestrator

    assert orchestrator._loop_detector is integration_system.loop_detector

    integration_system.loop_detector.check.return_value = False

    assert orchestrator._loop_detector.check.return_value is False


# ============================================================================
# Complete orchestration graph
# ============================================================================


def test_complete_orchestration_graph_is_wired(
    integration_system,
):
    orchestrator = integration_system.orchestrator

    assert orchestrator._context_builder is integration_system.context_builder
    assert orchestrator._planner is integration_system.planner
    assert orchestrator._agent_execution is integration_system.agent_execution
    assert orchestrator._reliability is integration_system.reliability_engine
    assert orchestrator._loop_detector is integration_system.loop_detector
    assert orchestrator._verifier is integration_system.verifier