from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_agent.core.agent_execution import AgentExecution, ExecutionRequest
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentPhase, AgentState
from gaia_agent.core.orchestration.models import (
    OrchestrationContext,
    PlanRuntime,
)
from gaia_agent.core.orchestration.orchestrator import Orchestrator

from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)

from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import (
    AgentError,
    ErrorCategory,
    ErrorSeverity,
)
from gaia_agent.reliability.failure_classifier import (
    FailureClassification,
)
from gaia_agent.reliability.policies.recovery_policy import (
    RecoveryAction,
    RecoveryPolicy,
)
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.engine import (
    ReliabilityAction,
    ReliabilityEngine,
)

from gaia_agent.agents.verifier import (
    VerificationResult,
    VerificationStatus,
)

from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator
from gaia_agent.context.ContextCompressor import ContextCompressor

from gaia_agent.llm.model import LLMModel

from gaia_agent.tools.registry import ToolRegistry


# ============================================================================
# Helpers
# ============================================================================


def _make_final_plan(
    action: str = "produce final answer",
) -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action=action,
                step_type=StepType.LLM,
                is_final_answer=True,
            )
        ]
    )


def _make_tool_then_final_plan() -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="run integration tool",
                step_type=StepType.TOOL,
                tool_name="test_tool",
                arguments={"value": 42},
            ),
            PlanStep(
                step_id=1,
                action="produce final answer",
                step_type=StepType.LLM,
                is_final_answer=True,
            ),
        ]
    )


def _make_state(
    request: str = "integration test request",
) -> AgentState:
    return AgentState(user_request=request)


def _make_llm_model() -> LLMModel:
    return LLMModel(
        provider="integration-test",
        model="qwen2.5:3b",
        max_tokens=4096,
        temperature=0.0,
    )


def _make_context_builder() -> ContextBuilder:
    """
    Real ContextBuilder.

    The sources are mocked because the purpose of this test is to verify
    cross-layer orchestration, not PostgreSQL/source implementations.
    """

    model = _make_llm_model()

    llm_client = MagicMock()

    budget = ContextBudget(
        max_tokens=4096,
    )

    policy = ContextPolicy(
        include_memory=False,
        include_conversation=True,
        include_history=True,
        include_runtime=True,
        include_attachments=True,
    )

    validator = ContextValidator(budget)

    compressor = ContextCompressor(
        client=llm_client,
        model=model,
        budget=budget,
        policy=policy,
    )

    conversation_source = MagicMock()
    conversation_source.get = AsyncMock(return_value=[])

    history_source = MagicMock()
    history_source.get = AsyncMock(return_value=[])

    memory_source = MagicMock()
    memory_source.get = AsyncMock(return_value=[])

    runtime_source = MagicMock()
    runtime_source.get = AsyncMock(
        return_value=[
            {
                "source": "integration-test",
                "request": "integration test request",
            }
        ]
    )

    attachment_source = MagicMock()
    attachment_source.get = AsyncMock(return_value=[])

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


def _make_real_agent_execution(
    *,
    llm_output: str = "integration test answer",
    llm_side_effect=None,
    tool_registry=None,
) -> tuple[AgentExecution, MagicMock]:
    """
    Real AgentExecution.

    Only external dependencies are mocked.
    """

    execution_policy = MagicMock()
    execution_policy.evaluate.return_value = SimpleNamespace(
        allowed=True,
        message="",
        reason="integration-test",
    )

    llm_executor = MagicMock()

    if llm_side_effect is not None:
        llm_executor.execute = AsyncMock(
            side_effect=llm_side_effect,
        )
    else:
        llm_executor.execute = AsyncMock(
            return_value=llm_output,
        )

    if tool_registry is None:
        tool_registry = MagicMock()

    execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=None,
        approval_policy=None,
        llm_executor=llm_executor,
        event_logger=None,
        metrics=None,
        tracer=None,
        token_tracker=None,
        error_handler=ErrorHandler(),
    )

    return execution, llm_executor


def _make_real_reliability_engine(
    *,
    should_retry: bool = False,
    recovery_action=None,
    recovery_recovered: bool = False,
) -> ReliabilityEngine:
    error_handler = ErrorHandler()

    failure_classifier = MagicMock()

    failure_classifier.classify.return_value = MagicMock(
        spec=FailureClassification,
    )

    retry_policy = MagicMock()

    retry_policy.evaluate.return_value = SimpleNamespace(
        should_retry=should_retry,
        delay=0,
        reason="integration retry",
    )

    retry = MagicMock()
    retry.delay = AsyncMock()

    recovery_policy = MagicMock()

    if recovery_action is None:
        recovery_action = SimpleNamespace(
            value="stop",
        )

    recovery_policy.evaluate.return_value = SimpleNamespace(
        action=recovery_action,
        reason="integration recovery",
    )

    recovery = MagicMock()

    recovery.recovered = recovery_recovered

    recovery.execute = AsyncMock(
        return_value=SimpleNamespace(
            recovered=recovery_recovered,
            reason="integration recovery executed",
        )
    )

    return ReliabilityEngine(
        error_handler=error_handler,
        failure_classifier=failure_classifier,
        retry_policy=retry_policy,
        recovery_policy=recovery_policy,
        retry=retry,
        recovery=recovery,
    )


def _make_termination_policy() -> MagicMock:
    """
    Real AgentLoop + mocked termination policy.

    We intentionally keep TerminationPolicy itself out of this test because
    its own behavior belongs to its unit tests.

    The AgentLoop lifecycle is real.
    """

    policy = MagicMock()

    def evaluate(termination_state):
        if (
            termination_state.final_answer_verified
            or termination_state.fatal_error
            or termination_state.human_aborted
            or termination_state.explicit_stop
            or termination_state.timed_out
        ):
            return SimpleNamespace(
                should_stop=True,
                reason="terminal state reached",
            )

        return SimpleNamespace(
            should_stop=False,
            reason="continue",
        )

    policy.evaluate.side_effect = evaluate

    return policy


def _make_orchestrator(
    *,
    planner,
    agent_execution,
    reliability_engine,
    verifier,
    context_builder=None,
):
    if context_builder is None:
        context_builder = _make_context_builder()

    loop_detector = MagicMock()
    loop_detector.check.return_value = False

    return Orchestrator(
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


def _make_verifier(
    *results: VerificationResult,
):
    verifier = MagicMock()

    verifier.verify = AsyncMock(
        side_effect=list(results),
    )

    return verifier


# ============================================================================
# 1. Full happy path
# ============================================================================


@pytest.mark.asyncio
async def test_real_agent_loop_happy_path_completes():
    planner = MagicMock()

    plan = _make_final_plan()

    planner.generate_plan = AsyncMock(
        return_value=plan,
    )

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_output="42",
    )

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason="integration verified",
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    termination_policy = _make_termination_policy()

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=termination_policy,
    )

    state = _make_state()

    result = await loop.run(state)

    assert result is state

    assert state.phase == AgentPhase.COMPLETED
    assert state.final_answer == "42"
    assert state.final_answer_ready is True
    assert state.final_answer_verified is True
    assert state.fatal_error is False

    planner.generate_plan.assert_awaited_once()
    llm_executor.execute.assert_awaited_once()
    verifier.verify.assert_awaited_once()

    assert orchestrator._state is None


# ============================================================================
# 2. Tool -> evidence -> final LLM -> verifier
# ============================================================================


@pytest.mark.asyncio
async def test_real_tool_execution_produces_evidence_for_verification():
    planner = MagicMock()

    plan = _make_tool_then_final_plan()

    planner.generate_plan = AsyncMock(
        return_value=plan,
    )

    tool = MagicMock()
    tool.execute.return_value = {
        "answer": 42,
        "source": "integration-tool",
    }

    tool_registry = MagicMock()
    tool_registry.get.return_value = tool

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_output="The answer is 42.",
        tool_registry=tool_registry,
    )

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason="tool evidence supports answer",
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state(
        "Use the tool and answer the question.",
    )

    result = await loop.run(state)

    assert result.phase == AgentPhase.COMPLETED

    assert tool_registry.get.called
    tool.assert_not_called()

    verifier.verify.assert_awaited_once()

    verification_input = verifier.verify.await_args.args[0]

    assert verification_input.question == state.user_request
    assert verification_input.candidate_answer == "The answer is 42."

    assert verification_input.raw_data

    evidence = verification_input.raw_data[0]
    assert evidence["tool_name"] == "test_tool"
    assert evidence["result"] == {
        "answer": 42,
        "source": "integration-tool",
    }
    assert evidence["succeeded"] is True
    assert evidence["step_id"] == 0

    assert llm_executor.execute.await_count == 1


# ============================================================================
# 3. Real AgentExecution -> Reliability RETRY
# ============================================================================


@pytest.mark.asyncio
async def test_real_agent_execution_failure_retries_through_reliability():
    planner = MagicMock()

    plan = _make_final_plan()

    planner.generate_plan = AsyncMock(
        return_value=plan,
    )

    first_error = AgentError(
        error_type="IntegrationFailure",
        message="temporary LLM failure",
        category=ErrorCategory.LLM_FAILURE,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        recoverable=True,
        source="integration-test",
        operation="llm",
    )

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_side_effect=[
            first_error,
            "recovered answer",
        ],
    )

    reliability = _make_real_reliability_engine(
        should_retry=True,
    )

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state()

    result = await loop.run(state)

    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "recovered answer"

    assert llm_executor.execute.await_count == 2

    assert len(orchestrator._context.execution_history) == 2

    first = orchestrator._context.execution_history[0]
    second = orchestrator._context.execution_history

    assert first.result.success is False
    assert second.result.success is True

    assert first.attempt == 1
    assert second.attempt == 2

    assert reliability.retry.delay.await_count == 1


# ============================================================================
# 4. REAL Reliability REPLAN contract
#
# This test is intentionally expected to expose the current Orchestrator
# integration gap.
# ============================================================================


@pytest.mark.asyncio
async def test_real_reliability_replan_reaches_new_plan():
    planner = MagicMock()

    initial_plan = _make_final_plan(
        action="initial answer",
    )

    replanned_plan = _make_final_plan(
        action="replanned answer",
    )

    planner.generate_plan = AsyncMock(
        return_value=initial_plan,
    )

    planner.replan = AsyncMock(
        return_value=replanned_plan,
    )

    failure = AgentError(
        error_type="RecoverableFailure",
        message="execution requires replanning",
        category=ErrorCategory.LLM_FAILURE,
        severity=ErrorSeverity.MEDIUM,
        retryable=False,
        recoverable=True,
        source="integration-test",
        operation="llm",
    )

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_side_effect=[
            failure,
            "answer after replan",
        ],
    )

    from gaia_agent.reliability.policies.recovery_policy import (
        RecoveryAction,
    )

    reliability = _make_real_reliability_engine(
        should_retry=False,
        recovery_action=RecoveryAction.REPLAN,
        recovery_recovered=True,
    )

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state(
        "Recover this failed execution.",
    )

    result = await loop.run(state)

    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "answer after replan"

    planner.replan.assert_awaited()

    assert llm_executor.execute.await_count == 2


# ============================================================================
# 5. Verification failure -> planner replan -> verification success
# ============================================================================


@pytest.mark.asyncio
async def test_verification_failure_replans_and_then_completes():
    planner = MagicMock()

    initial_plan = _make_final_plan(
        action="produce first answer",
    )

    replanned_plan = _make_final_plan(
        action="produce corrected answer",
    )

    planner.generate_plan = AsyncMock(
        return_value=initial_plan,
    )

    planner.replan = AsyncMock(
        return_value=replanned_plan,
    )

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_side_effect=[
            "incorrect answer",
            "correct answer",
        ],
    )

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.INVALID,
            reason="first answer is unsupported",
        ),
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason="corrected answer verified",
        ),
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state(
        "Answer and verify this question.",
    )

    result = await loop.run(state)

    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "correct answer"
    assert result.final_answer_verified is True

    assert verifier.verify.await_count == 2

    planner.replan.assert_awaited_once()

    assert llm_executor.execute.await_count == 2

    assert len(orchestrator._context.verification_history) == 2

    first_verification = (
        orchestrator._context.verification_history[0]
    )

    second_verification = (
        orchestrator._context.verification_history
    )

    assert first_verification.verified is False
    assert second_verification.verified is True


# ============================================================================
# 6. Verification budget exhaustion
# ============================================================================


@pytest.mark.asyncio
async def test_verification_budget_exhaustion_fails_agent():
    planner = MagicMock()

    initial_plan = _make_final_plan()

    replanned_plan = _make_final_plan(
        action="second attempt",
    )

    planner.generate_plan = AsyncMock(
        return_value=initial_plan,
    )

    planner.replan = AsyncMock(
        return_value=replanned_plan,
    )

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_side_effect=[
            "bad answer one",
            "bad answer two",
        ],
    )
    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.INVALID,
            reason="unsupported",
        ),
        VerificationResult(
            status=VerificationStatus.INVALID,
            reason="still unsupported",
        ),
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state()

    result = await loop.run(state)

    assert result.phase == AgentPhase.FAILED
    assert result.fatal_error is True

    assert result.final_answer == "bad answer two"
    assert result.final_answer_verified is False

    assert verifier.verify.await_count == 2
    assert llm_executor.execute.await_count == 2

    assert len(orchestrator._context.verification_history) == 2


# ============================================================================
# 7. Loop detector -> fatal termination
# ============================================================================


@pytest.mark.asyncio
async def test_loop_detection_stops_orchestration():
    planner = MagicMock()

    planner.generate_plan = AsyncMock(
        return_value=_make_final_plan(),
    )

    agent_execution, llm_executor = _make_real_agent_execution(
        llm_output="should never execute",
    )

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
        )
    )

    context_builder = _make_context_builder()

    loop_detector = MagicMock()
    loop_detector.check.return_value = True

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        loop_detector=loop_detector,
        verifier=verifier,
        event_logger=MagicMock(),
        metrics=MagicMock(),
        tracer=MagicMock(),
        max_execution_attempts=3,
        max_verification_attempts=2,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state(
        "Trigger loop detection.",
    )

    result = await loop.run(state)

    assert result.phase == AgentPhase.FAILED
    assert result.fatal_error is True

    assert llm_executor.execute.await_count == 0
    assert verifier.verify.await_count == 0


# ============================================================================
# 8. AgentLoop lifecycle: bind -> run -> unbind
# ============================================================================


@pytest.mark.asyncio
async def test_agent_loop_unbinds_orchestrator_after_completion():
    planner = MagicMock()

    planner.generate_plan = AsyncMock(
        return_value=_make_final_plan(),
    )

    agent_execution, _ = _make_real_agent_execution(
        llm_output="lifecycle answer",
    )

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state()

    assert orchestrator._state is None

    result = await loop.run(state)

    assert result.phase == AgentPhase.COMPLETED

    assert orchestrator._state is None


# ============================================================================
# 9. Planner contract is actually enforced at orchestration boundary
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_plan_is_rejected_before_execution():
    planner = MagicMock()

    invalid_plan = PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="tool step",
                step_type=StepType.TOOL,
                tool_name="test_tool",
                arguments={},
            ),
        ]
    )

    planner.generate_plan = AsyncMock(
        return_value=SimpleNamespace(
            steps=[
                SimpleNamespace(
                    step_id=0,
                    action="invalid",
                    step_type=StepType.TOOL,
                    tool_name="test_tool",
                    arguments={},
                    is_final_answer=False,
                )
            ]
        )
    )

    agent_execution, llm_executor = _make_real_agent_execution()

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    state = _make_state()

    orchestrator.bind_state(state)

    try:
        with pytest.raises(Exception):
            await orchestrator.generate_initial_plan()

        assert llm_executor.execute.await_count == 0
    finally:
        orchestrator.unbind()


# ============================================================================
# 10. Verification evidence contains execution metadata
# ============================================================================


@pytest.mark.asyncio
async def test_verification_evidence_preserves_execution_metadata():
    planner = MagicMock()

    planner.generate_plan = AsyncMock(
        return_value=_make_tool_then_final_plan(),
    )

    tool = MagicMock()
    tool.execute.return_value = "42"

    tool_registry = MagicMock()
    tool_registry.get.return_value = tool

    agent_execution, _ = _make_real_agent_execution(
        llm_output="42",
        tool_registry=tool_registry,
    )

    reliability = _make_real_reliability_engine()

    verifier = _make_verifier(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
        )
    )

    orchestrator = _make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state()

    result = await loop.run(state)

    assert result.phase == AgentPhase.COMPLETED

    verification_input = verifier.verify.await_args.args[0]

    assert len(verification_input.raw_data) == 1

    evidence = verification_input.raw_data[0]

    assert evidence["tool_name"] == "test_tool"
    assert evidence["result"] == "42"
    assert evidence["step_id"] == 0
    assert evidence["succeeded"] is True
    assert evidence["relevant"] is True

    assert "run_id" in evidence
    assert "plan_version" in evidence
    assert "attempt_id" in evidence
    assert evidence["step_status"] == "completed"


# ============================================================================
# 11. Architectural contract: REPLAN is a decision, not recovery op
# ============================================================================


@pytest.mark.asyncio
async def test_replan_decision_does_not_require_recovery_operation():
    reliability = _make_real_reliability_engine(
        recovery_action=RecoveryAction.REPLAN,
    )

    error = AgentError(
        error_type="TEST_FAILURE",
        message="tool failed",
    )

    result = await reliability.handle_failure(
        error=error,
        attempt=3,
        max_attempts=3,
    )

    assert result.action == ReliabilityAction.REPLAN
    assert result.recovery_attempted is False
    assert result.recovery_result is None


# ============================================================================
# 12. Recovery test: executes operation and validates change
# ============================================================================


@pytest.mark.asyncio
async def test_recovery_executes_operation_and_validates_change():
    recovery = Recovery()

    operation = AsyncMock(
        return_value={"changed": True}
    )

    result = await recovery.execute(
        error=AgentError(
            error_type="TEST_FAILURE",
            message="temporary failure",
        ),
        operation=operation,
        change_detector=lambda value: value["changed"],
    )

    assert result.recovered is True
    assert result.changed is True

    operation.assert_awaited_once()