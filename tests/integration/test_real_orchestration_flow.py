from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentPhase, AgentState
from gaia_agent.core.orchestration.models import (
    OrchestrationContext,
    PlanRuntime,
)
from gaia_agent.core.orchestration.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
)
from gaia_agent.observability.facade import ObservabilityFacade
from gaia_agent.planner.plan_schema import (
    PlanSchema,
    PlanStep,
    StepType,
)
from gaia_agent.planner.models import PlanningResult
from gaia_agent.planner.task_classifier import (
    TaskAnalysis,
    TaskIntent,
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
from gaia_agent.reliability.loop_detector import (
    LoopDetection,
    LoopType,
)
from gaia_agent.reliability.policies.recovery_policy import (
    RecoveryAction,
)
from gaia_agent.reliability.engine import (
    ReliabilityAction,
    ReliabilityEngine,
)
from gaia_agent.reliability.recovery import Recovery
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


def _make_task_analysis(
    *,
    intent: TaskIntent = TaskIntent.SELF_CONTAINED,
    needs_external_info: bool = False,
    recommended_first_tool: str | None = None,
    forbidden_tools: tuple[str, ...] = (),
) -> TaskAnalysis:
    return TaskAnalysis(
        intent=intent,
        needs_external_info=needs_external_info,
        recommended_first_tool=recommended_first_tool,
        forbidden_tools=forbidden_tools,
        analysis_text="integration-test task analysis",
    )


def _make_planning_result(
    plan: PlanSchema,
    *,
    intent: TaskIntent = TaskIntent.SELF_CONTAINED,
) -> PlanningResult:
    return PlanningResult(
        plan=plan,
        task_analysis=_make_task_analysis(
            intent=intent,
        ),
    )


def _make_state(
    request: str = "integration test request",
) -> AgentState:
    return AgentState(
        user_request=request,
    )


def _make_llm_model() -> LLMModel:
    return LLMModel(
        provider="integration-test",
        model="qwen2.5:3b",
        max_tokens=4096,
        temperature=0.0,
    )


def _make_context_builder() -> ContextBuilder:
    """Real ContextBuilder.

    External sources are mocked because these tests verify cross-layer
    orchestration rather than PostgreSQL/source implementations.
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

    validator = ContextValidator(
        budget,
    )

    compressor = ContextCompressor(
        client=llm_client,
        model=model,
        budget=budget,
        policy=policy,
    )

    conversation_source = MagicMock()
    conversation_source.get = AsyncMock(
        return_value=[],
    )

    history_source = MagicMock()
    history_source.get = AsyncMock(
        return_value=[],
    )

    memory_source = MagicMock()
    memory_source.get = AsyncMock(
        return_value=[],
    )

    runtime_source = MagicMock()
    runtime_source.get = AsyncMock(
        return_value=[
            {
                "source": "integration-test",
                "request": "integration test request",
            }
        ],
    )

    attachment_source = MagicMock()
    attachment_source.get = AsyncMock(
        return_value=[],
    )

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
    """Real AgentExecution.

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
    """Real AgentLoop + mocked termination policy.

    TerminationPolicy itself is tested separately. These tests focus on the
    AgentLoop lifecycle.
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


def _make_observability() -> MagicMock:
    """Observability is injected through the single public facade.

    The internal logger / metrics / tracer implementations are not part of
    this integration suite.
    """
    return MagicMock(
        spec=ObservabilityFacade,
    )


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
    loop_detector.check.return_value = LoopDetection(detected=False)

    return Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
        observability=_make_observability(),
        config=OrchestratorConfig(
            max_step_attempts=3,
            max_verification_attempts=2,
            max_replans=3,
        ),
    )


def _make_verifier(
    *results: VerificationResult,
):
    verifier = MagicMock()
    verifier.verify = AsyncMock(
        side_effect=list(results),
    )

    return verifier


def _capture_run_context(
    orchestrator,
) -> dict:
    """Capture the OrchestrationContext created during a run.

    AgentLoop clears its bound run context in cleanup, so tests
    that need the run history capture it at start().
    """
    captured = {}

    original_start = orchestrator.start

    async def capture(state, *, run=None):
        context = await original_start(state, run=run)
        captured["run"] = context
        return context

    orchestrator.start = capture

    return captured



@pytest.mark.asyncio
async def test_real_agent_loop_happy_path_completes():
    planner = MagicMock()
    plan = _make_final_plan()

    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            plan,
        ),
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

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state()

    result = await loop.run(
        state,
    )

    assert result is state
    assert state.phase == AgentPhase.COMPLETED
    assert state.final_answer == "42"
    assert state.final_answer_ready is True
    assert state.final_answer_verified is True
    assert state.fatal_error is False

    planner.generate_plan.assert_awaited_once()
    llm_executor.execute.assert_awaited_once()
    verifier.verify.assert_awaited_once()



@pytest.mark.asyncio
async def test_real_tool_execution_produces_evidence_for_verification():
    planner = MagicMock()
    plan = _make_tool_then_final_plan()

    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            plan,
        ),
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

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.COMPLETED

    assert tool_registry.get.called

    tool.assert_not_called()

    verifier.verify.assert_awaited_once()

    verification_input = verifier.verify.await_args.args[0]

    assert verification_input.question == state.user_request

    assert verification_input.candidate_answer == "The answer is 42."

    assert verification_input.raw_data

    evidence = verification_input.raw_data[0]

    assert evidence.tool_name == "test_tool"

    assert evidence.result == {
        "answer": 42,
        "source": "integration-tool",
    }

    assert evidence.succeeded is True
    assert evidence.step_id == 0

    assert llm_executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_real_agent_execution_failure_retries_through_reliability():
    planner = MagicMock()
    plan = _make_final_plan()

    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            plan,
        ),
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

    captured = _capture_run_context(orchestrator)

    state = _make_state()

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "recovered answer"

    assert llm_executor.execute.await_count == 2

    history = captured["run"].execution_history

    assert len(history) == 2

    first = history[0]
    second = history[1]

    assert first.result.success is False
    assert second.result.success is True

    assert first.attempt == 1
    assert second.attempt == 2

    assert reliability.retry.delay.await_count == 1


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
        return_value=_make_planning_result(
            initial_plan,
        ),
    )

    planner.replan = AsyncMock(
        return_value=_make_planning_result(
            replanned_plan,
        ),
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

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "answer after replan"

    planner.replan.assert_awaited_once()

    assert llm_executor.execute.await_count == 2


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
        return_value=_make_planning_result(
            initial_plan,
        ),
    )

    planner.replan = AsyncMock(
        return_value=_make_planning_result(
            replanned_plan,
        ),
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

    captured = _capture_run_context(orchestrator)

    state = _make_state(
        "Answer and verify this question.",
    )

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "correct answer"
    assert result.final_answer_verified is True

    assert verifier.verify.await_count == 2
    assert planner.replan.await_count == 1
    assert llm_executor.execute.await_count == 2

    history = captured["run"].verification_history

    assert len(history) == 2

    first_verification = history[0]
    second_verification = history[1]

    assert first_verification.verified is False
    assert second_verification.verified is True

@pytest.mark.asyncio
async def test_verification_budget_exhaustion_fails_agent():
    planner = MagicMock()
    initial_plan = _make_final_plan()

    replanned_plan = _make_final_plan(
        action="second attempt",
    )

    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            initial_plan,
        ),
    )

    planner.replan = AsyncMock(
        return_value=_make_planning_result(
            replanned_plan,
        ),
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

    captured = _capture_run_context(orchestrator)

    state = _make_state()

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.FAILED
    assert result.fatal_error is True

    assert result.final_answer == "bad answer two"
    assert result.final_answer_verified is False

    assert verifier.verify.await_count == 2
    assert llm_executor.execute.await_count == 2

    assert len(captured["run"].verification_history) == 2



@pytest.mark.asyncio
async def test_loop_detection_stops_orchestration():
    planner = MagicMock()
    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            _make_final_plan(),
        ),
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

    loop_detector.check.return_value = LoopDetection(
        detected=True,
        loop_type=LoopType.EXACT,
        similarity=1.0,
        reason="integration loop detected",
    )

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        loop_detector=loop_detector,
        verifier=verifier,
        observability=_make_observability(),
        config=OrchestratorConfig(
            max_step_attempts=3,
            max_verification_attempts=2,
            max_replans=3,
        ),
    )

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state(
        "Trigger loop detection.",
    )

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.FAILED
    assert result.fatal_error is True

    assert llm_executor.execute.await_count == 0
    assert verifier.verify.await_count == 0



@pytest.mark.asyncio
async def test_agent_loop_lifecycle_completes_cleanly():
    planner = MagicMock()
    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            _make_final_plan(),
        ),
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

    result = await loop.run(
        state,
    )

    assert result is state
    assert result.phase == AgentPhase.COMPLETED
    assert result.final_answer == "lifecycle answer"


@pytest.mark.asyncio
async def test_invalid_plan_is_rejected_before_execution():
    planner = MagicMock()
    invalid_plan = SimpleNamespace(
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

    planner.generate_plan = AsyncMock(
        return_value=SimpleNamespace(
            plan=invalid_plan,
            task_analysis=_make_task_analysis(),
        ),
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

    loop = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=_make_termination_policy(),
    )

    state = _make_state()

    result = await loop.run(
        state,
    )

    assert result is state
    assert result.phase == AgentPhase.FAILED
    assert result.fatal_error is True

    assert llm_executor.execute.await_count == 0


@pytest.mark.asyncio
async def test_verification_evidence_preserves_execution_metadata():
    planner = MagicMock()
    planner.generate_plan = AsyncMock(
        return_value=_make_planning_result(
            _make_tool_then_final_plan(),
        ),
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

    captured = _capture_run_context(orchestrator)

    state = _make_state()

    result = await loop.run(
        state,
    )

    assert result.phase == AgentPhase.COMPLETED

    verification_input = verifier.verify.await_args.args[0]

    assert len(verification_input.raw_data) == 1

    evidence = verification_input.raw_data[0]

    assert evidence.tool_name == "test_tool"
    assert evidence.result == "42"
    assert evidence.step_id == 0
    assert evidence.succeeded is True

    assert evidence.run_id == str(captured["run"].run_id)
    assert evidence.plan_version == 1
    assert evidence.attempt_id == "1"




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





@pytest.mark.asyncio
async def test_recovery_executes_operation_and_validates_change():
    recovery = Recovery()
    operation = AsyncMock(
        return_value={
            "changed": True,
        }
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