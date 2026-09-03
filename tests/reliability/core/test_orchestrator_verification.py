from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gaia_agent.agents.verifier import VerificationResult
from gaia_agent.core.orchestration.orchestrator import (
    MAX_VERIFICATION_ATTEMPTS,
    Orchestrator,
)
from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.reliability.errors import AgentError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_orchestrator():
    """
    Build a lightweight Orchestrator instance without running its
    production __init__ dependency graph.
    """
    orchestrator = Orchestrator.__new__(Orchestrator)

    orchestrator.state = SimpleNamespace(
        user_request="What is the answer?",
        final_answer="42",
        final_answer_ready=True,
        final_answer_verified=False,
        task_completed=False,
        verification_attempts=0,
        tool_error=None,
        evidence=[],
        plan=[],
        current_step=0,
        completed_steps=[],
        replan_count=0,
        last_failure_key=None,
        same_failure_count=0,
        fatal_error=False,
        execution_success=False,
        step_succeeded=False,
        blocked=False,
        recovery_attempted=False,
        tool_result=None,
        current_action=None,
        step_type=None,
        tool_name=None,
        tool_arguments={},
        execution_decision=None,
        approval_decision=None,
        risk_assessment=None,
    )

    orchestrator.metrics = Mock()

    orchestrator.context_builder = Mock()
    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(items=[])
    )

    orchestrator.verifier = Mock()

    orchestrator.reliability_engine = Mock()

    orchestrator.planner = Mock()
    orchestrator.error_handler = Mock()

    orchestrator.event_logger = Mock()
    orchestrator.tracer = Mock()

    orchestrator.answer_sanitizer = Mock()
    orchestrator.answer_sanitizer.sanitize.side_effect = (
        lambda value: str(value).strip()
    )

    orchestrator.loop_detector = Mock()

    orchestrator.emit_agent_failure = Mock()

    return orchestrator


def make_final_answer_step(
    *,
    step_id: int = 0,
    action: str = "Generate final answer.",
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        action=action,
        step_type=StepType.LLM,
        tool_name=None,
        arguments={},
        is_final_answer=True,
    )


# ---------------------------------------------------------------------------
# 1. Final answer must be captured as READY but NOT automatically VERIFIED
# ---------------------------------------------------------------------------


def test_capture_final_answer_does_not_mark_it_verified():
    orchestrator = make_orchestrator()

    orchestrator.state.tool_result = "42"

    step = make_final_answer_step()

    orchestrator._capture_step_result(step)

    assert orchestrator.state.final_answer == "42"
    assert orchestrator.state.final_answer_ready is True
    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False


# ---------------------------------------------------------------------------
# 2. Missing final answer must not silently succeed
# ---------------------------------------------------------------------------


def test_capture_final_answer_rejects_missing_result():
    orchestrator = make_orchestrator()

    orchestrator.state.tool_result = None

    step = make_final_answer_step()

    with pytest.raises(ValueError, match="returned no result"):
        orchestrator._capture_step_result(step)

    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False


# ---------------------------------------------------------------------------
# 3. Plan completion should trigger verification when answer is ready
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_completion_verifies_ready_unverified_answer():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False

    orchestrator._verify_final_answer = AsyncMock()

    await orchestrator._handle_plan_completion()

    orchestrator._verify_final_answer.assert_awaited_once()


# ---------------------------------------------------------------------------
# 4. Plan completion should create final-answer step when answer isn't ready
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_completion_creates_final_answer_step_when_not_ready():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer_ready = False
    orchestrator.state.final_answer_verified = False
    orchestrator.state.plan = []

    await orchestrator._handle_plan_completion()

    assert len(orchestrator.state.plan) == 1

    step = orchestrator.state.plan[0]

    assert step.step_id == 0
    assert step.step_type == StepType.LLM
    assert step.tool_name is None
    assert step.is_final_answer is True

    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False


# ---------------------------------------------------------------------------
# 5. Already verified answer should not trigger verification again
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_completion_does_not_reverify_verified_answer():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = True
    orchestrator.state.task_completed = True

    orchestrator._verify_final_answer = AsyncMock()

    await orchestrator._handle_plan_completion()

    orchestrator._verify_final_answer.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6. Verifier PASS should mark the answer verified and completed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_pass_marks_answer_verified():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "42"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False

    orchestrator.state.evidence = [
        SimpleNamespace(
            tool_name="python_interpreter",
            succeeded=True,
            result="42",
        )
    ]

    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(
            items=[]
        )
    )

    orchestrator.verifier.verify = AsyncMock(
        return_value=VerificationResult(
            verified=True,
            reason="Strong evidence supports the answer.",
        )
    )

    # Deterministic verification must be uncertain so that the
    # Orchestrator actually reaches VerifierAgent.
    orchestrator.reliability_engine.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            result=VerificationResult(
                verified=True,
                reason="Strong evidence supports the answer.",
            ),
            reason=None,
            error=None,
        )
    )

    await orchestrator._verify_final_answer()

    assert orchestrator.state.final_answer_ready is True
    assert orchestrator.state.final_answer_verified is True
    assert orchestrator.state.task_completed is True
    assert orchestrator.state.tool_error is None

    orchestrator.metrics.increment.assert_any_call(
        "answers_verified"
    )


# ---------------------------------------------------------------------------
# 7. Verifier FAIL must NOT complete the task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_failure_never_marks_answer_completed():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "21"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False
    orchestrator.state.verification_attempts = 0

    orchestrator.state.evidence = [
        SimpleNamespace(
            tool_name="web_search",
            succeeded=True,
            result="Unrelated page mentioning 21.",
        )
    ]

    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(items=[])
    )

    verification_failure = VerificationResult(
        verified=False,
        reason="The evidence does not answer the question.",
    )

    orchestrator.reliability_engine.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            result=verification_failure,
            reason=None,
            error=None,
        )
    )

    # Prevent the test from entering the real recovery implementation.
    orchestrator._handle_verification_failure = AsyncMock()

    await orchestrator._verify_final_answer()

    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False

    orchestrator._handle_verification_failure.assert_awaited_once()


# ---------------------------------------------------------------------------
# 8. No evidence must never reach a successful completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_with_no_evidence_cannot_complete():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "42"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False
    orchestrator.state.evidence = []

    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(items=[])
    )

    # If the verifier is reached, make it explicitly reject.
    orchestrator.reliability_engine.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            result=VerificationResult(
                verified=False,
                reason="No successful evidence is available.",
            ),
            reason=None,
            error=None,
        )
    )

    orchestrator._handle_verification_failure = AsyncMock()

    await orchestrator._verify_final_answer()

    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False


# ---------------------------------------------------------------------------
# 9. Verification failure before budget exhaustion should attempt recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_failure_before_budget_attempts_recovery():
    orchestrator = make_orchestrator()

    orchestrator.state.verification_attempts = 1
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False

    orchestrator._check_recovery_budget = Mock(
        return_value=True
    )

    orchestrator.reliability_engine.failure_classifier = Mock()

    classification = Mock()

    orchestrator.reliability_engine.failure_classifier.classify.return_value = (
        classification
    )

    recovery_decision = Mock()
    recovery_decision.action.name = "REPLAN"

    orchestrator.reliability_engine.recovery_policy = Mock()
    orchestrator.reliability_engine.recovery_policy.evaluate.return_value = (
        recovery_decision
    )

    failed_step = make_final_answer_step()

    orchestrator.state.plan = [failed_step]
    orchestrator.state.current_step = 0

    new_step = PlanStep(
        step_id=0,
        action="Use a different source to answer the question.",
        step_type=StepType.TOOL,
        tool_name="web_search",
        arguments={"query": "answer"},
        is_final_answer=True,
    )

    orchestrator.planner.replan_step = AsyncMock(
        return_value=new_step
    )

    orchestrator._replace_failed_step = Mock()
    orchestrator._prepare_step = Mock()

    await orchestrator._handle_verification_failure(
        AgentError(
            error_type="AnswerVerificationError",
            message="Evidence does not support answer.",
            source="verifier",
            operation="verify_answer",
            recoverable=True,
        )
    )

    orchestrator.planner.replan_step.assert_awaited_once()

    orchestrator._replace_failed_step.assert_called_once_with(
        new_step
    )

    orchestrator._prepare_step.assert_called_once_with(
        new_step
    )

    assert orchestrator.state.final_answer is None
    assert orchestrator.state.final_answer_ready is False
    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False


# ---------------------------------------------------------------------------
# 10. Exhausted verification budget must never complete the task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_verification_budget_keeps_answer_unverified():
    orchestrator = make_orchestrator()

    orchestrator.state.verification_attempts = (
        MAX_VERIFICATION_ATTEMPTS
    )

    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False
    orchestrator.state.tool_error = None

    error = Mock()

    await orchestrator._handle_verification_failure(
        error
    )

    assert orchestrator.state.final_answer_ready is True
    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False

    assert (
        orchestrator.state.tool_error
        == "Verification attempts exhausted; "
        "final answer remains unverified."
    )

    orchestrator.metrics.increment.assert_called_once_with(
        "answers_unverified"
    )


# ---------------------------------------------------------------------------
# 11. LLM-positive verification should not be downgraded merely because
#     the evidence is web_search evidence.
#
#     This protects the new VerifierAgent contract:
#       web evidence -> semantic LLM judgment.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_verified_web_evidence_can_complete_task():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "reverse"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False

    orchestrator.state.evidence = [
        SimpleNamespace(
            tool_name="web_search",
            succeeded=True,
            result=(
                "The source directly states that the answer is reverse."
            ),
        )
    ]

    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(items=[])
    )

    # Force deterministic verification to be uncertain so the
    # VerifierAgent path is exercised.
    orchestrator.reliability_engine.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            result=VerificationResult(
                verified=True,
                reason="The source directly answers the question.",
            ),
            reason=None,
            error=None,
        )
    )

    await orchestrator._verify_final_answer()

    assert orchestrator.state.final_answer_verified is True
    assert orchestrator.state.task_completed is True
    assert orchestrator.state.tool_error is None

    orchestrator.metrics.increment.assert_any_call(
        "answers_verified"
    )


# ---------------------------------------------------------------------------
# 12. Verifier returning no result must not complete the task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_verifier_result_does_not_complete_task():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "42"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False

    orchestrator.state.evidence = [
        SimpleNamespace(
            tool_name="python_interpreter",
            succeeded=True,
            result="42",
        )
    ]

    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(items=[])
    )

    orchestrator.reliability_engine.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            result=None,
            reason=None,
            error=None,
        )
    )

    await orchestrator._verify_final_answer()

    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False
    assert (
        orchestrator.state.tool_error
        == "Verifier returned no result."
    )


# ---------------------------------------------------------------------------
# 13. Verifier execution failure must not complete the task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_execution_failure_does_not_complete_task():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "42"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False

    orchestrator.state.evidence = [
        SimpleNamespace(
            tool_name="python_interpreter",
            succeeded=True,
            result="42",
        )
    ]

    orchestrator.context_builder.build = AsyncMock(
        return_value=SimpleNamespace(items=[])
    )

    orchestrator.reliability_engine.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=False,
            result=None,
            reason="Verifier execution failed.",
            error=Mock(),
        )
    )

    await orchestrator._verify_final_answer()

    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False
    assert (
        orchestrator.state.tool_error
        == "Verifier execution failed."
    )


# ---------------------------------------------------------------------------
# 14. Final answer step must be uniquely identifiable
# ---------------------------------------------------------------------------


def test_find_final_answer_step_returns_latest_final_step():
    orchestrator = make_orchestrator()

    normal_step = PlanStep(
        step_id=0,
        action="Search for evidence.",
        step_type=StepType.TOOL,
        tool_name="web_search",
        arguments={"query": "answer"},
        is_final_answer=False,
    )

    final_step = make_final_answer_step(
        step_id=1
    )

    orchestrator.state.plan = [
        normal_step,
        final_step,
    ]

    result = orchestrator._find_final_answer_step()

    assert result is final_step
    assert result.is_final_answer is True


# ---------------------------------------------------------------------------
# 15. Verification recovery must clear the previous candidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_recovery_clears_previous_candidate():
    orchestrator = make_orchestrator()

    orchestrator.state.final_answer = "wrong"
    orchestrator.state.final_answer_ready = True
    orchestrator.state.final_answer_verified = False
    orchestrator.state.task_completed = False
    orchestrator.state.verification_attempts = 1

    failed_step = make_final_answer_step()

    orchestrator.state.plan = [failed_step]
    orchestrator.state.current_step = 0

    orchestrator._check_recovery_budget = Mock(
        return_value=True
    )

    orchestrator.reliability_engine.failure_classifier = Mock()

    classification = Mock()

    orchestrator.reliability_engine.failure_classifier.classify.return_value = (
        classification
    )

    recovery_decision = Mock()
    recovery_decision.action.name = "REPLAN"

    orchestrator.reliability_engine.recovery_policy = Mock()
    orchestrator.reliability_engine.recovery_policy.evaluate.return_value = (
        recovery_decision
    )

    new_step = PlanStep(
        step_id=0,
        action="Find better evidence.",
        step_type=StepType.TOOL,
        tool_name="web_search",
        arguments={"query": "better evidence"},
        is_final_answer=True,
    )

    orchestrator.planner.replan_step = AsyncMock(
        return_value=new_step
    )

    orchestrator._replace_failed_step = Mock()
    orchestrator._prepare_step = Mock()

    await orchestrator._handle_verification_failure(
        AgentError(
            error_type="AnswerVerificationError",
            message="Wrong answer.",
            source="verifier",
            operation="verify_answer",
            recoverable=True,
        )
    )

    assert orchestrator.state.final_answer is None
    assert orchestrator.state.final_answer_ready is False
    assert orchestrator.state.final_answer_verified is False
    assert orchestrator.state.task_completed is False