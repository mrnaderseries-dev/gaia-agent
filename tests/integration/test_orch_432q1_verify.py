"""432q1 matrix T4/T5/T10: verification failure/uncertain + final-answer exec fail."""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gaia_agent.agents.verifier import VerificationResult, VerificationStatus
from gaia_agent.context.models import FinalContext
from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.core.agent_state import AgentState, AgentPhase
from gaia_agent.core.orchestration.models import OrchestrationAction
from gaia_agent.core.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from gaia_agent.planner.models import PlanningResult
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.task_classifier import TaskClassifier
from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.errors import AgentError, ErrorCategory, ErrorSeverity
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.loop_detector import LoopDetection
from gaia_agent.reliability.policies.recovery_policy import RecoveryPolicy
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.retry import Retry
def _ta():
    return TaskClassifier().classify("Calculate 2 + 2", available_files=(), available_tools=("python_interpreter",))

def _plan_final():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Fin", step_type=StepType.LLM, is_final_answer=True)])

def _plan_tool():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Run", step_type=StepType.TOOL, tool_name="python_interpreter", arguments={"code": "1"}), PlanStep(step_id=1, action="Fin", step_type=StepType.LLM, is_final_answer=True)])

def _mk(*, verify=None, exec_side=None, plan=None, rplan=None):
    cb = MagicMock(); cb.build = AsyncMock(return_value=FinalContext(items=[{"t": 1}], token_count=1))
    pl = MagicMock()
    pl.generate_plan = AsyncMock(return_value=PlanningResult(plan=plan or _plan_final(), task_analysis=_ta()))
    pl.replan = AsyncMock(return_value=PlanningResult(plan=rplan or _plan_final(), task_analysis=_ta()))
    ex = MagicMock()
    if isinstance(exec_side, list):
        ex.execute = AsyncMock(side_effect=exec_side)
    else:
        ex.execute = AsyncMock(return_value=exec_side if exec_side is not None else ExecutionResult(success=True, output="42", evidence=(), error=None, step_id=0, tool_name=None))
    rel = ReliabilityEngine(error_handler=ErrorHandler(), failure_classifier=FailureClassifier(), retry_policy=RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0), recovery_policy=RecoveryPolicy(allow_replanning=True), retry=Retry(jitter_ratio=0.0), recovery=Recovery(error_handler=ErrorHandler()))
    loop = MagicMock(); loop.check = MagicMock(return_value=LoopDetection(detected=False)); loop.record = MagicMock()
    ver = MagicMock(); ver.verify = AsyncMock(return_value=verify or VerificationResult(status=VerificationStatus.VERIFIED))
    orch = Orchestrator(context_builder=cb, planner=pl, agent_execution=ex, reliability_engine=rel, loop_detector=loop, verifier=ver, observability=None, config=OrchestratorConfig())
    return SimpleNamespace(orch=orch, state=AgentState(user_request="Calculate 2 + 2"), pl=pl, ex=ex, ver=ver)

@pytest.mark.asyncio
async def test_t4_verify_invalid_replans_and_clears_answer():
    h = _mk(verify=VerificationResult(status=VerificationStatus.INVALID, reason="wrong"), rplan=_plan_final())
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.REPLAN
    assert run.plan_runtime.replan_count == 1 and run.plan_runtime.plan_version == 2
    assert h.state.final_answer is None and h.state.final_answer_ready is False
    assert h.state.final_answer_verified is False and h.state.task_completed is False
    assert h.state.verification_attempts == 1 and len(run.verification_history) == 1

@pytest.mark.asyncio
async def test_t5_verify_uncertain_replans_if_budget_left():
    h = _mk(verify=VerificationResult(status=VerificationStatus.INSUFFICIENT_EVIDENCE, reason="thin"))
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.REPLAN
    assert h.state.final_answer_verified is False and h.state.task_completed is False

@pytest.mark.asyncio
async def test_t10_final_answer_exec_failure_retries_then_completes():
    err = AgentError(error_type="L", message="llm down", category=ErrorCategory.LLM_FAILURE, severity=ErrorSeverity.MEDIUM, retryable=True, recoverable=True, source="t", operation="e")
    bad = ExecutionResult(success=False, output=None, evidence=(), error=err, step_id=0, tool_name=None)
    good = ExecutionResult(success=True, output="42", evidence=(), error=None, step_id=0, tool_name=None)
    h = _mk(exec_side=[bad, good])
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.RETRY
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.COMPLETE
    assert h.state.final_answer_verified is True

@pytest.mark.asyncio
async def test_verify_budget_exhausted_fails_never_completes():
    h = _mk(verify=VerificationResult(status=VerificationStatus.INVALID, reason="wrong"))
    h.orch.config = OrchestratorConfig(max_verification_attempts=1)
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    assert h.state.final_answer_verified is False and h.state.task_completed is False
    assert h.state.phase is AgentPhase.FAILED

