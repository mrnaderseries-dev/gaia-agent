"""432q1 matrix T8/T11/T12: loop, context failure, terminal idempotency."""
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
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.loop_detector import LoopDetection
from gaia_agent.reliability.policies.recovery_policy import RecoveryPolicy
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.retry import Retry

def _ta():
    return TaskClassifier().classify("Calculate 2 + 2", available_files=(), available_tools=("python_interpreter",))

def _plan():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Run", step_type=StepType.TOOL, tool_name="python_interpreter", arguments={"code": "1"}), PlanStep(step_id=1, action="Fin", step_type=StepType.LLM, is_final_answer=True)])

def _mk(*, loop_det=False, ctx_exc=None):
    cb = MagicMock()
    if ctx_exc is not None:
        cb.build = AsyncMock(side_effect=ctx_exc)
    else:
        cb.build = AsyncMock(return_value=FinalContext(items=[{"t": 1}], token_count=1))
    pl = MagicMock()
    pl.generate_plan = AsyncMock(return_value=PlanningResult(plan=_plan(), task_analysis=_ta()))
    pl.replan = AsyncMock(return_value=PlanningResult(plan=_plan(), task_analysis=_ta()))
    ex = MagicMock(); ex.execute = AsyncMock(return_value=ExecutionResult(success=True, output="x", evidence=(), error=None, step_id=0, tool_name="python_interpreter"))
    rel = ReliabilityEngine(error_handler=ErrorHandler(), failure_classifier=FailureClassifier(), retry_policy=RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0), recovery_policy=RecoveryPolicy(allow_replanning=True), retry=Retry(jitter_ratio=0.0), recovery=Recovery(error_handler=ErrorHandler()))
    loop = MagicMock(); loop.check = MagicMock(return_value=LoopDetection(detected=loop_det, reason="loop" if loop_det else "")); loop.record = MagicMock()
    ver = MagicMock(); ver.verify = AsyncMock(return_value=VerificationResult(status=VerificationStatus.VERIFIED))
    orch = Orchestrator(context_builder=cb, planner=pl, agent_execution=ex, reliability_engine=rel, loop_detector=loop, verifier=ver, observability=None, config=OrchestratorConfig())
    return SimpleNamespace(orch=orch, state=AgentState(user_request="Calculate 2 + 2"), ex=ex, pl=pl, loop=loop)

@pytest.mark.asyncio
async def test_t8_loop_detected_fails_without_execution_or_replan():
    h = _mk(loop_det=True)
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    assert h.ex.execute.await_count == 0
    h.pl.replan.assert_not_awaited()
    assert h.state.phase is AgentPhase.FAILED
    assert h.state.task_completed is False

@pytest.mark.asyncio
async def test_t11_context_failure_fails_without_execution():
    h = _mk(ctx_exc=RuntimeError("ctx down"))
    run = await h.orch.start(h.state)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    assert h.ex.execute.await_count == 0
    assert run.plan_runtime.plan is None
    assert h.state.phase is AgentPhase.FAILED

@pytest.mark.asyncio
async def test_t12a_completed_does_not_execute():
    h = _mk()
    run = await h.orch.start(h.state)
    h.state.transition(h.state.phase.__class__.COMPLETED, reason=h.state.transition_history[-1].reason) if False else None
    from gaia_agent.core.agent_state import TransitionReason as TR
    h.state.transition(AgentPhase.EXECUTING, reason=TR.PLAN_READY)
    h.state.transition(AgentPhase.VERIFYING, reason=TR.EXECUTION_COMPLETED)
    h.state.final_answer_verified = True
    h.state.complete()
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.TERMINATE
    assert h.ex.execute.await_count == 0

@pytest.mark.asyncio
async def test_t12b_failed_does_not_continue():
    h = _mk()
    run = await h.orch.start(h.state)
    from gaia_agent.core.agent_state import TransitionReason as TR
    h.state.transition(AgentPhase.EXECUTING, reason=TR.PLAN_READY)
    h.state.fail(reason=TR.EXECUTION_FAILED, fatal=True)
    assert h.state.phase is AgentPhase.FAILED
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.TERMINATE
    assert h.ex.execute.await_count == 0
    assert h.state.task_completed is False
