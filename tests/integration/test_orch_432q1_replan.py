"""432q1 matrix T3/T6/T9/T14: replan failure, caps, invalid-plan atomicity."""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gaia_agent.agents.verifier import VerificationResult, VerificationStatus
from gaia_agent.context.models import FinalContext
from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.core.agent_state import AgentState, AgentPhase
from gaia_agent.core.orchestration.models import OrchestrationAction, OrchestrationContext
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

def _plan2():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Run", step_type=StepType.TOOL, tool_name="python_interpreter", arguments={"code": "1"}), PlanStep(step_id=1, action="Fin", step_type=StepType.LLM, is_final_answer=True)])

def _plan1():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Fin", step_type=StepType.LLM, is_final_answer=True)])

def _mk(*, bad=None, pl_replan=None, gen=None, cfg=None):
    cb = MagicMock(); cb.build = AsyncMock(return_value=FinalContext(items=[{"t": 1}], token_count=1))
    pl = MagicMock()
    pl.generate_plan = AsyncMock(return_value=PlanningResult(plan=gen or _plan2(), task_analysis=_ta()))
    pl.replan = pl_replan if pl_replan is not None else AsyncMock(return_value=PlanningResult(plan=_plan1(), task_analysis=_ta()))
    err = AgentError(error_type="R", message="bad", category=ErrorCategory.TOOL_ARGUMENT_ERROR, severity=ErrorSeverity.HIGH, retryable=False, recoverable=True, source="t", operation="e")
    ex = MagicMock(); ex.execute = AsyncMock(return_value=bad if bad is not None else ExecutionResult(success=False, output=None, evidence=(), error=err, step_id=0, tool_name="python_interpreter"))
    rel = ReliabilityEngine(error_handler=ErrorHandler(), failure_classifier=FailureClassifier(), retry_policy=RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0), recovery_policy=RecoveryPolicy(allow_replanning=True), retry=Retry(jitter_ratio=0.0), recovery=Recovery(error_handler=ErrorHandler()))
    loop = MagicMock(); loop.check = MagicMock(return_value=LoopDetection(detected=False)); loop.record = MagicMock()
    ver = MagicMock(); ver.verify = AsyncMock(return_value=VerificationResult(status=VerificationStatus.VERIFIED))
    orch = Orchestrator(context_builder=cb, planner=pl, agent_execution=ex, reliability_engine=rel, loop_detector=loop, verifier=ver, observability=None, config=cfg or OrchestratorConfig())
    return SimpleNamespace(orch=orch, state=AgentState(user_request="Calculate 2 + 2"), pl=pl, ex=ex)

@pytest.mark.asyncio
async def test_t3_replan_raise_fails_without_replacing_plan():
    h = _mk(pl_replan=AsyncMock(side_effect=RuntimeError("planner down")))
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    before = list(run.plan_runtime.plan.steps)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    assert h.state.phase is AgentPhase.FAILED
    assert [s.action for s in run.plan_runtime.plan.steps] == [s.action for s in before]
    assert run.plan_runtime.plan_version == 1 and run.plan_runtime.replan_count == 0

@pytest.mark.asyncio
async def test_t6_max_replans_blocks_planner_call():
    h = _mk(cfg=OrchestratorConfig(max_replans=0))
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    h.pl.replan.assert_not_awaited()

@pytest.mark.asyncio
async def test_t9_invalid_replan_plan_not_installed():
    h = _mk(pl_replan=AsyncMock(return_value=SimpleNamespace(plan=SimpleNamespace(steps=[]), task_analysis="nope")))
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    before = list(run.plan_runtime.plan.steps)
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    assert [s.action for s in run.plan_runtime.plan.steps] == [s.action for s in before]
    assert run.plan_runtime.plan_version == 1 and run.plan_runtime.replan_count == 0

@pytest.mark.asyncio
async def test_t14_invalid_initial_plan_not_installed():
    bad = SimpleNamespace(plan=SimpleNamespace(steps=[]), task_analysis="nope")
    h = _mk(gen=None, pl_replan=AsyncMock(return_value=PlanningResult(plan=_plan1(), task_analysis=_ta())))
    h.pl.generate_plan = AsyncMock(return_value=bad)
    run = OrchestrationContext(user_request="Calculate 2 + 2")
    run2 = await h.orch.start(h.state, run=run)
    assert run2 is run
    o = await h.orch.step(h.state, run)
    assert o.action is OrchestrationAction.FAIL
    assert run.plan_runtime.plan is None and run.plan_runtime.plan_version == 0
    assert h.ex.execute.await_count == 0

