"""432q1 matrix T1/T2/T7/T13: retry, replan-after-retry, budget, idempotency."""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gaia_agent.agents.verifier import VerificationResult, VerificationStatus
from gaia_agent.context.models import FinalContext
from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.core.agent_state import AgentState
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
def _analysis():
    return TaskClassifier().classify("Calculate 2 + 2", available_files=(), available_tools=("python_interpreter",))

def _tool_plan():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Run tool", step_type=StepType.TOOL, tool_name="python_interpreter", arguments={"code": "1+1"}), PlanStep(step_id=1, action="Final", step_type=StepType.LLM, is_final_answer=True)])

def _final_plan():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Replanned final", step_type=StepType.LLM, is_final_answer=True)])

def _ok(output="4", step_id=0, tool=None):
    return ExecutionResult(success=True, output=output, evidence=(), error=None, step_id=step_id, tool_name=tool)

def _bad(err):
    return ExecutionResult(success=False, output=None, evidence=(), error=err, step_id=0, tool_name="python_interpreter")

def _transient():
    return AgentError(error_type="T", message="timeout", category=ErrorCategory.TIMEOUT, severity=ErrorSeverity.MEDIUM, retryable=True, recoverable=False, source="t", operation="e")

def _recoverable():
    return AgentError(error_type="R", message="bad args", category=ErrorCategory.TOOL_ARGUMENT_ERROR, severity=ErrorSeverity.HIGH, retryable=False, recoverable=True, source="t", operation="e")

def _verified():
    return VerificationResult(status=VerificationStatus.VERIFIED)

def _mk(*, exec_side=None, replan=None):
    cb = MagicMock(); cb.build = AsyncMock(return_value=FinalContext(items=[{"t": 1}], token_count=1))
    pl = MagicMock()
    pl.generate_plan = AsyncMock(return_value=PlanningResult(plan=_tool_plan(), task_analysis=_analysis()))
    pl.replan = AsyncMock(return_value=PlanningResult(plan=replan or _final_plan(), task_analysis=_analysis()))
    ex = MagicMock()
    if isinstance(exec_side, list):
        ex.execute = AsyncMock(side_effect=exec_side)
    else:
        ex.execute = AsyncMock(return_value=exec_side if exec_side is not None else _ok())
    rel = ReliabilityEngine(error_handler=ErrorHandler(), failure_classifier=FailureClassifier(), retry_policy=RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0), recovery_policy=RecoveryPolicy(allow_replanning=True), retry=Retry(jitter_ratio=0.0), recovery=Recovery(error_handler=ErrorHandler()))
    loop = MagicMock(); loop.check = MagicMock(return_value=LoopDetection(detected=False)); loop.record = MagicMock()
    ver = MagicMock(); ver.verify = AsyncMock(return_value=_verified())
    orch = Orchestrator(context_builder=cb, planner=pl, agent_execution=ex, reliability_engine=rel, loop_detector=loop, verifier=ver, observability=None, config=OrchestratorConfig())
    return SimpleNamespace(orch=orch, state=AgentState(user_request="Calculate 2 + 2"), pl=pl, ex=ex)

@pytest.mark.asyncio
async def test_t1_retry_then_success():
    h = _mk(exec_side=[_bad(_transient()), _ok("tool-ok", step_id=0, tool="python_interpreter"), _ok("4", step_id=1)])
    run = await h.orch.start(h.state)
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.EXECUTE
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.RETRY
    assert run.plan_runtime.replan_count == 0
    h.pl.replan.assert_not_awaited()
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.EXECUTE
    assert run.current_attempt == 0
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.COMPLETE
    assert h.state.task_completed is True and h.state.final_answer_verified is True
    assert len(run.execution_history) == 3

@pytest.mark.asyncio
async def test_t2_retry_then_replan():
    h = _mk(exec_side=[_bad(_transient()), _bad(_recoverable()), _ok("4", step_id=0)])
    run = await h.orch.start(h.state)
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.EXECUTE
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.RETRY
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.REPLAN
    assert run.plan_runtime.replan_count == 1 and run.plan_runtime.plan_version == 2
    assert run.current_attempt == 0
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.COMPLETE

@pytest.mark.asyncio
async def test_t7_max_attempts_terminates():
    h = _mk(exec_side=_bad(_transient()))
    run = await h.orch.start(h.state)
    outs = []
    for _ in range(8):
        o = await h.orch.step(h.state, run)
        outs.append(o)
        if o.terminal:
            break
    assert outs[-1].action is OrchestrationAction.FAIL
    assert h.ex.execute.await_count == 3
    assert h.state.task_completed is False

@pytest.mark.asyncio
async def test_t13_retry_then_terminal_idempotent():
    h = _mk(exec_side=[_bad(_transient()), _ok("tool-ok", step_id=0, tool="python_interpreter"), _ok("4", step_id=1)])
    run = await h.orch.start(h.state)
    await h.orch.step(h.state, run)
    await h.orch.step(h.state, run)
    await h.orch.step(h.state, run)
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.COMPLETE
    assert [r.step_id for r in run.execution_history[:2]] == [0, 0]
    n = h.ex.execute.await_count
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.TERMINATE
    assert h.ex.execute.await_count == n

