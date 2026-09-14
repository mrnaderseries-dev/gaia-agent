"""Invalid-replan atomicity: invalid plan must not mutate prior runtime."""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gaia_agent.agents.verifier import VerificationResult, VerificationStatus
from gaia_agent.context.models import FinalContext
from gaia_agent.core.agent_execution import ExecutionResult
from gaia_agent.core.agent_state import AgentPhase, AgentState
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
def _initial_plan():
    return PlanSchema(steps=[PlanStep(step_id=0, action="Run tool", step_type=StepType.TOOL, tool_name="python_interpreter", arguments={"code": "1+1"}), PlanStep(step_id=1, action="Final", step_type=StepType.LLM, is_final_answer=True)])
def _mk(*, exec_result, invalid_result):
    cb = MagicMock(); cb.build = AsyncMock(return_value=FinalContext(items=[{"t": 1}], token_count=1))
    pl = MagicMock(); pl.generate_plan = AsyncMock(return_value=PlanningResult(plan=_initial_plan(), task_analysis=_analysis())); pl.replan = AsyncMock(return_value=invalid_result)
    ex = MagicMock(); ex.execute = AsyncMock(return_value=exec_result)
    rel = ReliabilityEngine(error_handler=ErrorHandler(), failure_classifier=FailureClassifier(), retry_policy=RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0), recovery_policy=RecoveryPolicy(allow_replanning=True), retry=Retry(jitter_ratio=0.0), recovery=Recovery(error_handler=ErrorHandler()))
    loop = MagicMock(); loop.check = MagicMock(return_value=LoopDetection(detected=False)); loop.record = MagicMock()
    ver = MagicMock(); ver.verify = AsyncMock(return_value=VerificationResult(status=VerificationStatus.VERIFIED))
    orch = Orchestrator(context_builder=cb, planner=pl, agent_execution=ex, reliability_engine=rel, loop_detector=loop, verifier=ver, observability=None, config=OrchestratorConfig())
    return SimpleNamespace(orch=orch, state=AgentState(user_request="Calculate 2 + 2"))
@pytest.mark.asyncio
async def test_invalid_replan_does_not_mutate_prior_runtime():
    rec = AgentError(error_type="R", message="bad args", category=ErrorCategory.TOOL_ARGUMENT_ERROR, severity=ErrorSeverity.HIGH, retryable=False, recoverable=True, source="t", operation="e")
    failing = ExecutionResult(success=False, output=None, evidence=(), error=rec, step_id=0, tool_name="python_interpreter")
    bad_plan = PlanSchema.model_construct(steps=[])
    assert isinstance(bad_plan, PlanSchema)
    invalid = PlanningResult(plan=bad_plan, task_analysis=_analysis())
    h = _mk(exec_result=failing, invalid_result=invalid)
    run = await h.orch.start(h.state)
    assert (await h.orch.step(h.state, run)).action is OrchestrationAction.EXECUTE
    old_plan = run.plan_runtime.plan
    old_ver = run.plan_runtime.plan_version
    old_re = run.plan_runtime.replan_count
    old_cur = run.plan_runtime.current_step
    old_done = set(run.plan_runtime.completed_steps)
    old_splan = list(h.state.plan)
    assert old_plan is not None and old_ver == 1
    out = await h.orch.step(h.state, run)
    assert out.action is OrchestrationAction.FAIL
    assert run.plan_runtime.plan is old_plan
    assert run.plan_runtime.plan_version == old_ver
    assert run.plan_runtime.replan_count == old_re
    assert run.plan_runtime.current_step == old_cur
    assert run.plan_runtime.completed_steps == old_done
    assert h.state.plan == old_splan
    assert h.state.phase is AgentPhase.FAILED
    assert getattr(out.error, "error_type", "") == "InvalidReplannedPlan"
