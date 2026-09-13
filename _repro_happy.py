import asyncio, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
sys.path.insert(0, "src")

from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from gaia_agent.observability.facade import ObservabilityFacade
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.models import PlanningResult
from gaia_agent.planner.task_classifier import TaskAnalysis, TaskIntent
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.failure_classifier import FailureClassification
from gaia_agent.agents.verifier import VerificationResult, VerificationStatus
from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator
from gaia_agent.context.ContextCompressor import ContextCompressor
from gaia_agent.llm.model import LLMModel


def _final_plan():
    return PlanSchema(steps=[PlanStep(step_id=0, action="produce final answer", step_type=StepType.LLM, is_final_answer=True)])

def _analysis():
    return TaskAnalysis(intent=TaskIntent.SELF_CONTAINED, needs_external_info=False, recommended_first_tool=None, forbidden_tools=(), analysis_text="t")

def _ctx():
    class DummyClient:
        async def generate(self, *_a, **_k):
            return ""
    model = LLMModel(provider="t", model="t", max_tokens=100, temperature=0.0)
    pol = ContextPolicy(); bud = ContextBudget(max_tokens=12000); val = ContextValidator(budget=bud)
    comp = ContextCompressor(client=DummyClient(), model=model, budget=bud, policy=pol)
    def _src(v):
        s = MagicMock(); s.get = AsyncMock(return_value=v); return s
    return ContextBuilder(policy=pol, budget=bud, validator=val, compressor=comp,
        attachment_source=_src([]), conversation_source=_src([]), history_source=_src([]),
        memory_source=_src([]), runtime_source=_src([{"source": "integration-test"}]))

def _execution(llm_output="42"):
    ep = MagicMock(); ep.evaluate.return_value = SimpleNamespace(allowed=True, message="", reason="t")
    llm = MagicMock(); llm.execute = AsyncMock(return_value=llm_output)
    execution = AgentExecution(tool_registry=MagicMock(), execution_policy=ep, risk_assessor=None,
        approval_policy=None, llm_executor=llm, event_logger=None, metrics=None, tracer=None,
        token_tracker=None, error_handler=ErrorHandler())
    return execution, llm

def _reliability():
    fc = MagicMock(); fc.classify.return_value = MagicMock(spec=FailureClassification)
    rp = MagicMock(); rp.evaluate.return_value = SimpleNamespace(should_retry=False, delay=0, reason="t")
    retry = MagicMock(); retry.delay = AsyncMock()
    recpol = MagicMock(); recpol.evaluate.return_value = SimpleNamespace(action=SimpleNamespace(value="stop"), reason="t")
    recovery = MagicMock(); recovery.execute = AsyncMock(return_value=SimpleNamespace(recovered=False, reason="t"))
    return ReliabilityEngine(error_handler=ErrorHandler(), failure_classifier=fc, retry_policy=rp,
        recovery_policy=recpol, retry=retry, recovery=recovery)

def _termination():
    p = MagicMock()
    def evaluate(s):
        if s.final_answer_verified or s.fatal_error or s.human_aborted or s.explicit_stop or s.timed_out:
            return SimpleNamespace(should_stop=True, reason="terminal")
        return SimpleNamespace(should_stop=False, reason="continue")
    p.evaluate.side_effect = evaluate
    return p

def _orchestrator(planner, ae, rel, ver):
    ld = MagicMock(); ld.check.return_value = False
    return Orchestrator(context_builder=_ctx(), planner=planner, agent_execution=ae,
        reliability_engine=rel, loop_detector=ld, verifier=ver,
        observability=MagicMock(spec=ObservabilityFacade),
        config=OrchestratorConfig(max_step_attempts=3, max_verification_attempts=2, max_replans=3))

async def main():
    planner = MagicMock()
    planner.generate_plan = AsyncMock(return_value=PlanningResult(plan=_final_plan(), task_analysis=_analysis()))
    ae, llm = _execution("42")
    ver = MagicMock(); ver.verify = AsyncMock(return_value=VerificationResult(status=VerificationStatus.VERIFIED, reason="ok"))
    orch = _orchestrator(planner, ae, _reliability(), ver)
    loop = AgentLoop(orchestrator=orch, termination_policy=_termination())
    state = AgentState(user_request="What is 42?")
    result = await loop.run(state)
    print("PHASE:", result.phase)
    print("FINAL_ANSWER:", result.final_answer)
    print("FINAL_ANSWER_VERIFIED:", result.final_answer_verified)
    print("TOOL_ERROR:", result.tool_error)
    print("FATAL_ERROR:", result.fatal_error)
    print("TRANSITIONS:", [f"{t.from_phase.value}->{t.to_phase.value}({t.reason.value})" for t in state.transition_history])
    print("VERIFY CALLED:", ver.verify.called)
    print("LLM CALLED TIMES:", llm.execute.await_count)

asyncio.run(main())