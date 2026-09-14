import sys
import asyncio

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")
sys.path.insert(0, r"c:\Users\user\gaia-agent\tests\integration")

from unittest.mock import AsyncMock, MagicMock

from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentPhase
from gaia_agent.agents.verifier import VerificationResult, VerificationStatus

import test_real_orchestration_flow as tf


async def main() -> None:
    planner = MagicMock()
    plan = tf._make_final_plan()
    planner.generate_plan = AsyncMock(return_value=tf._make_planning_result(plan))

    agent_execution, llm_executor = tf._make_real_agent_execution(llm_output="42")
    reliability = tf._make_real_reliability_engine()
    verifier = tf._make_verifier(
        VerificationResult(status=VerificationStatus.VERIFIED, reason="integration verified")
    )
    orchestrator = tf._make_orchestrator(
        planner=planner,
        agent_execution=agent_execution,
        reliability_engine=reliability,
        verifier=verifier,
    )
    loop = AgentLoop(orchestrator=orchestrator, termination_policy=tf._make_termination_policy())

    state = tf._make_state()
    result = await loop.run(state)

    print("PHASE", result.phase)
    print("TOOL_ERROR", repr(result.tool_error))
    print("TERMINATION_REASON", repr(result.termination_reason))
    print("FINAL_ANSWER", repr(result.final_answer))
    print("FINAL_ANSWER_VERIFIED", result.final_answer_verified)
    print("EXEC_RESULTS", [(r.success, type(r.error).__name__ if r.error else None, str(r.error)[:200] if r.error else None) for r in result.execution_results])
    run = loop.run
    if run is not None:
        print("RUN_OUTCOMES?")
        print("RUN_FINAL_ANSWER", repr(run.final_answer))
        print("RUN_VERIFICATIONS", [(v.status.value, v.result.reason) for v in run.verification_history])
        print("RUN_EXECUTIONS", [(rec.step_id, rec.result.success, type(rec.result.error).__name__ if rec.result.error else None, str(rec.result.error)[:300] if rec.result.error else None) for rec in run.execution_history])
        print("TRANSITIONS", [(t.from_phase.value if hasattr(t, "from_phase") else "?", t.to_phase.value if hasattr(t, "to_phase") else "?") for t in result.transition_history])


asyncio.run(main())
