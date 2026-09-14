import sys
import asyncio

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from unittest.mock import AsyncMock, MagicMock

from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.core.orchestration.models import OrchestrationContext, OrchestrationAction


async def main() -> None:
    context_builder = MagicMock()
    context_builder.build = AsyncMock(side_effect=RuntimeError("boom"))

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=MagicMock(),
        agent_execution=MagicMock(),
        reliability_engine=MagicMock(),
        loop_detector=MagicMock(),
        verifier=MagicMock(),
    )

    state = AgentState(user_request="test request")
    run = OrchestrationContext(user_request="test request")

    outcome = await orchestrator.step(state, run)

    print("ACTION", outcome.action)
    print("REASON", outcome.reason)
    print("ERROR_TYPE", type(outcome.error).__name__ if outcome.error else None)
    print("STATE_PHASE", state.phase)

    assert outcome.action is OrchestrationAction.FAIL
    assert isinstance(outcome.error, Exception)
    assert type(outcome.error).__name__ == "AgentError"
    print("CATCHALL_OK")


asyncio.run(main())
