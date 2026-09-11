from __future__ import annotations

import logging
from typing import Any

from gaia_agent.core.agent_state import AgentState, AgentPhase
from gaia_agent.core.orchestration.orchestrator import AgentOrchestrator
from gaia_agent.core.policies.termination import TerminationReason

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
    ) -> None:
        self.orchestrator = orchestrator
        self.state: AgentState | None = None

    async def run(
        self,
        state: AgentState,
    ) -> AgentState:
        if not isinstance(state, AgentState):
            raise TypeError(
                "AgentLoop.run() requires an AgentState."
            )

        user_request = state.user_request.strip()

        if not user_request:
            raise ValueError(
                "AgentState.user_request cannot be empty."
            )

        state.user_request = user_request
        self.state = state

        self.orchestrator.bind_state(state)
        self.orchestrator.emit_agent_started()

        termination = None

        while True:
            termination = self.check_termination()

            if termination.should_stop:
                break

            if not state.plan:
                await self.orchestrator.generate_initial_plan()

            current_iteration = state.iteration

            await self.orchestrator.run_iteration()

            state.iteration = current_iteration + 1

        if (
            termination
            and termination.reason == TerminationReason.COMPLETED
            and state.final_answer_verified
        ):
            self.orchestrator.emit_agent_completed()

        return state

    def check_termination(self) -> Any:
        if self.state is None:
            class DummyTermination:
                should_stop = True
                reason = TerminationReason.TERMINATION
            return DummyTermination()

        if hasattr(self.orchestrator, "check_termination"):
            return self.orchestrator.check_termination()

        class DefaultTermination:
            should_stop = self.state.is_terminal
            reason = self.state.termination_reason
        return DefaultTermination()