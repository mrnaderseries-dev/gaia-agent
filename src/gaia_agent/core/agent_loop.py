from __future__ import annotations

from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
)
from gaia_agent.core.orchestration.orchestrator import (
    Orchestrator,
)
from gaia_agent.core.policies.termination import (
    TerminationDecision,
    TerminationPolicy,
    TerminationReason,
    TerminationState,
)


class AgentLoop:

    def init(
        self,
        *,
        orchestrator: Orchestrator,
        termination_policy: TerminationPolicy,
    ) -> None:

        self.orchestrator = orchestrator
        self.termination_policy = (
            termination_policy
        )

        self._state: AgentState | None = None

    @property
    def state(self):
        return self._state

    async def run(
        self,
        state: AgentState,
    ) -> AgentState:

        if not state.user_request.strip():
            raise ValueError(
                "AgentState.user_request cannot be empty."
            )

        self._state = state

        self.orchestrator.bind_state(
            state
        )

        self.orchestrator.emit_agent_started()

        try:

            while True:

                decision = (
                    self.check_termination()
                )

                if decision.should_stop:
                    break

                await self.orchestrator.run_iteration()

                

            if (
                state.phase
                == AgentPhase.COMPLETED
            ):

                self.orchestrator.emit_agent_completed()

            return state

        finally:

            self.orchestrator.unbind()

    def check_termination(
        self,
    ) -> TerminationDecision:

        state = self._require_state()

        termination_state = (
            TerminationState(
                iteration=state.iteration,
                final_answer_ready=(
                    state.final_answer_ready
                ),
                final_answer_verified=(
                    state.final_answer_verified
                ),
                fatal_error=(
                    state.fatal_error
                ),
                human_aborted=(
                    state.human_aborted
                ),
                explicit_stop=(
                    state.explicit_stop
                ),
                timed_out=(
                    state.timed_out
                ),
                verification_attempts=(
                    state.verification_attempts
                ),
            )
        )

        decision = (
            self.termination_policy.evaluate(
                termination_state
            )
        )

        if decision.should_stop:
            state.termination_reason = (
                decision.reason
            )

        return decision

    def _require_state(
        self,
    ) -> AgentState:

        if self._state is None:
            raise RuntimeError(
                "AgentState is not bound."
            )

        return self._state