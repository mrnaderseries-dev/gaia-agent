from __future__ import annotations

from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.core.policies.termination import (
    TerminationDecision,
    TerminationPolicy,
    TerminationState,
    TerminationReason,
)


class AgentLoop:

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        termination_policy: TerminationPolicy | None = None,
        max_iterations: int | None = None,
    ) -> None:

        self.orchestrator = orchestrator
        if termination_policy is not None:
            self.termination_policy = termination_policy
        else:
        
            self.termination_policy = termination_policy 
        self.max_iterations = max_iterations
        self._state: AgentState | None = None

    async def run(
        self,
        state: AgentState,
    ) -> AgentState:

        user_req = (
            getattr(state, "user_request", None) 
            or getattr(state, "user_question", None) 
            or getattr(state, "question", None)
        )
        
        if not user_req or not str(user_req).strip():
            setattr(state, "user_request", "Perform the evaluation task.")
        else:
            setattr(state, "user_request", user_req)

        self._state = state
        
        self.orchestrator.bind_state(state)
        self.orchestrator.emit_agent_started()

        termination = None

        while True:

            termination = self.check_termination()

            if termination.should_stop:
                break

            if not getattr(state, "plan", None) or len(state.plan) == 0:
                if hasattr(self.orchestrator, "generate_initial_plan"):
                    await self.orchestrator.generate_initial_plan()
                elif hasattr(self.orchestrator, "plan"):
                    pass

            await self.orchestrator.run_iteration()

            state.iteration += 1
            
            print("\n--- ITERATION ---")
            print("iteration:", state.iteration)
            print("plan:", state.plan)
            print("current_step:", state.current_step)
            print("current_action:", state.current_action)
            print("step_type:", state.step_type)
            print("tool_result:", state.tool_result)
            print("tool_error:", state.tool_error)
            print("final_answer:", state.final_answer)
            print("final_answer_ready:", state.final_answer_ready)
            print("final_answer_verified:", state.final_answer_verified)
            print("blocked:", state.blocked)

        if termination and (
            termination.reason == TerminationReason.COMPLETED
            and state.final_answer_verified
        ):
            self.orchestrator.emit_agent_completed()

        return state

    def check_termination(
        self,
    ) -> TerminationDecision:

        state = self._require_state()

        termination_state = TerminationState(
            iteration=state.iteration,
            final_answer_ready=(
                getattr(
                    state,
                    "final_answer_ready",
                    False,
                )
            ),
            final_answer_verified=(
                getattr(
                    state,
                    "final_answer_verified",
                    False,
                )
            ),
            verification_attempts=(
                getattr(
                    state,
                    "verification_attempts",
                    0,
                )
            ),
            fatal_error=(
                getattr(
                    state,
                    "fatal_error",
                    False,
                )
            ),
            human_aborted=(
                getattr(
                    state,
                    "human_aborted",
                    False,
                )
            ),
            explicit_stop=(
                getattr(
                    state,
                    "explicit_stop",
                    False,
                )
            ),
            timed_out=(
                getattr(
                    state,
                    "timed_out",
                    False,
                )
            ),
        )

        decision = self.termination_policy.evaluate(
            termination_state
        )

        if decision.should_stop:
            state.termination_reason = decision.reason

        return decision

    def _require_state(
        self,
    ) -> AgentState:

        if self._state is None:
            raise RuntimeError(
                "AgentState is not bound."
            )

        return self._state