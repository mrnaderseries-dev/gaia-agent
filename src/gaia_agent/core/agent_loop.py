from __future__ import annotations

from gaia_agent.core.agent_state import (
    AgentPhase,
    AgentState,
)
from gaia_agent.core.orchestration.models import (
    OrchestrationAction,
    OrchestrationContext,
    OrchestrationOutcome,
)
from gaia_agent.core.orchestration.orchestrator import (
    Orchestrator,
)
from gaia_agent.core.policies.termination import (
    TerminationDecision,
    TerminationPolicy,
    TerminationState,
)


class AgentLoop:
    """
    Top-level lifecycle driver for the agent.

    AgentLoop is intentionally thin.

    It owns:
        - binding the AgentState
        - starting the orchestration run
        - repeatedly asking the Orchestrator for one step
        - asking the TerminationPolicy whether execution may continue
        - stopping on terminal / approval / termination conditions

    It does NOT own:
        - planning
        - task classification
        - strategy selection
        - tool execution
        - retry
        - recovery
        - replanning
        - verification
        - evidence handling

    Those responsibilities belong to the corresponding layers.
    """

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        termination_policy: TerminationPolicy,
    ) -> None:
        self.orchestrator = orchestrator
        self.termination_policy = termination_policy

        self._state: AgentState | None = None
        self._run: OrchestrationContext | None = None

    @property
    def state(self) -> AgentState | None:
        return self._state

    @property
    def run_context(self) -> OrchestrationContext | None:
        return self._run

    async def run_agent(
        self,
        state: AgentState,
    ) -> AgentState:
        self._validate_state(state)

        self._state = state

        try:
            run = await self.orchestrator.start(
                state
            )

            self._run = run

            while True:
                if state.is_terminal:
                    break

                decision = self.check_termination()

                if decision.should_stop:
                    self._terminate_from_policy(
                        state,
                        decision,
                    )
                    break

                outcome = await self.orchestrator.step(
                    state,
                    run,
                )

                if self._handle_outcome(
                    state,
                    run,
                    outcome,
                ):
                    break

            return state

        finally:
            self._cleanup()

    async def run(
        self,
        state: AgentState,
    ) -> AgentState:
        return await self.run_agent(state)

    def check_termination(
        self,
    ) -> TerminationDecision:
        state = self._require_state()

        termination_state = TerminationState(
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

    def _handle_outcome(
        self,
        state: AgentState,
        run: OrchestrationContext,
        outcome: OrchestrationOutcome,
    ) -> bool:
        action = outcome.action

        if action is OrchestrationAction.COMPLETE:
            self._complete(
                state,
                run,
            )
            return True

        if action is OrchestrationAction.FAIL:
            self._fail(
                state,
                outcome,
            )
            return True

        if action is OrchestrationAction.TERMINATE:
            self._terminate(
                state,
                reason=outcome.reason,
            )
            return True

        if (
            action
            is OrchestrationAction.WAIT_FOR_APPROVAL
        ):
            state.waiting_for_approval = True
            state.blocked = True
            return True

        return False

    @staticmethod
    def _complete(
        state: AgentState,
        run: OrchestrationContext,
    ) -> None:
        """Apply the final orchestration projection.

        The Orchestrator normally transitions the state to COMPLETED when
        verification passes.  AgentLoop only reconciles the projection and
        safely handles callers/tests that return COMPLETE immediately after
        verification without duplicating lifecycle ownership.
        """
        if run.final_answer is not None:
            state.final_answer = run.final_answer

        state.final_answer_ready = run.final_answer is not None
        state.final_answer_verified = run.is_verified
        state.verification_attempts = run.verification_attempts

        if not run.is_verified:
            raise RuntimeError(
                "Orchestrator returned COMPLETE without a verified final answer."
            )

        if state.phase is AgentPhase.COMPLETED:
            state.task_completed = True
            return

        if state.phase is AgentPhase.TERMINATED:
            raise RuntimeError("Cannot complete a terminated agent.")

        state.complete()

    @staticmethod
    def _fail(
        state: AgentState,
        outcome: OrchestrationOutcome,
    ) -> None:
        if state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.TERMINATED,
        }:
            return

        if state.phase is AgentPhase.FAILED:
            return

        if outcome.error is not None:
            state.tool_error = str(
                outcome.error
            )

        elif outcome.reason:
            state.tool_error = (
                outcome.reason
            )

        if state.can_transition_to(
            AgentPhase.FAILED
        ):
            state.fail(
                reason=TransitionReason.EXECUTION_FAILED,
                fatal=state.fatal_error,
            )

    @staticmethod
    def _terminate(
        state: AgentState,
        *,
        reason: str | None,
    ) -> None:
        if state.phase is AgentPhase.TERMINATED:
            return

        if state.phase is AgentPhase.COMPLETED:
            return

        if reason is not None:
            state.termination_reason = reason

        if state.can_transition_to(
            AgentPhase.TERMINATED
        ):
            state.terminate()

    @staticmethod
    def _terminate_from_policy(
        state: AgentState,
        decision: TerminationDecision,
    ) -> None:
        state.termination_reason = (
            decision.reason
        )

        if state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.TERMINATED,
        }:
            return

        if state.can_transition_to(
            AgentPhase.TERMINATED
        ):
            state.terminate()

    @staticmethod
    def _validate_state(
        state: AgentState,
    ) -> None:
        if not isinstance(
            state,
            AgentState,
        ):
            raise TypeError(
                "AgentLoop.run() expects AgentState."
            )

        if not state.user_request.strip():
            raise ValueError(
                "AgentState.user_request cannot be empty."
            )

        if state.phase is AgentPhase.TERMINATED:
            raise ValueError(
                "Cannot run an already terminated AgentState."
            )

    def _require_state(
        self,
    ) -> AgentState:
        if self._state is None:
            raise RuntimeError(
                "AgentState is not bound."
            )

        return self._state

    def _cleanup(self) -> None:
        self._state = None
        self._run = None