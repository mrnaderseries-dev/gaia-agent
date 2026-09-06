from __future__ import annotations

from dataclasses import fields

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
        self.termination_policy = termination_policy
        self.max_iterations = max_iterations
        self._state: AgentState | None = None

    @property
    def state(self) -> AgentState | None:
        return self._state

    @state.setter
    def state(
        self,
        value: AgentState,
    ) -> None:
        self._state = value

    # -----------------------------------------------------------------
    # Diagnostic helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _safe_repr(
        value,
        max_chars: int = 12000,
    ) -> str:

        try:
            text = repr(value)
        except Exception as exc:
            text = (
                f"<repr failed: "
                f"{type(exc).__name__}: {exc}>"
            )

        if len(text) > max_chars:
            return (
                text[:max_chars]
                + (
                    f"\n... [TRUNCATED; "
                    f"original length={len(text)}]"
                )
            )

        return text

    @classmethod
    def _print_state_snapshot(
        cls,
        state: AgentState,
        label: str,
    ) -> None:

        print()
        print("=" * 100)
        print(label)
        print("=" * 100)

        try:
            for field_info in fields(state):

                name = field_info.name

                try:
                    value = getattr(
                        state,
                        name,
                    )
                except Exception as exc:
                    value = (
                        f"<read failed: "
                        f"{type(exc).__name__}: {exc}>"
                    )

                print(
                    f"[STATE] {name} = "
                    f"{cls._safe_repr(value)}"
                )

        except Exception as exc:

            print(
                "[STATE SNAPSHOT ERROR]",
                f"{type(exc).__name__}: {exc}",
            )

        print(
            "=" * 100,
            flush=True,
        )

    # -----------------------------------------------------------------
    # Main agent loop
    # -----------------------------------------------------------------

    async def run(
        self,
        state: AgentState,
    ) -> AgentState:

        user_req = (
            getattr(
                state,
                "user_request",
                None,
            )
            or getattr(
                state,
                "user_question",
                None,
            )
            or getattr(
                state,
                "question",
                None,
            )
        )

        if (
            not user_req
            or not str(user_req).strip()
        ):
            setattr(
                state,
                "user_request",
                "Perform the evaluation task.",
            )
        else:
            setattr(
                state,
                "user_request",
                user_req,
            )

        self.state = state

        self.orchestrator.bind_state(
            state
        )

        self.orchestrator.emit_agent_started()

        termination = None

        print()
        print("=" * 100)
        print("AGENT EXECUTION STARTED")
        print("=" * 100)
        print(
            "[REQUEST]",
            state.user_request,
            flush=True,
        )

        self._print_state_snapshot(
            state,
            "INITIAL AGENT STATE",
        )

        while True:

            # ---------------------------------------------------------
            # Check termination BEFORE next iteration
            # ---------------------------------------------------------

            termination = (
                self.check_termination()
            )

            print()
            print(
                "[TERMINATION CHECK]",
                f"should_stop={termination.should_stop}",
                f"reason={termination.reason}",
                flush=True,
            )

            if termination.should_stop:

                print()
                print(
                    "[TERMINATION] Agent loop stopping.",
                    flush=True,
                )

                self._print_state_snapshot(
                    state,
                    "STATE AT TERMINATION",
                )

                break

            # ---------------------------------------------------------
            # Generate initial plan if necessary
            # ---------------------------------------------------------

            if (
                not getattr(
                    state,
                    "plan",
                    None,
                )
                or len(state.plan) == 0
            ):

                print()
                print(
                    "[PLANNER] No plan available.",
                    flush=True,
                )

                if hasattr(
                    self.orchestrator,
                    "generate_initial_plan",
                ):

                    print(
                        "[PLANNER] Generating initial plan...",
                        flush=True,
                    )

                    await (
                        self.orchestrator
                        .generate_initial_plan()
                    )

                    print(
                        "[PLANNER] Initial plan generated.",
                        flush=True,
                    )

                elif hasattr(
                    self.orchestrator,
                    "plan",
                ):

                    print(
                        "[PLANNER] Orchestrator.plan exists; "
                        "continuing.",
                        flush=True,
                    )

            # ---------------------------------------------------------
            # Current iteration
            # ---------------------------------------------------------

            current_iteration = (
                getattr(
                    state,
                    "iteration",
                    0,
                )
            )

            print()
            print("=" * 100)
            print(
                f"ITERATION {current_iteration + 1}"
            )
            print("=" * 100)

            print(
                "[ITERATION BEFORE]",
                current_iteration,
                flush=True,
            )

            self._print_state_snapshot(
                state,
                "STATE BEFORE ORCHESTRATOR ITERATION",
            )

            # ---------------------------------------------------------
            # Execute orchestrator iteration
            # ---------------------------------------------------------

            print()
            print(
                "[ORCHESTRATOR] run_iteration() START",
                flush=True,
            )

            try:

                await (
                    self.orchestrator
                    .run_iteration()
                )

            except Exception as exc:

                print()
                print(
                    "[ORCHESTRATOR ERROR]",
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                self._print_state_snapshot(
                    state,
                    "STATE AFTER ORCHESTRATOR EXCEPTION",
                )

                raise

            print()
            print(
                "[ORCHESTRATOR] run_iteration() END",
                flush=True,
            )

            state.iteration += 1

            print()
            print(
                "[ITERATION AFTER]",
                state.iteration,
                flush=True,
            )

            # ---------------------------------------------------------
            # Full post-iteration state
            # ---------------------------------------------------------

            self._print_state_snapshot(
                state,
                f"FULL STATE AFTER ITERATION "
                f"{state.iteration}",
            )

            # ---------------------------------------------------------
            # Important individual diagnostic fields
            # ---------------------------------------------------------

            print()
            print("-" * 100)
            print("ITERATION SUMMARY")
            print("-" * 100)

            print(
                "[PLAN]",
                self._safe_repr(
                    state.plan
                ),
                flush=True,
            )

            print(
                "[CURRENT STEP]",
                state.current_step,
                flush=True,
            )

            print(
                "[CURRENT ACTION]",
                self._safe_repr(
                    state.current_action
                ),
                flush=True,
            )

            print(
                "[STEP TYPE]",
                self._safe_repr(
                    state.step_type
                ),
                flush=True,
            )

            print(
                "[TOOL NAME]",
                state.tool_name,
                flush=True,
            )

            print(
                "[TOOL ARGUMENTS]",
                self._safe_repr(
                    state.tool_arguments
                ),
                flush=True,
            )

            print(
                "[TOOL RESULT]",
                self._safe_repr(
                    state.tool_result
                ),
                flush=True,
            )

            print(
                "[TOOL ERROR]",
                self._safe_repr(
                    state.tool_error
                ),
                flush=True,
            )

            print(
                "[EXECUTION SUCCESS]",
                state.execution_success,
                flush=True,
            )

            print(
                "[STEP SUCCEEDED]",
                state.step_succeeded,
                flush=True,
            )

            print(
                "[BLOCKED]",
                state.blocked,
                flush=True,
            )

            print(
                "[WAITING FOR APPROVAL]",
                state.waiting_for_approval,
                flush=True,
            )

            print(
                "[RETRY COUNT]",
                state.retry_count,
                flush=True,
            )

            print(
                "[RECOVERY ATTEMPTED]",
                state.recovery_attempted,
                flush=True,
            )

            print(
                "[REPLAN COUNT]",
                state.replan_count,
                flush=True,
            )

            print(
                "[LOOP SALVAGE ATTEMPTED]",
                state.loop_salvage_attempted,
                flush=True,
            )

            print(
                "[SAME FAILURE COUNT]",
                state.same_failure_count,
                flush=True,
            )

            print(
                "[SAME PLAN COUNT]",
                state.same_plan_count,
                flush=True,
            )

            print(
                "[LAST FAILURE KEY]",
                self._safe_repr(
                    state.last_failure_key
                ),
                flush=True,
            )

            print(
                "[FINAL ANSWER]",
                self._safe_repr(
                    state.final_answer
                ),
                flush=True,
            )

            print(
                "[FINAL ANSWER READY]",
                state.final_answer_ready,
                flush=True,
            )

            print(
                "[FINAL ANSWER VERIFIED]",
                state.final_answer_verified,
                flush=True,
            )

            print(
                "[VERIFICATION ATTEMPTS]",
                state.verification_attempts,
                flush=True,
            )

            print(
                "[TASK COMPLETED]",
                state.task_completed,
                flush=True,
            )

            print(
                "[FATAL ERROR]",
                state.fatal_error,
                flush=True,
            )

            print(
                "[TERMINATION REASON]",
                state.termination_reason,
                flush=True,
            )

            print(
                "[EVIDENCE]",
                self._safe_repr(
                    state.evidence
                ),
                flush=True,
            )

            print(
                "[EXECUTION RESULTS]",
                self._safe_repr(
                    state.execution_results
                ),
                flush=True,
            )

            print(
                "[MESSAGES]",
                self._safe_repr(
                    state.messages
                ),
                flush=True,
            )

            print(
                "-" * 100,
                flush=True,
            )

        # -------------------------------------------------------------
        # Final agent completion
        # -------------------------------------------------------------

        if termination and (
            termination.reason
            == TerminationReason.COMPLETED
            and state.final_answer_verified
        ):
            self.orchestrator.emit_agent_completed()

        print()
        print("=" * 100)
        print("AGENT EXECUTION FINISHED")
        print("=" * 100)

        self._print_state_snapshot(
            state,
            "FINAL AGENT STATE",
        )

        return state

    # -----------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------

    def check_termination(
        self,
    ) -> TerminationDecision:

        state = self._require_state()

        termination_state = TerminationState(
            iteration=getattr(
                state,
                "iteration",
                0,
            ),
            final_answer_ready=getattr(
                state,
                "final_answer_ready",
                False,
            ),
            final_answer_verified=getattr(
                state,
                "final_answer_verified",
                False,
            ),
            verification_attempts=getattr(
                state,
                "verification_attempts",
                0,
            ),
            fatal_error=getattr(
                state,
                "fatal_error",
                False,
            ),
            human_aborted=getattr(
                state,
                "human_aborted",
                False,
            ),
            explicit_stop=getattr(
                state,
                "explicit_stop",
                False,
            ),
            timed_out=getattr(
                state,
                "timed_out",
                False,
            ),
        )

        if self.termination_policy is not None:
            decision = (
                self.termination_policy
                .evaluate(
                    termination_state
                )
            )

        else:

            decision = TerminationDecision(
                should_stop=False
            )

        if decision.should_stop:

            state.termination_reason = (
                decision.reason
            )

        return decision

    # -----------------------------------------------------------------
    # State requirement
    # -----------------------------------------------------------------

    def _require_state(
        self,
    ) -> AgentState:

        if self._state is None:
            raise RuntimeError(
                "AgentState is not bound."
            )

        return self._state