"""Apply atomic replan fix to orchestrator.py"""
import pathlib

p = pathlib.Path("src/gaia_agent/core/orchestration/orchestrator.py")
t = p.read_text(encoding="utf-8")

# --- Fix 1: _install_planning_result atomicity ---
old1 = '''    def _install_planning_result(
        self,
        state: AgentState,
        run: OrchestrationContext,
        planning_result: Any,
    ) -> None:
        run.install_planning_result(
            planning_result
        )

        plan = run.plan_runtime.plan

        if plan is None:
            raise ValueError(
                "Planner returned no plan."
            )

        self._validate_plan(plan)

        if state.phase is not AgentPhase.PLANNING:'''

new1 = '''    def _install_planning_result(
        self,
        state: AgentState,
        run: OrchestrationContext,
        planning_result: Any,
    ) -> None:
        # Atomic install: validate the candidate BEFORE mutating
        # OrchestrationContext / PlanRuntime. An invalid plan must never
        # become the active plan (no version bump, no reset, no clearing).
        candidate = getattr(planning_result, "plan", None)

        if candidate is None:
            raise ValueError(
                "Planner returned no plan."
            )

        if not isinstance(candidate, PlanSchema):
            raise TypeError(
                "Planning result must contain a PlanSchema."
            )

        self._validate_plan(candidate)

        run.install_planning_result(
            planning_result
        )

        plan = run.plan_runtime.plan

        if state.phase is not AgentPhase.PLANNING:'''

assert old1 in t, "old1 not found"
t = t.replace(old1, new1, 1)

# --- Fix 2: _plan try-block separation ---
old2 = '''        try:
            planning_result = await self.planner.generate_plan(
                run.user_request,
                context,
            )

            self._install_planning_result(
                state,
                run,
                planning_result,
            )

        except PlannerRecoveryRequired as exc:
            error = self._planner_error(
                exc,
                operation="generate_plan",
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.PLAN_REJECTED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )'''

new2 = '''        try:
            planning_result = await self.planner.generate_plan(
                run.user_request,
                context,
            )
        except PlannerRecoveryRequired as exc:
            error = self._planner_error(
                exc,
                operation="generate_plan",
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.PLAN_REJECTED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        try:
            self._install_planning_result(
                state,
                run,
                planning_result,
            )
        except PlannerRecoveryRequired as exc:
            error = self._planner_error(
                exc,
                operation="generate_plan",
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.PLAN_REJECTED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )

        except (TypeError, ValueError) as exc:
            error = AgentError(
                error_type="InvalidPlan",
                message=str(exc) or "Planner returned an invalid plan.",
                category=ErrorCategory.PLAN_RECOVERY_ERROR,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                recoverable=False,
                source="Orchestrator",
                operation="generate_plan",
                original_exception=exc,
            )

            self._fail(
                state,
                error,
            )

            self._emit(
                EventType.PLAN_REJECTED,
                run,
                error=error,
            )

            return OrchestrationOutcome(
                action=OrchestrationAction.FAIL,
                error=error,
                reason=error.message,
            )'''

assert old2 in t, "old2 not found"
t = t.replace(old2, new2, 1)

p.write_text(t, encoding="utf-8")
print("Both fixes applied successfully.")
