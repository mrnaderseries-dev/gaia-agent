# debugging_432q1 — Orchestration production-readiness pass

## 1. Current Architecture Understanding

Orchestration is a thin step-driver over explicit ownership boundaries:

- AgentState owns lifecycle phase transitions only.
- Orchestrator owns step driving, plan install, verify, replan orchestration.
- PlanRuntime owns plan, current_step, completed_steps, plan_version, replan_count.
- OrchestrationContext owns task_analysis, plan_runtime, execution/verification
  histories, final_answer, iteration, current_attempt, termination_reason.
- AgentState plan/counters/final_* fields are projections synced from the above.
- Planner contract: generate_plan/replan -> PlanningResult{plan, task_analysis}.
- PlanSchema enforces: sequential step_ids, exactly one final LLM step, last.
- step() increments iteration, short-circuits COMPLETED/FAILED/TERMINATED to
  TERMINATE, builds context, plans if no plan, else executes current step.
- _execute checks loop BEFORE execution (detected -> FAIL, no replan).
  Success records loop, marks complete, advances (non-final), resets attempt.
  Final-answer steps go to _verify.
- _handle_failure delegates to ReliabilityEngine RETRY|REPLAN|STOP.
- _replan enforces max_replans, installs via install_planning_result, bumps
  replan_count, resets attempt.
- _verify records history, syncs state; VERIFIED -> COMPLETE; budget out -> FAIL;
  else recoverable AnswerVerificationFailed -> _replan.
- AgentLoop maps COMPLETE/FAIL/TERMINATE/WAIT and is idempotent on terminal.
## 2. Existing Orchestration Contracts

- start(state, *, run=None) -> OrchestrationContext; IDLE->PLANNING.
- step(state, run) -> OrchestrationOutcome; terminal={COMPLETE,FAIL,TERMINATE}.
- Config{max_step_attempts=3, max_verification_attempts=2, max_replans=3}.
- PlanningResultLike{plan: PlanSchema, task_analysis: TaskAnalysis}.
- install_planning_result: TypeError unless exact types; set_plan(reset=True).
- set_plan: always plan_version+=1; resets position when requested.
- Reliability: RETRY while attempt<max AND transient; REPLAN if recoverable.
- Classifier: TIMEOUT/NETWORK/RATE_LIMIT->TRANSIENT; plan/tool-arg->RECOVERABLE.
- LoopDetector.check pure; record only on success.
- verified=(status==VERIFIED) when status present.
- complete() raises unless final_answer_verified.


## 3. Baseline Test Results

- E2E tests/integration/test_orchestrator_recovery_e2e.py: PASS (1 passed).
- tests/integration full BEFORE fix: 8 failed / 38 passed (stale API only).
- tests/reliability+planner+context+agents BEFORE fix: pre-existing failures
  unrelated to orchestration (verified by stashing 432q1 changes and rerunning:
  test_engine allow_replan kwarg, verifier evidence tuple unpack, etc).

## 4. Debugging Strategy

REAL ReliabilityEngine + REAL runtime/state; mock only CB/Planner/Exec/Loop/
Verifier boundaries. Drive REAL Orchestrator.start/step and AgentLoop.run.
Classify red as A(test)/B(mock)/C(prod fix)/D(architect note).

## 5. Failure Matrix

T1 retry-then-success; T2 retry-fail-replan; T3 replan-raise; T4 verify-invalid;
T5 verify-uncertain; T6 max-replans; T7 max-attempts; T8 loop; T9 invalid-replan;
T10 llm-exec-fail; T11 context-fail; T12a completed; T12b failed; T13 idem;
T14 invalid-initial-plan. Details expanded as tests land.

## 6. Lifecycle Invariants

I1 retry never bumps replan_count; I2 only valid plans installed;
I3 verified only via VERIFIED; I4 completed only verified; I5 terminal never
executes; I6 attempt increments/resets, budget terminates; I7 replan cap blocks;
I8 verify budget exhausts to FAIL; I9 loop to FAIL; I10 iteration synced.

## 7. Planned Tests

Split into focused files (editor size limit; same matrix):

- tests/integration/test_orch_432q1_retry.py: T1, T2, T7, T13.
- tests/integration/test_orch_432q1_replan.py: T3, T6, T9, T14.
- tests/integration/test_orch_432q1_verify.py: T4, T5, T10 + verify-budget.
- tests/integration/test_orch_432q1_edge.py: T8, T11, T12a, T12b.
- Existing E2E recovery test kept as regression baseline (untouched, still green).

## 8. Production Risks

R1: CONFIRMED as test expectation; current_attempt is reset but legacy tests that
assert attempt==1 encode an older contract. New matrix does not depend on it.
R2: CONFIRMED BUG, FIXED (see Production Bugs Found B1).
R3: retry_count/recovery_attempted dead projections (documented, out of scope).
R4: COMPLETED+FAILED both map to TERMINATE action (works via AgentLoop+state).
R5: stale suites encode removed API; fixed for integration scope. Reliability
old-orchestration suites remain stale/out of scope (prove pre-existing).

## Architectural Findings

None requiring redesign. Loop->FAIL (no replan), retry/replan separation,
PlanRuntime ownership, verification gate, and AgentLoop mapping all hold under
the matrix. Stale suites outside integration scope are test debt, not proof of
an architectural defect.

## Test/Mock Mismatches

M1: New matrix initially asserted PLAN on first step; production returns EXECUTE
with reason plan_ready (plan+execute in one step). Fixed tests (Category A).
M2: Old integration suites used bind_state/unbind/run_iteration/_build_context/
_loop_detector, start_recovery/RECOVERING, user_id, ContextBuilder(State),
planner positional args + list context, planner returning bare PlanSchema.
Fixed tests to PlanningResult + ContextRequest + FinalContext contracts.
M3: test_orchestrator_verification* and old reliability orchestration suites
encode a removed orchestrator (MAX_VERIFICATION_ATTEMPTS, _capture_step_result,
_same_execution, plan_id, StepType.FINAL_ANSWER). Left untouched as pre-existing
debt; new matrix covers the behavior with the real contract instead.

## Production Bugs Found

B1 (Category C, genuine bug): `src/gaia_agent/core/orchestration/orchestrator.py`,
`_replan()`: `replan_count` was incremented BEFORE install, and only
`PlannerRecoveryRequired` was caught. An invalid replan therefore raised an
uncaught `TypeError`/`ValueError` AFTER bumping the counter, corrupting budget
accounting and escaping the FAIL contract (T9 red: `replan_count==1` after an
invalid replan).
Fix: install first, then bump `replan_count`/`state.replan_count`/`run.current_attempt`;
catch `(TypeError, ValueError)` from install/validate and convert to terminal `FAIL`
with `InvalidReplannedPlan` (`PLAN_RECOVERY_ERROR`, non-retryable/non-recoverable).
Preserves architecture: no contract change; invalid plans still never become active;
counters now truthful; outcome stays `FAIL`.

B2 (Category C, genuine bug): `_verify_final_answer()` left the failed candidate
in `state.final_answer`/`final_answer_ready` and `run.final_answer` after
`INVALID`/uncertain verification before replanning. A subsequent COMPLETE path or
reader could observe a stale failed answer alongside `replan_count>0`.
Fix: clear `state.final_answer`/`final_answer_ready` and `run.final_answer` before
`_replan()` on the not-verified path. Preserves architecture: verification gate
unchanged; COMPLETE still requires VERIFIED; T4 now asserts cleared candidate.

B3 (Category C, genuine bug): `_install_planning_result()` called
`run.install_planning_result(planning_result)` BEFORE
`_validate_plan(plan)`. An invalid plan could mutate `PlanRuntime` (change
`plan`, increment `plan_version`, reset `current_step`/`completed_steps`)
before validation failed, violating the invariant that an invalid plan must
never become active.
Fix: extract the candidate `plan` attribute, validate `isinstance(PlanSchema)`
and call `_validate_plan(candidate)` BEFORE calling
`run.install_planning_result(...)`. If validation raises, the
`OrchestrationContext` / `PlanRuntime` remains unchanged.

B4 (Category C, genuine bug): `_plan()` combined `planner.generate_plan()` and
`_install_planning_result()` in a single `try` block that only caught
`PlannerRecoveryRequired`. A `TypeError`/`ValueError` from an invalid initial
plan propagated uncaught through `_plan()` to `step()`'s broad
`except Exception`, producing a generic `AgentError` instead of a clear
`InvalidPlan` error.
Fix: split into two `try` blocks — `generate_plan()` with only
`PlannerRecoveryRequired` catch (programming errors propagate as generic),
and `_install_planning_result()` with both `PlannerRecoveryRequired` and
`(TypeError, ValueError)` catches (the latter → `InvalidPlan` error type).

## Atomicity Invariant (Post-Fix Verification)

The full error-handling separation is now consistent across both `_plan` and `_replan`:

- **Planner execution** (`planner.generate_plan()` / `planner.replan()`):
  only `PlannerRecoveryRequired` is caught → `FAIL` with `PlannerRecoveryRequired`
  error. Any other exception (e.g. `TypeError`/`ValueError` from a programming bug
  inside the planner) propagates as a generic internal error, NOT mislabeled
  as an invalid plan.

- **Returned plan validation** (`_install_planning_result()` → candidate extraction,
  `_validate_plan()`, install): catches `(TypeError, ValueError)` → `FAIL` with
  `InvalidPlan` (initial plan) or `InvalidReplannedPlan` (replanned plan).

- **`replan_count`** is incremented ONLY after `_install_planning_result()` succeeds,
  so an invalid replan leaves `plan_version`, `replan_count`, `current_step`,
  `completed_steps`, `plan`, and `state.plan` all unchanged.

New test: `test_orchestrator_invalid_replan_atomicity.py` proves this by
simulating a replan that returns an invalid empty `PlanSchema` and asserting
that `plan_version`, `replan_count`, `current_step`, `completed_steps`,
`state.plan`, and `state.phase` are all preserved, with `outcome.action is FAIL`
and `error_type == "InvalidReplannedPlan"`.

## Final Test Results

- `pytest tests/integration -q -p no:cacheprovider`: **47 passed** (46 original + 1 new atomicity test).
- `pytest tests/integration/test_orchestrator_recovery_e2e.py -q -p no:cacheprovider`: **1 passed** (baseline preserved).
- `pytest tests/integration/test_orchestrator_invalid_replan_atomicity.py -v -p no:cacheprovider`: **1 passed**.
- Broader reliability/planner/context/agents suites: pre-existing failures
  unrelated to orchestration (proven by stash+rerun); intentionally not redesigned.

## Remaining Risks

- Old tests/reliability orchestration + verification suites still encode the
  removed API; they will keep failing at collection until separately updated.
- AgentState.retry_count/recovery_attempted remain unsynced projections.
- Verifier semantics (deterministic gate vs LLM) were exercised only via mocks;
  no live-LLM run in this environment.

## Final Assessment

Orchestration lifecycle is production-ready for the tested scenarios: retry,
replan, budgets, loop, invalid-plan atomicity, final-answer gating,
context-failure containment, terminal idempotency, and E2E recovery all pass
with minimal contract-preserving fixes (B1/B2) and corrected contracts (M1/M2).
Broader repo suites have unrelated pre-existing debt and must not be read as
orchestration regressions.

- tests/reliability old orchestration tests: stale imports/APIs.
