# DEBUGGING_TEST.md — GAIA Wave 1 · Core Correctness Forensic Diagnosis

> **Artifact type:** DIAGNOSTIC REPORT ONLY. No production source file was modified during this
> investigation. No test was modified or deleted. This document is the hand-off artifact for Wave 1
> implementation: a senior engineer must be able to implement the fix plan below **without repeating this
> investigation**.

- **Repository:** https://github.com/mrnaderseries-dev/gaia-agent
- **Commit examined:** `1b40b08` ("Create context_manager.py"), branch `main`
- **Date of investigation:** 9 Sep 2026
- **Environment:** Windows, Python 3.12 (`.venv`), pytest 9.1.1 (pytest-timeout NOT installed)
- **Files examined:** `src/gaia_agent/**` production tree, `tests/**`, root/`src` evaluation harnesses,
  archived `evaluation_results.jsonl`
- **Commands executed (read-only diagnostics):** `git log/status`, `Select-String` field audits over `src`+`tests`,
  `pytest tests/planner` (**29 passed**), `pytest tests/reliability` (**95 passed / 7 failed**),
  `pytest tests/integration tests/planner/test_planner_runtime_contract.py` (**6 passed**)
- **No network/LLM-dependent test was executed (ollama is not running in this environment)**.

---

## 0. EXECUTIVE SUMMARY (TL;DR)

GAIA currently produces **100 % failed evaluation runs** (archived `evaluation_results.jsonl` shows every single
record ending in `AttributeError: 'AgentState' object has no attribute 'iteraion'` —the historical typo has since been
removed from `src` but was never regression-locked). Beyond that historical crash, the current Core has **five
root causes** that explain "incorrect states, repeated work, lost history, incorrect termination, unreliable
evaluation":

1. **P0 — Recovery/replanning for tool-execution & plan-generation failures is dead-wired.**
   `ReliabilityEngine._execute_recovery()` requires `recovery_change_detector` and refuses to run recovery when it is
   `None` (`reliability/engine.py:122-133`). All three production `reliability_engine.execute()` call sites pass
   `recovery_operation` but **never pass `recovery_change_detector`**
   (`orchestrator.py:222-228`, `orchestrator.py:286-295`, `orchestrator.py:1104-1111`). Result: a tool
   failure never triggers `_recover_execution()`/`_replan_full_plan()`;the `ReliabilityEngine` exhausts its
   retry budget and returns "Recovery operation requires a change detector";the orchestrator merely sets
   `tool_error` and returns;the **same step is re-attempted next iteration** until `MAX_ITERATIONS` (20) stops
   the loop at an unrelated "max_iterations" terminal. Recovery/RecoveryPolicy/Replanning exist but
   **cannot fire**. (Verification-recovery and loop-recovery call `planner.replan_step()` directly и DO work;only
   the engine-mediated tool/plan recovery is dead.)

2. **P0 — `AgentState` is a flat, flag/counter bag with no lifecycle state machine.** No component owns any
   field; writers are ad-hoc;several flags have no writer at all, several are never read by any policy;
   `task_completed` is ignored by `TerminationPolicy`; `human_aborted`/`explicit_stop` have zero writers in the
   entire production tree; `executed_step_fingerprints`, `same_plan_count`, `execution_results`, `retry_count`,
   `state.recovery_attempted` are effectively **dead fields** (never written, never incremented, or only reset);

3. **P1 — Execution identity is split-brain and non-deterministic.** `AgentState.executed_step_fingerprints`
   (dead set( vs `Orchestrator.execution_history` (live,**cleared by `bind_state()`**
   (`orchestrator.py:90`).  `Orchestrator._execution_fingerprint()` is `json.dumps(...sort_keys)`-based —
   **no semantic argument normalization** (list order, `1` vs `1.0`, whitespace matter`);it does NOT include
   plan identity, strategy identity, nor a stable step identity besides hard-coded `step_id`;and **all hard-coded
   LLM final-answer actions share one fingerprint** (`"LLM | generate final answer from gathered results"`),
   so X unrelated answer steps can be mistaken for "the same execution" (`if self._has_seen_execution(...)`).

4.. **P1 — Counter semantics are conflated and mis-reported.** One shared `replan_count` counts ALL recovery
   types (tool-recovery, plan-recovery, verification-recovery, loop-recovery(`orchestrator.py:144`));
   `state.retry_count` is never incremented anywhere in `src` (only reset at `orchestrator.py:917`);
   `state.recovery_attempted` is never set True anywhere in `src` (only reset False at 781/918);
   `verification_attempts` is **NOT reset by `_apply_full_plan()` or verification-recovery**
   (`orchestrator.py:1341-1343` resets only the trio) — so a fresh plan/candidate inherits the previous
   candidate's verification budgetand can be declared `ANSWER_UNVERIFIED_BUDGET` afterits **first** verification.



5.. **P1 — Human-in-the-loop is disconnected;approval-blocked steps spin to `MAX_ITERATIONS`.**
   `AgentExecution.execute()` raises `ApprovalBlockedError` immediately when approval is required
   (`agent_execution.py:198-216`);**nothing in production ever calls `HumanApprovalHandler.request_approval()`**;
   `waiting_for_approval=True` has no consumer;`state.blocked=True` is cleared atothes start of the next step
   (`orchestrator.py:847`);`RecoveryPolicy` maps `UNKNOWN`→`STOP`;the loop then re-attemptts the same blocked step
   every iteration until `MAX_ITERATIONS` — no `fatal_error`, no `HUMAN_ABORTED`, no `EXPLICIT_STOP`.

Secondary issues (P2) include:**test/code contract drift** — `PlanSchema` validators require "exactly one
final-answer LLM step,last" (`plan_schema.py:119-146`) na "TOOL step cannot be final" (`plan_schema.py:82-85`),
while 6 test fixtures build single-step TOOL-only plans → **7 currently-failing tests**;the 7th failing test
constructs the orchestrator via `Orchestrator.__new__` and omits `correlation_id` → `AttributeError` in
`_emit_agent_failure` (`tests/reliability/core/test_orchestrator_verification.py:25`, `orchestrator.py:1597`);
and the **evaluation subsystem** (`src/gaia_agent/evaluation/`)is 0-byte stubs while the real harnesses live at
`src/run_evaluation.py` and `src/gaia_agent/diag_eval.py` — which disagree on missing-answer encoding
(`None` vs string `"0"` at `diag_eval.py:344-348`).

---

## PART  १ — COMPLETE STATE FIELD AUDIT

Method: exhaustive `Select-String` search over `src/**/*.py` and `tests/**/*.py` (excluding `.venv`,
`__pycache__`(; every reference cited below as `file:line`. Definition/owner/writer/resetter/incrementer are
given relative to **current commit `1b40b08`**. Scope legend: **TASK** = survives whole run; **PLAN** =
scoped to current plan; **STEP** = per plan-step scratch; **ATTEMPT** = per execution/retry attempt;
**DERIVED** = computable from others; **TERMINAL** = terminal outcome signal.



| FIELD | SCOPE | OWNER | WRITERS | RESETTERS | INCREMENTERS | CURRENT MEANING | PROBLEMS |
|---|---|---|---|---|---|---|---|
| `user_request` | TASK | harness + AgentLoop | `agent_loop.py:145-155`; `diag_eval.py:265-312` | — | — | canonical question | alias fallbacks return None (`slots=True`); silent, unenforced |
| `user_id` | TASK | harness | harnesses only | — | — | user id | never read in core |
| `messages` | TASK | — | **NONE in `src`** | — | — | conversation buffer | DEAD: only reader is print `agent_loop.py:551-556` |
| `plan` | PLAN | Orchestrator | `orchestrator.py:396`; `:994`; `:810-817` | replaced whole list on replan | — | `PlanStep[]` | no plan identity;step ids restart at 0 per plan |
| `step_type` | STEP | Orchestrator | `orchestrator.py:836` | `:822-826` partial; `_prepare_step` overwrite | — | current step type | stale between iterations until next `_prepare_step` |
| `current_step` | PLAN/STEP | Orchestrator | `:397`(=0(;`:915`(`+=1`;`:1403` | `_apply_full_plan`→0 | `_mark_step_completed`:915 | 0-based index | normally valid;replan re-maps index space, breaking evidence linkage |
| `completed_steps` | PLAN | Orchestrator | `:909-913` | **`_apply_full_plan`:398 CLEAR on every full replan** | appends once | completed indices | replan wipes success history;duplicate guard value-based |
| `current_action` | STEP | Orchestrator | `:834` | next `_prepare_step` | — | action text | stale into verification/termination |
| `blocked` | STEP | AgentExecution | `agent_execution.py:171,190,201,219` | `orchestrator.py:407,822,847`(False( | — | per-step block scratch | **reset each `_prepare_step` → approval-block cannot persist;Root Cause 5** |
| `waiting_for_approval` | ATTEMPT | AgentExecution | `agent_execution.py:172,200,218` | overwritten next attempt | — | approval-required marker | no consumer;no approval acquisition path exists |
| `execution_results` | TASK | — | **NONE in `src`** | — | — | results buffer | DEAD (reader: print `:543-546`( |
| `tool_name` | STEP | Orchestrator | `:838` | next `_prepare_step` | — | current tool | — |
| `tool_arguments` | STEP | Orchestrator+AgentExecution | `:840-842`;`agent_execution.py:343` (mutated to validated copy( | `_prepare_step` overwrites | — | current args | **fingerprint `orchestrator.py:220` uses `step.arguments`,NOT validated executed args** |
| `tool_result` | STEP | AgentExecution | `agent_execution.py:400,524` | `_prepare_step`:844;`_replace_failed_step`:819 | — | last output | final-answer capture requires non-None `:881` |
| `tool_error` | STEP/TASK | many | `agent_execution.py:176,576`;orchestrator `:235,261,299,324,340,366,712,741,766,780,845,1114,1130,1167,1206`et al. | `agent_execution.py:401,525`;`_prepare_step`:845;`_replace_failed_step`:820;`_apply_full_plan`:401 | — | "last error string" for ANY subsystem | overused cross-subsystem;no category/step/attempt payload;stale |
| `risk_assessment` | STEP | AgentExecution | `:263` | `:411,826,851` | — | risk | correct per-step scratch |
| `approval_decision` | STEP | AgentExecution | `:196` | `:410,825,850` | — | approval decision | no consumer after set |
| `execution_decision` | STEP | AgentExecution | `:167` | `:409,824,849` | — | execution decision | no consumer after set (diagnostic only( |
| `iteration` | TASK outer-loop | AgentLoop | `agent_loop.py:334` (`+=1` after `run_iteration()`( | — | AgentLoop ONLY | completed outer passes | **Part 5**: counts outer passes not attempts;0-based vs label+1;`iteraion` legacy fixed but unregressed |
| `final_answer` | TASK | Orchestrator | `_capture_step_result`:899;`:403`(None(;`:1341`(None( | replan/verif-recovery | — | candidate answer | see Part  ्8 |

| `final_answer_ready` | TASK/TERMINAL | Orchestrator | `:900,1071,1187,1206`(True(;`:404,1342`(False( | `_apply_full_plan`;verif-recovery | — | "answer exists" | flag-trio redundancy;budget-exhaust sets True w/o verification (`:1206`( — deliberate, flag-based |
| `final_answer_verified` | TASK/TERMINAL | Orchestrator | `:1072,1188`(True(;`:405,1207,1343`(False( | `_apply_full_plan`;verif-recovery | — | "verification ok" | no single authority;many terminal paths unverified by design |
| `termination_reason` | TASK/TERMINAL | AgentLoop | `agent_loop.py:655-657` | — | — | terminal reason enum | set ONLY on `should_stop`;**external-timeout path never reaches it** → `None` |
| `fatal_error` | TASK | Orchestrator | `orchestrator.py:147,171,300,768` | `:565,1406`(False( | — | fatal flag | salvage **clears** fatal (`:1406`(;no stable fatal chain;payload only in tool_error/events |
| `human_aborted` | TASK/TERMINAL | — | **NONE in `src`** | — | — | abort signal | DEAD — unreachable terminal(only `termination.py:57-60` reads( |
| `explicit_stop` | TASK/TERMINAL | — | **NONE in `src`** | — | — | stop signal | DEAD — unreachable terminal(only `termination.py:63-66` reads( |
| `timed_out` | TASK/TERMINAL | diag_eval harness | `diag_eval.py:649` ONLY (not core( | — | — | external timeout | core has NO timeout producer;termination unauthoritative |
| `retry_count` | TASK | — | `orchestrator.py:917` (**=0 reset only**( | — | **NONE** | expected retry tally | DEAD — retries live inside `ReliabilityEngine/Retry` local `attempts`;never mirrored to state;metrics always 0 |
| `recovery_attempted` | TASK | — | `orchestrator.py:781,918` (**=False only**( | — | **NONE** | "recovery happened" | DEAD — real signal is `ReliabilityResult.recovery_attempted` (read `:230,297`(;state never True → metrics misreport |
| `replan_count` | TASK | Orchestrator | `_check_recovery_budget`:144 (`+=1` per budget check( | — | `_check_recovery_budget` ONLY | "total replans" | **counts budget checks, not replans**;single budget shared by tool/plan/verification/loop recovery (`MAX_REPLANS=2`:40( |
| `loop_salvage_attempted` | TASK | Orchestrator | `:1397` | — | — | salvage-once guard | OK — guarded `:1390` |
| `same_failure_count` | TASK | Orchestrator | `:161-170` | **`:167` =1 on new key** | `:165` (same key( | consecutive same-category failures | key = `category:error_code` (`:156-159`( — no step identity |
| `same_plan_count` | TASK | — | **NONE in `src`** | — | — | expected repeated-plan tally | DEAD (reader: print `agent_loop.py:477-480`( |
| `last_failure_key` | TASK | Orchestrator | `:162,168` | `:168` on new key | — | last failure discriminator | excludes step/plan identity |
| `executed_step_fingerprints` | TASK | — | **NONE in `src`** | — | — | expected history set | **DEAD** — actual history is `Orchestrator.execution_history` (`:81(`,**cleared by `bind_state()` `:90`** |
| `verification_attempts` | TASK | Orchestrator | `_verify_final_answer`:1017-1019 (`+1`( | **NONE** | `_verify_final_answer` ONLY | total verifications per run | **never reset on replan/recovery** → budget pollution;two independent gates (`MAX_VERIFICATION_ATTEMPTS=2`:45 vs `TerminationPolicy(max_verification_attempts=2`:41-42(;`main.py:135-137` constructs policy WITHOUT the param → default ( |
| `execution_success` | STEP | AgentExecution | `agent_execution.py:173,202,403,527` | implicit deny paths only | — | "any success so far" | failure paths don't clear → **stale True + `tool_error≠None` reachable**;Part 10 |
| `step_succeeded` | STEP | AgentExecution | `:174,203,222,226`(False(;`:402,526`(True( | each `execute()` start | — | last atomic attempt ok | OK scratch;read by validator `:864` |
| `task_completed` | TASK/TERMINAL | Orchestrator | `:1073,1189`(True(;`:1208`(False( | verification-budget branch | — | "task done" | **TerminationPolicy NEVER reads it**;redundant SSOT with ready+verified;silently stale-able |
| `evidence` | TASK | AgentExecution | `agent_execution.py:409-421,529-541` (`ToolResultRecord`( | — | — | evidence ledger | `step_id` = `state.current_step` at write time (`:411-412`( → breaks on replan;never pruned |
| `{user_question,question,prompt,…}` | TASK (intended( | — | none (alias loop `diag_eval.py:286-295` cannot set: `slots=True` → `hasattr` False → skipped( | — | — | historical aliases | dead;silent fallbacks mask typos (`agent_loop.py:123-139`( |

**Part  ्१ verdict:** 30 declared fields + 4 legacy aliases. Of these, **10+ are dead / effectively dead**
(`messages`, `execution_results`, `human_aborted`, `explicit_stop`, `retry_count`, `recovery_attempted`,
`same_plan_count`, `executed_step_fingerprints`, `approval_decision`, `execution_decision`,
`waiting_for_approval`(no consumer(,and **zero fields have a formal owner or reset contract** — all resets are
ad-hoc code-site specific. The flag-trio (`final_answer_ready`/`final_answer_verified`/`task_completed`) plus
`termination_reason` constitute a de-facto lifecycle with **no state machine** behind it.

---

## PART  २ — SINGLE SOURCE OF TRUTH AUDIT

| Duplicated concept | Components that currently own it | Correct owner? | Can states diverge? | Concrete divergence example | Recommended SSOT |
|---|---|---|---|---|---|
| execution history | (a( `Orchestrator.execution_history: set[str]` (`orchestrator.py:81`(; (b( `AgentState.executed_step_fingerprints: set[str]` (`agent_state.py:99-101`( — **never written**; (c( `LoopDetector._history/_exact_counts/_plan_history` (`loop_detector.py:69-82`( | (a( only real owner; (b( dead junk; (c( different purpose (pattern detection( | **YES** — three disjoint stores;(b( can never reflect (a(;(c( detects loops but duplicates (a('s job | (a( records → (b( silently empty forever;or future writer writes only (b( → (a( bypass | single `ExecutionHistory` store, owned by Orchestrator task-scope, NOT cleared mid-task |
| final-answer state | `final_answer_ready` (`state.py:61`( + `final_answer_verified` (`:63`( + `task_completed` (`:84`( + `termination_reason` (`:65`( | Orchestrator writes trio;AgentLoop writes reason | Partially — trio written atomically at 3 sites (`orchestrator.py:1071-1073,1187-1189,1206-1208`( but **nothing enforces it**;no invariant checker | future code sets `task_completed=True` without verified → termination仍 checks only ready+verified → undetected | **Explicit lifecycle enum**;`task_completed` becomes DERIVED;`termination_reason` becomes the single terminal witness |
| iteration | `AgentState.iteration` (`:57`( + `TerminationPolicy.max_iterations` (`termination.py:40,103`( + `config.agent_max_iterations` (`config.py:15`(? unused by main( | AgentLoop increments;policy reads | Ownership split across 3 files;two constants never reconciled | `main.py:135-137` hardcodes 20 ignoring `config.agent_max_iterations=20` (coincidence?(;policy default 23 (`termination.py:40`( if constructed without arg | single `IterationBudget` policy input, injected once |
| counters | `retry_count` (`state.py:75`(; `replan_count` (`:91`(; `verification_attempts` (`:104`(; engine-local `attempts`/`recovery_count` (`engine.py:209-211`(; policy budgets (MAX_REPLANS`:40`, MAX_VERIFICATION_ATTEMPTS`:45`, RetryPolicy.max_attempts( | Nobody writes `retry_count`;nobody resets `verification_attempts`;engine-local doubles are invisible | **YES** — `replan_count` mixes tool/plan/verify/loop recovery (`orchestrator.py:144`(;`verification_attempts` survives plan reset | tool fails 3× n every one of 20 iterations (`recovery_change_detector` missing( → `retry_count` reports 0 despite real retries,and `replan_count` 0 despite repeated recovery attempts (both wrong( | budget owners per CONCERN: retry-budget (per attempt(, replan-budget (per plan(, verification-budget (per candidate(; counters derived from ledger |
| blocked/approval | `blocked` (`state.py:34`( + `waiting_for_approval` (`:35`( + `approval_decision` (`:53`( + `ApprovalPolicy` (approval.py( + `HumanApprovalHandler` (handler.py( | AgentExecution writes flags;policy decides;handler never invoked | NO runtime owner — flags written, never consumed | approval required → flags set → `ApprovalBlockedError` raised → loop re-runs same step 20× (`_prepare_step`:847 clears `blocked`(;nobody calls `request_approval()` | single `ApprovalGate` component that owns wait/reject/modify lifecycle and sets terminal (`HUMAN_ABORTED`/`EXPLICIT_STOP`( |
| verification budget | orchestrator `MAX_VERIFICATION_ATTEMPTS=2` (`:45`( + `TerminationPolicy(max_verification_attempts=2` (`:41-42`( | Two constants,one in each system | NO single owner;equal today by coincidence | constant `:45` ship of range w/o touching policy (or vice versa( → behavior silently differs | single config constant injected into both |
| tool error | `tool_error` (state( + `AgentError` (errors.py( + `FailureClassification` (failure_classifier.py( | `ErrorHandler` classifies;state stores only last message string | state keeps only string — category/severity/attempt **discarded** on write (`:235` etc( | recovery gate fail → `tool_error="Recovery operation requires a change detector…"` (engine.py:130( while `fatal_error=False` → loop keeps re-attempting | state stores `AgentError` (or list(, NOT `str`;derived render helper |

---

## PART  ३ — STATE INVARIANTS

Legend: **VALID** = enforceable today; **PARTIALLY VALID** = true today but unenforced / site-dependent; **BROKEN** =
currently violable; **MISSING** = no code/policy expresses it.



| INVARIANT | CLASS | EVIDENCE / VERDICT |
|---|---|---|
| `final_answer_verified=True` ⇒ `final_answer` exists | **PARTIALLY VALID** | both verified-writers (`1072,1188`( are preceded by non-None candidate gates (`:881,1021`(;but no enforcement — set-able externally |
| `final_answer_verified=True` ⇒ verification actually PASSed | **PARTIALLY VALID** | only two writers, both only after PASS (`:1070-1073,1140-1189`(;unenforced;no verification-result payload stored |
| `final_answer_ready=True` ⇒ `final_answer` exists | **VALID-ish** | all True-writers (`900,1071,1187,1206`( run only with non-None candidate;no guard |
| `task_completed=True` ⇒ verified+ready | **PARTIALLY VALID** | only writers set atomically (`1073,1189`(;**no invariant check**;`1208` sets completed=False — consistent;Termination ignores it |
| `iteration` never decreases | **VALID** | single incrementer (`agent_loop.py:334`(;no decrement |
| `retry_count` never decreases | **BROKEN** | never incremented (dead(;and reset to 0 at `:917` violates monotonicity implied by name |
| `replan_count` never decreases | **VALID but semantically wrong** | single incrementer `:144`;increments on budget CHECK not actual replan |
| `verification_attempts` never decreases | **VALID but poisonous** | single incrementer `:1017-1019`;never reset — leaks across plans/candidates → budget pollution |
| `current_step` references a valid step | **PARTIALLY VALID** | guards at `:203,:539,:1461`;salvage sets to found idx (`:1399-1404`(;`_mark_step_completed`:915 может push ==len by design |
| `completed_steps` ⊆ actually completed steps | **PARTIALLY VALID** | appended only after `_capture_step_result`;but `_apply_full_plan` CLEARs (`:398`( while evidence survives → "completed" truth lost |
| `execution_history` persists across replan | **BROKEN** | history NOT cleared on replan (good( BUT cleared on EVERY `bind_state()` (`:90`( — mid-task rebind wipes anti-loop memory;token twin `executed_step_fingerprints` never synced |
| exactly one final step,last,LLM | **VALID (schema-level)** | `plan_schema.py:119-146` pydantic enforcement — main anti-drift guard;Part 12 (6 tests violate it( |
| `termination_reason ≠ None` ⇔ loop exited | **BROKEN** | `run()` breaks only on `should_stop` (`:200-213`(;external timeout aborts `run()` → reason=None,timed_out=True |
| `fatal_error=True` ⇒ stops | **BROKEN** | `_handle_loop` salvage CLEARs fatal (`:1406`(;recovery-gate failures set only `tool_error` → no fatal at all → tool-failure loops until max_iterations |
| `execution_success=True` ⇒ `tool_error is None` | **BROKEN** | failure paths (`agent_execution.py:448-476,572-600`( set tool_error but leave stale success=True;`_validate_execution_result` `:863-866` catches nothing |
| `step_succeeded=True` ⇒ `execution_success=True` | **VALID** | both set together `:402-403,526-527`;deny sets both False |
| `execution_success=False ∧ step_succeeded=True` | **UNREACHABLE today** | no writer produces this combo |
| `blocked=True` ⇒ no side-effects executed | **VALID-ish** | `_validate_execution_result`:857-858;raise-before-tool `:186-188,214-216` |
| `human_aborted=True` ⇒ HUM_ABORTED reason | **MISSING** | no writer for flag;policy handles it (`:57-61`( but dead |
| per-candidate verification budget | **BROKEN** | `verification_attempts` global per RUN;replans/recovery never reset → second candidate inherits first's budget |
| one replan = one plan replacement | **BROKEN** | `replan_count` increments per `_check_recovery_budget` call (`:144`( — ≥1 counts per single recovery event (execution/verification/loop paths each invoke it( |

**Missed invariants (MISSING):** every invariant above has NO runtime `check/assert` — the core has zero
post-state validation;`AgentLoop._print_state_snapshot` prints only.There is no invariant module,
no `__post_init__` guard,and no test locks most of this table.

---

## PART  ्४ — ACTUAL STATE TRANSITION GRAPH (RECONSTRUCTED FROM CODE)

This is the REAL lifecycle as implemented today — NOT the ideal one. State writes happen at the cited
lines. Vertical flow is one `AgentLoop.run()` pass.Indentation = nesting inside one `iteration`.

```
[init]  harness builds AgentState (user_request only(   (src/run_evaluation.py:99-102; diag_eval.py:270-273(
   |
   v
[bind]  AgentLoop.run() → orchestrator.bind_state(state(  (agent_loop.py:157-161(;
            execution_history.clear()  (orchestrator.py:90(
   |
   v
[termination check]  TerminationPolicy.evaluate(...)  (agent_loop.py:188-190; termination.py:52-112(
   |        precedence: HUMAN_ABORTED > EXPLICIT_STOP > TIMED_OUT > FATAL_ERROR
   |        > COMPLETED (ready∧verified( > ANSWER_UNVERIFIED_BUDGET (ready∧attempts≥2(
   |        > MAX_ITERATIONS (iteration≥max(; else continue
   v
[plan?]       if not plan:  (agent_loop.py:219-263( → orchestrator.generate_initial_plan()
            │        └──────► _create_plan → ReliabilityEngine.execute(generate_plan…
            │                   retry×3 → recovery gate FAILS (no change detector(  (engine.py:122-133(
            │                   → tool_error set, return   (orchestrator.py:339-349(
            v
[run_iteration]  orchestrator.run_iteration()  (agent_loop.py:305-310(
            ├── context = context_builder.build(state(
            ├── if plan empty → create (above(; if tool_error → return
            ├── if current_step >= len(plan( → _handle_plan_completion()
            │        ├── if not final_answer_ready → _create_final_answer_step() (append "Generate final answer…"(
            │        └── elif not final_answer_verified → _verify_final_answer() (attempts+=1(
            │               ├── deterministic PASS 1070-73 → ready=verified=completed=True
            │               ├── deterministic FAIL → _handle_verification_failure → recoverable REPLAN
            │               │        ├── attempts≥2 → ready=True,verified=False,task_completed=False
            │               │        │        (terminal-unverified, or orchestrator.py:1206-1208(
            │               │        └── else replan_step() DIRECT (planner( → _replace_failed_step
            │               │             → final_answer=None,ready=verified=False (attempts NOT reset(
            │               └── verifier.verify → verified? → verified/ready/completed=True else same failure path
            ├── loop_detector.check(...)  (orchestrator.py:209-217( → detected?
            │        → _handle_loop: budget check + planner.replan_step DIRECT (works(
            │          + replace+prepare  OR salvage final step (1390-1415(
            ├── _prepare_step(step(  (orchestrator.py:219,828-851( — resets blocked=False,
            │        decisions/risk=None, tool_result/error=None
            ├── _record_execution(step(  (orchestrator.py:220,129-136( → history.add(fingerprint(
            ├── ReliabilityEngine.execute(agent_execution.execute…( (orchestrator.py:222-228(
            │        ├── execute(): execution_policy → risk → approval → TOOL/LLM
            │        │   ├── policy deny → blocked=True,waiting=False,exec_success=step_succ=False
            │        │   │        raise ApprovalBlockedError (agent_execution.py:169-188(
            │        │   ├── approval required → waiting=True,blocked=True, raise ApprovalBlockedError
            │        │   │        (:198-216( — NO human resolution loop;handler NEVER invoked
            │        │   └── tool/llm success → tool_result,exec_success=step_succ=True,evidence.append
            │        │        (agent_execution.py:400-421,524-541(  failure → tool_error set,raise
            │        ├── retry loop ×3  (engine.py:313-381( (if TRANSIENT(
            │        └── recovery gate → FAILS (no change detector( → success=False  (engine.py:296-297(
            ├── if result.recovery_attempted: _handle_execution_recovery (UNREACHABLE today(
            ├── if not result.success: tool_error=reason;return  (orchestrator.py:234-244(
            │        (→ next iteration re-executes SAME step — no replan,no fatal,no advance(
            ├── if state.blocked: return  (orchestrator.py:246-248(
            ├── _capture_step_result(step( (orchestrator.py:250,868-900( → if final step:
            │        final_answer=sanitize(tool_result(,ready=True
            └── _mark_step_completed(step(  (orchestrator.py:252,902-922( → completed.append,
                 current_step+=1, retry_count=0,recovery_attempted=False
   |
   v
[post-iteration]  state.iteration += 1  (agent_loop.py:334(
   |
   v
[loop]  back to [termination check];when should_stop → break  (agent_loop.py:200-213(
   |
   v
[final emit]  if reason==COMPLETED ∧ verified: emit_agent_completed()  (:567-572(
   |
   v
[return state]  run() returns AgentState (possibly un-terminated flags(
```

**Invalid / questionable transitions reachable today:**

1. `tool failure` → `tool_error≠None` → next iteration re-executes **same** step, no replan, no
   `fatal_error`, no step advance — repeated work until `MAX_ITERATIONS` (Root Cause 1(.
2. `approval required` → `blocked=True,waiting_for_approval=True` → raises → next iteration
   `_prepare_step` clears `blocked` → same blocked step again → 20× blocked passes → `MAX_ITERATIONS`
   (Root Cause 5(.
3.. `verification budget exhausted` → terminal state `final_answer_ready=True ∧ final_answer_verified=False
   ∧ task_completed=False` (orchestrator.py:1206-1208( — reachable and INTENDED ("honest unverified"(
4.. `loop salvage` → `fatal_error` flipped False (`orchestrator.py:1406(` — fatal→non-fatal transition exists.


5.. `external timeout` → `timed_out=True` AND `termination_reason=None` (diag_eval.py:649( —
   terminal signal duplicated/absent;run() was aborted,no policy ran.



6.. `final_answer_verified=True` + `task_completed=True` can coexist with `termination_reason=None` until
   next termination check;then COMPLETED ( latency only(.



7.. `_handle_verification_failure` replan path resets trio but NOT `verification_attempts`
   (`:1341-1343( → 2nd candidate inherits budget (Root Cause  ्4(.



8.. Stale `execution_success=True` + `tool_error≠None` reachable after a failure following a success
   (agent_execution failure path leaves success flag;Part  ्१०(.



**Answer to the explicit checklist in the brief:**

- `final_answer=None ∧ final_answer_ready=True` — **unreachable today** (all True-writers have candidate(.
- `final_answer=None ∧ final_answer_verified=True` — **unreachable today**.
- `final_answer_ready=True ∧ final_answer_verified=False ∧ task_completed=True` — **unreachable today**
  (completed only written together with verified(.
- `task_completed=True ∧ final_answer_verified=False` — **unreachable today** (writers atomic(.
- `execution_success=False ∧ step_succeeded=True` — **unreachable today**.
  **However:** `execution_success=True ∧ tool_error≠None` — **REACHABLE** (stale success flag(;and
  the trio-atomicity is unprotected — no invariant module exists to catch future drift..

---

## PART  ्५ — ITERATION SEMANTICS

**What "iteration" means today:** number of completed **outer `AgentLoop` while-loop passes** —
i.e., executions of `orchestrator.run_iteration()` — incremented exactly once per pass at
`agent_loop.py:334` AFTER the orchestrator returns. It is **NOT** a tool-attempt counter,
**NOT** a plan-step counter, **NOT** a retry counter, **NOT** a recovery counter, **NOT** a
verification counter.



| Question | Answer (evidence( |
|---|---|
| Who owns `iteration`? | `AgentLoop` (sole writer `agent_loop.py:334`;readers: TerminationPolicy via `TerminationState`, events, prints, `RuntimeSource`( |
| Who increments it? | `AgentLoop.run()` only, once per outer pass |
| What starts an iteration? | start of each `while True:` body — no explicit start signal;implicit |
| What ends an iteration? | return from `orchestrator.run_iteration()` (any path: step done, failure returned, blocked, plan-completion, verification(;exceptions are re-raised so crash ends run( |
| Can one iteration execute multiple tools? | **YES — often**: `ReliabilityEngine` retries up to `retry_policy.max_attempts=3` (`main.py:193-197`(;`max_total_executions= retry× (recoveries+1(=9` (`engine.py:63-67`(;recovery would add more if wired;verification/loop replanning then runs in later passes. Net: one iteration typically = one plan-step + up to 3 tool attempts (retries(;multiple plan-steps per iteration are NOT normal (plan advances one step per pass( |
| Can retry happen inside an iteration? | **YES** — `Retry.execute` inside `ReliabilityEngine.execute` (`engine.py:313-381`(;invisible to state counters |
| Can recovery happen inside an iteration? | Intended YES (engine `_execute_recovery`+ recovery_operation(, but **DEAD** — no `recovery_change_detector` passed → engine refuses (`engine.py:122-133`;call sites 227,294,1110( |
| Can replanning happen inside an iteration? | YES — but only via DIRECT planner calls in `_handle_loop`/`_handle_verification_failure` (`:1288,1488`(;the engine-mediated tool/plan replan is dead |
| Can verification happen inside an iteration? | YES — at plan exhaustion (`:972-983`(;verification attempts increment within that iteration;the COMPLETED terminal therefore occurs on the iteration AFTER verification succeeded (loop must re-check( |
| Does `max_iterations` limit what it claims? | **NO**. It limits outer passes only (default `20` at `main.py:135-137`(;each pass can retry ×3 (=60+ tool attempts worst-case over 20 passes,and more if recoveries/verifications/replans stretch passes(;**it does NOT bound tool attempts,retries,recoveries,or wall-clock** |

**Concepts currently mixed (attested code sites(:**

| Concept | Real location today | Attested |
|---|---|---|
| iteration | `state.iteration` (outer pass( | `agent_loop.py:334` |
| execution attempt | engine-local `attempts` / `attempts_since_recovery` | `engine.py:209-211,313` |
| retry attempt | engine-local `attempts` inside `Retry` | `retry.py:70-76` |
| recovery attempt | engine-local `recovery_count` | `engine.py:211,300` |
| replan attempt | orchestrator-local `replan_count` via `_check_recovery_budget` | `orchestrator.py:144` |
| verification attempt | `state.verification_attempts` | `orchestrator.py:1017-1019` |

**Architectural recommendation (explicit(:** introduce an explicit `ATTEMPT`-scope counter ledger on the state
machine: `iteration` stays THE outer-pass counter;**new** per-phase budgets live inside each phase
(retry-budget per `AgentExecution.execute` call;replan-budget per plan/candidate;verification-budget per
candidate(;expose **derived** report fields (total tool attempts, total retries, total recoveries, total
replans( computed from the new ledgers so metrics stop lying. `max_iterations` should bound outer passes;
NEW `max_attempts_per_step` / `max_total_tool_attempts` should bound inner work (Wave 2+: see Part 16(.

---

## PART  ्६ — PLAN / REPLAN LIFECYCLE

Event chain: `_create_plan` (`orchestrator.py:280`( → `_generate_plan` (`:379`( → `_apply_full_plan`
(`:390-413`(;failures → `_replan_full_plan` (`:415-458`(;step failures → `_recover_execution`
(`:533-607`( → `_replace_failed_step` (`:787-826`( → `_prepare_step` (`:828-851`(;verification failures →
`replan_step` direct (`:1288`( → `_replace_failed_step` + trio reset (`:1341-1343`(;loop failures →
`_handle_loop` (`:1375-1543`(.

**What `_apply_full_plan()` resets (`:396-411`(:**

| State group | Reset? | Fields | Comment |
|---|---|---|---|
| plan | replaced | `plan=list(plan.steps)` | new plan object |
| step index | reset | `current_step=0`, `completed_steps.clear()` | restarts index space from 0 |
| tool scratch | reset | `tool_result=None`, `tool_error=None` | |
| final-answer trio | reset | `final_answer=None`, `final_answer_ready=False`, `final_answer_verified=False` | candidate discarded |
| block/decisions | reset | `blocked=False`, `execution_decision=None`, `approval_decision=None`, `risk_assessment=None` | |
| metrics | +1 | `plans_created` (`:413`( | |

**What `_apply_full_plan()` does NOT reset (the dangerous part(:**

| State group | NOT reset | Consequence |
|---|---|---|
| evidence | kept (`state.evidence` untouched( | GOOD — evidence survives replan (but `evidence[].step_id` now indexes a DIFFERENT plan → step_id linkage broken( |
| execution history | kept (`orchestrator.execution_history` untouched( | GOOD — anti-repeat guard survives replan BUT cleared by `bind_state()` (`:90`( — Part  ्७ |
| counters | kept: `replan_count`, `verification_attempts`, `retry_count`, `same_failure_count`, `last_failure_key`, `same_plan_count` | **BAD** — verification budget leaks across candidates (Part 4 ¶7(;failure fingerprinting across plans conflated |
| success flags | kept: `execution_success`, `step_succeeded` | **BAD** — stale success can leak into next plan (Part  ्१०( |
| terminal flags | kept: `fatal_error`, `termination_reason`, `timed_out`, `human_aborted`, `explicit_stop` | mostly irrelevant (terminal ends run(;but `fatal_error=False` after salvage (`:1406`( is explicit |
| task-level context | kept: `messages`, `execution_results`, `loop_salvage_attempted` | `loop_salvage_attempted` rightly persists (once-only( |

**Which state CLASS is destroyed incorrectly?**

- **TASK-LEVEL**: preserved (`user_request`, evidence( — correct.
- **PLAN-LEVEL**: fully replaced (`plan`, `current_step`, `completed_steps`( — correct for a FULL replan;
  but `completed_steps` wipe means success history is only reconstructable from `evidence`, not from the completed
  index set. (Recommendation: keep `completed_steps` as plan-scoped, but expose plan-scope id so evidence
  re-linkable.(
- **STEP-LEVEL**: `tool_*` scratch reset correctly.
- **ATTEMPT-LEVEL**: **should** reset on new candidate/plan but DOESN'T for `verification_attempts`(and
  `same_failure_count`/`last_failure_key` are only reset when key changes( — Root Cause  ्4 ventilation.

**Attack-case examples (concrete(:**

1\. Steps 1-2 succeed, step 3 fails → recovery budget gate fails (`_recover_execution` unreachable(
   → no replan at all → step 3 retried every iteration until MAX_ITERATIONS — successful steps 1-2
   remain completed,but no final answer ever attempted. ( = Root Cause 1's most visible symptom.(
2\. Verification fails on candidate #1 (`verification_attempts=1`( → verification-recovery replaces final
   step + trio reset but keeps `verification_attempts=1` → candidate #2's very FIRST verification makes
   `attempts=2` → if it fails, budget-exhaust branch immediately locks `ANSWER_UNVERIFIED_BUDGET`
   (`orchestrator.py:1202-1208`( — candidate #2 didn't get a single full verification. **Deterministic
   budget pollution.**
3\. Full replan via `_replan_full_plan` after tool failure is unreachable (Root Cause1(;if it WERE,
   it would reset trio but keep `replan_count` — consistent counter semantics (monotone(,fine..
4\. `_replace_failed_step` on a TOOL replacement (`:787-826`( resets blocked/decisions/tool scratch BUT does
   NOT reset `execution_success`/`step_succeeded` — after 3 retries fail then replace,success flags may
   stale.



---

## PART  ्७ — EXECUTION HISTORY / FINGERPRINT AUDIT

Components: `Orchestrator.execution_history: set[str]` (`:81`(; `_execution_fingerprint()` (`:102-120`(;
`_has_seen_execution()` (`:122-127`(; `_record_execution()` (`:129-136`(; `_same_execution()` (`:609-637`();
`_arguments_fingerprint()` (`:640-651`(; plus `AgentState.executed_step_fingerprints` (dead(; plus
`LoopDetector._fingerprint/_normalize` (`loop_detector.py:271-371`( — a SEPARATE hashing scheme.

| # | Question | Answer |
|---|---|---|---|
| 1 | Where does execution history live? | `Orchestrator.execution_history` (instance attr(, NOT on AgentState (never serialized,never restored( |
| 2 | Does it survive replanning? | YES for step-replace/full-replan (not cleared there(;but **NO for `bind_state()`** — cleared at `:90`.Any mid-task rebind (restore/resume/tests( wipes anti-repeat memory |
| 3 | Does `bind_state()` clear it? | **YES** (`orchestrator.py:90`( — with it the "same execution" protection per task disappears if rebinding occurs |
| 4 | Do multiple histories exist? | **YES** — 3: orchestrator set; `AgentState.executed_step_fingerprints` (never written(;LoopDetector deque/Counter (`loop_detector.py:69-82`( — different hash scheme,never sync |
| 5 | Are fingerprints deterministic? | Mostly (json.dumps sort_keys default=str( but **not stable across equivalent values** — #10 |
| 6 | Are arguments normalized? | **NO** — `_arguments_fingerprint` = `json.dumps(sort_keys=True, ensure_ascii=False, default=str(`;list order preserved;`1` vs `1.0` vs `"1"` differ;whitespace preserved — semantically-identical calls hash differently |
| 7 | Is plan identity represented? | **NO** — `PlanSchema` (plan_schema.py:104( has only `steps`;`make_plan` passes `plan_id=` and pydantic ignores extra model kwargs (extra=`ignore`( (indeed tests at `test_orchestrator.py:19` pass `plan_id=` and it's silently dropped(;fingerprint has no plan component either |
| 8 | Is step identity represented? | Only via `step_type+tool+args` (TOOL( or `step_type+action` (LLM(;`step_id` NOT in fingerprint — two identical actions at different steps collide (usually DESIRED for anti-loop(;but cross-PLAN collisions too (same action regenerated after a genuine replan is blocked( |
| 9 | Is strategy identity represented? | Implicit only — via tool_name/action text;`_strategy_family` exists in Planner (`planner.py:1389-1400`( but Orchestrator fingerprint doesn't use it |
| 10 | Can semantically-identical executions bypass detection? | **YES** — different list order, `1` vs `1.0`, value-wrapping (e.g., `{"q":["a","b"]}` vs `{"q":["b","a"]}`(, whitespace differences all yield different fingerprints yet same tool call semantics |
| 11 | Can semantically-different executions collide? | **YES (LLM steps(** — flat "LLM | action-lowercase" fingerprint;ALL final-answer steps share hardcoded actions: `"Generate final answer from gathered results."` (`:998`( or `"Answer the task strictly from the evidence already gathered"` (`:693`( — any two final-answer passes with same action collide (2nd one rejected by `_has_seen_execution`(;`_force_different_strategy` evidence-step ALSO collides after first use (`:701`( → "no new strategy is available" false-negative (`:762-770`( |

**Where `_record_execution` runs (`orchestrator.py:220`(:** BEFORE `AgentExecution.execute` — so the recorded
fingerprint is built from the **static plan step**, NOT from the **validated/executed** arguments
(`step.arguments` vs `state.tool_arguments` mutated at `agent_execution.py:343`(.Execution identity can
misrepresent what actually ran (e.g., tool contract validation may add defaults/coerce(.

**Recommended `ExecutionIdentity` concept (recommendation only — DO NOT implement yet(:** a value object
`ExecutionIdentity(step_type, tool_name, normalized_args, strategy_family, plan_generation](,` with
(a( `normalize_args` canonicalization (recursive dict-key sort, list sort only when order-insensitive per tool
spec, numeric canonicalization(; (b( plan-scoped `generation_id` (incremented per `_apply_full_plan`(;
(c( optional `expected_outcome_key`; (d( hash = `sha256` of canonical JSON (no `repr`,no raw json(;
(e( ONE store on the state machine (task-scope( — cleared ONLY when a NEW task binds,never mid-plan;
`executed_step_fingerprints` removed or aliased to the live store's serialized value at terminal.

---

## PART  ्८ — ANSWER LIFECYCLE

Desired lifecycle (brief's reference(: `NO_ANSWER → ANSWER_GENERATED → VERIFYING → VERIFIED → COMPLETED`.

**Actual lifecycle by writer:**

| Transition | Code site | Fields written |
|---|---|---|
| NO_ANSWER → ANSWER_GENERATED | `_capture_step_result` (`orchestrator.py:868-900`( — triggered when a step with `is_final_answer=True` completes | `final_answer=sanitizer(tool_result(`, `final_answer_ready=True` (`:899-900`(;;`final_answer_verified` untouched (False(;`task_completed` untouched (False( |
| ANSWER_GENERATED → VERIFYING | next plan-exhausted pass: `_handle_plan_completion` (`:977-983`( → `_verify_final_answer` (`:1012`( | `verification_attempts += 1` (`:1017-1019`(;then deterministic gate/LLM verifier |
| VERIFYING → VERIFIED | deterministic PASS (`:1070-1074`( or LLM verifier PASS + support (`:1187-1193`( | `final_answer_ready=True`, `final_answer_verified=True`, `task_completed=True`, `tool_error=None` (`:1187-1190`( |
| VERIFIED → COMPLETED | next `while`-top `check_termination` (`agent_loop.py:188-190`;`termination.py:81-88`( | `termination_reason=COMPLETED` (`agent_loop.py:655-657`(;then `emit_agent_completed` (`:567-572`( |
| ANSWER_GENERATED → (re(VERIFYING with NEW candidate( | verification failure → `_handle_verification_failure` (`:1196`( → replan_step (direct( → `_replace_failed_step` (`:1333-1339`( | `final_answer=None`, `final_answer_ready=False`, `final_answer_verified=False` (`:1341-1343`(;**`verification_attempts` NOT reset** (`:1017-1019` global( |
| VERIFYING → ANSWER_UNVERIFIED_BUDGET | `_handle_verification_failure` when `verification_attempts ≥ MAX_VERIFICATION_ATTEMPTS` (2( (`:1202-1208`( | `final_answer_ready=True`, `final_answer_verified=False`, `task_completed=False`, `tool_error="…remains unverified"` (`:1206-1213`(;then termination: `ANSWER_UNVERIFIED_BUDGET` (`termination.py:93-101`( |
| (any( → NO_ANSWER | `_apply_full_plan` (`:403-405`( (full replan( | trio reset (without resetting verification_attempts( |

**Where each stage lives:** answer creation = `_capture_step_result`;ready = same;verification begins =
`_verify_final_answer` (per plan-exhausted pass(;verification succeeds = `:1070-1074` (deterministic( or
`:1187-1193` (LLM(+support(;verification fails = `_handle_verification_failure`;task completion =
same as VERIFIED writers +`termination_reason` at next policy pass.

**Invalid transitions identified:**

1. `VERIFIED → UNVERIFIED-BUDGET` — **NOT possible** (budget branch gated on `final_answer_verified==False`
   implicitly? — actually `_handle_verification_failure` is only reachable when not yet verified (the `:981` gate goes
   to `_verify` only `if not final_answer_verified`(;so clean.(
2\. `VERIFIED → NO_ANSWER` — **possible mid-run** only if another full replan occurs after verified
   (would require loop continuing post-verified — it doesn't (verified causes stop at next check( — practically unreachable.
3\. `ANSWER_GENERATED → ANSWER_UNVERIFIED_BUDGET` after **1 verification attempt** — **REACHABLE**
   via budget pollution (`verification_attempts` carries across candidate/plan;see Part 4 ¶7, Part 6 ¶2(.
4\. `NO_ANSWER → (nothing( → MAX_ITERATIONS` — **REACHABLE** — the "task never reaches final
   answer" failure mode: tool-failure loop (Root Cause 1( or blocked loop (Root Cause 5( exhausts
   20 iterations with `final_answer=None`, `final_answer_ready=False`, `termination_reason=MAX_ITERATIONS`(.
5\. **No `VERIFYING` marker exists** — nothing in state says "verification in progress";a crash mid-verify
   leaves `final_answer_ready=True + final_answer_verified=False + termination_reason=None` —e.g., external
   timeout (`diag_eval.py:649`( produces exactly this mirror-image of the "unverified budget" terminal but
   с `termination_reason=None`.

**Answer-specific recommendations (see Part  ्१५ for the enum design(:** model the answer lifecycle as
explicit states (`NO_ANSWER → ANSWER_READY → VERIFYING → VERIFIED → TERMINAL_VERIFIED/TERMINAL_UNVERIFIED/…`(
with transitions legal only via the state machine;derive `final_answer_ready`, `final_answer_verified`,
`task_completed` as **reports** of that machine rather than independent booleans.

---

## PART  ्९ — TERMINATION AUDIT

`TerminationPolicy.evaluate` (`termination.py:52-112`( is the ONLY termination authority;read by
`AgentLoop.check_termination` (`agent_loop.py:590-659`( at top of every `while` pass;reason stored at
`agent_loop.py:655-657` when `should_stop`.(Callers: AgentLoop only.(

**Precedence (strictly sequential in policy(:**

| Priority | Condition | Reason | Code |
|---|---|---|---|
| 1 | `human_aborted` | HUMAN_ABORTED | `termination.py:57-61` |
| 2 | `explicit_stop` | EXPLICIT_STOP | `:63-67` |
| 3 | `timed_out` | TIMED_OUT | `:69-73` |
| 4 | `fatal_error` | FATAL_ERROR | `:75-79` |
| 5 | `final_answer_ready ∧ final_answer_verified` | COMPLETED | `:81-88` |
| 6 | `final_answer_ready ∧ verification_attempts ≥ max_verification_attempts` | ANSWER_UNVERIFIED_BUDGET | `:93-101` |
| 7 | `iteration ≥ max_iterations` | MAX_ITERATIONS | `:103-107` |
| — | else | continue (should_stop=False( | `:109-111` |

**Multiple-terminal-conditions question:** since evaluation is sequential, ONLYthe highest-priority true
condition is reported.** Fatal-error with already-verified-answer stops as `FATAL_ERROR` (not COMPLETED( —
the answer is lost without acknowledgment. Similarly `timed_out` with verified answer → TIMED_OUT swallows
completion. So,**YES, multiple terminal conditions can simultaneously be true,and precedence silently decides
which reason wins** — no audit trail of shadowed conditions in state.



| Terminal outcome | PRECONDITIONS | OWNER | STATE CHANGES | SIDE EFFECTS | FINAL STATE (typical( |
|---|---|---|---|---|---|
| COMPLETED | ready∧verified | policy | reason=COMPLETED (`:655`(;`emit_agent_completed` (`agent_loop.py:567-572,1562-1585`( | event AGENT_COMPLETED;metric agents_completed | ready=T,verified=T,task_completed=T,reason=COMPLETED;iteration≤max |
| ANSWER_UNVERIFIED_BUDGET | ready∧(not verified(∧attempts≥2 | orchestrator sets ready `:1206`;policy | reason set | no AGENT_FAILED event (no emit(;harness records unverified | ready=T,verified=F,task_completed=F,,reason=ANSWER_UNVERIFIED_BUDGET;candidate delivered unverified (intended( |
| MAX_ITERATIONS | iteration≥max (default 20 at `main.py:135-137`( | AgentLoop counter | reason set | "not completed";no failure event | **`final_answer` MAY be None** (tool-loop/blocked cases(;reason=MAX_ITERATIONS;fatal to:F — misleading: often means "stuck step" not "worked diligently" |
| FATAL_ERROR | fatal_error flag from `:147,171,300,768` | orchestrator | reason set | AGENT_FAILED event via `_emit_agent_failure` | fatal_error=T,tool_error=reason,reason=FATAL_ERROR |
| TIMED_OUT | timed_out (only `diag_eval.py:649` sets( | harness (NOT core( | — (**policy never runs on this path!(** | **no event**;`termination_reason` STAYS None | timed_out=T,reason=None — inconsistent terminal |
| HUMAN_ABORTED | human_aborted | — (no writer in src( | reason set | — | unreachable today |
| EXPLICIT_STOP | explicit_stop | — (no writer in src( | reason set | — | unreachable today |

**Architectural judgment:** termination is based on a **collection of independent flags**, NOT an authoritative
lifecycle state. Evidence: `task_completed` never consulted;(`timed_out` no core producer,and even when
harness sets it the policy never runs (loop aborted(;(`human_aborted`/`explicit_stop` no producers;(budget
double-encoded (Part 2(;precedence silently discards shadowed conditions.

---

## PART  ्१० — FAILURE STATE AUDIT

For each failure scenario, resulting state (current wiring(:

| Scenario | producer path | `execution_success` | `step_succeeded` | `fatal_error` | `tool_error` | `termination_reason` | `task_completed` | `final_answer_ready` | `final_answer_verified` |
|---|---|---|---|---|---|---|---|---|
| tool execution failure | `agent_execution.py:448-476` raise → engine retries×3 → recovery GATE-FAILS (`engine.py:122-133`( → `orchestrator.py:234-244` | **stale** (True if prior success( | False (reset `:222-226`( | False | gate reason | (eventually( MAX_ITERATIONS | False | prior (usually F( | False |
| LLM step failure | same shape (`:572-600`( | stale | False | False | message | MAX_ITERATIONS | False | prior | False |
| execution policy deny | `agent_execution.py:169-188` (blocked( | set False (`:173`( | False (`:174`( | False | deny message | MAX_ITERATIONS | False | prior | False |
| approval required | `:198-216` | set False (`:202`( | False (`:203`( | False | approval message | MAX_ITERATIONS | False | prior | False |
| planner failure (initial plan( | `_create_plan` engine-gate fail (`orchestrator.py:339-349`( | prior (stale( | prior | **False** | gate reason | MAX_ITERATIONS | False | prior | False |
| recovery failure (if recovery reached( | `_handle_execution_recovery` (`:712-724`( | prior | prior | **False (unless `:768`(** | reason | varies | False | prior | False |
| verification failure (recoverable( | `_handle_verification_failure` (`:1196+`( → replan rewired | prior | prior | False | trio reset | (later( ANSWER_UNVERIFIED_BUDGET or COMPLETED | False | False→True(when final( | False |
| verification budget exhausted | `:1202-1208` | prior | prior | False | "unverified" msg | ANSWER_UNVERIFIED_BUDGET | **False** | **True** | **False** |
| timeout | harness `diag_eval.py:649` | prior | prior | prior | prior | **None** | prior | prior | prior |
| fatal (budget/same-failure( | `orchestrator.py:146-178`;`_create_plan` `:300`;dup `:768` | **set False** at `:153,:177` | prior | **True** | reason | FATAL_ERROR | False | prior | prior |

**Contradictory combinations identified:**

1. **`execution_success=True ∧ tool_error≠None`** — REACHABLE after a tool/LLM failure following an earlier
   success (failure path leaves success flag;orchestrator validator only checks current-attempt flags — no
   cross-phase check(.
2\. **`fatal_error=True ∧ execution_success=True`** — REACHABLE when fatal branches at `:300,:768` don't clear
   success flag (only `:153,:177` do(.
3\. **`termination_reason=None ∧ timed_out=True`** — REACHABLE via external abort ( no policy pass(.
4\. **`final_answer_ready=True ∧ final_answer_verified=False ∧ task_completed=False`** — REACHABLE,intended
   (verification-budget terminal(.
5\. **`blocked=True ∧ waiting_for_approval=False`** — REACHABLE on policy-deny path (`:171-172`( — denied-but-not-
   approval;semantics blur.

**Tool-error "blocked/approval" loop example (concrete run-through(:** risk HIGH → approval required →
`waiting=True,blocked=True` → raise → engine classifies `ApprovalBlockedError` → UNKNOWN (severity MEDIUM,
retryable=F,recoverable=F( → RecoveryPolicy UNKNOWN→STOP → engine returns success=False → orchestrator sets
`tool_error`,returns → next iteration `_prepare_step` clears blocked → AgentExecution re-raises → … 20× →
MAX_ITERATIONS;`waiting_for_approval=True` persisted (until next attempt overwrites(.

---

## PART  ्११ — EVALUATION IMPACT

**Inventory of evaluation wiring (important — the named subsystem is EMPTY(:**

| Artifact | Status | Notes |
|---|---|---|
| `src/gaia_agent/evaluation/__init__.py`, `evaluator.py`, `gaia.py`, `submission.py` | **all 0 bytes (stubs;** the "evaluation subsystem" named inthe brief does not exist as code | `src/gaia_agent/evaluation/run_20260903_172634.log` is the only non-empty file (a log( |
| `src/run_evaluation.py` (5105 B( | REAL GAIA-benchmark harness | loads `gaia-benchmark/GAIA`「2023_level1」validation split;writes `evaluation_results.jsonl`;records `answer=result.final_answer` (**None** on failure(,`final_answer_ready/verified`,`tool_error`,`fatal_error`,`termination_reason` (as `str`(,`status` |
| `src/gaia_agent/diag_eval.py` (HF Unit-4 harness( | REAL HF-scoring harness | `build_state()` with alias loop (dead due `slots=True`(;`asyncio.wait_for` **timeout=420s** (`:642-645`(`→ sets `state.timed_out=True` (`:649`(;`extract_result` records **`"answer": str(answer… else **"0"**`** ( `:344-348`(;`termination_reason` via `str(...)` (`:353-355`(;per-question `_eval_results_hf20.json` |
| `src/_diag_eval.py`, `src/_diag_run.py`, `src/run_loop.py`, `src/load_results.py`, `src/submit.py`, `src/smoke_test_fixes.py`, `src/test_debug_fixes.py`, `src/tempCodeRunnerFile.py`, `src/gaia_agent/{tempCodeRunnerFile,agents/tempCodeRunnerFile}.py` | **stale dev leftovers** | duplicate/misleading harnesses;should be archived (Wave 2 hygiene( |
| `evaluation_results.jsonl` (repo root( | archived REAL output | **every record( `status`:`"error"`, `error`:`"AttributeError: 'AgentState' object has no attribute 'iteraion'"`** — 50+ records,100% failure. |

**Core-state bugs → evaluation impact mapping:**

| Core bug | Evaluation symptom (concrete( |
|---|---|
| Recovery dead-wiring (Root Cause 1( | missing final answers / false-failure:tool fails → 20 same-step iterations → `MAX_ITERATIONS`,`final_answer=None`,`answer=null` in `evaluation_results.jsonl` → GAIA score contribution 0;`completed` metric doesn't decrement but `verified`/«answers» do |
| Blocked/approval loop (Root Cause 5( | premature termination with `final_answer=None` + `termination_reason=None` (if harness-timeout aborts run( or `MAX_ITERATIONS`;`waiting_for_approval=True` never reported by harnesses (not recorded( |
| `iteraion` legacy/or future field-typo | **historical:** `AttributeError` on EVERY archived eval record — 100% failure. Current `src` has NO `iteraion` anymore (searched 0 hits in `.py`(;the archived JSONL is the только evidence.** **No regression test locks the correct spelling** — a rename/typo can silently reintroduce total failure (slots=True → runtime AttributeError in run 1( |
| flag-trio/metric mismatches | misleading metrics:`task_completed` recorded diplomatically (`diag_eval.py:368-373,863-871(` but never consumed by termination;`execution_success` stale True inflates "execution succeeded" counts;`retry_count`/`recovery_attempted` always 0/False → "no retries/recoveries" even when engine retried 3× per step every iteration |
| missing-answer encoding divergence | `run_evaluation.py:110,145` → `answer=None`;`diag_eval.py:344-348` → `answer="0"` string — **identical missing-answer renders as None in one harness and "0" in another** (numeric "0" ambiguity corrupts any downstream scoring that treats "0" as a real answer( |
| verification budget pollution (Root Cause 4( | unverified answers being accepted as terminal `ANSWER_UNVERIFIED_BUDGET` after only 1 attempt on 2nd candidate (delivered as `final_answer` nonetheless(;orchestrator's support gate `evidence_supports_candidate` (`:1155-1185`( exists but budget branch `:1202-1208` bypasses support→ accept-unverified-as-final |
| `termination_reason=None` on timeouts | `run_evaluation.py:115-119` prints None;`diag_eval.py:353-355` prints "None";evaluation can't distinguish "ran out of time" from "crashed" (error field separate( |
| max_iterations counts outer passes | tasks with many failures "use up" 20 iterations without producing useful tool calls (wasted RL/ollama spend;long wall-clock (timeout-prone(→ timeouts recorded as `termination_reason=None` |
| `evaluation/` stubs + multiple harnesses | no single canonical scorer;HF harness and GAIA harness diverge in answer/status/error encodings;benched results not reproducible from repo alone (missing scoreground-truth application here( |

**The explicit "iteraion" question — EVERY occurrence in the repository:**

| Location | Kind |
|---|---|
| `evaluation_results.jsonl:1..N` (53+ records( | **error strings only**: "AttributeError: 'AgentState' object has no attribute 'iteraion'" — historical artifacts of crashes,wrote by an old build |
| **`src/**/*.py`** | **0 occurrences** — the typo has been fixed (field is `iteration`) and no production reader/writer uses the misspelling today |

**Conclusion for Part 11:** current `src` is clean of the literal typo, but (a( no regression test enforces it,(b( the archived results prove the WHOLE GAIA run crashed on run #1 — the most impactful historical correctness bug in this repo,(c( analogous **silent alias fallbacks** (`getattr(state,"user_question",None(` etc.( still mask typos by returning None — engineering-wise,ther is no `__getattr__` guard on `AgentState` to explode on unknown attrs (slots=True raises, which IS the guard( — the crash is precisely because `slots=True` and a typo'd writer somewhere historically. Currently that guard is the ONLY protection and tests don't lock it.

---

## PART  ्१२ — TEST COVERAGE AUDIT

**Current runs:** `pytest tests/planner` → **29 passed**;`pytest tests/reliability` → **95 passed / 7 failed**
(9.81s(;`pytest tests/integration tests/planner/test_planner_runtime_contract.py` → **6 passed**.
`tests/test_conversation.py`,`test_database_connection.py`,`test_memmory.py` → **"no tests ran"** (0 `test_`-named
functions().pytest-timeout plugin is not installed.

**Broken tests — reason classification (NOT production bugs,except #6 reveals fragility(:**

| Test | Failure | Classification |
|---|---|---|
| `orchestration/test_orchestrator.py::test_replan_same_execution_must_stop` | `ValidationError`: `PlanSchema` requires exactly one final-answer step;fixture `make_plan` builds TOOL-only 1-step plan with `plan_id=` kwarg (silently ignored( | **CONTRACT DRIFT** — fixtures predate `plan_schema.py:119-146` final-answer invariant |
| `...::test_replan_different_tool_must_be_executable` | same | CONTRACT DRIFT |
| `p0_3_orchestrator_recovery.py::test_full_replan_rejects_repeated_plan` | same (`make_plan` `:64`( | CONTRACT DRIFT |
| `...::test_full_replan_with_new_strategy_replaces_plan` | same | CONTRACT DRIFT |
| `test_orchestrator_verification.py::test_verification_failure_before_budget_attempts_recovery` | `ValidationError`: TOOL step with `is_final_answer=True` (fixture `:399`( — **`plan_schema.py:82-85` now rejects it** | CONTRACT DRIFT |
| `...::test_verification_recovery_clears_previous_candidate` | same (`:699`( | CONTRACT DRIFT |
| `...::test_verifier_execution_failure_does_not_complete_task` | `AttributeError: 'Orchestrator' object has no attribute 'correlation_id'` — fixture via `Orchestrator.__new__` (`:25`( omits `correlation_id`/`event_logger`;production `_emit_agent_failure` (`:1587-1618`( accesses it | **TEST-HARNESS GAP** — bypasses `__init__`;also reveals `_emit_agent_failure` unguarded deps |

**Invariant → test mapping (what exists(:**

| Invariant (Part 3( | Existing tests | Coverage |
|---|---|---|
| 1 initial state | none | **MISSING** |
| 2 successful execution (plan→tool→final→verified→COMPLETED( | partial: verification-gate tests (`test_agent_loop_verification_gate.py`(;planner runtime contract | WEAK: no full-stack full-loop success test |
| 3 failed execution (tool fail( | `test_orchestrator_verification` (partial(;planner runtime | WEAK: no test asserts tool-failure loop/state fields |
| 4 retry | engine tests (`test_engine.py`( | engine-level only;`state.retry_count` never asserted (it's dead( |
| 5 recovery | `p0_3_orchestrator_recovery.py` (mocked( | **tests mock `_recover_execution`/budget — never exercise the real no-change-detector wiring → Root Cause  ्1 invisible to suite** |
| 6 replan | `p0_3_...`,planner runtime | WEAK: full-replan mocks budget;no assert on `verification_attempts` reset |
| 7 repeated execution | `test_orchestrator.py::test_same_execution_logic` et al | fingerprint equality only;no bind_state-clear / normalization / LLM-collision tests |
| 8 answer generation | `test_orchestrator_verification.py::test_capture_final_answer_*` | ✅ decent |
| 9 verification success | `test_orchestrator_verification.py` PASS tests | ✅ |
| 10 verification failure | same + `test_verifier_evidence_relevance.py` | ✅ decent |
| 11 verification budget exhausted | `test_orchestrator_verification_budget.py` + `test_agent_loop_verification_gate.py` | ✅ but pins CURRENT global-counter semantics (not per-candidate( |
| 12 max iterations | `test_agent_loop_verification_gate` policy-only | partial — no AgentLoop end-to-end |
| 13 timeout | **MISSING** | harness-only path untested |
| 14 explicit stop | **MISSING** (no producer( | none |
| 15 human abort | **MISSING** (no producer( | none |
| 16 fatal error | partial (`test_error_classification`,engine( | no state-level fatal assertions |
| 17 plan replacement | `p0_3_...` (2 fail( | broken fixtures |
| 18 state rebinding | **MISSING** | no `bind_state()` history/semantics test |
| 19 execution history persistence | **MISSING** | none |
| 20 fingerprint behavior | `test_orchestrator.py` `_same_execution` (2 pass( | narrow — no normalization/collision |
| 21 evaluation execution | **MISSING** | no test around harness encoding |
| 22 `iteraion` regression | **MISSING** | no attr-whitelist / typo test |
| 23 approval/blocked loop | **MISSING** | no test for 20× blocked step |
| 24 tool-failure recovery wiring | **MISSING** | **no test asserts `reliability_engine.execute(..., recovery_change_detector=...)` passed** |

**Required regression tests (Wave 1 minimum(:** 1) initial-state defaults; 2) happy-path full
loop→COMPLETED; 3) tool-failure→REPLAN (wired change detector( with assert new step runs next pass;->orchestrator sanity: 4) retry accounting; 5) recovery accounting; 6) replan resets
verification budget (per-candidate(; 7) bind_state task-scope semantics (history persists per task(;
8) fingerprint normalization/LLM-final uniqueness; 9) approval-required→terminal block (not 20×
loop(; 10) external-timeout→`termination_reason=TIMED_OUT`; 11) eval encoding (None≠"0",reason
stability(; 12) `AgentState` attr-whitelist (no `iteraion`,no alias fallback(; 13) `_apply_full_plan`
preserves evidence/counters correctly; 14) `max_iterations` bounds outer passes and NOT tool attempts;
15) human_aborted/explicit_stop producers→terminal paths. All 15 must ship with Wave 1 cal (Part 16(.

---

## PART  ्१३ — ROOT CAUSE ANALYSIS

Separating ROOT CAUSES from SECONDARY BUGS from SYMPTOMS. Every downstream error is grouped under
its originating root cause — this is a **dependency graph**, not a blame list.

```
ROOT CAUSE  1 (P0): ReliabilityEngine recovery is wired dead (recovery_change_detector never passed
              by Orchestrator  engine.py:122-133 refuses recovery without it(
   │   call sites: orchestrator.py:222-228 (tool execution(; :286-295 (plan generation(; :1104-1111 (verif —
   │   no recovery_operation at all(
   ▼
   affected components: Orchestrator.run_iteration (:234-244(, ReliabilityEngine.recovery/Recovery class,
      Planner.replan/replan_step/get_alternative_strategy (never invoked for tool/plan failures(
   ▼
   observed symptoms: same-step retry loops till MAX_ITERATIONS;tool_error="...change detector...";
      recovery/step_replans metrics 0;`replan_count` 0;no new strategy;no fresh evidence;fatal NEVER set
   ▼
   evaluation impact: missing final answers (answer=null(;false-failure;every GAIA question with tool failure
      scores 0;iterations burned → timeouts;"max_iterations" reason misleading

ROOT CAUSE  2 (P0): AgentState = flat un-owned flag/counter bag with no lifecycle state machine
   │   evidence: 30+ fields,10+ dead;no owner contract;task_completed ignored by TerminationPolicy;
   │   human_aborted/explicit_stop без producers;state.recovery_attempted/retry_count dead;
   │   verification_attempts global (never per-candidate renewed(;execution_success stale across phase
   ▼
   affected components: ALL (AgentLoop, Orchestrator, AgentExecution, policies, eval harnesses(
   ▼
   observed symptoms: state contradiction (execution_success=True + tool_error≠None(;budget pollution;
      blocked-approval 20× loop;metrics lie;termination_reason=None timeouts;dead fields mislead
      readers (diag_eval reports retry_count=0(;harnesses diverge encodingмаmissing answers
   ▼
   evaluation impact: misleading metrics;unreliable "verified" signals;false-failure/missing-answer
      inflation in reports

ROOT CAUSE  3 (P1): Execution identity split-brain & non-normalized (
   │   evidence: AgentState.executed_step_fingerprints dead twin;Orchestrator.execution_history cleared by
   │   bind_state(:90(;fingerprint json.dumps non-normalized;no plan/strategy identity;LLM final-answer
   │   fingerprints collide on hardcoded actions (:693,:998(;loop_detector uses separate sha256 scheme
   ▼
   affected components: Orchestrator._has_seen_execution/_force_different_strategy/
      _recover_execution;_handle_verification_failure
   ▼
   observed symptoms: "Recovery produced an already executed step" false-negatives ("no new strategy is
      available"( after 2nd evidence-step;equivalent calls bypass anti-repeat;recorded fp ≠ executed args
      (validated at agent_execution.py:343 after record at :220(
   ▼
   evaluation impact: spurious verification-recovery failures;answers never generated;replan churn

ROOT CAUSE  4 (P1): Counter semantics conflated / mis-reported (
   │   one replan_count for all recovery types (:144(;retry_count never incremented;recovery_attempted never
   │   True;verification_attempts never reset on plan/candidate replace (:1341-1343(;config vs policy
   │   constants split (MAX_VERIFICATION_ATTEMPTS:45 vs TerminationPolicy(:41-42(main.py:135-137(
   ▼
   affected: TerminationPolicy;diag_eval/run_evaluation metrics;orchestrator budgets
   ▼
   symptoms: 2nd candidate loses whole verification budget;metric reports 0 retries/recoveries;
      USER-visible "replan_count" inaccurate (budget-checks ≠ replans(
   ▼
   evaluation impact: wrong budgets → premature ANSWER_UNVERIFIED_BUDGET;wrong metrics in eval jsonl

ROOT CAUSE  5 (P1): Human-in-the-loop disconnected + terminal flags без producers (
   │   ApprovalBlockedError raised (:198-216( without calling HumanApprovalHandler.request_approval(;
   │   blocked cleared at _prepare_step:847 → next iteration re-blocked;RecoveryPolicy UNKNOWN→STOP;
   │   human_aborted/explicit_stop отсут producable paths → their terminal reasons unreachable
   ▼
   affected: AgentExecution, Orchestrator loop, TerminationPolicy, human/handler.py
   ▼
   symptoms:blocked/approval steps spin 20 iterations → MAX_ITERATIONS (no fatal(;waiting_for_approval
      beacon stuck;HUMAN_ABORTED/EXPLICIT_STOP никогда появляются
   ▼
   evaluation impact: eval harnesses никогда see human decisions;time-wasting runs→timeouts
```

**SECONDARY BUGS (grouped under their parent root cause(:** stale `execution_success` (RC2(;
`evidence[].step_id` index-shift on replan (RC2(;`completed_steps` wipe on replan (RC2/RC3(;
`final_answer`/`tool_error` "0"-vs-None harness divergence (RC2(;`termination_reason=None` on
timeout (RC2(;`same_failure_count` keyed without step identity (RC2/RC4(;`timed_out` producer external
only (RC2/RC5(;test-contract drift & `__new__`-harness fragility (RC2 — infrastructure(.

**SYMPTOMS (not causes(:** "GAIA enters incorrect states, repeats work, loses history,
terminates incorrectly, unreliable evaluation behavior" — each maps 1:1 to RC1-RC5 above.



---

## PART  ्१४ — YOUR ENGINEERING OPINION (REQUIRED(

### P0-1 — Recovery/replanning is dead-wired for tool-executionand plan-generation failures

- **WHAT I OBSERVED:** `Orchestrator` passes `recovery_operation=self._recover_execution` (resp.
  `_replan_full_plan`( but NEVER `recovery_change_detector` at the only three `reliability_engine.execute()`
  call sites (`orchestrator.py:222-228,286-295,1104-1111`(. `ReliabilityEngine._execute_recovery`
  explicitly returns failure when the detector is `None` (`engine.py:122-133`(,so recovery never runs.
-
- **WHY IT IS A REAL PROBLEM:** every tool/LLM failure burns 3 retries per iteration and thenloop re-attempts

  same step until `MAX_ITERATIONS=20` — no strategy change,no new evidence,no fatal signal,no
  `replan_count` move;20-iteration GAIA questions die with `final_answer=None`. The whole Recovery/Recovery-
  Policy/Replanning subsystem (an existing capability( is silently DISABLED. **Tests don't catch it**
  because they mock `_recover_execution`/budgets instead of exercisingthe engine wiring (Part 12,item  ्5(.
-
- **ROOT CAUSE:** a contract mismatch: `ReliabilityEngine` REQUIRES a change-detector callback (by
  design — "recovery is only safe when we can verify it produced a meaningful change"(,but Orchestrator
  never supplies one (and for verification,no `recovery_operation` either(.
-
- **MY RECOMMENDED SOLUTION:** pass `recovery_change_detector=` a real plan-change detector at ALL THREE
  call sites (e.g., `lambda new_step: new_step is not None and not self._same_execution(new_step,failed_step(`
  for step recovery;`lambda plan: ... not repeat_plan(plan,old_plan(` for full replan(;decide ONE
  owner for recovery decision (orchestrator supplies detectors;planner is sole producer of replacements(;add
  per-call `max_recoveries` consistent with `MAX_REPLANS`;make verification-recovery use the SAME engine path
  (or an explicitly-labeled non-engine path( so ALL four recovery types share identical budget semantics;
  log the old→new step/plan pair as evidence.

- **WHY I PREFER THIS:** it reactivates an existing,already-designed capability without new concepts;
  thechange-detector contract is intentionally safety-preserving (never re-run identical execution( — exactly
  what Wave 1 wants;minimal diff on 3 call sites + 1 detector function.

- **ALTERNATIVES CONSIDERED:** (a( bypass engine and call `_recover_execution` directly from a new
  `except` in `run_iteration` — rejected: duplicates engine retry/recovery-budget logic,second source of
  truth;(b( remove the change-detector requirement from engine — rejected: weakens the safety invariant enforced
  intentionally;(c( auto-derive detector from `_same_execution` inside engine — rejected: engine shouldn't
  know plan semantics;coupling..
-
- **RISKS:** recovery budget now actually consumed → loops now stop at replan-limit with `fatal_error=True`
  (correct(;behavioral change for existing passing mocked tests (they assert mocked wiring(;planner replan
  may still return same-step (`_recover_execution` already guards `_same_execution`+`_has_seen_execution`
  at `:592`( (residual(;double budget accounting (engine `max_recoveries=2` + orchestrator
  `MAX_REPLANS=2`( must be reconciled to ONE budget..
-
- **AFFECTED COMPONENTS:** `orchestrator.py` (3 call sites(,`reliability/engine.py` (accept detector(,
  `reliability/policies/recovery_policy.py` (already REPLAN-happy(,`planner.py` (replan(,tests
  `p0_3_orchestrator_recovery.py` (must mutate fixtures to valid PlanSchema(;new detector tests..
-
- **REGRESSION TEST REQUIRED:** (i( engine-level: execute fails 3×,change-detector supplied,
  recovery_operation returns different step → `success=True`,recovery_count=1;(ii( orchestrator-level: tool
  execute raises 3× → next `run_iteration` executes the REPLACED step (not the original(;(iii( gate-level:
  detector returns same step → engine refuses,`reason` mentions "change";(iv( `MAX_REPLANS` respected:
  after 2 replans → `fatal_error=True`..
-
- **CONFIDENCE: HIGH.** (Direct code-path proof at engine.py:122-133 vs call sites;Part 5,13.(

---

### P0-2 — `AgentState` is a flat un-owned flag/counter bag with no lifecycle state machine

- **WHAT I OBSERVED:** 30+ fields with ad-hoc writers,10+ dead fields (`messages`, `execution_results`,
  `human_aborted`, `explicit_stop`, `retry_count`, `recovery_attempted`, `same_plan_count`,
  `executed_step_fingerprints`, `approval_decision`, `execution_decision`(;`task_completed` written by
  orchestrator but ignored by TerminationPolicy;`human_aborted`/`explicit_stop` have ZERO producers;ther
  is no `__post_init__` validation,no invariant module..
-
- **WHY IT IS A REAL PROBLEM:** correctness is currently a property of *which code happened to run in
  which order*,not of any central model. New writers silently break invariant combos (Part 3(;
  diagnostics/evals read fields that nobody maintains (retry_count always 0,recovery_attempted always
  False(;the `iteraion`-style typos became 100%-crash precisely because there's no enforced field contract..

- **ROOT CAUSE:** state was grown organically ("one flag per need"( without a lifecycle/stage model or an
  owner/reset/scope contract per field.

- **MY RECOMMENDED SOLUTION (design at Part  ्१५ B-G(:** introduce an explicit `LifecyclePhase` enum
  + an explicit `AnswerState` enum;replace the trio flags + `task_completed` with DERIVED properties над
  the enums;make `final_answer_ready`/`final_answer_verified` read-only properties of `AnswerState`;
  keep `termination_reason` as sole terminal witness;assign per-field OWNERSHIP (Part 1 table( in code
  comments + a `state_invariants.py` module checked in `AgentLoop` after every iteration (debug builds(;
  delete dead fields;keep `blocked`/`waiting_for_approval` inside ONE `ApprovalGate` datum.
-
- **WHY I PREFER THIS:** preserves all existing capabilities (memory,retrieval,planner,reliability,risk,
  approval,observability,verification,recovery,orchestration( — it changes state's *shape*,not the
  components' responsibilities;it makes the rest of Wave 1 provable.,

- **ALTERNATIVES CONSIDERED:** (a( minimum patch: add per-field `__post_init__` asserts — rejected:
  asserts alone can't model sequencing/lifecycle and remain verbose;(b( full rewrite: replace AgentState
  with an orchestration DSL — explicitly out-of-scope (Wave 23+((;(c( split state into per-scope
  dataclasses NOW — rejected for Wave 1: broad blast radius;keep as follow-up evolution after enums land(.

- **RISKS:** derived-property migration touches many sites (verifier/orchestrator/termination/eval(;tests
  that pin flag semantics (test_orchestrator_verification_budget,test_agent_loop_verification_gate( must be
  updated deliberately (NOT weakened(;serialization of state (if any consumer persists AgentState( changes..

- **AFFECTED COMPONENTS:** `core/agent_state.py`, `core/agent_loop.py`, `core/orchestration/orchestrator.py`,
  `core/policies/termination.py`, `agents/verifier.py` (read-only(,`evaluation/*` harnesses (report props(,
  all tests asserting raw flags.

- **REGRESSION TEST REQUIRED:** Part 12 items 1,2,3,6,8,9,10,11,12,13,15 — plus a
  "state model" test: every state transition leaves `AnswerState` legal && derived flags agree with the enum;
  plus attr-whitelist test (no alias/typo attrs(..

- **CONFIDENCE: HIGH** for the diagnosis;MEDIUM for the exact enum shape (sequencing trade-offs at
  Part  ्१६(.

---

### P1-3 — Execution identity split-brain & non-normalized fingerprints

- **WHAT I OBSERVED:** dead twin field `executed_step_fingerprints` (agent_state.py:99(;live store `Orchestrator.execution_history` cleared by `bind_state()` (`:90`(;fingerprint = `json.dumps(sort_keys…)` with NO normalization (`_arguments_fingerprint` `:640-651`(;recorded BEFORE argument validation (`:220` vs `agent_execution.py:343`(;hardcoded LLM final-answer actions collide (`:693,:998`(.

- **WHY REAL:** anti-repeat protection is both over- and under-inclusive: equivalent tool calls bypass it;(X final-answer attempts falsely rejected ("already executed step"( → "no new strategy is available"( `:762-770`( kills legit replan chains..

- **ROOT CAUSE:** two separate identity schemes (orchestrator JSON vs LoopDetector sha256-repr(,no canonical `ExecutionIdentity`,no plan/strategy identity,no argument canonicalization..

- **MY RECOMMENDED SOLUTION:** the `ExecutionIdentity` value object (Part  ्७ reco( with canonical `normalize_args`;plan-scoped `generation_id`;ONE task-scoped store on `AgentState` (orchestrator-owned(;fingerprint = sha256 of canonical JSON;record AFTER validation/prepare;include strategy_family;special-case LLM final-answer steps to include a per-generation nonce (so final-answer re-generation isn't "same execution"(..

- **ALTERNATIVES:** (a( keep two stores and document — rejected: split-brain already bit us;(b( include full args repr — rejected: huge fingerprints,non-canonical;(c( drop anti-repeat entirely and rely onc LoopDetector — rejected: loop_detector only detects AFTER≥3 repeats,not for 2-attempt verification chains..

- **RISKS:** fingerprint format change invalidates mocked fingerprint tests;normalization must be tool-spec-aware (order-insensitive args( — risk of over-normalization (dropping meaningful order for e.g. `python_interpreter` code(;mitigate: per-tool `arg_order_significant` spec flag..

- **AFFECTED:** `orchestrator.py` (fingerprint/record/has_seen(,`core/evidence.py` (optional(,`planner.py` (strategy_family export(,`agent_execution.py` (record timing(..

- **REGRESSION TESTS:** normalization equivalence (list order,1vs1.0,whitespace(;LLM-final uniqueness;bind_state-persists-per-task;record-after-validate (mutated args hashed(;plan_generation_id isolation..

- **CONFIDENCE: HIGH.**

---

### P1-4 — Counter semantics conflated / mis-reported (retry/recovery/replan/verification counters(

- **WHAT I OBSERVED:** `replan_count` increments per budget CHECK (`:144`(;`retry_count` never incremented (only reset `:917`(;`state.recovery_attempted` never True (only False at 781,918(;`verification_attempts` global,never reset on plan/candidate replace (`:1341-1343`(;two independent verification-budget constants (`:45` vs `termination.py:41-42`(.

- **WHY REAL:** metrics heal all across the board (eval jsonl reports retries=0/recoveries=False always(;and the verification-budget pollution is a **correctness** bug: 2nd candidate loses its whole budget → premature `ANSWER_UNVERIFIED_BUDGET` (delivered unverified(..

- **ROOT CAUSE:** counters were bolted на as reporting fields without a corresponding writer/reset contract (Part 1(;budget ownership split between orchestrator constant and policy parameter..

- **MY RECOMMENDED SOLUTION:** per-scope budgets (Part  ्५( — retry budget per attempt-batch,replan budget per plan-generation,verification budget per candidate(;derive all report counters from a single `counters` ledger;one config object injected into orchestrator + termination policy (Part  ्१५ H(..

- **ALTERNATIVES:** (a( keep single counters but add resets — rejected: doesn't fix semantic mixing (replan_count counting verification-recovery etc(;(b( delete counters — rejected: observability is an existing capability..

- **RISKS:** budget-semantics change flips some today-passing tests (budget tests pin global counter( — update tests deliberately;config plumbing touches main.py..

- **AFFECTED:** `orchestrator.py`, `termination.py`, `main.py`, `diag_eval.py`/`run_evaluation.py` (readers(,`reliability/engine.py` (expose per-attempt stats(..

- **REGRESSION TESTS:** Part  ्१२ items 4,5,6,11,14..

- **CONFIDENCE: HIGH.**

---

### P1-5 — Human-in-the-loop disconnected; terminal flags `human_aborted`/`explicit_stop` without producers

- **WHAT I OBSERVED:** `AgentExecution.execute()` raises `ApprovalBlockedError` instantly on approval-required (`agent_execution.py:198-216`(;`HumanApprovalHandler.request_approval()` never called anywhere in src (handler.py exists(;`blocked` reset at each `_prepare_step` (`orchestrator.py:847`(;`human_aborted`/`explicit_stop` zero writers;RecoveryPolicy maps UNKNOWN→STOP.

- **WHY REAL:** approval-required actions don't wait for a human — they burn 20 iterations re-blocking;evaluation runs can never complete those tasks;the two terminal reasons are unreachable (dead policy branches(;human decisions (approve/reject/modify( are impossible although the modules exist..

- **ROOT CAUSE:** the approval chain was designed (policies+handler( but never wired: no call site for `request_approval`,no stored `pending_action`,no accept/reject transition into the loop..

- **MY RECOMMENDED SOLUTION (Wave 1(:** blockthe STEP (persist `waiting_for_approval` + `blocked`,store the pending `ApprovalRequest` on state( and **stop the loop immediately** with a NEW terminal handling (e.g., `HUMAN_ABORTED` when reject/abort,`EXPLICIT_STOP` when stop(;wire the lightweight in-loop approval hook (optional,async( via `HumanApprovalHandler` through `AgentExecution`;Wave 2 can add full interactive resume..

- **ALTERNATIVES:** (a( full interactive approval loop now — rejected: needs persistence/resume infra (Wave 2(;(b( classify ApprovalBlocked as FATAL → at least stops — rejected: loses "waiting" semantics,but acceptable fallback if wave-1 scope shrink needed..

- **RISKS:** changing blocked-step behavior invalidates mocked approval tests (none exist today — nothing asserts the 20× loop,so low-risk(;new terminal states must be added to eval harnesses' reason mapping..

- **AFFECTED:** `agent_execution.py`, `orchestrator.py`, `termination.py`, `core/human/handler.py` (wire(,`diag_eval.py`/`run_evaluation.py` (reason mapping(..

- **REGRESSION TESTS:** approval-required → terminal block (not loop(;reject → HUMAN_ABORTED flag+reason;timeout path → TIMED_OUT reason;explicit_stop producer test..

- **CONFIDENCE: HIGH** (behavior gap proven by zero call-sites of `request_approval`(.



---

### Existing designs that I judge CORRECT (explicit statement(:

1. `PlanSchema`/`PlanStep` pydantic validators (exactly-one-final-answer,last,LLM;TOOL-no-final( — CORRECT and already protective;the 6 failing tests are STALE FIXTURES,not design flaws. Keep validators;fix fixtures (plan_id param added properly or removed(.
2. `ReliabilityEngine` retry/recovery architecture (retry→classify→recovery-with-change-detector( — CORRECT design;the bug is the missing wiring,not the engine..

3.. `AnswerSanitizer` pipeline;`VerifierAgent` deterministic-first verification gate (deterministic PASS/FAIL before LLM(;and `evidence_supports_candidate` — CORRECT design,intended to be a mandatory gate fore LLM-verification (currently bypassed only bythe budget-exhaust non-verification path(..

4.. `LoopDetector` with plan-history & normalized sha256 fingerprints — CORRECT for pattern detection;it should remain INDEPENDENT from the execution-history anti-repeat store (different purposes(..

5.. `ContextBudget`/`ContextCompressor`/`ContextValidator` stack — CORRECT as-is(notevidence of correctness issues found in this investigation(..

6.. Per-step `step_succeeded` scratch reset at each `execute()` start — CORRECT..

---

## PART  ्१५ — ARCHITECTURAL DECISIONS

**A. How should `AgentState` change?** → **REFACTORED (one option(.** Not minimally patched,not substantially
redesigned: introduce explicit lifecycle enums + derived properties + ownership contracts,keep the existing
field names/reporting surface (so evaluation/tests/diagnostics keep working(,delete only confirmed-dead fields..
Rationale: the blast radius of a scattered patch is large but tolerable;the value (provability( is exactly what
Wave 1 needs;a full redesign belongs to Waves  ्2+ when multi-turn/resume/persistence land..

**B. Lifecycle representation?** → **HYBRID (explicit state machine for phase + answer,booleans only as
derived views(.** Concretely: `AgentPhase(PLANNING,EXECUTING,VERIFYING,WAITING_APPROVAL,FINALIZED(`
+ `AnswerState(NO_ANSWER,ANSWER_READY,VERIFYING,VERIFIED,UNVERIFIED_BUDGET(`;
`final_answer_ready`/`final_answer_verified`/`task_completed` become **read-only properties** of these enums;
`termination_reason` stays the single terminal witness enum (already exists(. Booleans alone were proven insufficient
(Part  ्११(;a pure enum FSMs alone would ripple через serialization/evals;hybrid keeps both worlds..

**C. Where should TASK-level state live?** → `AgentState` itself (as today( — `user_request`, `evidence`,
`messages`( plus the new `phase`/`answer_state` enum;the task-scoped execution-history store ALSO on AgentState
(orchestrator-populated( so it survives rebind/serialization;`bind_state` must NOT clear task-scoped state..

**D. Where should PLAN-level state live?** → `AgentState.plan` + `completed_steps` + a NEW `plan_generation_id`
(int incremented by `_apply_full_plan`( — plan identity now explicit;`evidence[].step_id` stores `(plan_generation_id,
step_id(` or resolves via generation id so replan doesn't orphan evidence..

**E. Where should STEP-level state live?** → `AgentState` scratch fields (as today( BUT with a formal owner
(Orchestrator prepares,AgentExecution executes,and BOTH reset under one `_enter_step`/`_exit_step` contract(;
`step_type/current_action/tool_name/tool_arguments/tool_result/tool_error/blocked/risk/decisions` all reset in ONE
place (the new `_exit_step(`,preventingstale-leak class bugs (Part  ्१०(..

**F. Where should ATTEMPT-level state live?** → a new `AttemptLedger` dataclass on `AgentState`
(per attempt: `attempt_id,phase,tool,args_hash,start,end,result,error,fingerprint(( appended by the
implementing component (AgentExecution appends tool/LLM attempts;Orchestrator appends replan/verification
attempts(;budgets (retry/replan/verification( consume ledger entries,derived counters come fromlen(ledger(
— THIS replaces `retry_count`/`recovery_attempted`/`replan_count`/`verification_attempts` as sources of truth
(keep last 4 as mon structured report fields only(..

**G. Where should execution history live?** → ONE `ExecutionHistory` store on `AgentState` (task-scope(,
owned by Orchestrator;cleared ONLY on task init/rebind-NEW-task;(never on mid-task rebind/replan(;its
fingerprints come from the canonical `ExecutionIdentity` (Part  ्७( — replaces `Orchestrator.execution_history`
and deletes the dead `executed_step_fingerprints`..

**H. Where should counters live?** → NEW `AttemptLedger` + per-scope budgets (Part  ्५ F(;store
`replan_count`, `verification_attempts`, `retry_count`, `recovery_attempted` as **derived report properties**
overledger/filters (not independently writable fields(;`same_failure_count`/`last_failure_key` move INTO the
recovery-budget component (per-plan( and include step/plan identity in the key (fixes Part  ्१० #3(..

**I. SINGLE SOURCE OF TRUTH for completion?** → `AnswerState == VERIFIED` AND `AgentPhase == FINALIZED`
(derived `task_completed = (answer_state is VERIFIED(`.Termination uses phase/policy-state,NOT a flag.;
`termination_reason` remains the single terminal WITNESS (set once(;eval harnesses read reason+answer_state..

**J. SINGLE SOURCE OF TRUTH for answer verification?** → the verifier result itself: store
`verification_result: VerificationResult | None` on state (или последний `VerificationAttempt` in ledger(;
`final_answer_verified` derives from `verification_result.verified`;`verification_attempts` per-candidate counter
derives from ledger entries for phase==VERIFYING (per `plan_generation_id`(..

**K. Which current fields should become DERIVED?** → `final_answer_ready`, `final_answer_verified`,
`task_completed` (from `AnswerState`(;`retry_count`,`recovery_attempted`,`replan_count`,
`verification_attempts` (from `AttemptLedger`(;`iteration` stays stored (sole outer-pass counter(;
`execution_success` becomes "last step succeeded" derived from last ledger entry.

**L. Which current fields should REMAIN (stored(?** → `user_request`,`user_id`,`messages`,`plan`,
`current_step`,`completed_steps`,`step_type`,`current_action`,`tool_name`,`tool_arguments`,`tool_result`,
`tool_error`,`blocked`,`waiting_for_approval`(+ pending ApprovalRequest(,`risk_assessment`,
`approval_decision`,`execution_decision`,`iteration`,`final_answer`,`termination_reason`,`evidence`,
`replan_count/verification_attempts/retry_count/recovery_attempted` (as report mirrors(,
`same_failure_count`,`last_failure_key`,`loop_salvage_attempted`,`timed_out`,`fatal_error`,
`human_aborted`,`explicit_stop` (now WITH producers wiring(..

**M. Which changes belong to Wave 1?** → (1( wire recovery/change-detector; (2( add
`AgentPhase`/`AnswerState` + derive trio/task_completed; (3( `AttemptLedger` + derived counters
+ per-candidate verification budget reset; (4( `ExecutionIdentity` canonical store on state (не clear on
replan/rebind(; (5( wire approval stop/terminal handling (loop-stop,not 20×(; (6( one config for
max_iterations/max_verification_attempts/max_replans injected from main(; (7( clear-step-reset & no-stale
success flags; (8( fix dead fields (delete `executed_step_fingerprints`, `same_plan_count`,
`execution_results` or repurpose(; (9( make `iteraion`-style typos impossible (attr-whitelist + regression
test(; (10( fix 7 failing tests fixtures (fixture plans conform to validators;orchestrator factory
for tests(; (11( eval harness normalization (single answer-encoding,reason mapping incl timeouts(..

**N. Which changes MUST wait for Wave 2/3/4?** → Wave ्2: interactive approval resume+pending
request persistence;multi-task state serialization/restore (needs `ExecutionHistory` on disk(;per-tool
`arg_order_significant` spec-driven normalization edge cases;`_diag*`/stale harness archival and single
canonical scorer (GAIA submission(;Wave ्3+: plan-patch/merge (partial plan replacement rather than full
replan(,multi-agent orchestration,state-diff logging;Wave ्4+: RL/cost optimization (inner budgets
beyond correctness(, streaming/eval infrastructure..

**O. Which existing capabilities must explicitly remain untouched?** → (1( AgentLoop/Orchestrator/
AgentExecution layering;(2( Memory, retrieval, context-builder stack (ContextBudget/Compressor/Validator(;(
3( Planner+plan_schema validators (keep,keep enforcing(;(4( ReliabilityEngine retry/classify/recovery
architecture (fix wiring only(;(5( Risk assessor+ApprovalPolicy (keep policies;wire handler(;(6(
Verifier deterministic-first + evidence_supports_candidate;(7( LoopDetector pattern detection;(8( Observability
(event/metrics/tracer/token_tracker(;(9( ToolRegistry/contract_validator/tool set;(10( Evaluation purposes
(GAIA score,HF unit-4 scoring( — harnesses keep working against existing result files format where
possible,. NO capability may be removed;dead-field deletion is a refactor,not a removal of behavior..

---

## PART  ्१६ — WAVE 1 FILE-BY-FILE CHANGE PLAN

Ordering suggested: **(S1( state model → (S2( engine wiring → (S3( counters/identity → (S4( approval →
(S5( harnesses → (S6( tests**. Each file lists CURRENT PROBLEM, CHANGE REQUIRED, DEPENDENCIES, RISK,
TESTS REQUIRED.（DO NOT implement yet — this is the plan.(

| FILE | CURRENT PROBLEM | CHANGE REQUIRED | DEPENDENCIES | RISK | TESTS REQUIRED |
|---|---|---|---|---|---|
| `src/gaia_agent/core/agent_state.py` | flat un-owned bag;dead fields;trio flags unguarded;no phase/answer state | add `AgentPhase`,`AnswerState` enums;add `verification_result`, `plan_generation_id`, `attempt_ledger`, `pending_approval` fields;turn `final_answer_ready/verified/task_completed/retry_count/recovery_attempted/replan_count/verification_attempts` into `@property` derived;delete `executed_step_fingerprints`,`same_plan_count`,`execution_results`;move `execution_history` into state (task-scope(;`__post_init__` invariant self-check;attr-whitelist (`__setattr__` guard for unknown names in debug( | S1 standalone | wide ripple;tests pinning flags | 1,2,3,6,8-15 from Part  ्१२;attr-whitelist/typo test |
| `src/gaia_agent/core/policies/termination.py` | flag-collection precedence;`task_completed` unused;budget constant duplicated | evaluate over `phase`/`answer_state` (+ existing flags for compatibility(;read `max_verification_attempts` from injected config (NOT self default alone(;add `WAITING_APPROVAL`-terminal branch could be via phase+flag (see agent_execution( | S1 | terminal semantics change flips gate tests | gate tests updated deliberately;new terminal-state tests (timeout,approval,HUMAN,EXPLICIT( |
| `src/gaia_agent/core/agent_loop.py` | increments iteration blindly;no post-state invariant check;timeout not in-loop;massive print noise | after each `run_iteration()` run invariant checker (debug gated(;loop-check invariant before increment;keep printing (diagnostics existing capability(;NO logic removal | S1 | perf (print volume( — keep behind env flag | loop-invariant regression tests |
| `src/gaia_agent/core/orchestration/orchestrator.py` | recovery_change_detector never passed (P0-1(;`_check_recovery_budget` conflation;verification budget global;execution_history cleared at bind_state;fingerprint non-normalized;record before validation;stale success flags on `_exit_step` | (a( pass `recovery_change_detector` at 3 engine call sites + `max_recoveries` aligned;(b( split budget checks per type (replan-budget per plan,verification-budget per candidate( with `verification_attempts` reset on candidate replace;(c( store `execution_history` on state via task-scope (clear only on NEW task(;(d( adopt `ExecutionIdentity` (canonical normalize,`plan_generation_id`,LLM-answer nonce(;record AFTER validate/prepare;(e( add `_exit_step` reset discipline (clear tool scratch+success flags on every exit of step execution(; (f( wire `pending_approval` — blocked step → loop STOP via phase=WAITING_APPROVAL (+ optional async `request_approval`(; (g( `_apply_full_plan` increments `plan_generation_id`,resets per-plan budgets (NOT task ones( | S1,S2 | largest diff;mocked tests engineers assert old wiring | Part 12 items 3,5,6,7,8,9,13,14,19,20,23 |
| `src/gaia_agent/core/agent_execution.py` | failure paths leave stale `execution_success`;approval raise без handler;waits? none;record timing (fingerprint after validation( должен быть orchestrator-side | (a( on ANY except path set `execution_success=False, step_succeeded=False` (b( `pending_approval` set + optional `request_approval()` via injected handler (async(;(c( expose per-attempt start/end to `AttemptLedger` (append ( | S1,S2 | approval tests new;failure-flag tests new | items 3,4,9,23 |
| `src/gaia_agent/reliability/engine.py` | NO bug itself;(recovery требует detector (by design( — keep | (optionally( accept richer `recovery_change_detector` docs;expose per-attempt stats (attempts/recovery_count( to ledger callback;unchanged else | S2 | low | engine tests extend (item 5(;existing pass unchanged |
| `src/gaia_agent/reliability/policies/recovery_policy.py` | UNKNOWN→STOP blocks blocked-steps | keep (correct(;ensure ApprovalBlocked classified FATAL/STOP explicitly instead of UNKNOWN (classifier change( | S2 | low | classification tests (human/approval( |
| `src/gaia_agent/reliability/failure_classifier.py` + `error_handler.py` | ApprovalBlocked→UNKNOWN (severity MEDIUM( | classify `ApprovalBlockedError` as HIGH/PERMANENT STOP with explicit category | S2 | low | `test_error_classification.py` extend |
| `src/gaia_agent/planner/planner.py` | `replan_step` unused for tool failures (dead(;plan identity absent | export `strategy_family(step(`;ensure `replan()`/`replan_step` NEVER emit TOOL-final steps (validator already guards(;plan keeps final-answer step valid | S2 | planner tests pass today — keep them green | runtime-contract tests extend (item 6,17( |
| `src/gaia_agent/agents/verifier.py` | returns result but orchestrator only stores triplet | keep `VerificationResult` as-is;orchestrator stores it — NO verifier change needed (Wave 1( | S1 | none | existing verifier tests stay green |

| `src/gaia_agent/core/human/handler.py` + `models.py` | exists but never called | wire into `AgentExecution` approval path (optional async hook(;REJECT/ABORT sets `human_aborted`;MODIFY rewrites `tool_arguments`;STOP sets `explicit_stop` | S4 | new interactive surface (CLI only( | new approval lifecycle tests |
| `src/gaia_agent/config.py` + `main.py` | constants split;`config.agent_max_iterations` unused | single `LimitsConfig(max_iterations,max_verification_attempts,max_replans,max_attempts_per_step(` injected into TerminationPolicy+Orchestrator+ReliabilityEngine;main uses it;`iteraion`-typo impossible via state attr-whitelist | S1 | config plumbing | config round-trip tests |
| `src/gaia_agent/observability/events.py` (+`logger`,`metrics`( | event `iteration` only | add `phase`,`attempt_id`,`reason` metadata;no schema break | S1 | none | event-shape tests |
| `src/gaia_agent/core/evidence.py` + `tools/contract_validator.py` + `tools/registry.py` | `ToolResultRecord.step_id` plain int;fingerprint uses raw args | (Wave-1 optional( add `plan_generation_id` to record;expose `arg_order_significant` per tool spec for canonicalize;else defer | S3 | low | evidence-link tests (item 13,20( |
| `src/run_evaluation.py` + `src/gaia_agent/diag_eval.py` | divergent encodings (None vs "0"(;timeout→reason None;dead alias loop | shared `result_encoding` helper (answer None vs empty;reason enum→json stable(;report `answer_state`,`phase`,`iteration`,`last_attempts`;timeout records TIMED_OUT reason by re-checking after `wait_for` (loop-checked inside run(;remove alias dead loop | S5 | harness output change — document | eval-encoding tests (item 21,10,11( |
| `src/gaia_agent/evaluation/*.py` (stubs( | 0-byte package | Wave 2 — replace with canonical scorer OR delete stubs so `from gaia_agent.evaluation import evaluator` fails loudly (recommend DELETE in Wave 1 to avoid silent ImportError later( | S5 | low | import smoke test |
| cleanup `src/_diag_*`,`src/tempCodeRunnerFile.py`,`src/gaia_agent/{,agents}tempCodeRunnerFile.py`,`src/run_loop.py`,`src/test_debug_fixes.py`,`src/smoke_test_fixes.py` | dev leftovers,duplicate harnesses | **Wave 2 hygiene** — archive;not Wave 1 except marking in `pyproject` if they break tooling | — | none | n/a |
| `tests/**` (fixtures + new suite( | 6 fixture failures (PlanSchema validators(;1 `__new__`-harness gap;missing invariant tests | update fixtures to valid plan shapes (final-answer step;no TOOL-final(;add `tests/core/test_state_machine.py`,`test_recovery_wiring.py`,`test_execution_identity.py`,`test_approval_terminal.py`,`test_eval_encoding.py`,`test_agent_state_whitelist.py`;update gate/budget mocks to new derived fields;keep all 15 Wave-1 regression engines green | S1-S6 | test churn additive;expect ~30 new tests | the 15 items + existing ~129 pass |
| `src/gaia_agent/diag_eval.py` results file schema (`_eval_results_hf20.json`( | raw flags | include `phase`,`answer_state`,`verification_result`,`attempt_ledger` summary | S5 | output compat | backward-compat reader test |

---

## PART  ्१७ — FINAL VERDICT

**TOP 5 ROOT CAUSES** (evidence → Part 13(:

1. **ReliabilityEngine recovery is dead-wired** — `recovery_change_detector` никогда passed by Orchestrator
   (`orchestrator.py:222-228,286-295,1104-1111`;engine gate `engine.py:122-133`( → tool/plan failures never
   replan,loop till `MAX_ITERATIONS`,`final_answer=None`..
2. **AgentState is a flat flag bag with no lifecycle state machine / no ownership contracts** —
   10+ dead fields,`task_completed` unread by termination,unreachable terminal flags
   (`human_aborted`,`explicit_stop`(,no invariant enforcement (`agent_state.py`;Part 1/3(..
3. **Execution identity split-brain & non-normalized** — dead twin +
   bind_state-cleared live store + json-style fingerprints + LLM-final-answer collisions
   (`agent_state.py:99`,`orchestrator.py:81,90,102-136,693,998`;Part  ्७(..
4. **Counter semantics conflated & never reset** — shared replan budget,dead retry/recovery counters,
   verification budget leaking across candidates (`orchestrator.py:144,917,1341-1343`;Part  ्५/6/8(..
5. **Human-in-the-loop disconnected + dead terminal flags** — `ApprovalBlockedError` unhandled loop,
   `request_approval()` никогда called,flag producers absent (`agent_execution.py:198-216`;Part  ्१०(..

**MOST DANGEROUS CURRENT BUG:** the dead recovery wiring (RC1( — it silently disables the ENTIRE
recovery/replanning capability,so tool failures both fail the task (no strategy change( AND burn the full
iteration budget AND produce misleading metrics;combined with the flag-bag (RC2( it reproduces
"GAIA enters incorrect states, repeats work, loses history, terminates incorrectly" on virtually every
non-trivial GAIA question. Runner-up: the historical `iteraion` AttributeError (100% eval failure( — proof of
the price of no state contract.

**MOST IMPORTANT ARCHITECTURAL PROBLEM:** absence of an explicit lifecycle/state machine and per-field
ownership contract on `AgentState` — every other root cause (1,3,4,5( is made possible or undetectable by it.

**MY RECOMMENDED WAVE 1 DESIGN:** keep all components and their responsibilities;add three small skeletal
abstractions — (i( `AgentPhase`/`AnswerState` enums with derived flags,(ii( `AttemptLedger` + per-scope
budgets,(iii( canonical `ExecutionIdentity` store on AgentState — wire the existing recovery engine with change
detectors,reset per-candidate budgets,stop-and-terminal on approval-required,update harnesses to a single
encoding,and lock everything with the 15 regression test areas (Part  ्१२(.

**CHANGES I STRONGLY RECOMMEND:**
1. Pass `recovery_change_detector` (+ aligned `max_recoveries`( at all `reliability_engine.execute` call sites.
2. Introduce `AgentPhase`+`AnswerState`,derive trio/task_completed,delete dead fields,attr-whitelist.
3. Wire approval to stop-and-terminal (with optional `request_approval` hook( and make
   `human_aborted`/`explicit_stop`/`timed_out` reachable + dashboarded via in-loop policy..
4. Per-candidate verification budget (reset on plan/candidate replace( + single config for all limits..
5. Canonical `ExecutionIdentity` + single task-scoped history on AgentState (never cleared on replan/rebind(..
6. Fix the 7 failing test fixtures to the (correct( PlanSchema contract + orchestrator test factory..

**CHANGES I DO NOT RECOMMEND:**
1. Removing the change-detector requirement from `ReliabilityEngine` (safety invariant(..
2. Full `AgentState` redesign/DSL in Wave 1 (out of scope;leave for Waves  ्2+ as follow-up evolution(..
3. Deleting `LoopDetector`'s independent pattern detection (its purpose differs from anti-repeat(..
4. Removing `PlanSchema` final-answer validators to make old fixtures pass (keep contract;fix fixtures(..
5. Interactive resume/persistence for approvals in Wave 1 (needs infra;land in Wave 2(..

**RISKS:** (1( budget semantics change may flip currently-passing gate/budget tests — update deliberately and
review diff-to-test ratio;(2( derived-property migration ripple through harnesses/serialization — keep
backward-compatible output fields;(3( newly-wired recovery will actually consume `MAX_REPLANS` on hard tasks —
expect behavior change (fewer loops,more `fatal_error`(;(4( tool-spec normalization (arg order( requires
judgment calls — start conservative (normalize only dict-key order + scalar canonicalization(;(5( approval
stop changes evaluation behavior for risk tasks — document new terminal reasons in harnesses..

**DEPENDENCIES:** this plan's S1→S6 ordering;`PlanSchema` validators (keep( act as the plan-contract anchor;
`config.py`+`main.py` become the single knob layer;no new third-party libraries required (pydantic,asyncio,
dataclasses already present(..

**CONFIDENCE: HIGH** — every root cause is anchored to concrete,cited code paths;the only MEDIUM-confidence
area is the exact fidelity of the derived-property migration (enum choice details(,for which the Wave 1
implementation must re-run the full suite + the 15 regression areas..

---

## APPENDIX — EVIDENCE INDEX & REPRODUCTION

### A. Key file/line citations (fast lookup for Wave 1 implementation(

| Concern | Citation |
|---|---|
| Recovery gate refusal | `src/gaia_agent/reliability/engine.py:120-133` |
| Engine call sites WITHOUT detector | `src/gaia_agent/core/orchestration/orchestrator.py:222-228` (tool execution(;`:286-295` (plan generation(;`:1104-1111` (verification — no `recovery_operation` either( |
| Retry loop (invisible to state( | `src/gaia_agent/reliability/engine.py:313-381`;`reliability/retry.py:70-110` |
| `bind_state` clears history | `src/gaia_agent/core/orchestration/orchestrator.py:83-90` |
| Fingerprint (non-normalized( | `orchestrator.py:102-120`;`_arguments_fingerprint:640-651`;record-before-validate `:219-220` vs `agent_execution.py:339-343` |
| LLM final-answer collisions | `orchestrator.py:693`;`orchestrator.py:998`;`_force_different_strategy:701-704` |
| `_apply_full_plan` reset set | `orchestrator.py:390-413` (and what it does NOT reset,Part 6( |
| verification attempts global | `orchestrator.py:1017-1019` (increment(;`:1202-1208` (budget-exhaust(;`:1341-1343` (trio reset WITHOUT attempts( |
| replan budget shared | `orchestrator.py:138-180` (`_check_recovery_budget`(;`:40-42` constants |
| Approval raise without handler | `agent_execution.py:198-216`;`core/human/handler.py` (never called( |
| Termination precedence | `core/policies/termination.py:52-112` (main(;`agent_loop.py:188-190,590-659` (caller( |
| Iteration increment | `agent_loop.py:334` |
| Stale `execution_success` on failure | `agent_execution.py:448-476,572-600` (set tool_error only(;orchestrator validator `:854-866` |
| `iteraion` archive evidence | `evaluation_results.jsonl` lines 1..N (error strings(;0 occurrences in `src/**/*.py` |
| Dead fields | `agent_state.py:37-39` (`execution_results`(;`:99-101` (`executed_step_fingerprints`(;`:96` (`same_plan_count`(;`:75` (`retry_count` — only reset at `orchestrator.py:917`(;`:77` (`recovery_attempted` — only False at `781,918`( |
| PlanSchema final-answer contract | `planner/plan_schema.py:104-153` (exactly one,last,LLM(;`:74-101` (TOOL-no-final( |
| Test failures | Section header commands;7 failures classified in Part  ्१२ |
| `iteraion`/alias fallback masking | `agent_loop.py:123-139` (`getattr` fallbacks(;`diag_eval.py:286-295` (dead alias loop due `slots=True`( |

### B. How to reproduce the analysis

```powershell
# 1) state audit (all read/writes of the 30 fields)
cd c:\Users\user\gaia-agent
$fields = @('iteration','current_step','final_answer','task_completed','retry_count','replan_count')
foreach ($f in $fields) { Get-ChildItem -Recurse src,tests -Filter *.py |
  Where-Object { $_.FullName -notmatch 'venv|__pycache__' } |
  Select-String -Pattern ([regex]::Escape($f)) |
  ForEach-Object { "$($_.Path):$($_.LineNumber)" } }

# 2) iteraion check
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch 'venv|__pycache__|\.git' } |
  Select-String -Pattern 'iteraion'   # → only evaluation_results.jsonl error strings

# 3) run suites (pytest-timeout NOT installed)
.venv\Scripts\python.exe -m pytest tests\planner tests\reliability tests\integration -q
```

### C. Verify-no-modification statement

At the time of writing, `git status` shows **no production source or test file modified by this
investigation**. The only repository artifact produced is this file (`DEBUGGING_TEST.md`). The
`evaluation_results.jsonl` file is pre-existing archived output, read-only here.

### D. Limitations / gaps of this investigation (honest disclosure(

1. **No live LLM run**: ollama is not reachable in this environment;all runtime claims rest on static
   tracing + the mocked/unit suite. The `evaluation_results.jsonl` (real run( provides one real-world data
   point (100 % `iteraion` crash — from an older build(.
2. **`evaluation_results.jsonl` provenance**: it was produced by an older revision (the typo'ed field(;
   it is used here ONLY as evidence of the historical class of failure,not as a measure of today's behavior.
3. **`ReliabilityEngine` budget worst-case math** is traced statically;exact per-task tool-attempt. counts
   would require a live run with attempt-ledger logging (Wave 1 addition(.
4. **Some branch outcomes** (e.g., whether `RecoveryPolicy` returns STOP on UNKNOWN for a specific test
   input( were confirmed against mocked tests;the approval-20×-loop claim is a static trace (no test asserts
   it today(.
5. **`context/` and `memory/` subsystems**: inspected for state-field usage only;no deep audit of their
   internal correctness was performed (out of Wave-1 Core scope(.

---
*End of DEBUGGING_TEST.md — forensic diagnosis complete. Ready for Wave 1 implementation per Part 16.*