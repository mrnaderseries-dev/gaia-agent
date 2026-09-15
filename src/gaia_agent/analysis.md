# GAIA Architecture Analysis

## Session Start

- Repository: c:\Users\user\gaia-agent (git: main @ ff7ed26, in sync with origin/main)
- Working directory: c:\Users\user\gaia-agent\src\gaia_agent
- Python environment: Python 3.12.0 (system) + project .venv (Python 3.12.0, gaia-agent 0.1.0 installed editable)
- Entrypoint: `python -m gaia_agent.main` (src/gaia_agent/main.py — composition root `create_agent()` + default request "Calculate 2 + 2.")
- Test suite: tests/ with subdirs agents, context, integration, planner, reliability, tools + top-level test_conversation.py, test_database_connection.py, test_memmory.py
- Initial git status: clean tracked tree; only untracked runtime log artifacts from previous sessions (rt_*.txt, run_*.txt, runtime_*.txt, output logs) — no uncommitted code changes (`git diff --stat` empty)
- LLM backend: local Ollama at http://localhost:11434, models present: `qwen2.5:3b` (text, tools-capable), `nomic-embed-text:latest` (embeddings). NOTE: main.py configures VISION_MODEL=`gemma3` which is NOT present in Ollama — flagged as potential vision capability gap.

## Objective

Audit the existing architecture, run the real application through main, diagnose/fix real runtime failures with minimal production changes, test progressively harder GAIA tasks (math, text, web, vision, file, multi-hop, recovery), and document everything here. No redesign, no shortcuts, no hardcoded answers.

## Architecture Audit

Inspected from source (Phase 1). Layer map with ownership:

| Layer | File | Owns | Notes |
|---|---|---|---|
| AgentLoop | core/agent_loop.py | run loop, termination policy invocation, outcome->state mapping | Calls orchestrator.start/step, TerminationPolicy.evaluate; refuses COMPLETE without verified answer (RuntimeError) |
| Orchestrator | core/orchestration/orchestrator.py | planning/installing/executing/verifying/replanning steps | Holds OrchestrationContext (run identity, PlanRuntime, execution/verification history) |
| PlanRuntime | core/orchestration/models.py (PlanRuntime) | installed plan, current_step, completed_steps, plan_version, replan_count | AgentState receives projections only |
| Planner | planner/planner.py | classification, plan generation/replan, validation | TaskClassifier + StrategySelector + SemanticPlanValidator + LoopDetector delegation |
| PlanSchema | planner/plan_schema.py | plan structural contract | exactly one final LLM step, last, tool_name=None, arguments={} |
| AgentExecution | core/agent_execution.py | tool + LLM step execution, risk/approval policies | Builds ToolResultRecord evidence for TOOL steps; LLM steps produce metadata only |
| Reliability | reliability/engine.py + retry/recovery/policies | failure classification, retry/replan decisions | Orchestrator decides replan; engine returns ReliabilityAction |
| LoopDetector | reliability/loop_detector.py | EXACT/STRUCTURAL/SEMANTIC repetition detection | check_plan() (intra-plan, skips final-answer steps) + check()/record() (cross-plan history) |
| Verification | agents/verifier.py | final-answer verification | deterministic first, then LLM semantic; strict INSUFFICIENT_EVIDENCE fallbacks |
| Context | context/ContextBuilder.py + sources | builds FinalContext for planner/execution | |
| Observability | observability/facade.py + logger/metrics/tracer | events, spans, metrics | Orchestrator._emit/_span; correlation via ObservabilityContext |
| LLM | llm/service.py, provider/ollama.py | generation, structured output | OllamaClient timeout=120s, format=json-schema |
| AgentState | core/agent_state.py | lifecycle ownership, projections | _ALLOWED_TRANSITIONS; complete() refuses unverified answers |
| TerminationPolicy | core/policies/termination.py | stop decisions | COMPLETED / MAX_ITERATIONS / ANSWER_UNVERIFIED_BUDGET / FATAL_ERROR etc. |

## Runtime Validation

Run 1 (main_run_1.txt, `python -m gaia_agent.main`, exit 0) and Run 2 (main_run_2.txt, clean cmd redirect, exit 0). Both runs FAILED the task:

- FINAL ANSWER: "4" (produced, later reset to None by _verify failure path)
- PHASE: AgentPhase.FAILED
- TASK COMPLETED: False
- FINAL ANSWER READY: False (reset) / FINAL ANSWER VERIFIED: False
- FATAL ERROR: False
- REPLAN COUNT: 0, RETRY COUNT: 0
- ITERATION: 3, VERIFICATION ATTEMPTS: 1
- TERMINATION REASON: None
- Transitions: IDLE->PLANNING->EXECUTING->VERIFYING->PLANNING(VERIFICATION_FAILED)->FAILED(EXECUTION_FAILED)
- Executions: #1 python_interpreter step 0 success output=4; #2 LLM final step 1 success output=4
- TOOL ERROR: 'Replanned plan repeats the failed execution.'
- stderr: "Planner replan generation/validation failed:" + httpx/httpcore ReadTimeout traceback from OllamaClient during Planner.replan LLM call (planner.py:416 -> ollama.py:68 -> _request)
- Same terminal error string appears in 6 prior session logs (rt_2plus2.txt, run_2plus2.txt, runtime_final.txt, run_new_2plus2.txt, run_verify.txt, runtime_A/B.txt) - persistent, reproducible failure.

## Repository Mapping (Hypothesis Verification Phase — MAPPING ONLY, no changes)

### Planner
- Actual file: src/gaia_agent/planner/planner.py
- Class: Planner
- Relevant methods: generate_plan(); replan() (:366); _validate_generated_plan() (:499); _validate_recovery_strategy() (:724, early-returns unless failed_step is TOOL); get_alternative_strategy() (:783); _tool_and_final_plan() (:977); _build_replan_prompt() (:1118); _fingerprint_step() (:1654); strategy_family() (:1628)

### Replan Fingerprint
- Actual file/function: planner/planner.py :: _fingerprint_step (:1654-1669)
- Contract: `f"{step.step_type.value}|{step.tool_name or ''}|{json.dumps(step.arguments, sort_keys=True)}"`
- KEY FACT: for any LLM final-answer step this is the CONSTANT `"llm||{}"` because PlanSchema (planner/plan_schema.py: validate_step_contract :74-101, validate_steps :104-153) *forces* every final-answer step to be LLM, tool_name=None, arguments={}. Two valid plans can never differ in their final-answer-step fingerprint.

### Duplicate Detection
- Actual file/function: planner/planner.py :: _validate_generated_plan
- Intra-plan duplicate check (:556-565): fingerprints only NON-final steps (`if not step.is_final_answer`) -> "Plan contains duplicate tool executions."
- Failed-step check (:577-584): compares ALL steps' fingerprints against failed_step_fingerprint -> PlannerRecoveryRequired("Replanned plan repeats the failed execution.") — NO final-answer exclusion here. This is the inconsistency.

### Loop Detection
- Actual file/class: reliability/loop_detector.py :: LoopDetector — check (:75), record (:134), check_plan (:173), signature (:247), reset (:242)
- Contract: check_plan() skips `step.is_final_answer` (:189) — intra-plan. check() compares a candidate against persistent `_history` recorded across the whole run (EXACT/STRUCTURAL/SEMANTIC, threshold 0.88).
- Orchestrator usage: core/orchestration/orchestrator.py — check before execution (:488) -> on detection emits LOOP_DETECTED and FAILS with AgentError "ExecutionLoopDetected" (LOOP_DETECTED, HIGH, retryable=False, recoverable=False) (:493-524); record after successful execution (:631). `_replan` (:767-973) does NOT call loop_detector.reset() and does NOT salvage already-successful steps.

### Verification
- Actual files/classes/methods:
  - agents/verifier.py :: VerifierAgent.verify (:889) -> _filter_current_evidence (:178) -> _check_source_support (:502) -> deterministic_verification (:543) -> if not VERIFIED/INVALID -> _semantic_verify (:942; LLM call with output_schema=VerificationResult) -> _validate_llm_result (:972; non-VerificationResult or status None => INSUFFICIENT_EVIDENCE "invalid or incomplete verification result")
  - Helpers: _extract_structured_answer (:340), _extract_structured_answers (:359)
- Orchestrator side: core/orchestration/orchestrator.py :: _verify (:975-1152). Builds VerificationInput(question=run.user_request, candidate_answer=answer, raw_data=run.verification_evidence(), task_type=task_analysis.intent.value) (:1041-1048). On failure with budget remaining -> _replan(failed_step=step, ...) (:1146-1152) where `step` is the FINAL-ANSWER LLM step (routed from _execute :656 `if step.is_final_answer: return await self._verify(...)`).
- Evidence contract: OrchestrationContext.verification_evidence() (core/orchestration/models.py:459-483) = evidence of SUCCESSFUL executions only. ToolResultRecord records are built in core/agent_execution.py (~:950) for TOOL steps (python_interpreter => "strong" evidence); LLM steps produce NO evidence records.

### Ollama
- Actual file/class/method: llm/provider/ollama.py :: OllamaClient.generate (:35), _request (:94)
- Configuration: base_url http://localhost:11434, timeout=120.0s (default), structured output via payload["format"]=json-schema; httpx.ReadTimeout observed during replan generation.

### Call chain (verified from source)
User Request -> AgentLoop.run_agent (core/agent_loop.py) -> Orchestrator.start/step -> _plan (Planner.generate_plan -> PlanSchema -> PlanRuntime install) -> _execute (loop_detector.check -> AgentExecution -> Tool/LLM) -> reliability path (ReliabilityEngine.handle_failure -> RETRY/REPLAN/STOP) on errors -> final-answer step routed to _verify -> verification_evidence() -> VerifierAgent.verify (deterministic | Ollama LLM) -> VERIFIED: COMPLETE (AgentState.complete, TerminationPolicy COMPLETED) | NOT verified with budget: _replan -> Planner.replan -> _validate_generated_plan -> install -> re-execute | replan rejected (PlannerRecoveryRequired) -> _fail -> FAILED.

## FINAL READINESS AUDIT — 2026-09-15 (part 1: truth + tasks A/B)
- Tree: `M verifier.py`, `M planner.py`; untracked analysis.md, logs,
  temp probes (`readiness_probe/step_probe/classify_probe`, to delete).
- Config: TEXT=qwen2.5:3b, VISION=gemma3 (ABSENT in Ollama →
  CAPABILITY GAP, reconfirmed). Ollama has only qwen2.5:3b +
  nomic-embed-text. `datasets` NOT installed → run_evaluation loader
  untested (ENVIRONMENT ISSUE). Net OK (google → 200).
- Default re-run `python -m gaia_agent.main` → COMPLETED, '4',
  VERIFIED True, 0 replans. GOOD, still valid.
- Task A `25 * 17 + 43` (expect 468): step probe showed tool ran
  `result = 25 * 17` → `425` (dropped `+ 43`). BUG: planner-local
  `detect_simple_operation` matched only first two operands while
  canonical classifier detector returned None here (F1 vs F2 proven).
  Full run: verifier correctly rejected, replan repeated same code,
  LoopDetector correctly blocked (GOOD). Fix 3: planner-local
  detector delegates to canonical first, then full-expression
  fallback (compile-checked, div-zero guarded). After fix the step
  probe runs `result = 25 * 17 + 43` → `468`. Full E2E then hit a
  final-LLM ReadTimeout (184s) — MODEL LIMITATION, not arch bug.
- Task B reverse "architecture": plan was LLM-only, model
  hallucinated `ecnitrauC`, verifier correctly INSUFFICIENT →
  replan. MODEL LIMITATION + CAPABILITY GAP (no deterministic text
  code path; `_deterministic_fallback_code` has no text branch).
  Planner correctly avoided web_search. GOOD.
- Tasks C–K live runs deferred; unit-level guards proven (78/78
  adversarial). Remaining uncertainty recorded below.

## FINAL-pt2 (tests, perf, output contract)
- Non-integration suite: 57 failed / 321 passed WITH fixes; stashed
  baseline shows IDENTICAL 57/321 → all pre-existing. TEST ISSUE
  (stale PlanningResult/llm.calls/reason-string/`MAX_VERIFICATION`
  imports/engine TypeErrors). DO NOT chase. Contract suites:
  adversarial + registry = 128 passed, 2 skipped. Planner 108/6
  (stable pre-existing).
- Perf: task-A E2E 184s → final-LLM ReadTimeout (120s Ollama
  timeout); step execution correct. MODEL LIMITATION (qwen2.5:3b
  structured-output tail latency), not arch bug.
- Output contract: `src/run_evaluation.py` writes JSONL
  `{task_id, question, answer=final_answer, ...}` — clean shape,
  no leakage. GOOD. But `evaluation/*.py` are EMPTY stubs and no
  official HF Unit-4 scoring contract is vendored → CAPABILITY GAP;
  loader untested (no `datasets` pkg).

# GAIA READINESS GATE

Status: READY WITH KNOWN LIMITATIONS

## Blocking Issues (must fix before HF eval)
1. Multi-term arithmetic planner bug — FIXED this session (Fix 3).
   Re-verify E2E once tail latency allows; step-level proof: 468.
2. Deterministic text-transform path missing — OPEN. Reverse-string
   ran LLM-only and the model misspelled. Needs a text code-gen
   branch in `_deterministic_fallback_code` (or LLM-code tool call),
   else GAIA text tasks depend on qwen2.5:3b literals.
3. Tail latency (qwen2.5:3b structured output, 120s timeout) — OPEN
   policy/host issue: task-A E2E hit final-LLM ReadTimeout at 184s.
   Consider longer timeout / smaller prompts / faster model for eval.

## Non-Blocking Issues
- 57 pre-existing non-integration failures + 6 planner + 3
  verifier_strong: all baseline-identical, stale expectations.
  No production change warranted in this audit.
- 2 reliability modules unimportable (`MAX_VERIFICATION_ATTEMPTS`).
  TEST ISSUE; excluded from counts.

## Capability Gaps
- Vision: gemma3 absent (only qwen2.5:3b + nomic-embed-text). J.
- Evaluation: `evaluation/*.py` empty stubs; no vendored HF Unit-4
  scoring contract; `datasets` missing so loader untested. I/file,
  H/web, K/multi-hop live runs still deferred.

## Model Limitations
- qwen2.5:3b: slow structured output (replan/final), wrong
  reverse-string literal. Architecture reacted correctly both times.

## Runtime Evidence
- Default 2+2: COMPLETED '4' VERIFIED (0 replans). GOOD.
- Task A step probe: `result = 25 * 17 + 43` → `468`. GOOD.
- Task A E2E: FAIL on final-LLM ReadTimeout (evidence above).
- Task B: LLM-only, wrong literal, verifier INSUFFICIENT → replan.
  No web_search selected. GOOD guard, BAD literal.

## Test Evidence
- adversarial + registry: 128 passed, 2 skipped.
- planner: 108 passed / 6 pre-existing failed.
- non-integration: 321 passed / 57 pre-existing failed (baseline
  identical with fixes stashed).

## Hugging Face Readiness Matrix
| Class | Runtime proof | Verdict |
| arithmetic (2-term) | default E2E pass | PASS |
| arithmetic (multi-term) | step probe 468; E2E timeout | PARTIAL |
| text transform | wrong literal, no code path | NOT PASS |
| web/file/vision/multihop | not yet run live | UNPROVEN |
| submission shape | JSONL answer=final_answer | PASS (shape) |
| official scoring | no contract vendored | UNPROVEN |

## Final Decision
READY WITH KNOWN LIMITATIONS for HF prep on arithmetic-style
tasks; NOT READY for full GAIA Unit 4 until text code path +
latency policy + web/file/vision live proofs exist. Recommended
next action: add minimal text-transform code branch, raise or
bypass tail-latency risk for eval, then run live H/I/J/K probes
before touching the real evaluation set. Temp probes
(readiness_probe/step_probe/classify_probe) should be deleted
before commit.

## Hypothesis Verification (MAPPING PHASE — no code changes)

### Current Hypothesis
planner.py:577-584 excludes nothing from the failed-step fingerprint comparison. Because the failed_step in a verification-failure replan IS the final-answer LLM step (fingerprint "llm||{}"), and EVERY valid PlanSchema's final step has that same fingerprint, ANY structurally valid replan is rejected with "Replanned plan repeats the failed execution." The neighboring duplicate check (:556-565) and LoopDetector.check_plan (:189) both deliberately exclude final-answer steps; :577 is the lone outlier.

### Evidence For
- Source: _fingerprint_step normalizes final-answer steps to the constant "llm||{}"; PlanSchema structurally mandates it.
- Source: orchestrator.py:656 routes final-answer steps to _verify; :1146 passes that same `step` as failed_step to _replan.
- Source: duplicate check at :556 excludes final-answer steps; LoopDetector.check_plan at :189 skips them; :577 does not.
- Runtime: main_run_2.txt and six prior session logs all end with 'Replanned plan repeats the failed execution.' after a verification failure on a task whose tool step SUCCEEDED (output 4).
- In this run the LLM fallback replan path timed out (ReadTimeout), so the rejection came from the deterministic fallback plan (_tool_and_final_plan) — i.e., even a well-formed LLM replan would hit the same constant-fingerprint comparison.

### Evidence Against
- None found in source. Strictness for TOOL steps is intentional and must be preserved; the issue is only the constant fingerprint of mandatory final-answer steps.

### Still Unknown
1. WHY verification of the correct answer "4" failed (deterministic INSUFFICIENT_EVIDENCE vs semantic LLM result rejected by _validate_llm_result). verification_history stores the reason but main's report does not print it.
2. Whether, once the fingerprint check passes, the replanned plan can execute: LoopDetector._history persists across replans; a replanned python_interpreter step with identical arguments would EXACT-match history -> ExecutionLoopDetected -> FAIL (:488-524). No reset/salvage exists in _replan.
3. Whether the replan LLM ReadTimeout (120s) is a recurring blocker for qwen2.5:3b with the large replan prompt.

### Required Probes (not yet run)
- Probe A (pure python, no LLM): call Planner._validate_generated_plan directly with failed_step = final LLM step and a structurally valid replacement plan -> confirm the deterministic PlannerRecoveryRequired.
- Probe B (real Ollama): rebuild the exact evidence from the failed run (ToolResultRecord python_interpreter, result "4", succeeded=True) and call deterministic_verification + VerifierAgent.verify -> capture the actual verification status/reason.

## Session Continuation — 2026-09-15 (audit + verified runtime)

### Entry state
- `git status`: `M planner/planner.py`, `M tools/python.py` (+ untracked logs).
- Prior tree held Fix 1 (GOOD, kept) AND an INVALID python change
  (`json.dumps({"result": result})`). INVALIDATED: fixed contract is
  `result = 2 + 2` -> `"4"` (`test_registered_tool_execution_is_async`).
  Reverted `tools/python.py` to `str(result)` (+ removed `import json`).

### Phase 1 audit (actual source)
- A. `ToolResultRecord` (`core/evidence.py`): step_id, tool_name,
  arguments, result, succeeded, error, artifacts, evidence_type, source,
  run_id, attempt_id, plan_version, timestamp. Provenance PRESENT. GOOD.
- B. `AgentExecution._build_tool_evidence` (`:858-923`): successful TOOL
  steps become records with `arguments=request.arguments`. GOOD.
- C. `verification_evidence()` (`orchestration/models.py:459-483`):
  extends from successful execution history. Real run passes
  `ToolResultRecord(python_interpreter, {code...}, "4", True)`. GOOD.
- D. `VerifierAgent` (`agents/verifier.py`): numeric branch returned
  INSUFFICIENT_EVIDENCE ("appears but is not explicitly established")
  for bare `"4"` because it read only `result` text, never
  `arguments.code`. CAPABILITY GAP at evidence boundary. Adversarial
  guard `test_incidental_number_never_becomes_verified[42]` forbids a
  bare-number shortcut.
- E. Tests: adversarial suite (78), verifier_strong (30),
  registry string contract, planner suites. Inspected before changes.

### Hypothesis -> CONFIRMED
- HYPOTHESIS: verifier ignores `arguments.code` provenance, so
  authoritative `"4"` looks identical to incidental `"42"`.
- Probe: bare `"4"` -> INSUFFICIENT; `"4"`+code -> VERIFIED;
  bare `"42"` -> INSUFFICIENT (guard holds); mismatch -> INVALID.
- CONFIRMED. Fix placed at verifier/evidence boundary.

### Fix 2 — provenance-aware numeric verification
- File: `src/gaia_agent/agents/verifier.py` only.
- Added `_get_arguments`, `_python_code_declares_result_variable`
  (regex `result\s*=` on executed code), `_is_authoritative_computation`.
- Numeric branch: bare single-number result + provenance showing the
  executed code declared `result` -> authoritative. Match -> VERIFIED,
  mismatch -> INVALID, multiple distinct -> CONFLICTING_EVIDENCE.
  Runs only when no explicit labelled final exists; text/incidental
  paths untouched. Bare numbers WITHOUT provenance stay INSUFFICIENT.
- Preserves: python `"4"` contract, adversarial guard (78/78 pass),
  no Planner/LoopDetector/test changes.
- INVALIDATED history: python tool returning `{"result": 4}`.

### Validation 2026-09-15
- Probes: A INSUFFICIENT / B VERIFIED / C INSUFFICIENT / D INVALID /
  E VERIFIED (labelled text).
- `tests/tools/test_registry.py`: 50 passed, 2 skipped.
- `tests/agents`: 105 passed, 3 failed in test_verifier_strong.py —
  BASELINE-PROVEN pre-existing (same 3 fail with fix stashed).
- `tests/planner`: 108 passed, 6 failed — pre-existing (Fix 1 note).
- REAL: `python -m gaia_agent.main` -> COMPLETED, answer '4',
  VERIFIED True, 1 attempt, 0 replans. LoopDetector untouched.
- Next: 25*17+43, reverse "architecture", web/vision/file tasks.

## Fixes (continued)

### Fix 1 — Replan fingerprint rejects the mandatory final-answer step
- File: src/gaia_agent/planner/planner.py
- Class/Method: Planner._validate_generated_plan (failed-step repetition check, ~:577)
- Exact root cause: the check compared EVERY step of the replacement plan against the failed step's fingerprint. After a verification failure the failed_step IS the final-answer LLM step, whose fingerprint is the constant "llm||{}" (PlanSchema forces tool_name=None, arguments={} for final steps). Therefore every structurally valid replan was rejected with PlannerRecoveryRequired("Replanned plan repeats the failed execution.") — making post-verification-failure recovery impossible. Matches main_run_2.txt + 6 prior session logs.
- Minimal change: added `if not step.is_final_answer` to the comparison generator — identical exclusion convention already used by the duplicate-fingerprint check (:556-560) and LoopDetector.check_plan (:189).
- Why architecture-preserving: no contract redesign; strict duplicate detection for executable/tool steps fully preserved (proved by Probe A control); only the mandatory, contract-defined final-answer step is exempt.
- Test: Probe A (no LLM, real Planner): failed_fp="llm||{}"; replacement plan ACCEPTED; control plan with duplicate tool step correctly REJECTED ("Plan contains duplicate tool executions.").
- Focused tests: pytest tests/planner -q -> 6 failed, 108 passed. BASELINE CHECK: same 6 failures with the fix stashed (git stash) -> pre-existing, unrelated to Fix 1. See Remaining Problems.
- Runtime result: pending main re-run.

## Remaining Issues

- 6 pre-existing planner test failures (baseline-verified via git stash): test_planner_consumes_final_context_and_preserves_attachment_evidence, test_planner_accepts_valid_llm_plan, test_planner_recovers_from_plan_missing_final_answer, test_planner_rejects_unknown_tool_and_uses_fallback, test_planner_preserves_valid_tool_arguments, test_planner_recovery_produces_different_valid_strategy — to be triaged (production bug vs stale expectation) after main works.
- VISION_MODEL `gemma3` is not available in local Ollama (only qwen2.5:3b, nomic-embed-text). To be confirmed during vision task phase.

## Modified Files

- src/gaia_agent/planner/planner.py — Fix 1 (one-line exclusion in failed-step fingerprint check).
- src/gaia_agent/analysis.md — this document.
- Probe script: C:\Users\user\AppData\Local\Temp\gaia_probe_a.py (temp dir, not in repo).

## Final Status

(in progress)

---

# FINAL READINESS AUDIT — 2026-09-16 (session 3)

Entry state: `git status` = branch main @ ff7ed26 (in sync with origin/main);
modified (uncommitted) `agents/verifier.py` (+87) and `planner/planner.py`
(+118/-22); untracked analysis.md + historical run logs. `git diff` is the
already-documented Fix 1/2/3/4 set (fingerprint exclusion, provenance-aware
numeric verification, canonical+full-expression arithmetic detector,
deterministic quoted-literal text transform). No new production edit made yet
in this session — this session begins with evidence gathering only.

## Current truth (re-established, not inherited)

- Ollama live `/api/tags` → HTTP 200, models = `nomic-embed-text:latest`,
  `qwen2.5:3b`. `gemma3` ABSENT → **VISION CAPABILITY GAP reconfirmed now**.
- venv Python 3.12.0; `gaia_agent` imported from `src/` (editable).
  `datasets`=False (so `run_evaluation.py` loader is STILL untested),
  `pypdf`=False, `pandas`=False;
  PIL/numpy/openpyxl/faster_whisper/ddgs/bs4/markdownify/requests=True.
- Tool registry exposes exactly 7 tools: `file_reader`, `transcribe_audio`,
  `analyze_image`, `analyze_excel`, `python_interpreter`, `web_search`,
  `visit_webpage`. (No `pandas`/`pypdf` → PDF/text-extraction depth is limited.)
- Evaluation package `gaia_agent/evaluation/{gaia,evaluator,submission}.py`
  are EMPTY (0 bytes) → no vendored HF Unit-4 scoring contract. UNPROVEN.
- `python -m gaia_agent.main` (default "Calculate 2 + 2.") →
  COMPLETED, final_answer='4', ready=True, **verified=True**, replan=0,
  retry=0, iteration=3, verification_attempts=1,
  IDLE→PLANNING→EXECUTING→VERIFYING→COMPLETED. GOOD (unchanged).

## Baseline proof for the 9 failing contract tests

- Question: are the 3 `test_verifier_strong` + 6 planner failures caused by the
  uncommitted fixes?
- Probe: copied both changed files aside → `git checkout -- ` both →
  `pytest tests/agents/test_verifier_strong.py tests/planner -q` → **9 failed,
  135 passed, identical 9 test ids** → restored both files
  (`git diff --stat` again 183 insertions / 22 deletions).
- Classification: **TEST ISSUE (stale expectations)**, not a production
  regression. See the §8 triage below for per-failure reasoning.

## Task A — multi-term arithmetic (`Calculate 25 * 17 + 43`), expect 468

Runtime: **FAILED** (no answer), elapsed 165.93s.

| Step | Observed |
|---|---|
| planner LLM call | LLMOutputError "Ollama structured output failed schema validation" after 32.41s → deterministic fallback plan used |
| plan step0 | TOOL python_interpreter `{"code": "result = 25 * 17 + 43\n"}` — full expression ✔ (Fix 3 GOOD) |
| exec #1 | python → `468` ✔ |
| exec #2 | final LLM step → `"458"` ✘ (wrong literal) |
| verification | `INVALID` / "The authoritative computation result contradicts the candidate.", raw_data = [ToolResultRecord python_interpreter result=468, arguments={"code": "result = 25 * 17 + 43\n"}] ✔ provenance present (Fix 2 GOOD) |
| replan | LLM replan call prompt_chars=10608 → httpx.ReadTimeout after **122.66s** (>120s client timeout) → deterministic alternative `_tool_and_final_plan` reproduced the IDENTICAL tool+final plan |
| install | accepted (Fix 1 GOOD — no more "Replanned plan repeats the failed execution") |
| re-execute | LoopDetector EXACT → "The exact execution step was repeated." → FAILED, fatal_error=True |

Classifications:
- Fix 1/2/3 = **GOOD** (each proven at runtime by the behaviour above).
- final answer `458`: **MODEL LIMITATION** (qwen2.5:3b).
- 10.6k-char replan prompt exceeding 120s: **MODEL LIMITATION + timeout config**.
- Recovery after a verification failure: **BUG / DESIGN RISK** (evidence below).

### LoopDetector constant-signature defect (evidence)

- `LoopDetector.signature()` (reliability/loop_detector.py:268-272) builds the
  EXACT signature from `{step_type, tool, arguments}`. PlanSchema forces every
  final-answer step to be LLM, tool_name=None, arguments={} → the EXACT
  signature of *any* final-answer step is the constant
  `{"arguments": {}, "step_type": "llm", "tool": ""}`.
- `Orchestrator._execute` (orchestrator.py:631) calls `loop_detector.record(step)`
  for EVERY successful step, including the final-answer step, and
  `check()` (:488) runs BEFORE `_execute`'s final-answer routing (:656).
- Therefore the first successful answer is recorded, and the intended recovery
  (re-synthesise the answer from the already-collected evidence) is ALWAYS an
  EXACT loop match → `LOOP_DETECTED` (retryable=False, recoverable=False) → FAIL.
- The same convention already exempts final-answer steps elsewhere:
  `LoopDetector.check_plan` skips `step.is_final_answer` (:189), Planner's
  duplicate check (:556) and the Fix-1 failed-step check (:583) do the same.
  The cross-plan `check`/`record` pair is the lone outlier.
- Task A also showed the deterministic replan fallback re-proposing the
  already-succeeded TOOL step (`get_alternative_strategy` → `_tool_and_final_plan`)
  before the `_has_successful_evidence` → `_llm_only_plan` branch (planner.py:
  455-497), i.e. the "re-answer" recovery is unreachable even before LoopDetector.

Remaining uncertainty: whether the same dead-end triggers on GAIA-style tasks
where the first answer happens to be right (then no verification failure occurs
and the path is never exercised). Next step: batch tasks B–K.

## Task B — deterministic text transform (`Reverse the string "architecture"`)

Runtime: **FAILED**, elapsed 211.16s.

| Step | Observed |
|---|---|
| plan step0 | TOOL python_interpreter `{"code": "result = \"architecture\"[::-1]\n"}` — deterministic quoted-literal text transform ✔ (**Fix 4 GOOD**, no web_search selected) |
| exec #1 | python → `erutcetihcra` ✔ (the correct reverse) |
| exec #2 | final LLM step → `rircihcau` ✘ (model "reversed" it mentally and got it wrong) |
| verification | `INSUFFICIENT_EVIDENCE` after a 52.17s semantic LLM call; the reasoning text even claims the reverse of 'architecture' is 'racitircuha' |
| replan | LLM replan call SUCCEEDED this time (10812 chars, 118.33s) and returned the SAME tool step but with single quotes → exact signature differs, structural signature identical |
| re-execute | LoopDetector **STRUCTURAL** → "The same execution structure and strategy were repeated." → FAILED |

Classification: two facts worth separating —
- Planner did NOT use web search for a deterministic transform, and did produce
  the correct deterministic code: **GOOD**.
- The answer was wrong because the model answered without the tool output, and
  the recovery was blocked: **BUG/DESIGN RISK** + **MODEL LIMITATION**.

## Task C — multi-number reasoning (warehouse bottles, expect 88)

Runtime: **FAILED**, elapsed 342.48s.

| Step | Observed |
|---|---|
| plan | **LLM-only** — a single final-answer step, no tool, task_type=arithmetic |
| exec #1 | final LLM step → a verbose chain-of-thought ending "…is 88." (88 happens to be correct, but nothing establishes it) |
| verification | `INSUFFICIENT_EVIDENCE` / "No successful current independent evidence is available for verification." — **GOOD**: strict, no hallucinated verification, no incidental-number acceptance |
| replan | returned the correct recovery: an LLM-only plan (re-answer) |
| re-execute | LoopDetector **EXACT** → "The exact execution step was repeated." → FAILED |

**This is direct runtime proof of the constant-signature defect**: the replanned
plan's ONLY step is the mandatory final-answer LLM step, whose EXACT signature is
the constant `{"arguments": {}, "step_type": "llm", "tool": ""}`; because the
previous identical step was `record()`-ed at orchestrator.py:631, the legitimate
re-answer is detected as a repeated execution and the run is killed
(retryable=False, recoverable=False). No answer is ever retried — for any task —
once verification fails.

Also observed: a multi-term arithmetic *word problem* is planned as LLM-only
(no computation, therefore no evidence), so it can never be verified. That is a
planner-strategy CAPABILITY GAP, distinct from the recovery defect.

## Context propagation defect: tool results never reach the final-answer step

- Hypothesis: the final-answer LLM step answers WITHOUT the tool output, so its
  literal is a model guess (Tasks A: 458 vs 468; B: `rircihcau` vs `erutcetihcra`).
- Evidence inspected:
  - `Orchestrator._build_context` (orchestrator.py:228-250) constructs
    `ContextRequest(user_request, attachments, plan, current_step,
    completed_steps, iteration)` and **omits `tool_name`, `blocked`,
    `tool_result`, `tool_error`, `current_action`, `step_type`**.
  - `context/sources/runtime.py` builds `RuntimeContext(... tool_result=request.tool_result ...)`
    and `context/sources/history.py` uses `current_action`/`step_type`, so those
    fields are already consumed by the context layer — they are simply never filled.
  - `context/request_builder.py::ContextRequestBuilder.from_state(state)` is the
    canonical builder and DOES populate all of them (`tool_result=state.tool_result`,
    `tool_name=state.tool_name`, `tool_error=state.tool_error`, …). It is used by
    tests (`tests/context/test_attachment_contract.py`,
    `tests/integration/test_agent_cross_layer.py`,
    `tests/reliability/core/test_agent_state_contract_integrity.py`) but **not**
    by the Orchestrator, which re-implements a partial copy inline.
  - `git log -S "tool_result=state.tool_result" -- orchestrator.py` → no commit
    ever populated it there → the inline copy was never wired (not a regression).
  - `LLMExecutor._format_context` stringifies the `FinalContext`, so the final
    prompt therefore contains a `RuntimeContext(..., tool_result=None, tool_error=None)`
    and a `HistoryContext(plan=[...], completed_steps=[0], current_action=None, step_type=None)`
    — i.e. the plan shape but never the produced value.
- Classification: **BUG** (evidence never reaches the answer synthesis).
  Consequence for GAIA: any task whose answer exists only in tool output
  (web page, file, spreadsheet, image, audio) cannot be answered from evidence —
  the model must recall/guess it.
- Decision: fix at the smallest responsible layer — populate the six already-
  supported fields in `_build_context`, mirroring `ContextRequestBuilder.from_state`.
  (Confirmed by prompt capture after the fix; see Validation section.)

## Pre-fix batch progress (evidence log)

- A: FAILED 165.93s | B: FAILED 211.16s | C: FAILED 342.48s
- LLM call latency per task: 30–130s per call; `LLMOutputError: Ollama structured
  output failed schema validation` occurred on 1/3 calls (A), 1/4 (B), 3/4 (C)
  → qwen2.5:3b frequently cannot satisfy the PlanSchema JSON schema.
- Task B's replan call succeeded at 118.33s — only 1.7s inside the 120s client
  timeout: latency is on the edge of the timeout, not comfortably inside it.

## Pre-fix results for the remaining representative tasks

All were run through the real composition root (`create_agent`) with the
instrumented harness, one fresh agent per task. **Every pre-fix task failed**
(A, B, C, D, E, E2, F, G, H, I, J, K).

| Task | Plan | Tool result | Final-step output | Verification | Recovery outcome |
|---|---|---|---|---|---|
| A 25*17+43 | python | 468 ✔ | "458" ✘ | INVALID (authoritative) | EXACT loop → FAILED |
| B reverse | python | erutcetihcra ✔ | rircihcau ✘ | INSUFFICIENT_EVIDENCE | STRUCTURAL loop → FAILED |
| C bottles | LLM-only | — | "…= 88" (unproven) | INSUFFICIENT_EVIDENCE ✔ | EXACT loop (final-only plan) → FAILED |
| D 17*23 | web_search (classifier said factual_search) | irrelevant results | "491" ✘ | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |
| E quetzal | web_search ✔ query | Wikipedia result ✔ | "Guatemala City" (ungrounded) | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |
| E2 2019 Nobel | web_search ✔ query | Wikipedia result ✔ | "Frances H. Arnold" (ungrounded) | INSUFFICIENT_EVIDENCE ✔ | STRUCTURAL loop → FAILED |
| F missing file | web_search (no file tool) | results | "file does not exist" | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |
| G 10/0 | LLM-only | — | refused/explained | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |
| H Mozilla CEO | web_search ✔ query | Wikipedia result ✔ | "…I need to perform a web search" | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |
| I attached CSV | **web_search** (not file tool) | irrelevant results | "context does not include … the result of the tool execution" | verifier **ReadTimeout crash** | FAILED (ReadTimeout) |
| J image | **web_search** (not analyze_image) | irrelevant results | "no attached image … in the context" | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |
| K attached CSV | **web_search** (not file tool) | irrelevant results | "context does not include the actual content of the 'sales.csv' file" | INSUFFICIENT_EVIDENCE | EXACT loop → FAILED |

What the pre-fix evidence proves:

- **GOOD — tool layer**: `python_interpreter` (A, B) and `web_search` (E, E2, H)
  executed and produced real results; evidence records carried `tool_name`,
  `arguments`, `result`, `succeeded`, `evidence_type`, `source`.
- **GOOD — verification strictness**: the verifier never rubber-stamped an
  ungrounded answer (INVALID for A; INSUFFICIENT_EVIDENCE for B, C, E, E2, H,
  K, J). Incidental/intermediate numbers were not accepted (Task C).
- **BUG — evidence never reaches the answer step**: the model itself reported
  this in Tasks I, K, J ("the provided context does not include … the result of
  the tool execution"). Root cause and fix below.
- **BUG — recovery is impossible after any verification failure**: EXACT
  (constant final-step signature) or STRUCTURAL (same tool/strategy family)
  loop detection blocked every replacement plan; 0 retries, 1 replan, then FAIL.
- **BUG/CAPABILITY GAP — local artifacts never reach the planner**:
  `available_files` is a Planner constructor argument that main.py never sets,
  so `StrategySelector`'s LOCAL_FILE/IMAGE branches can never select
  `file_reader`/`analyze_image`, and the deterministic/emergency planners fall
  back to `web_search` (Tasks I, J, K, F). Attachment tasks are unanswerable
  regardless of model quality.
- **MODEL LIMITATION**: qwen2.5:3b frequently cannot satisfy the `PlanSchema`
  JSON schema (1/3 to 3/4 of planner calls: "Ollama structured output failed
  schema validation"), invents wrong literals (458, 491, rircihcau,
  Frances H. Arnold) and answers from memory instead of the evidence.
- **ENVIRONMENT / timeout configuration**: planner/replan/verifier prompts of
  7k–11k chars repeatedly exceed the client's 120s timeout (A, E, F, H, I, K;
  and Task I's verifier crash). Task B's replan landed at 118.33s.
- **PERFORMANCE**: elapsed per question 165.93s (A), 211.16s (B), 320.30s (G),
  342.48s (C), 343.59s (E), 344.06s (I), 377.61s (D), 388.10s (H), 403.58s (K),
  420.95s (F), 421.16s (J). GAIA validation (165 questions) at this rate is many
  hours of wall-clock — not fatal, but it bounds what can be evaluated.
- **UNPROVEN (probe design gap)**: Tasks F and G were intended to exercise
  `FailureClassifier → ReliabilityEngine → retry/recovery`, but in both the
  *tool* did not fail (F: web_search "succeeded"; G: no tool in the plan), so
  the failure path was never entered. The reliability path is instead evidenced
  by the existing orchestrator integration tests (see Test Evidence).
- **UNPROVEN**: `CONFLICTING_EVIDENCE` was not produced end-to-end (Task D as
  written returned one web result and an ungrounded number). It needs a
  controlled verifier-level probe with two disagreeing authoritative
  computations.

## Fixes applied in this session (smallest correct layer, no redesign)

### Fix 5 — execution projections were never put into the context request
- File: `core/orchestration/orchestrator.py`, `Orchestrator._build_context`.
- Root cause: the inline `ContextRequest(...)` omitted `tool_name`, `blocked`,
  `tool_result`, `tool_error`, `current_action`, `step_type`. `RuntimeSource`
  and `HistorySource` already consume those fields, and
  `context/request_builder.py::ContextRequestBuilder.from_state()` already
  populates them (used by tests); the orchestrator's inline copy was never
  wired (`git log -S "tool_result=state.tool_result"` shows no commit ever
  populated it there → never-wired, not a regression).
- Change: populate the six fields from `AgentState`.
- Effect: a running step now sees the plan *and* the value the previous tool
  actually produced, so the final-answer step can report evidence instead of
  guessing it.

### Fix 6 — a final-answer step is not a repeatable execution (LoopDetector)
- File: `reliability/loop_detector.py`, `LoopDetector.check` and
  `LoopDetector.record`.
- Root cause: PlanSchema forces final-answer steps to `LLM/None/{}`, so their
  EXACT signature is the constant
  `{"arguments": {}, "step_type": "llm", "tool": ""}`. `Orchestrator._execute`
  recorded it after the first successful answer, so every later answer attempt
  was an EXACT loop and the run died (retryable=False, recoverable=False).
- Change: return `LoopDetection(detected=False)` and skip recording when
  `step.is_final_answer` — the same convention `check_plan`
  (loop_detector.py:189), the Planner duplicate check (:556) and Planner Fix 1
  (:583) already use.
- Why loop protection is not weakened: a final-answer step performs no tool
  action and produces no evidence, and the number of answer attempts stays
  bounded by the Orchestrator's `max_verification_attempts=2` /
  `max_replans=3` / `max_iterations=20` budgets. Every action-step loop
  (exact/structural/semantic, tool and arguments) is still detected exactly as
  before — Task A/B style re-execution of a *tool* step is still blocked.

### Fix 7 — the run's attachments never reached the Planner
- File: `planner/planner.py`: `__init__` (+`_configured_files`), `create_plan`,
  `replan`, new `_sync_available_files(context)`.
- Root cause: `available_files` is a constructor argument; `main.py` builds the
  Planner without it (main.py:262-267), so it is always empty.
  `StrategySelector` requires `context.available_files` for FILE_READING
  (strategy_selector.py:137) and `_select_file()` reads `self.available_files`,
  so file/image strategies and the deterministic file fallback could never fire.
- Change: merge this run's `FinalContext` attachment paths/filenames into
  `available_files` at the start of `create_plan`/`replan` (nothing invented;
  only real attachments plus caller-configured files).
- Effect: LOCAL_FILE/IMAGE tasks can now select `analyze_excel` / `file_reader`
  / `analyze_image`, and `_classify` (which also passes `available_files`) sees
  the real files.

## Regression evidence for the fixes

- Command:
  `pytest tests/reliability tests/integration tests/planner tests/agents tests/context tests/tools/test_registry.py -q --continue-on-collection-errors`
- **Pure HEAD** (`git stash push` of all uncommitted work, then restored):
  `57 failed, 431 passed, 2 skipped, 2 errors`
- **This session's tree** (Fix 1-4 + Fix 6 + Fix 7):
  `57 failed, 431 passed, 2 skipped, 2 errors` → **no regression introduced**.
- The 2 collection errors are stale imports of `MAX_VERIFICATION_ATTEMPTS`
  from `gaia_agent.core.orchestration.orchestrator` (it is now an
  `OrchestratorConfig` field) in
  `tests/reliability/core/test_orchestrator_verification.py` and
  `..._verification_budget.py` → TEST ISSUE, present at HEAD too.
- The 57 failures cluster in `tests/reliability/core/*` and `tests/agents/*`
  and reference removed/renamed APIs (`Orchestrator._same_execution`,
  `deterministic_verification` as a 2-tuple, `evidence_supports_candidate`,
  `PlanSchema` vs `PlanningResult` returns, older reason strings) → TEST ISSUE
  (stale expectations), identical at HEAD.

## §8 triage of the previously documented 9 failures

| Test | Verdict |
|---|---|
| verifier_strong::test_llm_can_resolve_conflicting_evidence | TEST ISSUE — asserts `len(llm.calls)==1`; the deterministic labelled-result branch now decides first (strictly stronger than asking the model). Production OK. |
| verifier_strong::test_deterministic_reason_is_preserved | TEST ISSUE — expected reason string predates the current wording ("labelled final result in current strong evidence"). |
| verifier_strong::test_verifier_strong_end_to_end_contract | TEST ISSUE — same `llm.calls==1` assumption in its conflict case. |
| planner_final_context_contract::test_planner_consumes_final_context_and_preserves_attachment_evidence | TEST ISSUE — asserts `isinstance(plan, PlanSchema)`; `create_plan` returns `PlanningResult`. The rest of that file (context items reaching the prompt) passes, so attachment/context plumbing works. |
| planner_runtime_contract::test_planner_accepts_valid_llm_plan | TEST ISSUE — `PlanSchema` vs `PlanningResult` return-type staleness. |
| planner_runtime_contract::test_planner_recovers_from_plan_missing_final_answer | TEST ISSUE — same return-type staleness. |
| planner_runtime_contract::test_planner_rejects_unknown_tool_and_uses_fallback | TEST ISSUE — the fallback DOES occur (log: "Unknown or unavailable tool" then fallback); only the returned-type assertion is stale. |
| planner_runtime_contract::test_planner_preserves_valid_tool_arguments | TEST ISSUE — same return-type staleness. |
| planner_runtime_contract::test_planner_recovery_produces_different_valid_strategy | TEST ISSUE — `replan` returns `PlanningResult(plan=…)`, so `isinstance(result, PlanSchema)` fails while the recovery plan itself is correct. |

Conclusion: none of the 9 (nor the 57 overall) represent a production defect
affecting the current runtime contract; they assert obsolete return types,
reason strings or removed private helpers. No production change was made to
satisfy them and no test was modified.
