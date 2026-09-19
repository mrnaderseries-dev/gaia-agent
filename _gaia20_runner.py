"""Real GAIA Unit-4 20-question diagnostic runner.

Runs the official 20 HF Agents-Course Unit 4 GAIA questions through the REAL
composition root (gaia_agent.main.create_agent -> AgentLoop -> Orchestrator ->
Planner -> AgentExecution -> Evidence -> Verification -> Reliability) with the
real Ollama models, real tools and real (staged) attachments.

* Reuses ONE agent for all questions (mirrors src/run_evaluation.py).
* One question at a time, sequential (never concurrent).
* Captures plans, verifications, LLM call timings, executions, evidence and the
  final-answer prompt context via read-only instrumentation patches.
* Writes each result immediately to _gaia20_results.json (resumable).
* Never exposes the expected answer to the agent.

Usage:
    python _gaia20_runner.py                # all 20 (skips already-completed)
    python _gaia20_runner.py 3,4,5          # only selected 1-based indices
    python _gaia20_runner.py 1,2 --fresh    # ignore previous results
"""

from __future__ import annotations

import asyncio
import json
import os
import string
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from gaia_agent.main import create_agent  # noqa: E402
from gaia_agent.core.agent_state import AgentState  # noqa: E402
from gaia_agent.context.attachments import Attachment  # noqa: E402

import gaia_agent.agents.verifier as verifier_mod  # noqa: E402
import gaia_agent.planner.planner as planner_mod  # noqa: E402
import gaia_agent.planner.task_classifier as tc_mod  # noqa: E402
from gaia_agent.core.llm_executor import LLMExecutor  # noqa: E402
from gaia_agent.llm.provider.ollama import OllamaClient  # noqa: E402

QUESTIONS = json.loads((ROOT / "_gaia20_official.json").read_text(encoding="utf-8"))
RESULTS_PATH = ROOT / "_gaia20_results.json"
PER_QUESTION_TIMEOUT_S = float(os.environ.get("GAIA_Q_TIMEOUT", "420"))

PLANS: list[dict] = []
VERIFICATIONS: list[dict] = []
LLM_CALLS: list[dict] = []
CLASSIFICATIONS: list[dict] = []
ANSWER_CONTEXTS: list[dict] = []


def _reset() -> None:
    PLANS.clear()
    VERIFICATIONS.clear()
    LLM_CALLS.clear()
    CLASSIFICATIONS.clear()
    ANSWER_CONTEXTS.clear()


def _describe_plan(result: Any) -> dict:
    payload: dict = {"type": type(result).__name__}
    plan = getattr(result, "plan", result)
    steps = getattr(plan, "steps", None)
    if steps is None:
        payload["repr"] = str(result)[:400]
        return payload
    payload["steps"] = [
        {
            "index": getattr(s, "index", None),
            "step_type": str(getattr(s, "step_type", None)),
            "tool": getattr(s, "tool_name", None),
            "arguments": getattr(s, "arguments", None),
            "is_final_answer": getattr(s, "is_final_answer", None),
            "description": getattr(s, "description", None),
        }
        for s in steps
    ]
    return payload


def _wrap_method(cls: type, name: str, sink: list[dict]) -> None:
    if not hasattr(cls, name):
        return
    original = getattr(cls, name)

    async def wrapper(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        sink.append({"method": name, "plan": _describe_plan(result)})
        return result

    setattr(cls, name, wrapper)


_wrap_method(planner_mod.Planner, "generate_plan", PLANS)
_wrap_method(planner_mod.Planner, "create_plan", PLANS)
_wrap_method(planner_mod.Planner, "replan", PLANS)
_wrap_method(planner_mod.Planner, "replan_step", PLANS)


_orig_verify = verifier_mod.VerifierAgent.verify


async def _patched_verify(self, *args, **kwargs):
    start = time.time()
    data = args[0] if args else kwargs.get("data")
    entry: dict = {"seconds": None}
    if data is not None:
        entry["candidate_answer"] = getattr(data, "candidate_answer", None)
        entry["task_type"] = str(getattr(data, "task_type", None))
        entry["raw_data"] = [
            {
                "tool": getattr(i, "tool_name", None),
                "result": str(getattr(i, "result", None))[:300],
                "arguments": getattr(i, "arguments", None),
                "succeeded": getattr(i, "succeeded", None),
                "source": getattr(i, "source", None),
            }
            for i in (getattr(data, "raw_data", None) or [])
        ]
    try:
        result = await _orig_verify(self, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        entry["crash"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - start, 2)
        VERIFICATIONS.append(entry)
        raise
    entry["status"] = str(getattr(result, "status", None))
    entry["reason"] = str(getattr(result, "reason", None))[:600]
    entry["seconds"] = round(time.time() - start, 2)
    VERIFICATIONS.append(entry)
    return result


verifier_mod.VerifierAgent.verify = _patched_verify

_orig_classify = tc_mod.TaskClassifier.classify


def _patched_classify(self, *args, **kwargs):
    result = _orig_classify(self, *args, **kwargs)
    CLASSIFICATIONS.append(
        {
            "intent": str(getattr(result, "intent", None)),
            "needs_external_info": getattr(result, "needs_external_info", None),
            "recommended_first_tool": getattr(result, "recommended_first_tool", None),
            "forbidden_tools": list(getattr(result, "forbidden_tools", ()) or ()),
            "analysis_text": str(getattr(result, "analysis_text", ""))[:400],
        }
    )
    return result


tc_mod.TaskClassifier.classify = _patched_classify

_orig_generate = OllamaClient.generate


async def _patched_generate(self, *args, **kwargs):
    start = time.time()
    messages = kwargs.get("messages")
    if messages is None and len(args) > 0:
        for candidate in args:
            if isinstance(candidate, list):
                messages = candidate
                break
    try:
        prompt_chars = sum(
            len(str(m.get("content", ""))) if isinstance(m, dict) else len(str(m))
            for m in (messages or [])
        )
    except Exception:  # noqa: BLE001
        prompt_chars = -1
    entry: dict = {
        "prompt_chars": prompt_chars,
        "output_schema": (
            kwargs["output_schema"].__name__
            if kwargs.get("output_schema") is not None
            else None
        ),
    }
    try:
        result = await _orig_generate(self, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - start, 2)
        LLM_CALLS.append(entry)
        raise
    entry["seconds"] = round(time.time() - start, 2)
    entry["response_chars"] = len(str(result))
    LLM_CALLS.append(entry)
    return result


OllamaClient.generate = _patched_generate

_orig_build_messages = LLMExecutor._build_messages


def _patched_build_messages(request):
    messages = _orig_build_messages(request)
    try:
        user = messages[1]["content"]
        ANSWER_CONTEXTS.append({"chars": len(user), "user_message": user[:20000]})
    except Exception:  # noqa: BLE001
        pass
    return messages


LLMExecutor._build_messages = staticmethod(_patched_build_messages)


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_number_str(value: str) -> str:
    for symbol in ("$", "%", ","):
        value = value.replace(symbol, "")
    return value


def _normalize_str(value: str) -> str:
    value = value.replace("\n", " ")
    for char in string.punctuation:
        value = value.replace(char, "")
    value = value.replace(" ", "")
    return value.lower()


def score_answer(model_answer: Any, ground_truth: Any) -> dict:
    """Local approximation of the official GAIA quasi-exact-match scorer."""
    model_answer = "" if model_answer is None else str(model_answer)
    ground_truth = "" if ground_truth is None else str(ground_truth)

    if _is_float(ground_truth):
        normalized = _normalize_number_str(model_answer)
        try:
            return {
                "correct": float(normalized) == float(ground_truth),
                "mode": "number",
                "model_normalized": normalized,
            }
        except ValueError:
            return {"correct": False, "mode": "number", "model_normalized": normalized}

    parts = [p.strip() for p in ground_truth.split(",")]
    if len(parts) > 1:
        model_parts = [p.strip() for p in model_answer.split(",")]
        if len(model_parts) == len(parts):
            correct = True
            for expected, got in zip(parts, model_parts):
                if _is_float(expected):
                    correct = correct and _is_float(got) and (
                        float(_normalize_number_str(got)) == float(expected)
                    )
                else:
                    correct = correct and _normalize_str(got) == _normalize_str(expected)
            return {"correct": correct, "mode": "list"}

    return {
        "correct": _normalize_str(model_answer) == _normalize_str(ground_truth),
        "mode": "string",
        "model_normalized": _normalize_str(model_answer),
        "truth_normalized": _normalize_str(ground_truth),
    }


def _safe_repr(value: Any, limit: int = 1500) -> str:
    try:
        text = str(value)
    except Exception as exc:  # noqa: BLE001
        return f"<repr failed: {exc}>"
    return text[:limit]


def _summarize_state(state: AgentState) -> dict:
    return {
        "final_answer": state.final_answer,
        "final_answer_ready": state.final_answer_ready,
        "final_answer_verified": state.final_answer_verified,
        "task_completed": state.task_completed,
        "execution_success": state.execution_success,
        "phase": str(getattr(state.phase, "value", state.phase)),
        "tool_error": _safe_repr(state.tool_error, 600) if state.tool_error else None,
        "fatal_error": state.fatal_error,
        "termination_reason": (
            str(state.termination_reason) if state.termination_reason is not None else None
        ),
        "replan_count": state.replan_count,
        "retry_count": state.retry_count,
        "iteration": state.iteration,
        "verification_attempts": state.verification_attempts,
        "transitions": [
            f"{t.from_phase.value}->{t.to_phase.value}:{t.reason.value}"
            for t in state.transition_history
        ],
        "executions": [
            {
                "step_id": getattr(e, "step_id", None),
                "tool": getattr(e, "tool_name", None),
                "success": getattr(e, "success", None),
                "output": _safe_repr(getattr(e, "output", None), 800),
                "error": _safe_repr(getattr(e, "error", None), 400),
                "blocked": getattr(e, "blocked", None),
            }
            for e in state.execution_results
        ],
        "evidence": [
            {
                "tool": ev.tool_name,
                "arguments": ev.arguments,
                "result": _safe_repr(ev.result, 800),
                "succeeded": ev.succeeded,
                "source": getattr(ev, "source", None),
            }
            for ev in state.evidence
        ],
    }


async def _run_one(agent, row: dict, index: int) -> dict:
    task_id = row["task_id"]
    question = row["question"]

    attachments = []
    if row.get("file_path"):
        attachments = [
            Attachment(
                attachment_id=f"att-{task_id[:8]}",
                filename=Path(row["file_path"]).name,
                path=row["file_path"],
            )
        ]

    state = AgentState(user_request=question, attachments=attachments)

    print()
    print("=" * 100)
    print(f"QUESTION {index}/20  task_id={task_id}  level={row.get('level')}")
    print(f"[FILE] {row.get('file_name') or '-'}  staged={bool(attachments)}")
    print("-" * 100)
    print(question)
    print("=" * 100, flush=True)

    _reset()
    started = time.time()
    error = None
    try:
        await asyncio.wait_for(agent.run(state), timeout=PER_QUESTION_TIMEOUT_S)
    except asyncio.TimeoutError:
        error = f"TIMEOUT after {PER_QUESTION_TIMEOUT_S}s (state salvaged)"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print("[EXCEPTION]", traceback.format_exc()[-2500:], flush=True)
    elapsed = round(time.time() - started, 1)

    record = {
        "index": index,
        "task_id": task_id,
        "level": row.get("level"),
        "question": question,
        "file_name": row.get("file_name"),
        "attachment_staged": bool(attachments),
        "expected": row.get("expected"),
        "elapsed_s": elapsed,
        "error": error,
        "state": _summarize_state(state),
        "classifications": list(CLASSIFICATIONS),
        "plans": list(PLANS),
        "verifications": list(VERIFICATIONS),
        "llm_calls": list(LLM_CALLS),
        "answer_context_chars": [c["chars"] for c in ANSWER_CONTEXTS],
        "answer_contexts": [c["user_message"] for c in ANSWER_CONTEXTS],
    }
    answer = state.final_answer
    record["score"] = score_answer(answer, row.get("expected"))
    record["correct"] = record["score"]["correct"]

    print(
        f"[RESULT] answer={answer!r} expected={row.get('expected')!r} "
        f"correct={record['correct']} verified={state.final_answer_verified} "
        f"termination={state.termination_reason} elapsed={elapsed}s "
        f"error={error!r}",
        flush=True,
    )
    return record


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fresh = "--fresh" in sys.argv
    selected = set()
    if args:
        for chunk in args[0].split(","):
            if chunk.strip():
                selected.add(int(chunk))

    existing: dict[str, dict] = {}
    if RESULTS_PATH.exists() and not fresh:
        for rec in json.loads(RESULTS_PATH.read_text(encoding="utf-8")):
            existing[rec["task_id"]] = rec

    print(f"[cfg] timeout/question = {PER_QUESTION_TIMEOUT_S}s")
    print(f"[cfg] selected = {sorted(selected) if selected else 'ALL 20'}")
    print("[cfg] creating agent ...", flush=True)
    agent = await create_agent()
    print("[cfg] agent ready.", flush=True)

    results = list(existing.values())

    for index, row in enumerate(QUESTIONS, start=1):
        if selected and index not in selected:
            continue
        if not selected and row["task_id"] in existing:
            print(f"[skip] Q{index} already done: {row['task_id']}", flush=True)
            continue

        record = await _run_one(agent, row, index)
        results = [r for r in results if r["task_id"] != row["task_id"]]
        results.append(record)

        RESULTS_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    verified = sum(1 for r in results if r.get("state", {}).get("final_answer_verified"))
    errors = sum(1 for r in results if r.get("error"))
    print()
    print("=" * 100)
    print(
        f"[SUMMARY] executed={total} correct(local scorer)={correct} "
        f"verified={verified} errors={errors}"
    )
    print("=" * 100, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

