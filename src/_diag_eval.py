"""LOCAL DIAGNOSTIC EVALUATION - first 10 questions, NO submission.

Detects residual reasoning-level issues and architectural failures.
Mirrors run_evaluation.py but fixes its harness bugs:
  - API returns lowercase 'question' key (harness read 'Question' -> empty)
  - harness overwrote state.user_question with '' (key-casing bug)
  - attachments are staged via GET /files/{task_id} when available
"""
import sys
import os
import json
import time
import asyncio
import traceback
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from gaia_agent import main as user_agent
from gaia_agent.core.agent_state import AgentState

API_URL = "https://agents-course-unit4-scoring.hf.space"
MAX_QUESTIONS = 10
PER_QUESTION_TIMEOUT_S = 420
# Targeted re-run support: ONLY_INDICES="1,3,5" re-runs just those
# questions (used to retest failures with fixed planner guidance).
ONLY_INDICES = {
    int(i) for i in os.environ.get("ONLY_INDICES", "").split(",")
    if i.strip()
}
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_eval_results" + ("_retry" if ONLY_INDICES else "") + ".json",
)
STAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_files")


def stage_attachment(task_id: str, file_name: str) -> str | None:
    """Download the task attachment if the scoring API serves it."""
    if not file_name:
        return None
    try:
        r = requests.get(f"{API_URL}/files/{task_id}", timeout=30)
        if r.status_code != 200 or len(r.content) < 10:
            return None
        dest_dir = os.path.join(STAGE_DIR, task_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(file_name))
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    except Exception:
        return None


def build_state(question_text: str, attachment_path: str | None) -> AgentState:
    state = AgentState(user_id="default_user", user_request=question_text)
    # Only overwrite aliases when we actually have the text (harness bug fix).
    for attr in ("user_question", "prompt", "query", "question", "input", "task", "text"):
        if hasattr(state, attr) and question_text:
            try:
                setattr(state, attr, question_text)
            except Exception:
                pass
    if attachment_path:
        # Real artifact reference, not a guessed filename (Phase 4).
        full = (f"{question_text}\n\nThe attached file is available at the absolute path: "
                f"{attachment_path}")
        state.user_request = full
        if hasattr(state, "user_question"):
            state.user_question = full
    return state


def extract_result(state: AgentState, elapsed: float, error: str | None) -> dict:
    def g(attr, default=None):
        return getattr(state, attr, default)

    answer = g("final_answer")
    if hasattr(answer, "final_answer"):
        answer = answer.final_answer
    return {
        "answer": str(answer).strip() if answer is not None else "0",
        "elapsed_s": round(elapsed, 1),
        "termination_reason": str(g("termination_reason")),
        "final_answer_ready": bool(g("final_answer_ready", False)),
        "final_answer_verified": bool(g("final_answer_verified", False)),
        "task_completed": bool(g("task_completed", False)),
        "execution_success": bool(g("execution_success", False)),
        "blocked": bool(g("blocked", False)),
        "iterations": g("iteration", 0),
        "last_tool_error": str(g("tool_error"))[:400] if g("tool_error") else None,
        "replan_count": g("replan_count", 0),
        "error": error,
    }


async def main() -> None:
    print("Initializing Agent via create_agent()...", flush=True)
    agent_instance = await user_agent.create_agent()
    print("Agent initialized.", flush=True)

    questions = requests.get(f"{API_URL}/questions", timeout=60).json()
    subset = questions[:MAX_QUESTIONS]
    print(f"Fetched {len(questions)} questions; running first {len(subset)}.", flush=True)

    results = []
    for idx, q in enumerate(subset, 1):
        if ONLY_INDICES and idx not in ONLY_INDICES:
            continue
        task_id = q.get("task_id")
        question_text = (q.get("question") or q.get("Question") or "").strip()
        file_name = (q.get("file_name") or "").strip()
        print(f"\n{'='*70}\n[{idx}/{len(subset)}] task={task_id} file={file_name or '-'}", flush=True)
        print(f"Q: {question_text[:160]}", flush=True)

        attachment = stage_attachment(task_id, file_name)
        if file_name and not attachment:
            print(f"NOTE: attachment '{file_name}' not available from scoring API (404) "
                  f"and GAIA dataset is gated -> task may be unanswerable here.", flush=True)

        state = build_state(question_text, attachment)
        t0 = time.time()
        try:
            await asyncio.wait_for(agent_instance.run(state), timeout=PER_QUESTION_TIMEOUT_S)
            rec = extract_result(state, time.time() - t0, None)
        except asyncio.TimeoutError:
            # Salvage whatever the agent produced before the wall clock
            # hit (matches the real harness: state.final_answer still
            # exists on the bound state object).
            rec = extract_result(
                state,
                time.time() - t0,
                f"TIMEOUT after {PER_QUESTION_TIMEOUT_S}s (state salvaged)",
            )
        except Exception as e:
            traceback.print_exc()
            rec = extract_result(state, time.time() - t0, f"{type(e).__name__}: {e}")

        rec.update({"task_id": task_id, "index": idx, "question": question_text[:300],
                    "file_name": file_name, "attachment_staged": bool(attachment)})
        results.append(rec)
        print(f"ANSWER[{idx}]: {rec['answer'][:200]!r} "
              f"(verified={rec.get('final_answer_verified')} completed={rec.get('task_completed')} "
              f"term={rec.get('termination_reason')} {rec.get('elapsed_s')}s)", flush=True)

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\nLOCAL RUN SUMMARY (no submission):", flush=True)
    for r in results:
        print(f"  #{r['index']:>2} {r['task_id'][:8]} -> {r['answer'][:60]!r} "
              f"verified={r.get('final_answer_verified')} {r['error'] or ''}", flush=True)
    print(f"Results saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
