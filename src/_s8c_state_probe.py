import json
import sys

sys.path.insert(0, ".")
from gaia_agent.config import settings  # noqa: E402

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

d = json.loads(Path(r"..\_gaia20_results.json").read_text(encoding="utf-8"))
q1 = next(r for r in d if r["index"] == 1)
print("Q1 elapsed:", q1["elapsed_s"], "error:", q1["error"])
print("QUESTION:", q1["question"][:300])
print("expected:", q1["expected"])
st = q1["state"]
print("\n== STATE ==")
for k in ("final_answer", "final_answer_ready", "final_answer_verified",
          "task_completed", "execution_success", "phase", "termination_reason",
          "replan_count", "retry_count", "iteration", "verification_attempts"):
    print(f"{k}: {st.get(k)}")
print("transitions:", st.get("transitions"))
print("\n== CLASSIFICATIONS ==")
for c in q1["classifications"]:
    print(json.dumps(c, ensure_ascii=False)[:400])
print("\n== PLANS ==")
for p in q1["plans"]:
    print(json.dumps(p, ensure_ascii=False)[:600])
print("tool_error:", st.get("tool_error"))
print("fatal_error:", st.get("fatal_error"))
print("\n== EXECUTIONS (full) ==")
for e in st["executions"]:
    print(json.dumps(e, ensure_ascii=False, indent=1)[:1500])
for e in st["executions"]:
    print(json.dumps(e, ensure_ascii=False)[:500])
print("\n== EVIDENCE ==")
for e in st["evidence"]:
    print(json.dumps(e, ensure_ascii=False)[:400])
print("\n== VERIFICATIONS ==")
for v in q1["verifications"]:
    print(json.dumps(v, ensure_ascii=False)[:500])
print("\n== LLM CALLS ==")
for c in q1["llm_calls"]:
    print(json.dumps(c, ensure_ascii=False)[:300])
print("\n== ANSWER CONTEXT CHARS ==", q1["answer_context_chars"])
