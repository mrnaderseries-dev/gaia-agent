"""Focused diagnostic: run the agent on ONE HF question with full instrumentation.

Surfaces residual architecture failures (crashes, infinite loops, contract
violations, blocked actions) and reasoning issues without running all 20.
"""
import asyncio
import json
import sys
import traceback

import urllib.request

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.main import create_agent
from gaia_agent.core.agent_state import AgentState


def fetch_questions(limit: int = 3):
    with urllib.request.urlopen(
        "https://agents-course-unit4-scoring.hf.space/questions",
        timeout=15,
    ) as r:
        return json.loads(r.read())[:limit]


async def run_one(q: dict):
    print("\n" + "=" * 70)
    print("TASK_ID:", q.get("task_id"))
    print("QUESTION:", q.get("Question") or q.get("question"))
    print("=" * 70, flush=True)

    agent = await create_agent()

    state = AgentState(
        user_id=1,
        user_request=q.get("Question") or q.get("question") or "",
    )

    try:
        result = await agent.run(state)
    except Exception as exc:
        print("\n--- AGENT RAISED EXCEPTION ---")
        traceback.print_exc()
        print("AGENT_EXCEPTION:", type(exc).__name__, str(exc)[:300])
        return

    print("\n--- RESULT SUMMARY ---")
    print("final_answer:", repr(result.final_answer))
    print("final_answer_ready:", result.final_answer_ready)
    print("final_answer_verified:", result.final_answer_verified)
    print("execution_success:", result.execution_success)
    print("task_completed:", result.task_completed)
    print("fatal_error:", result.fatal_error)
    print("termination_reason:", result.termination_reason)
    print("iteration:", result.iteration)
    print("completed_steps:", result.completed_steps)
    print("replan_count:", result.replan_count)
    print("same_failure_count:", result.same_failure_count)
    print("tool_error:", result.tool_error)
    print("blocked:", result.blocked)
    print("evidence_records:", len(getattr(result, "evidence", [])))
    print("\n--- PLAN ---")
    for step in result.plan:
        print(
            f"  #{step.step_id} {step.step_type.value} "
            f"tool={step.tool_name} action={step.action[:60]} "
            f"final={step.is_final_answer}"
        )


async def main():
    questions = fetch_questions(3)
    print(f"Fetched {len(questions)} questions for inspection:")
    for q in questions:
        print("  -", (q.get("Question") or q.get("question"))[:90])

    # Run only the first question end-to-end.
    await run_one(questions[0])


if __name__ == "__main__":
    asyncio.run(main())
