"""Full official-Q1 reproduction through the REAL composition root.

Read-only diagnostic: patches only in-process classes to record evidence.
No production file is modified, no credentials are read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler("_diag_q1_full_run.log", encoding="utf-8", mode="w")],
)

from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_state import AgentState
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.main import create_agent

QUESTION = (
    "How many studio albums were published by Mercedes Sosa between 2000 "
    "and 2009 (included)? You can use the latest 2022 version of english "
    "wikipedia."
)

diag = logging.getLogger("diag")

llm_calls: list[dict] = []

_original_request = OllamaClient._request


async def spy_request(self, payload):  # type: ignore[no-untyped-def]
    started = time.time()
    result = await _original_request(self, payload)
    elapsed = round(time.time() - started, 2)
    message = result.get("message", {}) if isinstance(result, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    call = {
        "elapsed_s": elapsed,
        "num_predict": payload.get("options", {}).get("num_predict"),
        "temperature": payload.get("options", {}).get("temperature"),
        "has_schema": "format" in payload,
        "messages": [
            {
                "role": m.get("role"),
                "chars": len(str(m.get("content", ""))),
            }
            for m in payload.get("messages", [])
        ],
        "content_chars": len(content),
        "content_head": content[:1500],
    }
    llm_calls.append(call)
    diag.info("LLM_CALL %s", json.dumps(call, ensure_ascii=False)[:3000])
    return result


OllamaClient._request = spy_request  # type: ignore[assignment]

exec_calls: list[dict] = []

_original_execute = AgentExecution.execute


async def spy_execute(self, request):  # type: ignore[no-untyped-def]
    started = time.time()
    result = await _original_execute(self, request)
    elapsed = round(time.time() - started, 2)
    record = {
        "elapsed_s": elapsed,
        "iteration": getattr(request, "iteration", None),
        "action": getattr(request, "action", None),
        "step_type": getattr(getattr(request, "step_type", None), "value", None),
        "tool_name": getattr(request, "tool_name", None),
        "arguments": getattr(request, "arguments", None),
        "success": getattr(result, "success", None),
        "error": str(getattr(result, "error", None))[:500],
        "output_head": str(getattr(result, "output", ""))[:800],
    }
    exec_calls.append(record)
    diag.info("EXEC %s", json.dumps(record, ensure_ascii=False)[:3000])
    return result


AgentExecution.execute = spy_execute  # type: ignore[assignment]


async def main() -> None:
    agent = await create_agent()
    state = AgentState(user_request=QUESTION, attachments=[])

    started = time.time()
    error = None
    try:
        await asyncio.wait_for(agent.run(state), timeout=600.0)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.time() - started, 1)

    run = agent.run_context

    summary = {
        "elapsed_s": elapsed,
        "error": error,
        "final_answer": state.final_answer,
        "final_answer_ready": state.final_answer_ready,
        "final_answer_verified": state.final_answer_verified,
        "verification_attempts": state.verification_attempts,
        "replan_count": state.replan_count,
        "retry_count": state.retry_count,
        "iteration": state.iteration,
        "phase": str(state.phase),
        "termination_reason": state.termination_reason,
        "fatal_error": state.fatal_error,
        "tool_error": str(state.tool_error)[:500],
        "evidence_count": len(getattr(state, "evidence", []) or []),
        "run_plan_version": getattr(run, "plan_version", None),
        "run_is_verified": getattr(run, "is_verified", None),
        "llm_calls": len(llm_calls),
        "llm_total_s": round(sum(c["elapsed_s"] for c in llm_calls), 2),
        "exec_calls": len(exec_calls),
    }
    diag.info("SUMMARY %s", json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("=" * 70)
    print("EXECUTIONS")
    for record in exec_calls:
        print(json.dumps(record, ensure_ascii=False, default=str)[:1500])


if __name__ == "__main__":
    asyncio.run(main())