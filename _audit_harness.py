from __future__ import annotations

"""Readiness-audit harness.

Runs the REAL application composition root (gaia_agent.main.create_agent)
against an arbitrary request and prints a compact JSON outcome report
including instrumentation of plans, LLM call timings and verification
results. No production code is modified.
"""

import asyncio
import json
import sys
import time
import traceback

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.main import create_agent
from gaia_agent.core.agent_state import AgentState
from gaia_agent.context.attachments import Attachment

import gaia_agent.agents.verifier as verifier_mod
import gaia_agent.planner.planner as planner_mod
from gaia_agent.llm.provider.ollama import OllamaClient

VERIFICATIONS: list[dict] = []
PLANS: list[dict] = []
LLM_CALLS: list[dict] = []


def _describe_plan(result) -> dict:
    payload: dict = {"type": type(result).__name__}
    plan = getattr(result, "plan", result)
    steps = getattr(plan, "steps", None)
    if steps is None:
        payload["repr"] = str(result)[:400]
        return payload
    payload["steps"] = [
        {
            "index": getattr(step, "index", None),
            "step_type": str(getattr(step, "step_type", None)),
            "tool": getattr(step, "tool_name", None),
            "arguments": getattr(step, "arguments", None),
            "is_final_answer": getattr(step, "is_final_answer", None),
            "description": getattr(step, "description", None),
        }
        for step in steps
    ]
    return payload


_orig_verify = verifier_mod.VerifierAgent.verify


async def _patched_verify(self, *args, **kwargs):
    start = time.time()
    entry: dict = {"seconds": None}
    data = args[0] if args else kwargs.get("data")
    if data is not None:
        entry["question"] = getattr(data, "question", None)
        entry["candidate_answer"] = getattr(data, "candidate_answer", None)
        entry["task_type"] = getattr(data, "task_type", None)
        raw = getattr(data, "raw_data", None) or []
        entry["raw_data"] = [
            {
                "type": type(item).__name__,
                "tool": getattr(item, "tool_name", None),
                "result": str(getattr(item, "result", None))[:200],
                "arguments": getattr(item, "arguments", None),
                "succeeded": getattr(item, "succeeded", None),
                "evidence_type": str(getattr(item, "evidence_type", None)),
                "source": getattr(item, "source", None),
            }
            for item in raw
        ]
    try:
        result = await _orig_verify(self, *args, **kwargs)
    except Exception as exc:
        entry["crash"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - start, 2)
        VERIFICATIONS.append(entry)
        raise
    entry["status"] = str(getattr(result, "status", None))
    entry["reason"] = getattr(result, "reason", None)
    entry["seconds"] = round(time.time() - start, 2)
    VERIFICATIONS.append(entry)
    return result


verifier_mod.VerifierAgent.verify = _patched_verify


def _wrap_planner(name: str) -> None:
    original = getattr(planner_mod.Planner, name)

    async def wrapper(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        PLANS.append({"method": name, "plan": _describe_plan(result)})
        return result

    setattr(planner_mod.Planner, name, wrapper)


for _name in ("generate_plan", "replan"):
    if hasattr(planner_mod.Planner, _name):
        _wrap_planner(_name)


_orig_generate = OllamaClient.generate


async def _patched_generate(self, *args, **kwargs):
    start = time.time()
    entry: dict = {}
    try:
        entry["prompt_chars"] = len(str(args[0] if args else kwargs))
        result = await _orig_generate(self, *args, **kwargs)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - start, 2)
        LLM_CALLS.append(entry)
        raise
    entry["seconds"] = round(time.time() - start, 2)
    LLM_CALLS.append(entry)
    return result


OllamaClient.generate = _patched_generate


def summarize(state: AgentState) -> dict:
    return {
        "final_answer": state.final_answer,
        "final_answer_ready": state.final_answer_ready,
        "final_answer_verified": state.final_answer_verified,
        "task_completed": state.task_completed,
        "phase": str(getattr(state.phase, "value", state.phase)),
        "tool_error": state.tool_error,
        "fatal_error": state.fatal_error,
        "termination_reason": (
            str(state.termination_reason)
            if state.termination_reason is not None
            else None
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
                "success": getattr(e, "success", None),
                "step_id": getattr(e, "step_id", None),
                "tool": getattr(e, "tool_name", None),
                "output": str(getattr(e, "output", None))[:400],
                "error": str(getattr(e, "error", None))[:400],
                "blocked": getattr(e, "blocked", None),
            }
            for e in state.execution_results
        ],
        "evidence": [
            {
                "tool": ev.tool_name,
                "arguments": ev.arguments,
                "result": str(ev.result)[:400],
                "succeeded": ev.succeeded,
                "evidence_type": str(getattr(ev, "evidence_type", None)),
            }
            for ev in state.evidence
        ],
        "artifacts": [str(a) for a in state.artifacts],
        "verifications": VERIFICATIONS,
        "plans": PLANS,
        "llm_calls": LLM_CALLS,
    }


def reset() -> None:
    VERIFICATIONS.clear()
    PLANS.clear()
    LLM_CALLS.clear()


async def run(task: str, attach: list[str]) -> dict:
    agent = await create_agent()
    attachments = [
        Attachment(
            attachment_id=f"att-{i}",
            filename=path.replace("\\", "/").split("/")[-1],
            path=path,
        )
        for i, path in enumerate(attach, start=1)
    ]
    state = AgentState(user_request=task, attachments=attachments)
    started = time.time()
    try:
        result = await agent.run(state)
    except Exception as exc:
        payload = {
            "task": task,
            "attachments": attach,
            "crash": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
            "elapsed_s": round(time.time() - started, 2),
            "verifications": VERIFICATIONS,
            "plans": PLANS,
            "llm_calls": LLM_CALLS,
        }
        print("AUDIT_RESULT_JSON_START")
        print(json.dumps(payload, default=str, indent=2))
        print("AUDIT_RESULT_JSON_END")
        return payload
    payload = summarize(result)
    payload["task"] = task
    payload["attachments"] = attach
    payload["elapsed_s"] = round(time.time() - started, 2)
    print("AUDIT_RESULT_JSON_START")
    print(json.dumps(payload, default=str, indent=2))
    print("AUDIT_RESULT_JSON_END")
    return payload


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "Calculate 2 + 2."
    attach_arg = sys.argv[2:] if len(sys.argv) > 2 else []
    asyncio.run(run(task_arg, attach_arg))