"""Focused Phase-2 reproduction of the Q1 planner structured-output failure.

Captures the RAW model output (LLMOutputError.raw_content) on schema validation
failure, and traces exactly which recovery/fallback branch create_plan takes.
Does NOT touch the official evaluation submission path.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from gaia_agent.main import create_agent
from gaia_agent.reliability.exception import LLMOutputError

Q1 = (
    "How many studio albums were published by Mercedes Sosa between "
    "2000 and 2009 (included)? You can use the latest 2022 version of "
    "english wikipedia."
)


async def main() -> None:
    print("Creating agent ...", flush=True)
    agent = await create_agent()
    planner = agent.orchestrator.planner
    client = planner.client

    calls: list[dict] = []
    orig_generate = client.generate

    async def wrapped(messages, **kwargs):
        t0 = time.time()
        schema = kwargs.get("output_schema")
        schema_name = schema.__name__ if schema else None
        try:
            result = await orig_generate(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            dt = round(time.time() - t0, 2)
            raw = getattr(exc, "raw_content", None)
            calls.append(
                {
                    "schema": schema_name,
                    "seconds": dt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_content_len": len(raw) if raw else None,
                    "raw_content": (raw[:3000] if raw else None),
                }
            )
            print(f"[CALL] schema={schema_name} FAILED in {dt}s: {type(exc).__name__}", flush=True)
            if raw:
                print("[RAW MODEL OUTPUT (truncated 3000):]", flush=True)
                print(raw[:3000], flush=True)
            raise
        dt = round(time.time() - t0, 2)
        snippet = str(result)[:500]
        calls.append(
            {
                "schema": schema_name,
                "seconds": dt,
                "error": None,
                "response_chars": len(str(result)),
                "response_snippet": snippet,
            }
        )
        print(f"[CALL] schema={schema_name} OK in {dt}s chars={len(str(result))}", flush=True)
        print(f"[RESPONSE] {snippet}", flush=True)
        return result

    client.generate = wrapped

    # ---- Focused: create_plan only ----
    print("=== create_plan(Q1) ===", flush=True)
    t0 = time.time()
    try:
        res = await planner.create_plan(Q1, context=None)
        dt = round(time.time() - t0, 2)
        plan = res.plan
        print(
            f"create_plan RETURNED in {dt}s. steps="
            f"{[(s.step_type.value, s.tool_name, s.is_final_answer) for s in plan.steps]}",
            flush=True,
        )
        print(f"create_plan plan = {plan.model_dump_json(indent=2)[:1500]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        dt = round(time.time() - t0, 2)
        print(f"create_plan RAISED in {dt}s: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    print("\n=== ALL LLM CALLS ===", flush=True)
    for c in calls:
        print(json.dumps(c, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
