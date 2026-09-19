"""Focused reproduction: capture the RAW Ollama structured output for the
official GAIA Q1 planner call and trace which recovery path is taken.

Read-only diagnostic. Patches only the in-process classes; no production
file is modified. No credentials are read or printed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.config import settings
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.llm.service import LLMService
from gaia_agent.planner.planner import Planner
from gaia_agent.reliability.loop_detector import LoopDetector
from gaia_agent.tools.registry import ToolRegistry

QUESTION = (
    "How many studio albums were published by Mercedes Sosa between 2000 "
    "and 2009 (included)? You can use the latest 2022 version of english "
    "wikipedia."
)

TEXT_MODEL = LLMModel(
    provider="ollama",
    model="qwen2.5:3b",
    max_tokens=1024,
    temperature=0.2,
)

VISION_MODEL = LLMModel(
    provider="ollama",
    model="moondream",
    max_tokens=768,
    temperature=0.0,
)

raw_calls: list[dict] = []

_original_parse = OllamaClient._parse_structured_output
_original_request = OllamaClient._request


async def spy_request(self, payload):  # type: ignore[no-untyped-def]
    started = time.time()
    result = await _original_request(self, payload)
    raw_calls.append(
        {
            "kind": "request",
            "elapsed": round(time.time() - started, 2),
            "num_predict": payload.get("options", {}).get("num_predict"),
            "has_format": "format" in payload,
        }
    )
    return result


def spy_parse(self, content, output_schema):  # type: ignore[no-untyped-def]
    raw_calls.append(
        {
            "kind": "raw",
            "schema": getattr(output_schema, "__name__", str(output_schema)),
            "content": content if isinstance(content, str) else repr(content),
        }
    )
    return _original_parse(self, content, output_schema)


OllamaClient._request = spy_request  # type: ignore[assignment]
OllamaClient._parse_structured_output = spy_parse  # type: ignore[assignment]


_original_repair = Planner._repair_and_validate_plan
_original_fallback = Planner._emergency_fallback_plan

repair_result: dict = {"called": False}
fallback_result: dict = {"called": False}


def spy_repair(self, raw_content, *, analysis):  # type: ignore[no-untyped-def]
    repair_result["called"] = True
    result = _original_repair(self, raw_content, analysis=analysis)
    repair_result["returned"] = (
        "None" if result is None else f"PlanSchema(steps={len(result.steps)})"
    )
    return result


def spy_fallback(self, *, user_question, failed_step=None, analysis=None):  # type: ignore[no-untyped-def]
    fallback_result["called"] = True
    result = _original_fallback(
        self,
        user_question=user_question,
        failed_step=failed_step,
        analysis=analysis,
    )
    fallback_result["steps"] = [
        {
            "step_id": step.step_id,
            "step_type": step.step_type.value,
            "tool_name": step.tool_name,
            "arguments": step.arguments,
            "is_final_answer": step.is_final_answer,
        }
        for step in result.steps
    ]
    return result


Planner._repair_and_validate_plan = spy_repair  # type: ignore[assignment]
Planner._emergency_fallback_plan = spy_fallback  # type: ignore[assignment]


async def main() -> None:
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=settings.ollama_timeout,
    )
    text_llm_service = LLMService(client=client, model=TEXT_MODEL)
    vision_llm_service = LLMService(client=client, model=VISION_MODEL)

    registry = ToolRegistry(
        base_dir=".",
        llm_service=text_llm_service,
        vision_llm_service=vision_llm_service,
    )
    available_tools = {spec.name: spec for spec in registry.get_tool_specs()}
    print("TOOLS:", sorted(available_tools), flush=True)

    planner = Planner(
        client=client,
        model=TEXT_MODEL,
        available_tools=available_tools,
        loop_detector=LoopDetector(max_history=50),
    )

    analysis = planner._classify(QUESTION)
    print("ANALYSIS intent:", analysis.intent.value, flush=True)
    print("ANALYSIS needs_external_info:", analysis.needs_external_info, flush=True)
    print("ANALYSIS recommended_first_tool:", analysis.recommended_first_tool, flush=True)
    print("ANALYSIS forbidden_tools:", analysis.forbidden_tools, flush=True)

    started = time.time()
    try:
        result = await planner.create_plan(user_question=QUESTION, context=None)
        elapsed = round(time.time() - started, 2)
        print("PLANNING_OK elapsed:", elapsed, flush=True)
        for step in result.plan.steps:
            print(
                "  STEP",
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "is_final_answer": step.is_final_answer,
                },
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.time() - started, 2)
        print("PLANNING_FAILED elapsed:", elapsed, flush=True)
        print("  type:", type(exc).__name__, flush=True)
        print("  message:", str(exc)[:2000], flush=True)

    print("=" * 70, flush=True)
    print("RAW OLLAMA CALLS:", flush=True)
    for index, call in enumerate(raw_calls):
        print(f"--- call {index} ---", flush=True)
        print(json.dumps(call, ensure_ascii=False, indent=2)[:6000], flush=True)

    print("=" * 70, flush=True)
    print("REPAIR PATH:", repair_result, flush=True)
    print("FALLBACK PATH:", fallback_result, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
