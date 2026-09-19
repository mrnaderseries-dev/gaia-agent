"""Direct Ollama structured-output probe (diagnostic) — captures raw schema'd content."""
import sys

sys.path.insert(0, "src")

import asyncio

from gaia_agent.agents.verifier import VerificationResult
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.planner.plan_schema import PlanSchema

PLAN_PROMPT = """Create a 2-step plan to answer: '''How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?'''
Step 1: web_search with a short keyword query. Step 2: the final answer step (step_type "llm", tool_name null, is_final_answer true).
JSON with fields: steps: list of {{step_id:int, action:str, step_type:"tool"|"llm", tool_name:str|null, arguments:object, is_final_answer:bool}}"""

VER_PROMPT = """Verify the candidate answer.
Question: How many studio albums did Mercedes Sosa publish 2000-2009?
Candidate: 2
Evidence: search snippet mentioning several albums in the period.
Respond with JSON: {{"status": "verified"|"invalid"|"insufficient_evidence"|"conflicting_evidence"|"unsupported", "reason": str}}"""


async def main():
    import gaia_agent.llm.provider.ollama as omod

    captured: list[str] = []

    original = omod.OllamaClient._parse_structured_output

    def spy(self, content, output_schema):
        captured.append(f"[{output_schema.__name__}] {content[:500]}")
        return original(self, content, output_schema)

    omod.OllamaClient._parse_structured_output = spy

    c = OllamaClient(timeout=180.0)
    m = LLMModel(
        provider="ollama", model="qwen2.5:3b", max_tokens=512, temperature=0.2
    )

    for label, schema, prompt in (
        ("PLAN", PlanSchema, PLAN_PROMPT),
        ("PLAN", PlanSchema, PLAN_PROMPT),
        ("VER", VerificationResult, VER_PROMPT),
    ):
        try:
            r = await c.generate(
                messages=[
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=m,
                output_schema=schema,
            )
            print(label, "OK ->", str(r)[:200])
        except Exception as e:  # noqa: BLE001
            print(label, "FAIL:", type(e).__name__, str(e)[:120])
        while captured:
            print("  RAW-SCHEMA'D:", captured.pop(0).replace("\n", " | "))


asyncio.run(main())
