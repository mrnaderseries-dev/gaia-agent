"""Probe: is the approval-block on the final-answer LLM step deterministic?

Read-only diagnostic.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.core.risk.analyzer import RiskAnalyzer
from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.models import RiskContext
from gaia_agent.core.risk.rules import RiskRules
from gaia_agent.core.policies.approval import ApprovalPolicy, ApprovalState
from gaia_agent.config import settings
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.llm.service import LLMService
from gaia_agent.tools.registry import ToolRegistry

TEXT_MODEL = LLMModel(
    provider="ollama",
    model="qwen2.5:3b",
    max_tokens=1024,
    temperature=0.2,
)

# Exact action string observed in the blocked official-Q1 run.
LLM_STEP_ACTION = "Synthesize the final answer using only the evidence obtained."


async def main() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout=settings.ollama_timeout)
    registry = ToolRegistry(
        base_dir=".",
        llm_service=LLMService(client=client, model=TEXT_MODEL),
    )

    rules = RiskRules(tool_registry=registry)
    analyzer = RiskAnalyzer(client=client, model=TEXT_MODEL)
    assessor = RiskAssessor(rules=rules, analyzer=analyzer)
    policy = ApprovalPolicy()

    for label, ctx in (
        ("final-answer LLM step (tool_name=None)", RiskContext(action=LLM_STEP_ACTION)),
        (
            "web_search tool step",
            RiskContext(
                action="Search for relevant independent evidence",
                tool_name="web_search",
                arguments={"query": "Mercedes Sosa studio albums 2000-2009"},
            ),
        ),
    ):
        print("=" * 70)
        print("CONTEXT:", label)
        print("capability:", rules.capability_for(ctx.tool_name))
        print("rule_factors:", sorted(f.value for f in rules.analyze(ctx)))
        for attempt in range(1, 3):
            started = time.time()
            assessment = await assessor.assess(ctx)
            decision = policy.evaluate(
                ApprovalState(
                    action_name=ctx.action,
                    tool_name=ctx.tool_name,
                    risk_assessment=assessment,
                )
            )
            print(
                json.dumps(
                    {
                        "attempt": attempt,
                        "elapsed_s": round(time.time() - started, 1),
                        "level": assessment.level.value,
                        "factors": [f.value for f in assessment.factors],
                        "confidence": assessment.confidence,
                        "approval_required": decision.approval_required,
                        "reason": decision.reason,
                        "message": decision.message,
                        "explanation": (assessment.explanation or "")[:300],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
