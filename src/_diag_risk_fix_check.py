"""Post-fix check: the exact blocked RiskContext must now be deterministic."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.core.policies.approval import ApprovalPolicy, ApprovalState
from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.models import RiskContext
from gaia_agent.core.risk.rules import RiskRules

ACTION = "Synthesize the final answer using only the evidence obtained."


class SentinelAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, context):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("LLM analyzer must not be consulted for a tool-less step")


async def main() -> None:
    analyzer = SentinelAnalyzer()
    assessor = RiskAssessor(rules=RiskRules(), analyzer=analyzer)

    assessment = await assessor.assess(RiskContext(action=ACTION))
    decision = ApprovalPolicy().evaluate(
        ApprovalState(
            action_name=ACTION,
            tool_name=None,
            risk_assessment=assessment,
        )
    )

    print("level:", assessment.level.value)
    print("factors:", [f.value for f in assessment.factors])
    print("analyzer_calls:", analyzer.calls)
    print("approval_required:", decision.approval_required)
    print("reason:", decision.reason)


if __name__ == "__main__":
    asyncio.run(main())
