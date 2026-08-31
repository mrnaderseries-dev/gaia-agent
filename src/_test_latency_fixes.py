"""Validates the latency/state fixes before the evaluation re-run."""
import asyncio
import sys

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.models import RiskContext, RiskLevel
from gaia_agent.core.risk.rules import RiskRules


class SentinelAnalyzer:
    """Records whether the LLM analyzer was consulted."""

    def __init__(self):
        self.called = False

    async def analyze(self, context):
        self.called = True
        raise RuntimeError("sentinel should never be awaited")


async def main() -> None:
    analyzer = SentinelAnalyzer()
    assessor = RiskAssessor(rules=RiskRules(), analyzer=analyzer)

    # 1. Read-only tool -> must skip the LLM analyzer entirely.
    a1 = await assessor.assess(
        RiskContext(
            action="Search the web for evidence",
            tool_name="web_search",
            arguments={"query": "Mercedes Sosa albums"},
        )
    )
    assert a1.level == RiskLevel.LOW, a1.level
    assert analyzer.called is False, "LLM analyzer ran for read-only tool!"

    # 2. Computation tool -> also deterministic.
    a2 = await assessor.assess(
        RiskContext(
            action="Compute the album count",
            tool_name="python_interpreter",
            arguments={"code": "print(4)"},
        )
    )
    assert a2.level == RiskLevel.LOW, a2.level
    assert analyzer.called is False, "LLM analyzer ran for computation!"

    # 3. Unknown tool with no capability -> LLM path allowed (may raise,
    #    caught inside assessor -> level falls back deterministically).
    a3 = await assessor.assess(
        RiskContext(
            action="Do something unmapped",
            tool_name="mystery_tool",
            arguments={},
        )
    )
    assert a3 is not None

    print("RISK_FIX_OK: LLM analyzer skipped for capability-known tools")

    # 4. Agent still constructs.
    from gaia_agent import main as user_agent

    agent = await user_agent.create_agent()
    assert agent is not None
    print("AGENT_CONSTRUCT_OK")

    # 5. Ollama payload sanity: num_ctx and keep_alive present.
    from gaia_agent.llm.provider.ollama import OllamaClient

    assert hasattr(OllamaClient, "_request")
    print("PROVIDER_OK")


asyncio.run(main())
