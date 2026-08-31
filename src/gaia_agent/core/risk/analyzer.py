from __future__ import annotations

from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.structured_output import (
    RiskAnalysisOutput,
)

from .models import (
    RiskAnalysis,
    RiskContext,
)


class RiskAnalyzer:

    def __init__(
        self,
        client: LLMClient,
        model: LLMModel,
    ) -> None:
        self.client = client
        self.model = model

    async def analyze(
        self,
        context: RiskContext,
    ) -> RiskAnalysis:

        prompt = self._build_prompt(context)

        # Fixed: Corrected the indentation to align properly inside the analyze method
        result = await self.client.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a risk analysis component of an AI agent. "
                        "Analyze the proposed action only. "
                        "Do not execute the action."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=self.model,
            output_schema=RiskAnalysisOutput,
        )

        return RiskAnalysis(
            level=result.risk_level,
            factors=tuple(result.factors),
            confidence=result.confidence,
            explanation=result.explanation.strip(),
        )

    def _build_prompt(
        self,
        context: RiskContext,
    ) -> str:

        return f"""
You are a risk analysis component of an AI agent.

You MUST NOT execute the action.

Analyze only the proposed action.

Action:
{context.action}

Tool:
{context.tool_name}

Arguments:
{context.arguments}

Evaluate:

- overall risk level
- risk factors
- confidence
- explanation

Risk levels:
LOW
MEDIUM
HIGH
CRITICAL

Risk factors:
FINANCIAL
PRIVACY
EXTERNAL_SIDE_EFFECT
DATA_MODIFICATION
DESTRUCTIVE
SECURITY
LEGAL
REPUTATIONAL

Important:

- Do not assume an action is safe because information is missing.
- Consider possible external side effects.
- Consider data modification and destructive consequences.
- Do not invent facts.
- Return only the required structured output.
"""  # Removed trailing 'zabetha' string


