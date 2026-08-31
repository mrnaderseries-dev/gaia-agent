from __future__ import annotations

from pydantic import BaseModel, Field

from gaia_agent.core.risk.models import (
    RiskFactor,
    RiskLevel,
)


class RiskAnalysisOutput(BaseModel):
    risk_level: RiskLevel

    factors: list[RiskFactor] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    explanation: str