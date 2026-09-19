from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

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

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return float(value) / 100.0
        return value