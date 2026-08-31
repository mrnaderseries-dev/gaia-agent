from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(str, Enum):
    FINANCIAL = "financial"
    PRIVACY = "privacy"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    DATA_MODIFICATION = "data_modification"
    DESTRUCTIVE = "destructive"
    SECURITY = "security"
    LEGAL = "legal"
    REPUTATIONAL = "reputational"


@dataclass(frozen=True, slots=True)
class RiskContext:
    action: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.action.strip():
            raise ValueError(
                "RiskContext action cannot be empty."
            )

        if self.tool_name is not None:
            if not self.tool_name.strip():
                raise ValueError(
                    "RiskContext tool_name cannot be empty."
                )


@dataclass(frozen=True, slots=True)
class RiskAnalysis:
    level: RiskLevel
    factors: tuple[RiskFactor, ...] = ()
    confidence: float = 0.0
    explanation: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "RiskAnalysis confidence must be between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    level: RiskLevel
    factors: tuple[RiskFactor, ...] = ()
    confidence: float | None = None
    explanation: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "RiskAssessment confidence must be between 0 and 1."
                )