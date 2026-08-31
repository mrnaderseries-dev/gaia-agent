
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..risk.models import (
    RiskAssessment,
    RiskFactor,
    RiskLevel,
)


@dataclass(frozen=True, slots=True)
class ApprovalState:
   

    action_name: str
    tool_name: str | None
    risk_assessment: RiskAssessment


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
   
    approval_required: bool
    reason: str | None = None
    message: str | None = None

    @classmethod
    def require_approval(
        cls,
        *,
        reason: str,
        message: str | None = None,
    ) -> "ApprovalDecision":

        return cls(
            approval_required=True,
            reason=reason,
            message=message,
        )

    @classmethod
    def no_approval_required(
        cls,
        *,
        message: str | None = None,
    ) -> "ApprovalDecision":

        return cls(
            approval_required=False,
            message=message,
        )


class ApprovalReason(str, Enum):
    RISK_THRESHOLD = "risk_threshold"
    MANDATORY_RISK_FACTOR = "mandatory_risk_factor"


class ApprovalPolicy:

    RISK_ORDER: dict[RiskLevel, int] = {
        RiskLevel.LOW: 0,
        RiskLevel.MEDIUM: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.CRITICAL: 3,
    }

    MANDATORY_APPROVAL_FACTORS: frozenset[RiskFactor] = frozenset(
        {
            RiskFactor.FINANCIAL,
            RiskFactor.DESTRUCTIVE,
            RiskFactor.EXTERNAL_SIDE_EFFECT,
        }
    )

    def __init__(
        self,
        minimum_risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> None:

        if minimum_risk_level not in self.RISK_ORDER:
            raise ValueError(
                f"Unsupported risk level: {minimum_risk_level!r}"
            )

        self.minimum_risk_level = minimum_risk_level

    def evaluate(
        self,
        state: ApprovalState,
    ) -> ApprovalDecision:

        self._validate_state(state)

        risk = state.risk_assessment

        mandatory_factors = (
            set(risk.factors)
            & self.MANDATORY_APPROVAL_FACTORS
        )

        
        if mandatory_factors:

            factors = ", ".join(
                factor.value
                for factor in sorted(
                    mandatory_factors,
                    key=lambda factor: factor.value,
                )
            )

            return ApprovalDecision.require_approval(
                reason=ApprovalReason.MANDATORY_RISK_FACTOR.value,
                message=(
                    "Human approval is required because "
                    f"the action contains mandatory risk factor(s): "
                    f"{factors}."
                ),
            )

      
        if self._risk_reaches_threshold(risk.level):

            return ApprovalDecision.require_approval(
                reason=ApprovalReason.RISK_THRESHOLD.value,
                message=(
                    "Human approval is required because "
                    f"the assessed risk level is "
                    f"{risk.level.value}, which reaches "
                    f"the configured threshold of "
                    f"{self.minimum_risk_level.value}."
                ),
            )

        return ApprovalDecision.no_approval_required(
            message="No human approval is required."
        )

    def _risk_reaches_threshold(
        self,
        risk_level: RiskLevel,
    ) -> bool:

        return (
            self.RISK_ORDER[risk_level]
            >= self.RISK_ORDER[self.minimum_risk_level]
        )

    @staticmethod
    def _validate_state(
        state: ApprovalState,
    ) -> None:

        if not state.action_name.strip():
            raise ValueError(
                "ApprovalState.action_name cannot be empty."
            )

        if state.tool_name is not None:
            if not state.tool_name.strip():
                raise ValueError(
                    "ApprovalState.tool_name cannot be empty."
                )

        if state.risk_assessment is None:
            raise ValueError(
                "ApprovalState.risk_assessment is required."
            )
