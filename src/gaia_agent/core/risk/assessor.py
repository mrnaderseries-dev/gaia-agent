from __future__ import annotations

from gaia_agent.core.risk.models import (
    RiskAnalysis,
    RiskAssessment,
    RiskContext,
    RiskFactor,
    RiskLevel,
)
from gaia_agent.planner.tool_spec import (
    TOOL_CAPABILITIES,
    ToolCapability,
)

from .analyzer import RiskAnalyzer
from .rules import RiskRules


class RiskAssessor:

    def __init__(
        self,
        rules: RiskRules,
        analyzer: RiskAnalyzer | None = None,
    ) -> None:
        self.rules = rules
        self.analyzer = analyzer

    async def assess(
        self,
        context: RiskContext,
    ) -> RiskAssessment:

        rule_factors = self.rules.analyze(context)

        # Deterministic-first risk assessment (analysis item #8): the LLM
        # analyzer costs a full structured-output generation per tool step,
        # which dominates latency on local models. Only consult it when the
        # deterministic rules cannot classify the action at all (unknown
        # tool, no risk factors, no declared capability).
        rule_level = self._level_from_rules(rule_factors)

        capability = TOOL_CAPABILITIES.get(
            (context.tool_name or "").lower().strip()
        )

        skip_llm = rule_level is not None or capability in {
            ToolCapability.READ_ONLY,
            ToolCapability.COMPUTATION,
            ToolCapability.NETWORK_READ,
        }

        llm_analysis: RiskAnalysis | None = None

        if self.analyzer is not None and not skip_llm:
            try:
                llm_analysis = await self.analyzer.analyze(
                    context
                )
            except Exception:

                llm_analysis = None

        factors = self._merge_factors(
            rule_factors,
            llm_analysis,
        )

        level = self._determine_level(
            rule_factors=rule_factors,
            llm_analysis=llm_analysis,
        )

        confidence = self._determine_confidence(
            rule_factors=rule_factors,
            llm_analysis=llm_analysis,
        )

        explanation = self._build_explanation(
            rule_factors,
            llm_analysis,
        )

        return RiskAssessment(
            level=level,
            factors=tuple(
                sorted(
                    factors,
                    key=lambda factor: factor.value,
                )
            ),
            confidence=confidence,
            explanation=explanation,
        )

    def _merge_factors(
        self,
        rule_factors: set[RiskFactor],
        llm_analysis: RiskAnalysis | None,
    ) -> set[RiskFactor]:

        factors = set(rule_factors)

        if llm_analysis is not None:
            factors.update(llm_analysis.factors)

        return factors

    def _determine_level(
        self,
        *,
        rule_factors: set[RiskFactor],
        llm_analysis: RiskAnalysis | None,
    ) -> RiskLevel:

        rule_level = self._level_from_rules(
            rule_factors
        )

        llm_level = (
            llm_analysis.level
            if llm_analysis is not None
            else RiskLevel.LOW
        )

        if rule_level is None:
            return llm_level

        return max(
            rule_level,
            llm_level,
            key=self._risk_rank,
        )

    @staticmethod
    def _risk_rank(
        level: RiskLevel,
    ) -> int:

        return {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }[level]

    def _level_from_rules(
        self,
        factors: set[RiskFactor],
    ) -> RiskLevel | None:

        if RiskFactor.FINANCIAL in factors:
            return RiskLevel.CRITICAL

        if RiskFactor.SECURITY in factors:
            return RiskLevel.CRITICAL

        if RiskFactor.DESTRUCTIVE in factors:
            return RiskLevel.CRITICAL

        if RiskFactor.LEGAL in factors:
            return RiskLevel.HIGH

        if RiskFactor.PRIVACY in factors:
            return RiskLevel.HIGH

        if RiskFactor.DATA_MODIFICATION in factors:
            return RiskLevel.HIGH

        if RiskFactor.EXTERNAL_SIDE_EFFECT in factors:
            return RiskLevel.MEDIUM

        if RiskFactor.REPUTATIONAL in factors:
            return RiskLevel.MEDIUM

        return None

    def _determine_confidence(
        self,
        *,
        rule_factors: set[RiskFactor],
        llm_analysis: RiskAnalysis | None,
    ) -> float | None:

        if llm_analysis is None:
            return None

        return llm_analysis.confidence

    def _build_explanation(
        self,
        rule_factors: set[RiskFactor],
        llm_analysis: RiskAnalysis | None,
    ) -> str:

        parts: list[str] = []

        if rule_factors:
            factors = ", ".join(
                factor.value
                for factor in sorted(
                    rule_factors,
                    key=lambda factor: factor.value,
                )
            )

            parts.append(
                f"Deterministic rules detected: {factors}."
            )

        if llm_analysis is not None:
            parts.append(
                "LLM analysis: "
                + (
                    llm_analysis.explanation
                    or "No explanation provided."
                )
            )
        else:
            parts.append(
                "LLM risk analysis was unavailable."
            )

        return " ".join(parts)