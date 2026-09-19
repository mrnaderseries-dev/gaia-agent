"""Regression: a tool-less (LLM) step must not be blocked by approval.

Live official GAIA Q1 evidence (reproduced against the shipped composition
root, ``_diag_q1_full_run.py``): the planner installed its fallback plan
``[web_search, final-answer LLM step]``; the ``web_search`` step executed and
the final-answer step was refused with

    Human approval is required because the assessed risk level is medium,
    which reaches the configured threshold of medium.

A step with ``tool_name=None`` is an LLM step: it executes no tool and only
synthesizes text from already-collected evidence. It was still routed to the
LLM action-risk analyzer (``capability_for(None)`` is ``None`` and no rule
fired), whose prompt forbids assuming safety when information is missing.
qwen2.5:3b answered MEDIUM on both probe attempts (``risk_threshold`` ->
``approval_required``). ``AgentExecution`` then raised ``ApprovalBlocked``,
the Orchestrator returned ``WAIT_FOR_APPROVAL`` and ``AgentLoop`` stopped the
run: ``answer=None``, ``error=None``, ``verified=false``.

These tests lock the fixed contract:

- a step with no tool is assessed deterministically and needs no approval;
- ``AgentExecution`` executes such a step end to end (no block);
- the LLM analyzer is STILL consulted for tools with an unresolved
  capability, and its MEDIUM verdict still requires approval;
- capability-known read/compute/network-read tools keep the existing
  no-LLM path;
- ``ApprovalPolicy`` itself is unchanged: MEDIUM risk still requires
  approval and mandatory risk factors still force approval.
"""
from __future__ import annotations

from typing import Any

import pytest

from gaia_agent.core.agent_execution import (
    AgentExecution,
    ExecutionRequest,
)
from gaia_agent.core.policies.approval import (
    ApprovalPolicy,
    ApprovalState,
)
from gaia_agent.core.policies.execution import ExecutionPolicy
from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.models import (
    RiskAnalysis,
    RiskAssessment,
    RiskContext,
    RiskFactor,
    RiskLevel,
)
from gaia_agent.core.risk.rules import RiskRules
from gaia_agent.planner.plan_schema import StepType
from gaia_agent.planner.tool_spec import ToolCapability, ToolSpec

FINAL_ANSWER_ACTION = "Synthesize the final answer using only the evidence obtained."


class _Registry:
    """Minimal registry: RiskRules only needs ``get_spec(name).capability``."""

    def __init__(self, specs: dict[str, ToolSpec]) -> None:
        self._specs = specs

    def get_spec(self, name: str) -> ToolSpec:
        return self._specs[name]


REGISTRY = _Registry(
    {
        "web_search": ToolSpec(
            name="web_search",
            description="Search the web.",
            arguments_schema={"type": "object", "properties": {}},
            capability=ToolCapability.NETWORK_READ,
        ),
        "python_interpreter": ToolSpec(
            name="python_interpreter",
            description="Execute Python code.",
            arguments_schema={"type": "object", "properties": {}},
            capability=ToolCapability.COMPUTATION,
        ),
    }
)


class RecordingAnalyzer:
    """Records consultation; returns the verdict the live model returned."""

    def __init__(
        self,
        verdict: RiskAnalysis | None = None,
    ) -> None:
        self.calls = 0
        self.verdict = verdict or RiskAnalysis(
            level=RiskLevel.MEDIUM,
            factors=(),
            confidence=0.7,
            explanation="probe verdict",
        )

    async def analyze(self, context: RiskContext) -> RiskAnalysis:
        self.calls += 1
        return self.verdict


class StubLLMExecutor:
    """Stands in for LLMExecutor: returns the answer text."""

    def __init__(self, text: str = "FINAL ANSWER: 4") -> None:
        self.text = text
        self.calls = 0

    async def execute(self, request: Any) -> str:
        self.calls += 1
        return self.text


def make_assessor(analyzer: RecordingAnalyzer) -> RiskAssessor:
    return RiskAssessor(
        rules=RiskRules(tool_registry=REGISTRY),
        analyzer=analyzer,
    )


def approval_for(assessment: RiskAssessment, *, tool_name: str | None) -> Any:
    return ApprovalPolicy().evaluate(
        ApprovalState(
            action_name=FINAL_ANSWER_ACTION,
            tool_name=tool_name,
            risk_assessment=assessment,
        )
    )


# ---------------------------------------------------------------------------
# The fix: a tool-less step is deterministic and never blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_less_step_is_assessed_without_the_llm_analyzer() -> None:
    analyzer = RecordingAnalyzer()
    assessor = make_assessor(analyzer)

    assessment = await assessor.assess(
        RiskContext(action=FINAL_ANSWER_ACTION)
    )

    assert analyzer.calls == 0, "no tool action exists to analyze"
    assert assessment.level is RiskLevel.LOW
    assert assessment.factors == ()

    decision = approval_for(assessment, tool_name=None)

    assert decision.approval_required is False


@pytest.mark.asyncio
async def test_agent_execution_runs_the_final_answer_step_end_to_end() -> None:
    """The exact production path that returned answer=None must now execute."""

    analyzer = RecordingAnalyzer()
    analyzer.calls = 0
    executor = StubLLMExecutor()

    execution = AgentExecution(
        tool_registry=REGISTRY,
        execution_policy=ExecutionPolicy(),
        risk_assessor=make_assessor(analyzer),
        approval_policy=ApprovalPolicy(),
        llm_executor=executor,
    )

    result = await execution.execute(
        ExecutionRequest(
            step_id=1,
            step_type=StepType.LLM,
            action=FINAL_ANSWER_ACTION,
            tool_name=None,
            arguments={},
            user_request="How many studio albums ...?",
        )
    )

    assert result.success is True
    assert result.blocked is False
    assert result.error is None
    assert result.output == "FINAL ANSWER: 4"
    assert executor.calls == 1
    assert analyzer.calls == 0


# ---------------------------------------------------------------------------
# Safety controls that must NOT be weakened
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_capability_still_escalates_to_the_llm_analyzer() -> None:
    analyzer = RecordingAnalyzer()
    assessor = make_assessor(analyzer)

    assessment = await assessor.assess(
        RiskContext(action="Do something unmapped", tool_name="mystery_tool")
    )

    assert analyzer.calls == 1, "unknown tool capability must still escalate"
    assert assessment.level is RiskLevel.MEDIUM

    decision = approval_for(assessment, tool_name="mystery_tool")

    assert decision.approval_required is True


@pytest.mark.asyncio
async def test_capability_known_tools_keep_the_no_llm_path() -> None:
    analyzer = RecordingAnalyzer()
    assessor = make_assessor(analyzer)

    for tool_name in ("web_search", "python_interpreter"):
        assessment = await assessor.assess(
            RiskContext(action="Search the web for evidence", tool_name=tool_name)
        )

        assert assessment.level is RiskLevel.LOW
        assert approval_for(assessment, tool_name=tool_name).approval_required is False

    assert analyzer.calls == 0


def test_approval_policy_still_blocks_medium_risk_actions() -> None:
    decision = approval_for(
        RiskAssessment(level=RiskLevel.MEDIUM),
        tool_name="http_tool",
    )

    assert decision.approval_required is True
    assert decision.reason == "risk_threshold"


def test_approval_policy_still_requires_mandatory_risk_factors() -> None:
    decision = approval_for(
        RiskAssessment(
            level=RiskLevel.LOW,
            factors=(RiskFactor.EXTERNAL_SIDE_EFFECT,),
        ),
        tool_name="email_tool",
    )

    assert decision.approval_required is True
    assert decision.reason == "mandatory_risk_factor"
