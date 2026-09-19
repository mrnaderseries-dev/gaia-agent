"""Regression tests: LoopDetector STRUCTURAL tier must respect argument values.

Session-8 BUG-3 evidence (GAIA Q1 fresh run, real models/tools): after a
correct verification refusal, the planner produced a legitimate replan --
``web_search("Mercedes Sosa albums 2000 2009")`` after
``web_search("many studio albums published Mercedes Sosa 2000 2009 included
use")`` -- but ``LoopDetector.check()`` declared a STRUCTURAL loop because the
structural signature contains only ``step_type + strategy_family + tool +
argument_keys`` (argument VALUES omitted). The orchestrator converts any
detection into a fatal non-recoverable ``ExecutionLoopDetected``, so the
documented recovery path (``verification_failed -> replan -> execute a better
evidence step -> verify``) was unreachable for every evidence-seeking task:
any replan that keeps the same tool died before executing and the run ended
with NO answer.

These tests lock the shipped contract (no per-question hacks):

1. A same-tool re-invocation with genuinely different argument values is NOT
   a loop (EXACT and SEMANTIC must also stay silent for it).
2. A same-tool re-invocation whose argument values are only cosmetically
   varied (quoting/spacing the exact normalization does not collapse) IS a
   STRUCTURAL loop.
3. An identical re-execution is still EXACT; a reworded-but-identical
   objective is still SEMANTIC.
4. The same value-aware rule applies to intra-plan repetition
   (``check_plan``).
5. Final-answer steps remain exempt from cross-plan detection.
"""
from __future__ import annotations

import pytest

from gaia_agent.planner.plan_schema import PlanStep, StepType
from gaia_agent.reliability.loop_detector import LoopDetector, LoopType


def _web_search_step(query: str) -> PlanStep:
    return PlanStep(
        step_id=0,
        action=f"Search the web for: {query}",
        step_type=StepType.TOOL,
        tool_name="web_search",
        arguments={"query": query},
        is_final_answer=False,
    )


def _python_step(code: str) -> PlanStep:
    return PlanStep(
        step_id=0,
        action="Execute the python code.",
        step_type=StepType.TOOL,
        tool_name="python_interpreter",
        arguments={"code": code},
        is_final_answer=False,
    )


def _final_step() -> PlanStep:
    return PlanStep(
        step_id=1,
        action="Produce the final answer.",
        step_type=StepType.LLM,
        tool_name=None,
        arguments={},
        is_final_answer=True,
    )


def _record(detector: LoopDetector, step: PlanStep, family: str = "web") -> None:
    detector.record(step, strategy_family=family)


def test_same_tool_different_query_is_not_a_loop() -> None:
    """GAIA Q1 contract: a refined search query after verification failure
    must be executable."""
    detector = LoopDetector()
    first = _web_search_step(
        "many studio albums published Mercedes Sosa 2000 2009 included use"
    )
    second = _web_search_step("Mercedes Sosa albums 2000 2009")

    _record(detector, first)
    detection = detector.check(second, strategy_family="web")

    assert detection.detected is False


def test_cosmetically_varied_arguments_are_still_structural_loop() -> None:
    """Session-2 quoting case: same python code, different quote style — the
    exact normalization does not collapse quotes, so the STRUCTURAL tier must
    still catch it now that values are compared by similarity."""
    detector = LoopDetector()
    _record(detector, _python_step('result = "architecture"[::-1]\n'), "code")
    detection = detector.check(
        _python_step("result = 'architecture'[::-1]\n"),
        strategy_family="code",
    )

    assert detection.detected is True
    assert detection.loop_type is LoopType.STRUCTURAL


def test_identical_reexecution_is_still_exact_loop() -> None:
    detector = LoopDetector()
    step = _web_search_step("Mercedes Sosa albums 2000 2009")
    _record(detector, step)
    detection = detector.check(
        _web_search_step("Mercedes Sosa albums 2000 2009"),
        strategy_family="web",
    )

    assert detection.detected is True
    assert detection.loop_type is LoopType.EXACT


def test_reworded_identical_objective_is_still_semantic_loop() -> None:
    """Same meaningful words, different order/wording: EXACT and STRUCTURAL
    must stay silent (values differ beyond normalization), SEMANTIC must
    fire."""
    detector = LoopDetector()
    _record(
        detector,
        _web_search_step("search mercedes sosa studio albums 2000 2009"),
    )
    detection = detector.check(
        _web_search_step("2000 2009 studio albums sosa mercedes search"),
        strategy_family="web",
    )

    assert detection.detected is True
    assert detection.loop_type is LoopType.SEMANTIC


def test_check_plan_allows_two_different_queries_for_the_same_tool() -> None:
    """A plan that searches twice with different queries is not a repeated
    execution."""
    detector = LoopDetector()
    plan = [
        _web_search_step("mercedes sosa discography studio albums"),
        _web_search_step("mercedes sosa albums 2000 2009 wikipedia"),
        _final_step(),
    ]
    detection = detector.check_plan(
        plan,
        strategy_family_resolver=lambda _step: "web",
    )

    assert detection.detected is False


def test_check_plan_still_rejects_cosmetic_duplicate() -> None:
    detector = LoopDetector()
    plan = [
        _python_step('result = "architecture"[::-1]\n'),
        _python_step("result = 'architecture'[::-1]\n"),
        _final_step(),
    ]
    detection = detector.check_plan(
        plan,
        strategy_family_resolver=lambda _step: "code",
    )

    assert detection.detected is True


def test_final_answer_step_remains_exempt() -> None:
    """Re-answering after a verification failure must never be a loop."""
    detector = LoopDetector()
    _record(detector, _final_step())
    detection = detector.check(_final_step(), strategy_family="web")

    assert detection.detected is False


@pytest.mark.parametrize(
    ("left", "right", "expected_loop"),
    [
        ("a b c", "a b c", True),  # identical -> EXACT tier catches first
        ("a b c", "a b c d", True),  # near-identical values -> STRUCTURAL
        ("a b c", "completely different", False),  # different work -> allowed
    ],
)
def test_argument_value_similarity_threshold(
    left: str, right: str, expected_loop: bool
) -> None:
    detector = LoopDetector()
    _record(detector, _web_search_step(left))
    detection = detector.check(
        _web_search_step(right),
        strategy_family="web",
    )

    assert detection.detected is expected_loop
