"""Regression tests: output-format directives vs text-transformation (Q6).

GAIA Q6 ("Given this table defining * ... counter-examples ... prove * is
not commutative ... comma separated list ... in alphabetical order") was
misclassified as TEXT_TRANSFORMATION because the output-formatting phrase
"alphabetical order" matched the generic transform keywords, which steered
the planner to an LLM-only plan and forbade the deterministic path.

These tests lock the general classification behavior (no per-question hacks).
"""
from __future__ import annotations

from gaia_agent.planner.task_classifier import (
    TaskClassifier,
    TaskIntent,
)

_Q6_TEXT = """Given this table defining * on the set S = {a, b, c, d, e}

|*|a|b|c|d|e|
|---|---|---|---|---|---|
|a|a|b|c|b|d|
|b|b|c|a|e|c|
|c|c|a|b|b|a|
|d|b|e|b|e|d|
|e|d|b|a|d|c|

provide the subset of S involved in any possible counter-examples that
prove * is not commutative. Provide your answer as a comma separated list
of the elements in the set in alphabetical order."""

_Q3_TEXT = (
    '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,'
    "ecnetnes siht dnatsrednu uoy fI"
)

_TOOLS = (
    "python_interpreter",
    "web_search",
    "visit_webpage",
    "analyze_excel",
    "file_reader",
    "analyze_image",
)


def _classify(text: str):
    return TaskClassifier().classify(
        text,
        available_files=[],
        available_tools=_TOOLS,
    )


def test_logic_table_counterexample_is_not_text_transformation() -> None:
    analysis = _classify(_Q6_TEXT)

    assert analysis.intent is not TaskIntent.TEXT_TRANSFORMATION
    assert "web_search" in analysis.forbidden_tools
    assert analysis.recommended_first_tool == "python_interpreter"


def test_output_order_directive_alone_is_not_text_transformation() -> None:
    analysis = _classify(
        "Which country won the 1928 Olympics hockey gold? Return the "
        "answer in alphabetical order."
    )

    assert analysis.intent is not TaskIntent.TEXT_TRANSFORMATION


def test_genuine_sort_still_text_transformation() -> None:
    analysis = _classify("Sort the letters of the word 'banana'.")

    assert analysis.intent == TaskIntent.TEXT_TRANSFORMATION


def test_genuine_reverse_still_text_transformation() -> None:
    analysis = _classify("Reverse the string abcdef.")

    assert analysis.intent == TaskIntent.TEXT_TRANSFORMATION


def test_reversed_gaia_style_question_still_detected() -> None:
    analysis = _classify(_Q3_TEXT)

    assert analysis.intent == TaskIntent.TEXT_TRANSFORMATION
