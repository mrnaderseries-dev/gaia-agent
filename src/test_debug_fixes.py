"""
Regression tests for the 2026-08-30 debugging cycle.

Covers the root causes found in the evaluation logs:

R1  Recovery.execute crashed with
    "TypeError: catching classes that do not inherit from BaseException
    is not allowed" because AgentError is a dataclass, not an Exception.
R2  Planner fired web_search for self-contained tasks (raw-question
    queries, reversed-text tasks misclassified).
R3  Recovery terminated instead of choosing a different strategy.
R4  Verification accepted LLM answers that the evidence does not
    support (false verification).
R5  Plans without a final-answer step were accepted (final_answer=None).
R6  Invented / placeholder file paths reached file tools.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_agent.core.evidence import ToolResultRecord
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.agents.verifier import (
    VerificationStatus,
    deterministic_verification,
    evidence_supports_candidate,
)
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType
from gaia_agent.planner.planner import Planner
from gaia_agent.planner.task_classifier import TaskClassifier, TaskIntent
from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.loop_detector import LoopDetector
from gaia_agent.reliability.recovery import Recovery, RecoveryResult


class FakeTool:
    def __init__(self, name, inputs):
        self.name = name
        self.description = f"fake {name}"
        self.inputs = inputs
        self.output_type = "string"


def make_planner(available_files=None):
    return Planner(
        client=object(),
        model="test-model",
        available_tools={
            "web_search": FakeTool(
                "web_search",
                {"query": {"type": "string"}},
            ),
            "visit_webpage": FakeTool(
                "visit_webpage",
                {"url": {"type": "string"}},
            ),
            "file_reader": FakeTool(
                "file_reader",
                {"file_path": {"type": "string"}},
            ),
            "analyze_excel": FakeTool(
                "analyze_excel",
                {
                    "file_path": {"type": "string"},
                    "question": {"type": "string"},
                },
            ),
            "analyze_image": FakeTool(
                "analyze_image",
                {
                    "image_path": {"type": "string"},
                    "question": {"type": "string"},
                },
            ),
            "python_interpreter": FakeTool(
                "python_interpreter",
                {"code": {"type": "string"}},
            ),
        },
        available_files=available_files or [],
    )


# ----------------------------------------------------------------------
# R1: Recovery exception handling
# ----------------------------------------------------------------------

def test_recovery_wraps_plain_exceptions():
    recovery = Recovery()

    async def broken_callback(error):
        raise ValueError("Recovery budget exceeded (max replans: 2).")

    result = asyncio.run(
        recovery.execute(
            error=AgentError(
                error_type="ToolExecutionError",
                message="original failure",
            ),
            operation=broken_callback,
        )
    )

    assert isinstance(result, RecoveryResult)
    assert result.recovered is False
    assert result.error is not None
    # The REAL reason must survive (no "catching classes ..." TypeError).
    assert "Recovery budget exceeded" in result.reason
    assert "catching classes" not in result.reason


def test_recovery_handles_agent_error():
    recovery = Recovery()

    async def agent_error_callback(error):
        raise AgentError(
            error_type="PlannerError",
            message="Planner produced a repeated plan.",
        )

    result = asyncio.run(
        recovery.execute(
            error=AgentError(
                error_type="ToolExecutionError",
                message="original failure",
            ),
            operation=agent_error_callback,
        )
    )

    assert result.recovered is False
    assert result.error is not None
    assert result.error.message == "Planner produced a repeated plan."


def test_recovery_success_path():
    recovery = Recovery()

    async def working_callback(error):
        return PlanStep(
            step_id=0,
            action="Search the web",
            step_type=StepType.TOOL,
            tool_name="web_search",
            arguments={"query": "birds"},
        )

    result = asyncio.run(
        recovery.execute(
            error=AgentError(
                error_type="ToolExecutionError",
                message="original failure",
            ),
            operation=working_callback,
        )
    )

    assert result.recovered is True
    assert result.result is not None


# ----------------------------------------------------------------------
# R2: classification / planner tool-first behavior
# ----------------------------------------------------------------------

def test_reversed_sentence_classified_as_text_transformation():
    classifier = TaskClassifier()
    analysis = classifier.classify(
        '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,'
        "ecnetnes siht dnatsrednu uoy fI",
        available_files=[],
        available_tools=["web_search", "python_interpreter"],
    )

    assert analysis.intent == TaskIntent.TEXT_TRANSFORMATION
    assert analysis.needs_external_info is False
    assert "web_search" in analysis.forbidden_tools


def test_factual_task_still_factual():
    classifier = TaskClassifier()
    analysis = classifier.classify(
        "What is the capital city of Australia?",
        available_files=[],
        available_tools=["web_search"],
    )

    assert analysis.intent == TaskIntent.FACTUAL_SEARCH


def test_build_search_query_strips_urls_and_stopwords():
    planner = make_planner()
    query = planner._build_search_query(
        "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, "
        "what is the highest number of bird species to be on camera "
        "simultaneously?"
    )

    assert "http" not in query
    assert "the" not in query.split()
    assert len(query.split()) <= 10
    assert "bird" in query


def test_validate_step_repairs_raw_question_query():
    planner = make_planner()
    raw_question = (
        "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, "
        "what is the highest number of bird species to be on camera "
        "simultaneously?"
    )
    planner._current_question = raw_question

    step = PlanStep(
        step_id=0,
        action="Search for relevant evidence",
        step_type=StepType.TOOL,
        tool_name="web_search",
        arguments={"query": raw_question},
    )

    planner._validate_step(step)

    repaired = step.arguments["query"]
    assert repaired != raw_question
    assert "http" not in repaired
    assert "bird" in repaired


def test_validate_step_rejects_placeholder_file_path():
    planner = make_planner(available_files=["sales_2023.xlsx"])

    step = PlanStep(
        step_id=0,
        action="Analyze the spreadsheet",
        step_type=StepType.TOOL,
        tool_name="analyze_excel",
        arguments={
            "file_path": "<file_path_to_sales_file>",
            "question": "total revenue?",
        },
    )

    try:
        planner._validate_step(step)
    except ValueError as exc:
        assert "placeholder" in str(exc).lower()
    else:
        raise AssertionError("placeholder path must be rejected")


def test_validate_step_repairs_invented_file_path():
    planner = make_planner(
        available_files=[
            "gaia_attachments/abc/kuznetzov_taxonomist_data.xlsx",
        ]
    )

    step = PlanStep(
        step_id=0,
        action="Analyze the data",
        step_type=StepType.TOOL,
        tool_name="analyze_excel",
        arguments={
            "file_path": "kuznetzov_taxonomist_data.xlsx",
            "question": "where deposited?",
        },
    )

    planner._validate_step(step)

    assert step.arguments["file_path"] == (
        "gaia_attachments/abc/kuznetzov_taxonomist_data.xlsx"
    )


def test_validate_step_rejects_unknown_file_path():
    planner = make_planner(available_files=["other.csv"])

    step = PlanStep(
        step_id=0,
        action="Analyze the data",
        step_type=StepType.TOOL,
        tool_name="analyze_excel",
        arguments={
            "file_path": "invented_data.xlsx",
            "question": "where deposited?",
        },
    )

    try:
        planner._validate_step(step)
    except ValueError as exc:
        assert "never invent file paths" in str(exc).lower()
    else:
        raise AssertionError("invented path must be rejected")


# ----------------------------------------------------------------------
# R5: plans must contain a final answer step
# ----------------------------------------------------------------------

def test_plan_without_final_answer_step_rejected():
    planner = make_planner()

    plan = PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="web_search",
                step_type=StepType.TOOL,
                tool_name="web_search",
                arguments={"query": "x"},
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="final_answer",
                step_type=StepType.TOOL,
                tool_name="python_interpreter",
                arguments={"code": "result = 1"},
                is_final_answer=False,
            ),
        ]
    )

    try:
        planner._validate_final_answer_structure(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("plan without final step must be rejected")


def test_valid_plan_with_final_answer_step_passes():
    planner = make_planner()

    plan = PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action="Search",
                step_type=StepType.TOOL,
                tool_name="web_search",
                arguments={"query": "x"},
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action="Synthesize the final answer",
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )

    planner._validate_final_answer_structure(plan)


# ----------------------------------------------------------------------
# R2b: emergency fallback must respect forbidden tools
# ----------------------------------------------------------------------

def test_emergency_fallback_respects_forbidden_web_search():
    planner = make_planner()
    classifier = TaskClassifier()

    analysis = classifier.classify(
        "Reverse the letters of the wordGAIA and answer",
        available_files=[],
        available_tools=sorted(planner.available_tools),
    )
    assert "web_search" in analysis.forbidden_tools

    plan = planner._emergency_fallback_plan(
        user_question="Reverse the letters of the wordGAIA and answer",
        failed_step=None,
        analysis=analysis,
    )

    used_tools = [
        step.tool_name
        for step in plan.steps
        if step.step_type == StepType.TOOL
    ]
    assert "web_search" not in used_tools


def test_emergency_fallback_factual_uses_short_query():
    planner = make_planner()
    classifier = TaskClassifier()

    question = (
        "Who nominated the only Featured Article on English Wikipedia "
        "about a dinosaur that was promoted in November 2016?"
    )
    analysis = classifier.classify(
        question,
        available_files=[],
        available_tools=sorted(planner.available_tools),
    )

    plan = planner._emergency_fallback_plan(
        user_question=question,
        failed_step=None,
        analysis=analysis,
    )

    search_steps = [
        step
        for step in plan.steps
        if step.tool_name == "web_search"
    ]
    assert search_steps, "factual fallback should search"
    query = search_steps[0].arguments["query"]
    assert query != question
    assert len(query.split()) <= 10


# MARKER_PART_6
