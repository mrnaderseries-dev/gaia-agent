from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

from gaia_agent.agents.answer_sanitizer import AnswerSanitizer
from gaia_agent.agents.verifier import VerifierAgent
from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextCompressor import ContextCompressor
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator
from gaia_agent.context.sources.conversation import ConversationSource
from gaia_agent.context.sources.history import HistorySource
from gaia_agent.context.sources.runtime import RuntimeSource
from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.llm_executor import LLMExecutor
from gaia_agent.core.orchestration.orchestrator import Orchestrator
from gaia_agent.core.policies.approval import ApprovalPolicy
from gaia_agent.core.policies.execution import ExecutionPolicy
from gaia_agent.core.policies.termination import TerminationPolicy
from gaia_agent.core.risk.analyzer import RiskAnalyzer
from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.rules import RiskRules
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.llm.service import LLMService
from gaia_agent.observability.events import EventLogger
from gaia_agent.observability.metrics import Metrics
from gaia_agent.observability.token_tracker import TokenTracker
from gaia_agent.observability.tracer import Tracer
from gaia_agent.planner.planner import Planner
from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.loop_detector import LoopDetector
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.retry import Retry
from gaia_agent.reliability.policies.recovery_policy import RecoveryPolicy
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.tools.registry import ToolRegistry


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_FILE = BASE_DIR / "evaluation_results.jsonl"
OLLAMA_BASE_URL = "http://localhost:11434"
TEXT_MODEL_NAME = "qwen2.5:3b"
VISION_MODEL_NAME = "gemma3"

TEXT_MODEL = LLMModel(
    provider="ollama",
    model=TEXT_MODEL_NAME,
    max_tokens=768,
    temperature=0.2,
)

VISION_MODEL = LLMModel(
    provider="ollama",
    model=VISION_MODEL_NAME,
    max_tokens=768,
    temperature=0.0,
)


def build_agent() -> AgentLoop:
    event_logger = EventLogger()
    metrics = Metrics()
    tracer = Tracer()
    token_tracker = TokenTracker()

    llm_client = OllamaClient(
        base_url=OLLAMA_BASE_URL,
        token_tracker=token_tracker,
    )

    text_llm_service = LLMService(
        client=llm_client,
        model=TEXT_MODEL,
    )

    vision_llm_service = LLMService(
        client=llm_client,
        model=VISION_MODEL,
    )

    tool_registry = ToolRegistry(
        base_dir=str(BASE_DIR),
        llm_service=text_llm_service,
        vision_llm_service=vision_llm_service,
        stt_backend=None,
        stt_model_size="base",
        stt_device="cpu",
        stt_compute_type="int8",
    )

    tool_specs = tool_registry.get_tool_specs()

    available_tools = {
        spec.name: spec
        for spec in tool_specs
    }

    print()
    print("Available tools:")

    for spec in tool_specs:
        print(f"  - {spec.name}")

    execution_policy = ExecutionPolicy()
    risk_rules = RiskRules()

    risk_analyzer = RiskAnalyzer(
        client=llm_client,
        model=TEXT_MODEL,
    )

    risk_assessor = RiskAssessor(
        rules=risk_rules,
        analyzer=risk_analyzer,
    )

    approval_policy = ApprovalPolicy()
    termination_policy = TerminationPolicy(
        max_iterations=20,
    )

    context_policy = ContextPolicy(
        include_memory=False,
        include_conversation=True,
        include_history=True,
        include_runtime=True,
    )

    context_budget = ContextBudget(
        max_tokens=8000,
    )

    context_validator = ContextValidator(
        budget=context_budget,
    )

    context_compressor = ContextCompressor(
        client=llm_client,
        model=TEXT_MODEL,
        budget=context_budget,
        policy=context_policy,
    )

    conversation_source = ConversationSource()
    history_source = HistorySource()
    runtime_source = RuntimeSource()

    context_builder = ContextBuilder(
        policy=context_policy,
        budget=context_budget,
        validator=context_validator,
        compressor=context_compressor,
        conversation_source=conversation_source,
        history_source=history_source,
        memory_source=None,
        runtime_source=runtime_source,
    )

    loop_detector = LoopDetector(
        max_history=50,
        max_sequence_length=10,
        exact_repetition_threshold=3,
        sequence_repetition_threshold=3,
    )

    planner = Planner(
        client=llm_client,
        model=TEXT_MODEL,
        available_tools=available_tools,
        loop_detector=loop_detector,
        base_dir=str(BASE_DIR),
    )

    error_handler = ErrorHandler()
    failure_classifier = FailureClassifier()

    retry_policy = RetryPolicy(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
    )

    recovery_policy = RecoveryPolicy(
        allow_replan=True,
    )

    retry = Retry()
    recovery = Recovery()

    reliability_engine = ReliabilityEngine(
        error_handler=error_handler,
        failure_classifier=failure_classifier,
        retry_policy=retry_policy,
        recovery_policy=recovery_policy,
        retry=retry,
        recovery=recovery,
    )

    llm_executor = LLMExecutor(
        client=llm_client,
        model=TEXT_MODEL,
        context_builder=context_builder,
    )

    agent_execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=approval_policy,
        llm_executor=llm_executor,
        event_logger=event_logger,
        metrics=metrics,
        tracer=tracer,
        token_tracker=token_tracker,
    )

    verifier = VerifierAgent(
        client=llm_client,
        model=TEXT_MODEL,
    )

    answer_sanitizer = AnswerSanitizer()

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        error_handler=error_handler,
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
        answer_sanitizer=answer_sanitizer,
        event_logger=event_logger,
        metrics=metrics,
        tracer=tracer,
    )

    agent = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=termination_policy,
    )

    return agent


def load_questions() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print("[ERROR] The 'datasets' package is not installed.")
        print("Install it with:")
        print("    pip install datasets")
        raise exc

    try:
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_level1",
            split="validation",
        )
    except Exception as exc:
        print("[ERROR] Failed to load GAIA dataset:")
        print(f"        {type(exc).__name__}: {exc}")
        return []

    questions: list[dict[str, Any]] = []

    for row in dataset:
        questions.append(dict(row))

    return questions


def save_result(result: dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


async def run_question(
    agent: AgentLoop,
    question: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    question_id = question.get(
        "task_id",
        index,
    )

    prompt = question.get("Question")

    if not isinstance(prompt, str):
        prompt = str(prompt or "")

    print()
    print("=" * 80)
    print(f"[QUESTION {index}] {question_id}")
    print("=" * 80)
    print(prompt)
    print()

    state = AgentState(
        user_id=1,
        user_request=prompt,
    )

    try:
        result = await agent.run(state)

        answer = getattr(
            result,
            "final_answer",
            None,
        )

        output = {
            "task_id": question_id,
            "question": prompt,
            "answer": answer,
            "final_answer_ready": getattr(
                result,
                "final_answer_ready",
                False,
            ),
            "final_answer_verified": getattr(
                result,
                "final_answer_verified",
                False,
            ),
            "status": "completed",
            "error": None,
        }

        print()
        print("[ANSWER]")
        print(answer)

        print()
        print(
            "[FINAL ANSWER READY]",
            getattr(
                result,
                "final_answer_ready",
                False,
            ),
        )

        print(
            "[FINAL ANSWER VERIFIED]",
            getattr(
                result,
                "final_answer_verified",
                False,
            ),
        )

        return output

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"

        print()
        print(f"[ERROR] Question {question_id} failed:")
        print(f"        {error_text}")

        return {
            "task_id": question_id,
            "question": prompt,
            "answer": None,
            "final_answer_ready": False,
            "final_answer_verified": False,
            "status": "error",
            "error": error_text,
        }


async def main() -> None:
    print("--- STARTING OFFICIAL GAIA AGENT EVALUATION ---")

    print()
    print(f"[INFO] Base directory: {BASE_DIR}")
    print(f"[INFO] Results file: {RESULTS_FILE}")
    print(f"[INFO] Text model: {TEXT_MODEL_NAME}")
    print(f"[INFO] Vision model: {VISION_MODEL_NAME}")
    print()

    try:
        agent = build_agent()
    except Exception as exc:
        print("[ERROR] Failed to create agent:")
        print(f"        {type(exc).__name__}: {exc}")
        return

    print()
    print("[INFO] Agent created successfully.")

    questions = load_questions()

    if not questions:
        print("[ERROR] No GAIA questions loaded.")
        return

    print()
    print(f"[INFO] Loaded {len(questions)} official GAIA questions.")

    completed = 0
    failed = 0
    verified = 0

    for index, question in enumerate(
        questions,
        start=1,
    ):
        result = await run_question(
            agent,
            question,
            index,
        )

        save_result(result)

        if result["status"] == "completed":
            completed += 1

            if result.get(
                "final_answer_verified",
                False,
            ):
                verified += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print("GAIA EVALUATION FINISHED")
    print("=" * 80)

    print(f"Total questions          : {len(questions)}")
    print(f"Completed                : {completed}")
    print(f"Failed                   : {failed}")
    print(f"Final answers verified   : {verified}")
    print(f"Results file             : {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())