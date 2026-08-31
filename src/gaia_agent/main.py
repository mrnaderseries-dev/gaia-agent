from __future__ import annotations

import asyncio
import sys

# Root-cause fix for the Windows console 'charmap' UnicodeEncodeError:
# task questions contain non-ASCII characters that the default ANSI
# stdout codec cannot encode, which crashed the agent with
# UnicodeEncodeError and submitted "Error" as the answer.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from gaia_agent.agents.answer_sanitizer import AnswerSanitizer
from gaia_agent.agents.verifier import VerifierAgent

from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextCompressor import ContextCompressor
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator

from gaia_agent.context.sources.conversation import ConversationSource
from gaia_agent.context.sources.history import HistorySource
from gaia_agent.context.sources.memory import MemorySource
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

from gaia_agent.memory.providers.in_memory import InMemoryMemoryRepository

from gaia_agent.memory.retrieval.dense_retriver import CandidateRetriever
from gaia_agent.memory.retrieval.embedding import OllamaEmbeddingProvider
from gaia_agent.memory.retrieval.lexical_retriever import LexicalRetriever
from gaia_agent.memory.retrieval.retriever import MemoryRetriever

from gaia_agent.observability.logger import EventLogger
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


async def create_agent() -> AgentLoop:
    print("Creating agent...")

    # ==========================================================
    # LLM
    # ==========================================================

    llm_client = OllamaClient(
        base_url="http://localhost:11434",
    )

    llm_model = LLMModel(
        provider="ollama",
        model="qwen2.5:3b",
        # Local 3B model: plans, verdicts and GAIA answers are short.
        # 4096 let every call ramble for minutes (primary timeout cause).
        max_tokens=768,
        temperature=0.2,
    )

    # ==========================================================
    # Observability
    # ==========================================================

    event_logger = EventLogger()
    metrics = Metrics()
    tracer = Tracer()
    token_tracker = TokenTracker()

    # ==========================================================
    # Tools
    # ==========================================================

    tool_registry = ToolRegistry(
        base_dir=".",
        model=llm_model,
        stt_backend=None,
    )

    tool_specs = tool_registry.get_tool_specs()

    # Planner requires name -> contract dict (Phase 1).
    available_tools = {
        spec.name: spec
        for spec in tool_specs
    }

    print("\nAvailable tools:")

    for tool in tool_specs:
        print(f"  - {tool.name}")

    # ==========================================================
    # Policies
    # ==========================================================

    execution_policy = ExecutionPolicy()

    risk_rules = RiskRules()

    risk_analyzer = RiskAnalyzer(
        client=llm_client,
        model=llm_model,
    )

    risk_assessor = RiskAssessor(
        rules=risk_rules,
        analyzer=risk_analyzer,
    )

    approval_policy = ApprovalPolicy()

    termination_policy = TerminationPolicy(
        max_iterations=20,
    )

    # ==========================================================
    # Context
    # ==========================================================

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
        model=llm_model,
        budget=context_budget,
        policy=context_policy,
    )

    conversation_source = ConversationSource()
    history_source = HistorySource()
    runtime_source = RuntimeSource()

    # ==========================================================
    # Memory
    # ==========================================================

    memory_repository = InMemoryMemoryRepository()

    embedding_provider = OllamaEmbeddingProvider()

    candidate_retriever = CandidateRetriever(
        embedding_provider=embedding_provider,
    )

    lexical_retriever = LexicalRetriever()

    memory_retriever = MemoryRetriever(
        candidate_retriever=candidate_retriever,
        lexical_retriever=lexical_retriever,
    )

    memory_source = MemorySource(
        retriever=memory_retriever,
        repository=memory_repository,
    )

    # ==========================================================
    # Context Builder
    # ==========================================================

    context_builder = ContextBuilder(
        policy=context_policy,
        budget=context_budget,
        validator=context_validator,
        compressor=context_compressor,
        conversation_source=conversation_source,
        history_source=history_source,
        memory_source=memory_source,
        runtime_source=runtime_source,
    )

    # ==========================================================
    # Loop Detector (created before Planner, which requires it)
    # ==========================================================

    loop_detector = LoopDetector(
        max_history=50,
        max_sequence_length=10,
        exact_repetition_threshold=3,
        sequence_repetition_threshold=3,
    )

    # ==========================================================
    # Planner
    # ==========================================================

    planner = Planner(
        client=llm_client,
        model=llm_model,
        available_tools=available_tools,
        loop_detector=loop_detector,
    )

    # ==========================================================
    # Reliability
    # ==========================================================

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

    # ==========================================================
    # LLM Executor
    # ==========================================================

    llm_executor = LLMExecutor(
        client=llm_client,
        model=llm_model,
        context_builder=context_builder,
    )

    # ==========================================================
    # Agent Execution
    # ==========================================================

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

    # ==========================================================
    # Verifier
    # ==========================================================

    verifier = VerifierAgent(
        client=llm_client,
        model=llm_model,
    )

    answer_sanitizer = AnswerSanitizer()

    # ==========================================================
    # Orchestrator
    # ==========================================================

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

    # ==========================================================
    # Agent Loop
    # ==========================================================

    agent = AgentLoop(
        orchestrator=orchestrator,
        termination_policy=termination_policy,
    )

    print("Agent created.")

    return agent


async def main() -> None:
    print("=== STARTING AGENT ===")

    try:
        agent = await create_agent()

        state = AgentState(
            user_id=1,
            user_request="Search the web and tell me the current population of France.",
        )

        print("State created.")
        print(f"User request: {state.user_request}")
        print("Starting agent.run()...")

        result = await agent.run(
            state,
        )

        print("agent.run() returned.")

        print("\n==============================")
        print("FINAL ANSWER")
        print("==============================")
        print(result.final_answer)

        print("\n==============================")
        print("TOOL ERROR")
        print("==============================")
        print(result.tool_error)

        print("\n==============================")
        print("FATAL ERROR")
        print("==============================")
        print(result.fatal_error)

        print("\n==============================")
        print("CURRENT STEP")
        print("==============================")
        print(result.current_step)

        print("\n==============================")
        print("PLAN")
        print("==============================")
        print(result.plan)

        print("\n==============================")
        print("TERMINATION")
        print("==============================")
        print(result.termination_reason)

        print("\n=== AGENT FINISHED ===")

    except Exception as exc:
        print("\n==============================")
        print("AGENT STARTUP ERROR")
        print("==============================")
        print(type(exc).__name__)
        print(str(exc))

        raise


if __name__ == "__main__":
    asyncio.run(main())