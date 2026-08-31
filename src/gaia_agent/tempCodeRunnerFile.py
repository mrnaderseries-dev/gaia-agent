from __future__ import annotations

import asyncio

from gaia_agent.context.ContextBuilder import ContextBuilder

from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.orchestration.orchestrator import Orchestrator

from gaia_agent.core.policies.termination import TerminationPolicy

from gaia_agent.observability.logger import EventLogger
from gaia_agent.observability.metrics import Metrics
from gaia_agent.observability.tracer import Tracer

from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.retry import Retry
from gaia_agent.reliability.recovery import Recovery

from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.policies.recovery_policy import RecoveryPolicy


async def create_agent() -> AgentLoop:

    # ==========================================================
    # Infrastructure
    # ==========================================================

    llm_client = ...
    llm_model = ...

    tool_registry = ...

    event_logger = EventLogger()
    metrics = Metrics()
    tracer = Tracer()

    token_tracker = ...

    # ==========================================================
    # Policies
    # ==========================================================

    execution_policy = ...
    risk_assessor = ...
    approval_policy = ...

    termination_policy = TerminationPolicy(
        max_iterations=20
    )

    # ==========================================================
    # Context
    # ==========================================================

    context_policy = ...
    context_budget = ...
    context_validator = ...
    context_compressor = ...

    conversation_source = ...
    history_source = ...
    memory_source = ...
    runtime_source = ...

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
    # Planner
    # ==========================================================

    planner = ...

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
    # AgentExecution
    # ==========================================================

    async def llm_executor(
        state: AgentState,
    ):
        return await ...

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

    verifier = ...
    answer_sanitizer = ...

    # ==========================================================
    # Loop Detector
    # ==========================================================

    loop_detector = ...

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

    return AgentLoop(
        orchestrator=orchestrator,
        agent_execution=agent_execution,
        termination_policy=termination_policy,
    )


async def main() -> None:

    agent = await create_agent()

    state = AgentState(
        user_id="test-user",
        user_request="What is the capital of France?",
    )

    result = await agent.run(
        state
    )

    print("\nFinal answer:")
    print(result.final_answer)

    print("\nTermination:")
    print(result.termination_reason)


if __name__ == "__main__":
    asyncio.run(main())
    