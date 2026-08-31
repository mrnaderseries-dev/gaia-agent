from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextCompressor import ContextCompressor
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator

from gaia_agent.context.sources.conversation import (
    ConversationContext,
    ConversationSource,
)
from gaia_agent.context.sources.history import HistoryContext, HistorySource
from gaia_agent.context.sources.memory import MemorySource
from gaia_agent.context.sources.runtime import RuntimeContext, RuntimeSource

from gaia_agent.core.agent_state import AgentState
from gaia_agent.conversation.models import Message

from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel

from gaia_agent.planner.planner import Planner
from gaia_agent.planner.plan_schema import PlanSchema, PlanStep, StepType


class FakeLLMClient(LLMClient):

    async def generate(
        self,
        messages,
        *,
        model,
        output_schema=None,
        **kwargs,
    ):
        if output_schema is PlanSchema:
            return PlanSchema(
                steps=[
                    PlanStep(
                        step_id=0,
                        action="Answer the user's request using the provided context.",
                        step_type=StepType.LLM,
                    )
                ]
            )

        return "compressed context"


class FakeMemorySource(MemorySource):

    def __init__(self):
        pass

    async def get(self, state):
        return []

    def is_available(self, state):
        return False


@pytest.mark.asyncio
async def test_planner_works_with_context_builder():

    state = AgentState(
        user_request="Explain the current task using the available context.",
        user_id="test-user",
    )

    state.messages = [
        Message(
            role="user",
            content="We are testing the planner with context.",
        ),
        Message(
            role="assistant",
            content="The context builder should provide this message.",
        ),
    ]

    state.current_step = 1
    state.completed_steps = [0]
    state.current_action = "Process the user request."

    policy = ContextPolicy(
        include_memory=False,
        include_conversation=True,
        include_history=True,
        include_runtime=True,
    )

    budget = ContextBudget(
        max_tokens=4000,
    )

    validator = ContextValidator(
        budget=budget,
    )

    client = FakeLLMClient()

    model = LLMModel(
        provider="fake",
        model="fake-model",
        max_tokens=1000,
        temperature=0.0,
    )

    compressor = ContextCompressor(
    client=client,
    model=model,
    budget=budget,
    policy=policy,
)

    context_builder = ContextBuilder(
        policy=policy,
        budget=budget,
        validator=validator,
        compressor=compressor,
        conversation_source=ConversationSource(),
        history_source=HistorySource(),
        memory_source=FakeMemorySource(),
        runtime_source=RuntimeSource(),
    )

    final_context = await context_builder.build(state)

    assert final_context.items
    assert final_context.token_count <= budget.max_tokens

    context_types = {
        type(item).__name__
        for item in final_context.items
    }

    assert "ConversationContext" in context_types
    assert "HistoryContext" in context_types
    assert "RuntimeContext" in context_types
    assert "MemoryContext" not in context_types

    planner = Planner(
        client=client,
        model=model,
        available_tools=[],
    )

    plan = await planner.generate_plan(
        user_question=state.user_request,
        context=final_context.items,
    )

    assert isinstance(plan, PlanSchema)
    assert plan.steps

    assert plan.steps[0].step_id == 0
    assert plan.steps[0].step_type == StepType.LLM
    assert plan.steps[0].tool_name is None