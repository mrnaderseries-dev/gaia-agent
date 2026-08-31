from __future__ import annotations

from typing import Any

from gaia_agent.context.ContextBuilder import (
    ContextBuilder,
    FinalContext,
)
from gaia_agent.core.agent_state import AgentState
from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel


class LLMExecutor:

    def __init__(
        self,
        *,
        client: LLMClient,
        model: LLMModel,
        context_builder: ContextBuilder,
    ) -> None:

        self.client = client
        self.model = model
        self.context_builder = context_builder

    async def execute(
        self,
        state: AgentState,
    ) -> str:

        context = await self.context_builder.build(
            state
        )

        messages = self._build_messages(
            state=state,
            context=context,
        )

        result = await self.client.generate(
            messages=messages,
            model=self.model,
        )

        return self._extract_text(result)

    def _build_messages(
        self,
        *,
        state: AgentState,
        context: FinalContext,
    ) -> list[dict[str, str]]:

        context_text = self._format_context(
            context.items
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are the execution component of an AI agent.\n"
                    "Execute the current planned action with high precision.\n\n"
                    "CRITICAL OUTPUT RULES FOR EXACT ANSWERS:\n"
                    "1. Always prioritize factual data provided in the Context over your pre-trained memory.\n"
                    "2. Extract and output ONLY the exact factual answer, entity, name, date, or number requested.\n"
                    "3. DO NOT wrap your response in conversational filler or full sentences (e.g., Output 'Beirut' instead of 'The capital of Lebanon is Beirut').\n"
                    "4. NEVER use approximation or hedge words such as 'approximately', 'about', 'around', or outdated constraints like 'as of 2023'.\n"
                    "5. Do not fabricate or infer facts not present in the provided context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n"
                    f"{state.user_request}\n\n"
                    f"Current action:\n"
                    f"{state.current_action}\n\n"
                    f"Context:\n"
                    f"{context_text}"
                ),
            },
        ]

    @staticmethod
    def _format_context(
        items: list[Any],
    ) -> str:

        if not items:
            return "(No external context is available.)"

        return "\n\n".join(
            str(item)
            for item in items
        )

    @staticmethod
    def _extract_text(
        result: Any,
    ) -> str:

        if isinstance(result, str):

            text = result.strip()

            if not text:
                raise ValueError(
                    "LLM returned an empty response."
                )

            return text

        content = getattr(
            result,
            "content",
            None,
        )

        if isinstance(content, str):

            text = content.strip()

            if not text:
                raise ValueError(
                    "LLM returned empty content."
                )

            return text

        text = getattr(
            result,
            "text",
            None,
        )

        if isinstance(text, str):

            text = text.strip()

            if not text:
                raise ValueError(
                    "LLM returned empty text."
                )

            return text

        raise TypeError(
            "LLM returned an unsupported response type: "
            f"{type(result).__name__}"
        )