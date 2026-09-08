from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel

@dataclass(frozen=True, slots=True)
class LLMExecutionRequest:
    user_request: str
    action: str

    context: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class LLMExecutor:
    """
    Executes one LLM request.

    Responsibilities:
        - validate request
        - construct LLM messages
        - call LLM client
        - normalize response

    Does NOT:
        - access AgentState
        - mutate AgentState
        - build context
        - plan
        - retry
        - recover
        - verify
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        model: LLMModel,
    ) -> None:
        self.client = client
        self.model = model

    async def execute(
        self,
        request: LLMExecutionRequest,
    ) -> str:
       

        self._validate_request(request)

        messages = self._build_messages(
            request
        )

        response = await self.client.generate(
            messages=messages,
            model=self.model,
        )

        return self._extract_text(response)
    @staticmethod
    def _validate_request(
        request: LLMExecutionRequest,
    ) -> None:

        if not isinstance(
            request,
            LLMExecutionRequest,
        ):
            raise TypeError(
                "LLMExecutor.execute() requires "
                "an LLMExecutionRequest."
            )

        if not request.user_request.strip():
            raise ValueError(
                "user_request cannot be empty."
            )

        if not request.action.strip():
            raise ValueError(
                "action cannot be empty."
            )
    @staticmethod
    def _build_messages(
        request: LLMExecutionRequest,
    ) -> list[dict[str, str]]:

        context_text = (
            LLMExecutor._format_context(
                request.context
            )
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are the execution component "
                    "of an AI agent.\n"
                    "Execute the current planned action "
                    "with high precision.\n\n"
                    "CRITICAL OUTPUT RULES:\n"
                    "1. Prioritize factual information "
                    "provided in the context.\n"
                    "2. Return the exact answer requested.\n"
                    "3. Do not add unnecessary conversational "
                    "filler.\n"
                    "4. Do not fabricate facts.\n"
                    "5. Do not use approximate answers when "
                    "the context contains an exact answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n"
                    f"{request.user_request}\n\n"
                    f"Current action:\n"
                    f"{request.action}\n\n"
                    f"Context:\n"
                    f"{context_text}"
                ),
            },
        ]
    @staticmethod
    def _format_context(
        context: Any,
    ) -> str:
        if context is None:
            return (
                "(No external context is available.)"
            )

        if isinstance(
            context,
            str,
        ):
            return context

        if isinstance(
            context,
            (list, tuple),
        ):
            if not context:
                return (
                    "(No external context is available.)"
                )

            return "\n\n".join(
                str(item)
                for item in context
            )

        return str(context)
    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:

        if isinstance(
            response,
            str,
        ):

            text = response.strip()

            if not text:
                raise ValueError(
                    "LLM returned an empty response."
                )

            return text

        content = getattr(
            response,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):

            text = content.strip()

            if not text:
                raise ValueError(
                    "LLM returned empty content."
                )

            return text

        text = getattr(
            response,
            "text",
            None,
        )

        if isinstance(
            text,
            str,
        ):

            text = text.strip()

            if not text:
                raise ValueError(
                    "LLM returned empty text."
                )

            return text

        raise TypeError(
            "Unsupported LLM response type: "
            f"{type(response).__name__}"
        )