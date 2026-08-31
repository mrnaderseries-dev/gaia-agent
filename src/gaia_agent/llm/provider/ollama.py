from __future__ import annotations

from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from ..client import LLMClient
from ..model import LLMModel
from gaia_agent.observability.token_tracker import TokenTracker

T = TypeVar("T", bound=BaseModel)


class OllamaClient(LLMClient):

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_tracker = token_tracker

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: LLMModel,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T | str:

        if model.provider.lower() != "ollama":
            raise ValueError(
                f"OllamaClient cannot use provider '{model.provider}'."
            )

        payload: dict[str, Any] = {
            "model": model.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
                # Keep the full planner prompt (tool contracts included):
                # Ollama's default 2048 ctx silently truncates input,
                # which caused hallucinated tool names.
                "num_ctx": 8192,
            },
            # Keep the model loaded between questions; reloading a local
            # model per call stalls the evaluation run.
            "keep_alive": "60m",
        }

        if tools:
            payload["tools"] = tools

        if output_schema is not None:
            payload["format"] = output_schema.model_json_schema()

        payload.update(kwargs)

        response = await self._request(payload)

        content = self._extract_content(response)

        if output_schema is None:
            return content

        return self._parse_structured_output(
            content,
            output_schema,
        )

    async def _request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        import httpx

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=300.0,
        ) as client:

            response = await client.post(
                "/api/chat",
                json=payload,
            )

            response.raise_for_status()

            return response.json()

    @staticmethod
    def _extract_content(
        response: dict[str, Any],
    ) -> str:

        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid Ollama response: missing message.content"
            ) from exc

    @staticmethod
    def _parse_structured_output(
        content: str,
        output_schema: type[T],
    ) -> T:

        try:
            return output_schema.model_validate_json(content)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse LLM output as {output_schema.__name__}: {content}"
            ) from exc