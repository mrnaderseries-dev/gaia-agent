from __future__ import annotations

import json
from typing import Any, Sequence, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from gaia_agent.llm.client import LLMClient, Message
from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.usage import TokenTrackerProtocol, TokenUsage
from gaia_agent.reliability.exception import (
    AuthenticationError,
    AuthorizationError,
    LLMFailure,
    LLMOutputError,
    RateLimitError,
)

T = TypeVar("T")


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
        token_tracker: TokenTrackerProtocol | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token_tracker = token_tracker

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        model: LLMModel,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        operation: str = "llm.generate",
        **kwargs: Any,
    ) -> T | str:
        normalized_messages = [
            self._normalize_message(message)
            for message in messages
        ]

        payload: dict[str, Any] = {
            "model": model.model,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
            },
        }

        if tools:
            payload["tools"] = tools

        if output_schema is not None:
            payload["format"] = self._schema_for_ollama(
                output_schema
            )

        response = await self._request(payload)

        usage = self._extract_usage(response)

        self._track_usage(
            operation=operation,
            usage=usage,
        )

        message = response.get("message")

        if not isinstance(message, dict):
            raise LLMOutputError(
                "Ollama response does not contain a valid message."
            )

        content = message.get("content", "")

        if output_schema is None:
            return str(content)

        return self._parse_structured_output(
            content,
            output_schema,
        )

    async def _request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"

        async with httpx.AsyncClient(
            timeout=self.timeout
        ) as client:
            response = await client.post(
                url,
                json=payload,
            )

        if response.status_code >= 400:
            self._raise_http_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMOutputError(
                "Ollama returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise LLMOutputError(
                "Ollama returned an invalid response object."
            )

        return data

    @staticmethod
    def _raise_http_error(
        response: httpx.Response,
    ) -> None:
        status = response.status_code
        message = (
            f"Ollama request failed "
            f"({status}): {response.text}"
        )

        if status == 401:
            raise AuthenticationError(message)

        if status == 403:
            raise AuthorizationError(message)

        if status == 429:
            raise RateLimitError(message)

        if status >= 500 or status == 408:
            raise LLMFailure(
                message,
                retryable=True,
                recoverable=False,
            )

        raise LLMFailure(
            message,
            retryable=False,
            recoverable=True,
        )

    def _normalize_message(
        self,
        message: Message,
    ) -> dict[str, Any]:
        role = message.get("role")
        content = message.get("content", "")

        if not isinstance(role, str):
            raise ValueError(
                "Message role must be a string"
            )

        normalized: dict[str, Any] = {
            "role": role,
            "content": content,
        }

        if "name" in message:
            normalized["name"] = message["name"]

        if "images" in message:
            images = message["images"]

            if not isinstance(images, list):
                raise ValueError(
                    "Message images must be a list"
                )

            normalized["images"] = images

        if "tool_calls" in message:
            normalized["tool_calls"] = message["tool_calls"]

        if "tool_call_id" in message:
            normalized["tool_call_id"] = message["tool_call_id"]

        return normalized

    def _schema_for_ollama(
        self,
        output_schema: type[T],
    ) -> dict[str, Any]:
        if not issubclass(output_schema, BaseModel):
            raise TypeError(
                "output_schema must be a Pydantic BaseModel subclass"
            )

        return output_schema.model_json_schema()

    def _extract_usage(
        self,
        response: dict[str, Any],
    ) -> TokenUsage:
        prompt_tokens = self._non_negative_int(
            response.get("prompt_eval_count")
        )

        completion_tokens = self._non_negative_int(
            response.get("eval_count")
        )

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _non_negative_int(value: Any) -> int:
        if value is None:
            return 0

        if isinstance(value, bool):
            return 0

        if isinstance(value, int):
            return max(value, 0)

        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0

        return max(parsed, 0)

    def _track_usage(
        self,
        *,
        operation: str,
        usage: TokenUsage,
    ) -> None:
        tracker = self.token_tracker

        if tracker is None:
            return

        tracker.record(
            operation,
            usage=usage,
        )

    def _parse_structured_output(
        self,
        content: Any,
        output_schema: type[T],
    ) -> T:
        if not isinstance(content, str):
            raise LLMOutputError(
                "Structured Ollama response content must be a string."
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(
                "Ollama returned invalid JSON for structured output."
            ) from exc

        if not issubclass(output_schema, BaseModel):
            raise TypeError(
                "output_schema must be a Pydantic BaseModel subclass"
            )

        try:
            return output_schema.model_validate(parsed)
        except ValidationError as exc:
            raise LLMOutputError(
                "Ollama structured output failed schema validation."
            ) from exc