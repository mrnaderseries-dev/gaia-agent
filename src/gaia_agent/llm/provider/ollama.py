from __future__ import annotations

from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from gaia_agent.llm.client import (
    LLMClient,
    Message,
)
from gaia_agent.llm.model import LLMModel
from gaia_agent.observability.token_tracker import TokenTracker


T = TypeVar("T", bound=BaseModel)


class OllamaClient(LLMClient):
    """
    Ollama implementation of the provider-independent LLMClient.

    Supports:

    - normal text generation
    - structured JSON output
    - tool calling
    - multimodal messages containing images
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_tracker = token_tracker

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        model: LLMModel,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T | str:

        if model.provider.lower() != "ollama":
            raise ValueError(
                "OllamaClient cannot use provider "
                f"'{model.provider}'."
            )

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
                "num_ctx": 8192,
            },
            "keep_alive": "60m",
        }

        if tools:
            payload["tools"] = tools

        if output_schema is not None:
            payload["format"] = (
                output_schema.model_json_schema()
            )

        payload.update(kwargs)

        response = await self._request(
            payload
        )

        self._track_usage(
            response
        )

        content = self._extract_content(
            response
        )

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

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=300.0,
            ) as client:

                response = await client.post(
                    "/api/chat",
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Unable to connect to Ollama at "
                f"{self.base_url}. "
                "Make sure Ollama is running."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Ollama returned HTTP "
                f"{exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Ollama returned an invalid JSON response."
            )

        return data

    @staticmethod
    def _normalize_message(
        message: Message,
    ) -> dict[str, Any]:

        if not isinstance(message, dict):
            raise TypeError(
                "Every LLM message must be a dictionary."
            )

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str):
            raise ValueError(
                "LLM message is missing a valid 'role'."
            )

        if not isinstance(content, str):
            raise ValueError(
                "Ollama chat messages require string "
                "'content'."
            )

        normalized: dict[str, Any] = {
            "role": role,
            "content": content,
        }

        images = message.get("images")

        if images is not None:
            if not isinstance(images, list):
                raise TypeError(
                    "Message 'images' must be a list."
                )

            normalized["images"] = [
                str(image)
                for image in images
            ]

  
        for key in (
            "name",
            "tool_calls",
            "tool_call_id",
        ):
            if key in message:
                normalized[key] = message[key]

        return normalized
    @staticmethod
    def _extract_content(
        response: dict[str, Any],
    ) -> str:

        try:
            message = response["message"]
            content = message["content"]

        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid Ollama response: missing "
                "message.content"
            ) from exc

        if not isinstance(content, str):
            raise ValueError(
                "Invalid Ollama response: "
                "message.content is not a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "Ollama returned an empty response."
            )

        return content
    @staticmethod
    def _parse_structured_output(
        content: str,
        output_schema: type[T],
    ) -> T:

        try:
            return output_schema.model_validate_json(
                content
            )

        except Exception as exc:
            raise ValueError(
                "Failed to parse Ollama output as "
                f"{output_schema.__name__}: {content}"
            ) from exc
    def _track_usage(
        self,
        response: dict[str, Any],
    ) -> None:
     
        if self.token_tracker is None:
            return
        prompt_tokens = response.get(
            "prompt_eval_count"
        )
        completion_tokens = response.get(
            "eval_count"
        )
        if not isinstance(
            prompt_tokens,
            int,
        ):
            prompt_tokens = 0
        if not isinstance(
            completion_tokens,
            int,
        ):
            completion_tokens = 0
        total_tokens = (
            prompt_tokens
            + completion_tokens
        )
        tracker = self.token_tracker
        record_method = getattr(
            tracker,
            "record",
            None,
        )

        if callable(record_method):
            try:
                record_method(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            except TypeError:
                pass