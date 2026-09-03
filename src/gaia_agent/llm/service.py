from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence, TypeVar

from .client import LLMClient, Message
from .model import LLMModel

T = TypeVar("T")


class LLMService:

    def __init__(
        self,
        client: LLMClient,
        model: LLMModel,
    ) -> None:
        self.client = client
        self.model = model

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        operation: str = "llm.generate",
        **kwargs: Any,
    ) -> T | str:
        return await self.client.generate(
            messages,
            model=self.model,
            output_schema=output_schema,
            tools=tools,
            operation=operation,
            **kwargs,
        )

    def generate_sync(
        self,
        messages: Sequence[Message],
        *,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        operation: str = "llm.generate",
        **kwargs: Any,
    ) -> T | str:
        coroutine = self.generate(
            messages,
            output_schema=output_schema,
            tools=tools,
            operation=operation,
            **kwargs,
        )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coroutine)
            return future.result()

    async def generate_image(
        self,
        image_path: str | Path,
        question: str,
        *,
        operation: str = "llm.vision",
        **kwargs: Any,
    ) -> str:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        image_bytes = path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("ascii")

        messages: list[Message] = [
            {
                "role": "user",
                "content": question,
                "images": [image_base64],
            }
        ]

        result = await self.generate(
            messages,
            operation=operation,
            **kwargs,
        )

        if not isinstance(result, str):
            return str(result)

        return result

    def generate_image_sync(
        self,
        image_path: str | Path,
        question: str,
        *,
        operation: str = "llm.vision",
        **kwargs: Any,
    ) -> str:
        
        coroutine = self.generate_image(
            image_path=image_path,
            question=question,
            operation=operation,
            **kwargs,
        )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coroutine)
            return future.result()
        