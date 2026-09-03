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
        if client is None:
            raise ValueError(
                "LLMService requires a valid LLMClient."
            )

        if model is None:
            raise ValueError(
                "LLMService requires a valid LLMModel."
            )

        self.client = client
        self.model = model
    async def generate(
        self,
        messages: Sequence[Message],
        *,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T | str:
        """
        Generate text or structured output asynchronously.
        """

        return await self.client.generate(
            messages,
            model=self.model,
            output_schema=output_schema,
            tools=tools,
            **kwargs,
        )
    def generate_sync(
        self,
        messages: Sequence[Message],
        *,
        output_schema: type[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T | str:
        """
        Synchronous bridge for synchronous tools.

        smolagents Tool.forward() is currently synchronous in this
        project, while our provider API is asynchronous.

        If called while an event loop is already running, execute
        the async request in a dedicated worker thread.

        This avoids:

            RuntimeError:
            asyncio.run() cannot be called from a running event loop
        """

        async def runner() -> T | str:
            return await self.generate(
                messages,
                output_schema=output_schema,
                tools=tools,
                **kwargs,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gaia-llm-sync",
        ) as executor:
            future = executor.submit(
                asyncio.run,
                runner(),
            )
            return future.result()
    async def generate_image(
        self,
        *,
        image_path: str | Path,
        question: str,
        output_schema: type[T] | None = None,
        **kwargs: Any,
    ) -> T | str:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Image path is not a file: {path}"
            )

        image_base64 = self._encode_image(
            path
        )

        messages: list[Message] = [
            {
                "role": "user",
                "content": question,
                "images": [image_base64],
            }
        ]

        return await self.generate(
            messages,
            output_schema=output_schema,
            **kwargs,
        )

    def generate_image_sync(
        self,
        *,
        image_path: str | Path,
        question: str,
        output_schema: type[T] | None = None,
        **kwargs: Any,
    ) -> T | str:
        """
        Synchronous image-analysis API for synchronous tools.
        """

        async def runner() -> T | str:
            return await self.generate_image(
                image_path=image_path,
                question=question,
                output_schema=output_schema,
                **kwargs,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gaia-vision-sync",
        ) as executor:
            future = executor.submit(
                asyncio.run,
                runner(),
            )
            return future.result()
    @staticmethod
    def _encode_image(
        path: Path,
    ) -> str:
        """
        Encode an image as Base64.

        Ollama's REST API accepts Base64 encoded images in the
        message's images array.
        """

        try:
            data = path.read_bytes()
        except OSError as exc:
            raise OSError(
                f"Failed to read image '{path}': {exc}"
            ) from exc

        if not data:
            raise ValueError(
                f"Image file is empty: {path}"
            )

        return base64.b64encode(
            data
        ).decode("ascii")
    