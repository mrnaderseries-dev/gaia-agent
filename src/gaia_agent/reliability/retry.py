from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Generic, TypeVar

from gaia_agent.reliability.errors import AgentError
from gaia_agent.reliability.error_handler import ErrorHandler


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryResult(Generic[T]):
    success: bool
    result: T | None = None
    attempts: int = 0
    error: AgentError | None = None


class Retry:

    def __init__(self, error_handler: ErrorHandler | None = None) -> None:
        self.error_handler = error_handler or ErrorHandler()

    async def delay(
        self,
        delay: float,
    ) -> None:
        """
        Wait before the next retry attempt.

        Split out from execute() so the ReliabilityEngine can apply
        the policy delay without double-executing the operation.
        """
        if delay < 0:
            raise ValueError(
                "delay cannot be negative."
            )

        if delay <= 0:
            return

        jitter = random.uniform(0, 0.1 * delay)

        await asyncio.sleep(delay + jitter)

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        max_attempts: int,
        delay: float = 2.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
    ) -> RetryResult[T]:

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be greater than 0."
            )

        if delay < 0:
            raise ValueError(
                "delay cannot be negative."
            )

        attempts = 0
        last_error: AgentError | None = None
        current_delay = delay

        while attempts < max_attempts:

            attempts += 1

            try:

                result = await operation()

                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                )

            except Exception as exc:
        
                if isinstance(exc, AgentError):
                    last_error = replace(
                        exc,
                        attempt=attempts,
                    )
                else:
                    last_error = self.error_handler.handle(
                        exc, source="retry", operation="execute", attempt=attempts
                    )

                if not last_error.retryable:
                    break

            if attempts < max_attempts and current_delay > 0:
           
                jitter = random.uniform(0, 0.1 * current_delay)
                sleep_time = min(current_delay + jitter, max_delay)
                
                await asyncio.sleep(sleep_time)
                current_delay *= backoff_factor

        return RetryResult(
            success=False,
            attempts=attempts,
            error=last_error,
        )