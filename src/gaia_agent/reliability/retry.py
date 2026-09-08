from __future__ import annotations

import asyncio
import random


class Retry:
    def __init__(self, *, jitter_ratio: float = 0.1) -> None:
        if jitter_ratio < 0:
            raise ValueError("jitter_ratio must be >= 0.")

        self.jitter_ratio = jitter_ratio

    async def delay(self, delay: float) -> None:
        if delay < 0:
            raise ValueError("delay cannot be negative.")

        if delay == 0:
            return

        jitter = random.uniform(
            0,
            self.jitter_ratio * delay,
        )

        await asyncio.sleep(delay + jitter)