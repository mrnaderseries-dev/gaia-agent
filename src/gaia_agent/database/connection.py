from __future__ import annotations

import asyncpg

from gaia_agent.config import settings


class Database:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        print("Inside Database.connect()")

        if self.pool is not None:
            print("Pool already exists")
            return

        print("Creating pool...")
        print(f"Database URL: {settings.database_url}")

        self.pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
        )

        print("Pool created!")

    async def close(self) -> None:
        print("Inside Database.close()")

        if self.pool is None:
            return

        await self.pool.close()
        self.pool = None

    async def health_check(self) -> bool:
        print("Inside health_check()")

        if self.pool is None:
            raise RuntimeError("Database is not connected.")

        async with self.pool.acquire() as connection:
            await connection.execute("SELECT 1")

        return True