from __future__ import annotations

import json
from uuid import UUID

import asyncpg

from ..memory_repository import MemoryRepository
from ..models import (
    Memory,
    MemorySource,
    MemoryType,
)


class PostgresMemoryRepository(
    MemoryRepository
):

    def __init__(
        self,
        pool: asyncpg.Pool,
    ) -> None:
        self.pool = pool

    async def create(
        self,
        memory: Memory,
    ) -> Memory:

        async with self.pool.acquire() as connection:

            await connection.execute(
                """
                INSERT INTO memories (
                    memory_id,
                    user_id,
                    content,
                    memory_type,
                    source,
                    importance,
                    confidence,
                    created_at,
                    updated_at,
                    active,
                    metadata
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10, $11
                )
                """,
                memory.memory_id,
                memory.user_id,
                memory.content,
                memory.memory_type.value,
                memory.source.value,
                memory.importance,
                memory.confidence,
                memory.created_at,
                memory.updated_at,
                memory.active,
                json.dumps(memory.metadata),
            )

        return memory

    async def get(
        self,
        memory_id: UUID,
    ) -> Memory | None:

        async with self.pool.acquire() as connection:

            row = await connection.fetchrow(
                """
                SELECT
                    memory_id,
                    user_id,
                    content,
                    memory_type,
                    source,
                    importance,
                    confidence,
                    created_at,
                    updated_at,
                    active,
                    metadata
                FROM memories
                WHERE memory_id = $1
                """,
                memory_id,
            )

        if row is None:
            return None

        return self._to_model(row)

    async def update(
        self,
        memory: Memory,
    ) -> Memory:

        async with self.pool.acquire() as connection:

            result = await connection.execute(
                """
                UPDATE memories
                SET
                    content = $1,
                    memory_type = $2,
                    source = $3,
                    importance = $4,
                    confidence = $5,
                    updated_at = $6,
                    active = $7,
                    metadata = $8
                WHERE memory_id = $9
                """,
                memory.content,
                memory.memory_type.value,
                memory.source.value,
                memory.importance,
                memory.confidence,
                memory.updated_at,
                memory.active,
                json.dumps(memory.metadata),
                memory.memory_id,
            )

        if result == "UPDATE 0":
            raise ValueError(
                f"Memory '{memory.memory_id}' "
                "does not exist."
            )

        return memory

    async def delete(
        self,
        memory_id: UUID,
    ) -> None:

        async with self.pool.acquire() as connection:

            result = await connection.execute(
                """
                DELETE FROM memories
                WHERE memory_id = $1
                """,
                memory_id,
            )

        if result == "DELETE 0":
            raise ValueError(
                f"Memory '{memory_id}' "
                "does not exist."
            )

    async def list_by_user(
        self,
        user_id: str,
        active_only:bool=True
    ) -> list[Memory]:

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT
                    memory_id,
                    user_id,
                    content,
                    memory_type,
                    source,
                    importance,
                    confidence,
                    created_at,
                    updated_at,
                    active,
                    metadata
                FROM memories
                WHERE user_id = $1
                AND active = TRUE
                ORDER BY importance DESC, updated_at DESC
                """,
                user_id,
            )

        return [
            self._to_model(row)
            for row in rows
        ]

    @staticmethod
    def _to_model(
        row: asyncpg.Record,
    ) -> Memory:

        return Memory(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=MemoryType(
                row["memory_type"]
            ),
            source=MemorySource(
                row["source"]
            ),
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            active=row["active"],
            metadata=row["metadata"] or {},
        )
    