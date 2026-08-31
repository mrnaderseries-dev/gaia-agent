from __future__ import annotations

from uuid import UUID

from .models import Memory
from gaia_agent.memory.memory_repository import MemoryRepository


class MemoryService:
    """
    Provides business operations for user memories.
    """

    def __init__(
        self,
        repository: MemoryRepository,
    ) -> None:
        self.repository = repository

    async def create_memory(self,memory: Memory) -> Memory:
        return await self.repository.create(memory)

    async def get_memory(self,memory_id: UUID) -> Memory | None:
        return await self.repository.get(memory_id)

    async def list_user_memories(self,user_id: str,active_only: bool = True) -> list[Memory]:

        if not user_id.strip():
            raise ValueError("user_id cannot be empty.")

        return await self.repository.list_by_user(user_id=user_id,active_only=active_only)

    async def update_memory(self,memory: Memory)-> Memory:

        existing = await self.repository.get(memory.memory_id)

        if existing is None:
            raise ValueError(
                f"Memory '{memory.memory_id}' "
                "does not exist."
            )

        return await self.repository.update(memory)

    async def delete_memory(self,memory_id: UUID) -> None:

        existing = await self.repository.get(memory_id)

        if existing is None:
            raise ValueError(
                f"Memory '{memory_id}' "
                "does not exist."
            )

        await self.repository.delete(memory_id)