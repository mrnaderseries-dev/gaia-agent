from __future__ import annotations

from uuid import UUID
from ..models import Memory  # type: ignore
from ..memory_repository import MemoryRepository

class InMemoryMemoryRepository(
    MemoryRepository
):
    """
    In-memory implementation of MemoryRepository.

    Data is stored in RAM and is lost when
    the application stops.
    """

    def __init__(self) -> None:
        self._memories: dict[UUID, Memory] = {}

    async def create(
        self,
        memory: Memory,
    ) -> Memory:

        if memory.memory_id in self._memories:
            raise ValueError(
                f"Memory '{memory.memory_id}' "
                "already exists."
            )

        self._memories[memory.memory_id] = memory

        return memory

    async def get(
        self,
        memory_id: UUID,
    ) -> Memory | None:

        return self._memories.get(memory_id)

    async def list_by_user(
        self,
        user_id: str,
        active_only: bool = True,
    ) -> list[Memory]:

        return [
            memory
            for memory in self._memories.values()
            if (
                memory.user_id == user_id
                and (
                    not active_only
                    or memory.active
                )
            )
        ]

    async def update(
        self,
        memory: Memory,
    ) -> Memory:

        if memory.memory_id not in self._memories:
            raise ValueError(
                f"Memory '{memory.memory_id}' "
                "does not exist."
            )

        self._memories[memory.memory_id] = memory

        return memory

    async def delete(
        self,
        memory_id: UUID,
    ) -> None:

        if memory_id not in self._memories:
            raise ValueError(
                f"Memory '{memory_id}' "
                "does not exist."
            )

        del self._memories[memory_id]