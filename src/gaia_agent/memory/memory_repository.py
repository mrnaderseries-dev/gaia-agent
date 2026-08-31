from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from .models import Memory


class MemoryRepository(ABC):
    """
    Abstract repository for persistent memory storage.
    """

    @abstractmethod
    async def create(self, memory: Memory) -> Memory:
        raise NotImplementedError

    @abstractmethod
    async def get(self,memory_id: UUID) -> Memory | None:
        raise NotImplementedError

    @abstractmethod
    async def list_by_user(self,user_id: str,active_only: bool = True) -> list[Memory]:
        raise NotImplementedError

    @abstractmethod
    async def update(self,memory: Memory)-> Memory:
        raise NotImplementedError

    @abstractmethod
    async def delete(self,memory_id: UUID) -> None:
        raise NotImplementedError