from __future__ import annotations

from dataclasses import dataclass

from gaia_agent.core.agent_state import AgentState
from gaia_agent.memory.models import Memory
from gaia_agent.memory.retrieval.retriever import MemoryRetriever
from gaia_agent.memory.memory_repository import MemoryRepository 
from .base import ContextSource


@dataclass(slots=True)
class MemoryContext:
    memories: list[Memory]


class MemorySource(ContextSource):

    def __init__(self, retriever: MemoryRetriever,repository:MemoryRepository):
        self.retriever = retriever
        self.repository=repository

    async def get(
        self,
        state: AgentState,
    ) -> list[MemoryContext]:
        memories=await self.repository.list_by_user(user_id=state.user_id,active_only=True)

        result = await self.retriever.retrieve(query=state.user_request,memories=memories,top_k=10)

        return [
            MemoryContext(
                memories=result,
            )
        ]

    def is_available(self, state: AgentState) -> bool:
        return bool(state.user_request)