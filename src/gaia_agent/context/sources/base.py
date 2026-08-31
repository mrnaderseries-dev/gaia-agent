from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gaia_agent.core.agent_state import AgentState


class ContextSource(ABC):

    @abstractmethod
    async def get(self, state: AgentState) -> list[Any]:
        
        raise NotImplementedError

    @abstractmethod
    def is_available(self, state: AgentState) -> bool:
        
        raise NotImplementedError