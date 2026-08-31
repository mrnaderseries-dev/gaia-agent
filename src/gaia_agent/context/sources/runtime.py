from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gaia_agent.core.agent_state import AgentState

from .base import ContextSource


@dataclass(slots=True)
class RuntimeContext:
    iteration: int
    tool_name: str | None
    blocked: bool
    tool_result: Any | None
    tool_error: str | None


class RuntimeSource(ContextSource):

    async def get(
        self,
        state: AgentState,
    ) -> list[Any]:

        return [
            RuntimeContext(
                iteration=state.iteration,
                tool_name=state.tool_name,
                blocked=state.blocked,
                tool_result=state.tool_result,
                tool_error=state.tool_error,
            )
        ]

    def is_available(
        self,
        state: AgentState,
    ) -> bool:

        return state is not None