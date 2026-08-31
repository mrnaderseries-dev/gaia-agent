from __future__ import annotations

from dataclasses import dataclass

from gaia_agent.core.agent_state import AgentState
from gaia_agent.conversation.models import Message
from .base import ContextSource


@dataclass(slots=True)
class ConversationContext:
    messages: list[Message]


class ConversationSource(ContextSource):

    async def get(
        self,
        state: AgentState,
    ) -> list[ConversationContext]:
        return [
            ConversationContext(
                messages=list(state.messages),
            )
        ]

    def is_available(self, state: AgentState) -> bool:
        return bool(state.messages)