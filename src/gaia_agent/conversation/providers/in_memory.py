from __future__ import annotations

from uuid import UUID

from ..models import Conversation, Message
from ..repository import ConversationRepository


class InMemoryConversationRepository(ConversationRepository):
    

    def __init__(self) -> None:
        self._conversations: dict[
            UUID, Conversation
        ] = {}

    async def create(self,conversation: Conversation,) -> Conversation:
        if conversation.conversation_id in self._conversations:
            raise ValueError( f"Conversation '{conversation.conversation_id}'  already exists.")

        self._conversations[
            conversation.conversation_id
        ] = conversation

        return conversation

    async def get(self,conversation_id: UUID) -> Conversation | None:
        return self._conversations.get(
            conversation_id
        )

    async def list_by_user(self,user_id: str) -> list[Conversation]:
        return [
            conversation
            for conversation in self._conversations.values()
            if conversation.user_id == user_id
        ]

    async def add_message(self,conversation_id: UUID,message: Message) -> None:
        conversation = self._conversations.get(conversation_id)

        if conversation is None:
            raise ValueError(f"Conversation '{conversation_id}' does not exist.")

        conversation.add_message(message)

    async def update(self,conversation: Conversation) -> Conversation:
        conversation_id = conversation.conversation_id

        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation f'{conversation_id}' does not exist.")

        self._conversations[conversation_id] = (conversation)

        return conversation

    async def delete(self,conversation_id: UUID) -> None:
        if conversation_id not in self._conversations:
            raise ValueError(
                f"Conversation "
                f"'{conversation_id}' "
                "does not exist."
            )

        del self._conversations[conversation_id]