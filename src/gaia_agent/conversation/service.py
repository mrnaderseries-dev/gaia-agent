from __future__ import annotations

from uuid import UUID

from .models import Conversation, Message
from .repository import ConversationRepository


class ConversationService:
    """
    Provides business operations for conversations.
    """

    def __init__(
        self,
        repository: ConversationRepository,
    ) -> None:
        self.repository = repository

    async def create_conversation(
        self,
        user_id: str,
    ) -> Conversation:
        """Create a new empty conversation."""

        if not user_id.strip():
            raise ValueError(
                "user_id cannot be empty."
            )

        conversation = Conversation(
            user_id=user_id
        )

        return await self.repository.create(
            conversation
        )

    async def get_conversation(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:
        """Retrieve a conversation by ID."""

        return await self.repository.get(
            conversation_id
        )

    async def list_user_conversations(
        self,
        user_id: str,
    ) -> list[Conversation]:
        """Retrieve all conversations belonging to a user."""

        if not user_id.strip():
            raise ValueError(
                "user_id cannot be empty."
            )

        return await self.repository.list_by_user(
            user_id
        )

    async def add_message(
        self,
        conversation_id: UUID,
        message: Message,
    ) -> None:
        """Add a message to an existing conversation."""

        conversation = await self.repository.get(
            conversation_id
        )

        if conversation is None:
            raise ValueError(
                f"Conversation '{conversation_id}' "
                "does not exist."
            )

        await self.repository.add_message(
            conversation_id,
            message,
        )

    async def get_messages(
        self,
        conversation_id: UUID,
    ) -> list[Message]:
        """Retrieve all messages of a conversation."""

        conversation = await self.repository.get(
            conversation_id
        )

        if conversation is None:
            raise ValueError(
                f"Conversation '{conversation_id}' "
                "does not exist."
            )

        return await self.repository.get_messages(
            conversation_id
        )

    async def update_conversation(
        self,
        conversation: Conversation,
    ) -> Conversation:
        """Update an existing conversation."""

        existing = await self.repository.get(
            conversation.conversation_id
        )

        if existing is None:
            raise ValueError(
                f"Conversation "
                f"'{conversation.conversation_id}' "
                "does not exist."
            )

        return await self.repository.update(
            conversation
        )

    async def delete_conversation(
        self,
        conversation_id: UUID,
    ) -> None:
        """Delete an existing conversation."""

        existing = await self.repository.get(
            conversation_id
        )

        if existing is None:
            raise ValueError(
                f"Conversation '{conversation_id}' "
                "does not exist."
            )

        await self.repository.delete(
            conversation_id
        )