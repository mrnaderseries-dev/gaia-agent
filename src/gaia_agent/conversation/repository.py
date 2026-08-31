from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from .models import Conversation, Message


class ConversationRepository(ABC):
    """
    Persistence contract for conversations and their messages.

    Responsible only for storing and retrieving conversation data.
    It does not contain business logic.
    """

    @abstractmethod
    async def create(
        self,
        conversation: Conversation,
    ) -> Conversation:
        """Persist a new conversation."""
        raise NotImplementedError

    @abstractmethod
    async def get(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:
        """Retrieve a conversation by its ID."""
        raise NotImplementedError

    @abstractmethod
    async def update(
        self,
        conversation: Conversation,
    ) -> Conversation:
        """Update an existing conversation."""
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        conversation_id: UUID,
    ) -> None:
        """Delete a conversation."""
        raise NotImplementedError

    @abstractmethod
    async def list_by_user(
        self,
        user_id: str,
    ) -> list[Conversation]:
        """Retrieve all conversations belonging to a user."""
        raise NotImplementedError

    @abstractmethod
    async def add_message(
        self,
        conversation_id: UUID,
        message: Message,
    ) -> None:
        """Persist a message inside an existing conversation."""
        raise NotImplementedError

    @abstractmethod
    async def get_messages(
        self,
        conversation_id: UUID,
    ) -> list[Message]:
        """Retrieve all messages belonging to a conversation."""
        raise NotImplementedError