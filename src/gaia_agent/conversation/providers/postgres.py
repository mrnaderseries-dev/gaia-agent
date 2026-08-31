from __future__ import annotations
import json

from datetime import datetime, timezone
from uuid import UUID

import asyncpg

from ..models import Conversation, Message, MessageRole
from ..repository import ConversationRepository


class PostgresConversationRepository(
    ConversationRepository
):
    """
    PostgreSQL implementation of ConversationRepository.

    Responsible only for persistence.
    Business logic belongs to ConversationService.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
    ) -> None:
        self.pool = pool

    async def create(
        self,
        conversation: Conversation,
    ) -> Conversation:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id,
                    user_id,
                    created_at,
                    updated_at,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5)
                """,
                conversation.conversation_id,
                conversation.user_id,
                conversation.created_at,
                conversation.updated_at,
                json.dumps(conversation.metadata),
            )

        return conversation

    async def get(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:

        async with self.pool.acquire() as connection:

            row = await connection.fetchrow(
                """
                SELECT
                    conversation_id,
                    user_id,
                    created_at,
                    updated_at,
                    metadata
                FROM conversations
                WHERE conversation_id = $1
                """,
                conversation_id,
            )

            if row is None:
                return None

            message_rows = await connection.fetch(
                """
                SELECT
                    message_id,
                    role,
                    content,
                    created_at,
                    metadata
                FROM messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                """,
                conversation_id,
            )

        messages = [
            Message(
                message_id=message_row["message_id"],
                role=MessageRole(
                    message_row["role"]
                ),
                content=message_row["content"],
                created_at=message_row["created_at"],
                metadata=(
                    message_row["metadata"] or {}
                ),
            )
            for message_row in message_rows
        ]

        return Conversation(
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            messages=messages,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=row["metadata"] or {},
        )

    async def list_by_user(
        self,
        user_id: str,
    ) -> list[Conversation]:

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT
                    conversation_id,
                    user_id,
                    created_at,
                    updated_at,
                    metadata
                FROM conversations
                WHERE user_id = $1
                ORDER BY updated_at DESC
                """,
                user_id,
            )

        return [
            Conversation(
                user_id=row["user_id"],
                conversation_id=row[
                    "conversation_id"
                ],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]
    async def add_message(
        self,
        conversation_id: UUID,
        message: Message,
    ) -> None:

        async with self.pool.acquire() as connection:

            async with connection.transaction():

                exists = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM conversations
                        WHERE conversation_id = $1
                    )
                    """,
                    conversation_id,
                )

                if not exists:
                    raise ValueError(
                        f"Conversation "
                        f"'{conversation_id}' "
                        "does not exist."
                    )

                await connection.execute(
                    """
                    INSERT INTO messages (
                        message_id,
                        conversation_id,
                        role,
                        content,
                        created_at,
                        metadata
                    )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6
                    )
                    """,
                    message.message_id,
                    conversation_id,
                    message.role.value,
                    message.content,
                    message.created_at,
                    json.dumps(message.metadata),
                )

                await connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = $1
                    WHERE conversation_id = $2
                    """,
                    datetime.now(timezone.utc),
                    conversation_id,
                )

    async def get_messages(
        self,
        conversation_id: UUID,
    ) -> list[Message]:

        async with self.pool.acquire() as connection:

            rows = await connection.fetch(
                """
                SELECT
                    message_id,
                    role,
                    content,
                    created_at,
                    metadata
                FROM messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                """,
                conversation_id,
            )

        return [
            Message(
                message_id=row["message_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                created_at=row["created_at"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    async def update(
        self,
        conversation: Conversation,
    ) -> Conversation:

        async with self.pool.acquire() as connection:

            result = await connection.execute(
                """
                UPDATE conversations
                SET
                    user_id = $1,
                    updated_at = $2,
                    metadata = $3
                WHERE conversation_id = $4
                """,
                conversation.user_id,
                conversation.updated_at,
                json.dumps(conversation.metadata),
                conversation.conversation_id,
            )

        if result == "UPDATE 0":
            raise ValueError(
                f"Conversation "
                f"'{conversation.conversation_id}' "
                "does not exist."
            )

        return conversation

    async def delete(
        self,
        conversation_id: UUID,
    ) -> None:

        async with self.pool.acquire() as connection:

            result = await connection.execute(
                """
                DELETE FROM conversations
                WHERE conversation_id = $1
                """,
                conversation_id,
            )
            if result == "DELETE 0":
              raise ValueError(
                f"Conversation "
                f"'{conversation_id}' "
                "does not exist."
            )