import asyncio

from gaia_agent.database.connection import Database
from gaia_agent.conversation.models import (
    Message,
    MessageRole,
)
from gaia_agent.conversation.service import (
    ConversationService,
)
from gaia_agent.conversation.providers.postgres import (
    PostgresConversationRepository,
)
from gaia_agent.config import settings


async def main() -> None:

    print("Creating database...")

    database = Database()

    try:
        print("Connecting...")
        await database.connect()
        print("Database connected.")

        if database.pool is None:
            raise RuntimeError(
                "Database pool was not created."
            )

        repository = (
            PostgresConversationRepository(
                database.pool
            )
        )

        service = ConversationService(
            repository
        )

        # 1. Create conversation
        print("Creating conversation...")

        conversation = (
            await service.create_conversation(
                "test-user"
            )
        )

        print(
            "Conversation created:",
            conversation.conversation_id,
        )

        # 2. Add user message
        user_message = Message(
            role=MessageRole.USER,
            content="Hello GAIA",
        )

        await service.add_message(
            conversation.conversation_id,
            user_message,
        )

        print("User message added.")

        # 3. Add assistant message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content="Hello! How can I help you?",
        )

        await service.add_message(
            conversation.conversation_id,
            assistant_message,
        )

        print("Assistant message added.")

        # 4. Get conversation
        loaded = await service.get_conversation(
            conversation.conversation_id
        )

        if loaded is None:
            raise RuntimeError(
                "Conversation was not found."
            )

        print("Conversation retrieved.")

        # 5. Verify messages
        print(
            "Messages:",
            len(loaded.messages),
        )

        for message in loaded.messages:
            print(
                message.role.value,
                "->",
                message.content,
            )

        if len(loaded.messages) != 2:
            raise RuntimeError(
                "Expected exactly 2 messages."
            )

        # 6. List user conversations
        conversations = (
            await service.list_user_conversations(
                "test-user"
            )
        )

        print(
            "User conversations:",
            len(conversations),
        )

        # 7. Delete conversation
        await service.delete_conversation(
            conversation.conversation_id
        )

        print("Conversation deleted.")

        # 8. Verify deletion
        deleted = await service.get_conversation(
            conversation.conversation_id
        )

        if deleted is not None:
            raise RuntimeError(
                "Conversation still exists."
            )

        print(
            "Deletion verified."
        )

        print(
            "\nConversation Layer test PASSED."
        )

    finally:
        print("Closing database...")
        await database.close()
        print("Database closed.")


if __name__ == "__main__":
    asyncio.run(main())