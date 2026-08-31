import asyncio

from gaia_agent.database.connection import Database
from gaia_agent.memory.models import (
    Memory,
    MemorySource,
    MemoryType,
)
from gaia_agent.memory.memory_service import (
    MemoryService,
)
from gaia_agent.memory.providers.postgres import (
    PostgresMemoryRepository,
)


async def main() -> None:

    print("Creating database...")

    database = Database()

    try:
        print("Connecting...")

        await database.connect()

        print("Database connected.")

        repository = PostgresMemoryRepository(
            database.pool
        )

        service = MemoryService(
            repository
        )

        print("Creating memory...")

        memory = Memory(
            user_id="test-user",
            content="User prefers concise technical explanations.",
            memory_type=MemoryType.PREFERENCE,
            source=MemorySource.USER,
            importance=0.9,
            confidence=1.0,
            metadata={
                "test": True,
                "category": "preference",
            },
        )

        created = await service.create_memory(
            memory
        )

        print(
            f"Memory created: "
            f"{created.memory_id}"
        )

        print("Getting memory...")

        retrieved = await service.get_memory(
            created.memory_id
        )

        assert retrieved is not None
        assert retrieved.content == memory.content

        print("Get passed.")

        print("Listing memories...")

        memories = await service.list_user_memories(
            "test-user"
        )

        assert len(memories) >= 1

        print("List passed.")

        print("Updating memory...")

        retrieved.content = (
            "User prefers concise technical explanations "
            "with practical examples."
        )

        updated = await service.update_memory(
            retrieved
        )

        assert "practical examples" in updated.content

        print("Update passed.")

        print("Deleting memory...")

        await service.delete_memory(
            created.memory_id
        )

        deleted = await service.get_memory(
            created.memory_id
        )

        assert deleted is None

        print("Delete passed.")

        print()
        print("Memory layer test passed!")

    finally:
        print("Closing database...")

        await database.close()

        print("Database closed.")


if __name__ == "__main__":
    asyncio.run(main())