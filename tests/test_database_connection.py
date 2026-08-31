import asyncio

from gaia_agent.database.connection import Database


async def main() -> None:
    print("Creating database...")

    database = Database()

    try:
        print("Connecting...")
        await database.connect()
        print("Connected!")

        result = await database.health_check()
        print(f"Database connected: {result}")

    finally:
        print("Closing...")
        await database.close()
        print("Closed!")


# التصحيح هنا: أضفنا الشرطات السفلية __ للـ main
if __name__ == "__main__":
    asyncio.run(main())
