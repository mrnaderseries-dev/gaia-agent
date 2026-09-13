import sys
sys.path.insert(0, "src")

import asyncio
from gaia_agent.main import create_agent


async def main() -> None:
    try:
        agent = await create_agent()
        print("CREATE_AGENT_OK", type(agent).__name__)
    except Exception as exc:
        print("CREATE_AGENT_FAILED", type(exc).__name__, str(exc)[:300])


asyncio.run(main())