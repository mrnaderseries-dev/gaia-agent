import sys
import asyncio
import traceback

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

try:
    from gaia_agent.main import create_agent

    agent = asyncio.run(create_agent())

    print("CREATE_AGENT_OK", type(agent).__name__)

except BaseException:
    traceback.print_exc()
    print("CREATE_AGENT_FAILED")
