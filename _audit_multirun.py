from __future__ import annotations

"""Multi-question probe: reuse ONE agent (the run_evaluation.py pattern)."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\user\gaia-agent")
sys.path.insert(0, str(ROOT / "src"))

from gaia_agent.main import create_agent
from gaia_agent.core.agent_state import AgentState

TASKS = [
    "Calculate 2 + 2.",
    "Calculate 3 + 3.",
    "Calculate 5 + 5.",
]


async def main() -> None:
    agent = await create_agent()
    for index, task in enumerate(TASKS, start=1):
        state = AgentState(user_request=task)
        try:
            result = await agent.run(state)
            print(
                json.dumps(
                    {
                        "index": index,
                        "task": task,
                        "answer": result.final_answer,
                        "verified": result.final_answer_verified,
                        "phase": str(getattr(result.phase, "value", result.phase)),
                        "tool_error": result.tool_error,
                        "replans": result.replan_count,
                        "iterations": result.iteration,
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "index": index,
                        "task": task,
                        "crash": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )


asyncio.run(main())