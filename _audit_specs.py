from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.main import create_agent


async def main() -> None:
    agent = await create_agent()
    orchestrator = agent.orchestrator
    execution = orchestrator.agent_execution
    registry = execution.tool_registry
    specs = registry.get_tool_specs()
    print("tool_count:", len(specs))
    for spec in specs:
        print(
            json.dumps(
                {
                    "name": spec.name,
                    "description": (getattr(spec, "description", "") or "")[:120],
                    "parameters": list(
                        (getattr(spec, "parameters", None) or {}).keys()
                        if isinstance(getattr(spec, "parameters", None), dict)
                        else []
                    ),
                    "risk": str(getattr(spec, "risk_level", None)),
                },
                default=str,
            )
        )

    for name in (
        "web_search",
        "read_file",
        "python_interpreter",
        "vision_analysis",
        "analyze_image",
        "read_spreadsheet",
        "transcribe_audio",
    ):
        tool = registry.get_tool(name) if hasattr(registry, "get_tool") else None
        print("resolve", name, "->", type(tool).__name__ if tool else None)


asyncio.run(main())