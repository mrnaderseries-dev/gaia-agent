"""Session-8 environment/runtime readiness probe (real runtime, no mocks)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gaia_agent.config import settings  # noqa: E402
from gaia_agent.main import VISION_MODEL, TEXT_MODEL, create_agent  # noqa: E402
import gaia_agent.main as main_mod  # noqa: E402


async def main() -> None:
    agent = await create_agent()
    registry = agent.orchestrator.planner.__dict__.get("available_tools", {})
    print("[tools]", sorted(registry))
    print("[timeout]", settings.ollama_timeout)
    print("[text_model]", TEXT_MODEL.model, "max_tokens", TEXT_MODEL.max_tokens)
    print("[vision_model]", VISION_MODEL.model, "max_tokens", VISION_MODEL.max_tokens)

    try:
        import youtube_transcript_api  # noqa: F401

        print("[youtube_transcript_api] importable=True")
    except Exception as exc:  # noqa: BLE001
        print("[youtube_transcript_api] importable=False", type(exc).__name__)

    try:
        from smolagents import YoutubeTranscriptTool  # noqa: F401

        print("[smolagents YoutubeTranscriptTool] available=True")
    except Exception as exc:  # noqa: BLE001
        print("[smolagents YoutubeTranscriptTool] available=False", type(exc).__name__)

    client = getattr(agent.orchestrator, "llm_client", None)
    print("[orchestrator]", type(agent.orchestrator).__name__)
    print("[main_module_ollama_base]", main_mod.OLLAMA_BASE_URL)


asyncio.run(main())
