"""Session-7 runtime probes (real runtime, no mocks).

A. Effective Ollama timeout wiring + one real text LLM call latency.
B. ToolRegistry dump: is youtube_transcript actually registered?
C. smolagents YoutubeTranscriptTool import path (Q2/Q7 capability probe).
D. Production analyze_image path (moondream) on the REAL GAIA Q4 chess image.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gaia_agent.config import settings  # noqa: E402


async def main() -> None:
    print("=== A. timeout wiring + real text LLM call ===")
    print("settings.ollama_timeout =", settings.ollama_timeout)
    from gaia_agent.llm.provider.ollama import OllamaClient
    from gaia_agent.llm.service import LLMService
    from gaia_agent.main import OLLAMA_BASE_URL, TEXT_MODEL, VISION_MODEL

    # Same construction as main.py::create_agent (lines ~140-144).
    client = OllamaClient(
        base_url=OLLAMA_BASE_URL,
        timeout=settings.ollama_timeout,
    )
    print("client.timeout =", client.timeout)

    t0 = time.perf_counter()
    try:
        resp = await client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "What is 17*4? Reply with the number only.",
                }
            ],
            model=TEXT_MODEL,
            operation="probe.warmup",
        )
        print(
            f"text LLM call OK in {time.perf_counter() - t0:.1f}s -> {resp!r}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"text LLM call FAILED after {time.perf_counter() - t0:.1f}s: "
            f"{type(exc).__name__}: {exc}"
        )

    print()
    print("=== B. tool registry dump ===")
    text_service = LLMService(client=client, model=TEXT_MODEL)
    vision_service = LLMService(client=client, model=VISION_MODEL)
    from gaia_agent.tools.registry import ToolRegistry

    registry = ToolRegistry(
        base_dir=".",
        llm_service=text_service,
        vision_llm_service=vision_service,
    )
    names = sorted(registry._tools_by_name)
    print("registered tools:", names)
    print(
        "youtube_transcript registered:",
        "youtube_transcript" in names,
    )

    print()
    print("=== C. youtube transcript import path ===")
    try:
        from smolagents import YoutubeTranscriptTool  # type: ignore

        print("YoutubeTranscriptTool importable:", YoutubeTranscriptTool)
    except ImportError as exc:
        print("YoutubeTranscriptTool ImportError:", exc)

    print()
    print("=== D. production vision path on the REAL Q4 image ===")
    q4 = (
        ROOT
        / "src/gaia_agent/evaluation_files/"
        "cca530fc-4052-43b2-b130-b30968d8aa44/"
        "cca530fc-4052-43b2-b130-b30968d8aa44.png"
    )
    print("image exists:", q4.exists())
    if q4.exists():
        print("image bytes:", q4.stat().st_size)
        from gaia_agent.tools.vision import AnalyzeImageTool

        tool = AnalyzeImageTool(
            llm_service=vision_service,
            base_dir=str(ROOT / "src"),
        )
        t0 = time.perf_counter()
        out = tool(
            image_path=str(q4),
            question=(
                "It is black's turn in this chess position. "
                "Provide the correct next move for black which "
                "guarantees a win, in algebraic notation."
            ),
        )
        print(
            f"analyze_image in {time.perf_counter() - t0:.1f}s -> {out!r}"
        )


asyncio.run(main())
