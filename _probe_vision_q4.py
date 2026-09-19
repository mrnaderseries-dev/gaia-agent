"""Direct vision probe on the staged GAIA Q4 chess image (diagnostic).

Mirrors the production path exactly: AnalyzeImageTool's prompt text +
LLMService.generate_image (base64 user message) + VISION_MODEL params.
"""
import sys

sys.path.insert(0, "src")

import asyncio
from pathlib import Path

from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.llm.service import LLMService

IMAGE = Path(
    "src/gaia_agent/evaluation_files/"
    "cca530fc-4052-43b2-b130-b30968d8aa44/"
    "cca530fc-4052-43b2-b130-b30968d8aa44.png"
)

QUESTION = (
    "Review the chess position provided in the image. It is black's turn. "
    "Provide the correct next move for black which guarantees a win. "
    "Please provide your response in algebraic notation."
)

PRODUCTION_PROMPT = (
    "You are solving a GAIA benchmark task using a visual input.\n\n"
    "Analyze the provided image carefully.\n"
    "Use ONLY information that is actually visible in the image.\n"
    "Do not invent missing information.\n"
    "If the question requires reading text, numbers, labels, a chart, "
    "a chess position, or a diagram, inspect the image carefully before "
    "answering.\n\n"
    f"Question:\n{QUESTION}\n\n"
    "Return the most precise answer possible."
)


async def probe(model_name: str) -> None:
    client = OllamaClient(timeout=300.0)
    model = LLMModel(
        provider="ollama",
        model=model_name,
        max_tokens=1024,
        temperature=0.0,
    )
    service = LLMService(client=client, model=model)
    started = asyncio.get_event_loop().time()
    try:
        answer = await service.generate_image(
            image_path=IMAGE,
            question=PRODUCTION_PROMPT,
            operation="llm.vision",
        )
        elapsed = asyncio.get_event_loop().time() - started
        print(f"[{model_name}] seconds={elapsed:.1f} len={len(answer)}")
        print(f"[{model_name}] answer={str(answer)[:400]!r}")
    except Exception as exc:  # noqa: BLE001
        elapsed = asyncio.get_event_loop().time() - started
        print(
            f"[{model_name}] FAIL after {elapsed:.1f}s: "
            f"{type(exc).__name__}: {exc}"
        )


async def main() -> None:
    print("image exists:", IMAGE.exists(), "bytes:", IMAGE.stat().st_size)
    await probe("moondream")
    await probe("qwen2.5vl:3b")


asyncio.run(main())
