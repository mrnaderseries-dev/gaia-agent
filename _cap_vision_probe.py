import asyncio

from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.llm.service import LLMService
from gaia_agent.llm.model import LLMModel


async def main() -> None:
    client = OllamaClient(timeout=300.0)
    service = LLMService(
        client=client,
        model=LLMModel(
            provider="ollama",
            model="moondream",
            max_tokens=128,
            temperature=0.0,
        ),
    )
    for image in ("_cap_vision_big.png", "_cap_vision_test.png"):
        try:
            answer = await service.generate_image(
                image_path=image,
                question=(
                    "Describe exactly what text is visible in this image. "
                    "Transcribe every word and number you can read."
                ),
            )
            print(image, "=>", repr(answer[:300]))
        except Exception as exc:
            print(image, "FAILED:", type(exc).__name__, str(exc)[:300])


asyncio.run(main())
