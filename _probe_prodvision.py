"""Production analyze_image control: pipeline sanity vs model limitation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gaia_agent.llm.provider.ollama import OllamaClient  # noqa: E402
from gaia_agent.llm.service import LLMService  # noqa: E402
from gaia_agent.main import OLLAMA_BASE_URL, VISION_MODEL  # noqa: E402
from gaia_agent.tools.vision import AnalyzeImageTool  # noqa: E402

client = OllamaClient(base_url=OLLAMA_BASE_URL, timeout=300.0)
service = LLMService(client=client, model=VISION_MODEL)
tool = AnalyzeImageTool(llm_service=service, base_dir=str(ROOT / "src"))

q4 = (
    ROOT
    / "src/gaia_agent/evaluation_files/"
    "cca530fc-4052-43b2-b130-b30968d8aa44/"
    "cca530fc-4052-43b2-b130-b30968d8aa44.png"
)

t0 = time.perf_counter()
out = tool(
    image_path=str(q4),
    question="Describe what you see in this image in one short sentence.",
)
print(
    f"production analyze_image (describe) in "
    f"{time.perf_counter() - t0:.1f}s -> {out!r}"
)
