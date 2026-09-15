from __future__ import annotations

import importlib.util
import json
import sys

print("executable:", sys.executable)
print("version:", sys.version)

import gaia_agent

print("gaia_agent:", gaia_agent.__file__)

for name in (
    "datasets",
    "httpx",
    "ollama",
    "pydantic",
    "pytest",
    "pypdf",
    "openpyxl",
    "PIL",
    "numpy",
    "pandas",
    "faster_whisper",
    "whisper",
    "torch",
    "ddgs",
    "duckduckgo_search",
    "bs4",
    "markdownify",
    "requests",
):
    print(f"{name}: {importlib.util.find_spec(name) is not None}")

try:
    import httpx

    response = httpx.get("http://localhost:11434/api/tags", timeout=10.0)
    print("ollama_tags_status:", response.status_code)
    models = [m.get("name") for m in response.json().get("models", [])]
    print("ollama_models:", json.dumps(models))
except Exception as exc:  # noqa: BLE001
    print("ollama_error:", f"{type(exc).__name__}: {exc}")