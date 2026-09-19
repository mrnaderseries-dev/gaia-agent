"""Ollama request-timeout contract.

Real GAIA runtime evidence (2026-09-17 diagnostic): planner/replan/verifier
calls on ~10k-character prompts repeatedly hit `httpx.ReadTimeout` at ~121 s
because `OllamaClient` defaulted to a hardcoded 120 s timeout, so the whole
question budget was consumed by a doomed request. The timeout is now a
`Settings` value wired through the composition root.
"""
from __future__ import annotations

import pytest

from gaia_agent.config import settings
from gaia_agent.llm.provider.ollama import OllamaClient


def test_ollama_client_default_timeout_unchanged() -> None:
    assert OllamaClient().timeout == 120.0


def test_ollama_client_honours_explicit_timeout() -> None:
    assert OllamaClient(timeout=240.0).timeout == 240.0
    assert OllamaClient(base_url="http://localhost:11434", timeout=300.0).timeout == 300.0


def test_settings_exposes_configurable_ollama_timeout() -> None:
    assert isinstance(settings.ollama_timeout, float)
    # Must exceed the old hardcoded value, otherwise the fix is a no-op.
    assert settings.ollama_timeout > 120.0


@pytest.mark.asyncio
async def test_composition_root_wires_configured_timeout(monkeypatch) -> None:
    import gaia_agent.main as main_mod

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(main_mod, "OllamaClient", _FakeClient)

    await main_mod.create_agent()

    assert captured.get("timeout") == settings.ollama_timeout
    assert captured.get("base_url") == main_mod.OLLAMA_BASE_URL
