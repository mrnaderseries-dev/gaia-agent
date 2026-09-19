"""Distinguish moondream wiring vs model limitation (no production change).

Raw /api/chat calls: does moondream EVER answer through this request shape,
on a trivial control image vs the real Q4 chess image?
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent


def ask(image: Path, question: str) -> str:
    b64 = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": "moondream",
        "messages": [
            {
                "role": "user",
                "content": question,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 768},
    }
    r = httpx.post(
        "http://localhost:11434/api/chat", json=payload, timeout=300.0
    )
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def main() -> None:
    chess = (
        ROOT
        / "src/gaia_agent/evaluation_files/"
        "cca530fc-4052-43b2-b130-b30968d8aa44/"
        "cca530fc-4052-43b2-b130-b30968d8aa44.png"
    )
    controls = [
        ROOT / "_cap_vision_shape.png",
        ROOT / "_cap_vision_test.png",
        ROOT / "_cap_vision_big.png",
        ROOT / "_audit_data/answer_big.png",
    ]
    ctrl = next((p for p in controls if p.exists()), None)
    print("control image:", ctrl)

    if ctrl is not None:
        q = "Describe this image in one sentence."
        try:
            print(f"[control] {q!r} -> {ask(ctrl, q)!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[control] FAILED: {type(exc).__name__}: {exc}")

    if chess.exists():
        for q in (
            "Describe this image in one sentence.",
            "What chess pieces are visible on the board?",
            "It is black's turn. What is the best move for black in "
            "algebraic notation?",
        ):
            try:
                print(f"[chess] {q!r} -> {ask(chess, q)!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"[chess] FAILED: {type(exc).__name__}: {exc}")


main()
