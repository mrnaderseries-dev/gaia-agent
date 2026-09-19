"""Session-8 HF evaluation readiness probe (no submission performed)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BASE = "https://agents-course-unit4-scoring.hf.space"


def fetch(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def main() -> None:
    spec = fetch("/openapi.json")
    schemas = spec["components"]["schemas"]
    print("[submit schema]", json.dumps(schemas.get("Submission"), indent=1))
    for name in ("QuestionAnswer", "Answer", "ScoreResponse"):
        if name in schemas:
            print(f"[{name}]", json.dumps(schemas[name], indent=1))

    questions = fetch("/questions")
    print("[questions] count =", len(questions))
    print("[questions] fields =", sorted(questions[0].keys()))

    official = json.loads((ROOT / "_gaia20_official.json").read_text(encoding="utf-8"))
    print("[local official set] rows =", len(official))
    print("[local official set] fields =", sorted(official[0].keys()))
    staged = [
        (row["task_id"][:8], bool(row.get("file_path")))
        for row in official
    ]
    print("[local official set] (task, staged) =", staged)


if __name__ == "__main__":
    main()
