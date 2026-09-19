"""Session-8 readiness check: convert the real GAIA run records into the
official submission rows using the production writer (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gaia_agent.evaluation.submission import (  # noqa: E402
    build_submission_rows,
    write_submission,
)


def main() -> None:
    results = json.loads((ROOT / "_gaia20_results.json").read_text(encoding="utf-8"))
    official = json.loads((ROOT / "_gaia20_official.json").read_text(encoding="utf-8"))

    answers = {
        record["task_id"]: (record.get("state") or {}).get("final_answer")
        for record in results
    }

    rows = build_submission_rows(
        [
            {"task_id": row["task_id"], "answer": answers.get(row["task_id"])}
            for row in official
        ]
    )

    print("[rows]", len(rows))
    for row in rows:
        print("   ", json.dumps(row, ensure_ascii=False))

    path = write_submission(
        [
            {"task_id": row["task_id"], "answer": answers.get(row["task_id"])}
            for row in official
        ],
        ROOT / "_s8_submission_preview.jsonl",
    )
    print("[written]", path, path.stat().st_size, "bytes")
    print("[covered task_ids]", sum(1 for row in rows if row["model_answer"] != ""))


if __name__ == "__main__":
    main()
