"""Build the official GAIA Unit-4 20-question diagnostic dataset.

Sources (both are official, no invention):
  * Questions: HF Agents-Course Unit 4 scoring API  -> /questions
  * Expected answers: cached official GAIA 2023_level1 validation split
    (huggingface datasets cache, Arrow file, includes "Final answer")

Output: _gaia20_official.json  (task_id, question, level, file_name,
        file_path, expected) + staged attachments under evaluation_files/.

The expected answers are used ONLY for offline scoring; they are never
placed into AgentState.user_request or any agent-visible context.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.ipc as ipc
import requests

ROOT = Path(__file__).resolve().parent
API_URL = "https://agents-course-unit4-scoring.hf.space"
OUT = ROOT / "_gaia20_official.json"
STAGE_DIR = ROOT / "src" / "gaia_agent" / "evaluation_files"
ARROW = (
    Path(os.path.expanduser("~"))
    / ".cache/huggingface/datasets/gaia-benchmark___gaia/2023_level1"
    / "0.0.0/682dd723ee1e1697e00360edccf2366dc8418dd9/gaia-validation.arrow"
)


def load_local_answers() -> dict[str, dict]:
    if not ARROW.exists():
        return {}
    table = ipc.open_stream(str(ARROW)).read_all()
    data = table.to_pylist()
    return {row["task_id"]: row for row in data}


def _stage_bytes(task_id: str, file_name: str, content: bytes) -> str:
    dest_dir = STAGE_DIR / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(file_name).name
    dest.write_bytes(content)
    print(f"[attach] {task_id}: saved {dest.name} ({len(content)} bytes)")
    return str(dest.resolve())


def stage_attachment(task_id: str, file_name: str, gaia_path: str = "") -> str | None:
    if not file_name:
        return None
    existing = list((STAGE_DIR / task_id).glob("*")) if (STAGE_DIR / task_id).exists() else []
    if existing:
        return str(existing[0].resolve())
    # Preferred source: official HF GAIA dataset repo (gated, token in HF cache).
    if gaia_path:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                "gaia-benchmark/GAIA", gaia_path, repo_type="dataset"
            )
            return _stage_bytes(task_id, file_name, Path(local).read_bytes())
        except Exception as exc:  # noqa: BLE001
            print(f"[attach] {task_id}: hub download failed: {type(exc).__name__}: {exc}")
    # Fallback: Unit-4 scoring API.
    try:
        r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"[attach] {task_id}: {type(exc).__name__}: {exc}")
        return None
    if r.status_code != 200 or len(r.content) < 10:
        print(f"[attach] {task_id}: API HTTP {r.status_code}")
        return None
    return _stage_bytes(task_id, file_name, r.content)


def main() -> None:
    questions = requests.get(f"{API_URL}/questions", timeout=60).json()
    local = load_local_answers()

    rows = []
    missing = []
    for q in questions:
        task_id = q["task_id"]
        ref = local.get(task_id, {})
        expected = ref.get("Final answer")
        if expected is None:
            missing.append(task_id)
        file_name = q.get("file_name") or ref.get("file_name") or ""
        path = stage_attachment(task_id, file_name, ref.get("file_path") or "")
        rows.append(
            {
                "task_id": task_id,
                "question": q["question"],
                "level": q.get("Level"),
                "file_name": file_name,
                "file_path": path,
                "expected": expected,
                "file_path_gaia": ref.get("file_path"),
            }
        )

    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {OUT} rows={len(rows)}")
    print(f"[ok] answers resolved: {sum(1 for r in rows if r['expected'] is not None)}/{len(rows)}")
    if missing:
        print("[warn] no expected answer for:", missing)
    print(f"[ok] attachments staged: {sum(1 for r in rows if r['file_path'])}/{sum(1 for r in rows if r['file_name'])}")


if __name__ == "__main__":
    main()
