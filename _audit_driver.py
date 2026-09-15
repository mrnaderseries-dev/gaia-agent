from __future__ import annotations

"""Audit task driver: python _audit_driver.py <task_id> [<task_id> ...]"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import _audit_harness as harness

TASKS = json.loads((ROOT / "_audit_tasks.json").read_text(encoding="utf-8"))


async def _main() -> None:
    for task_id in sys.argv[1:]:
        spec = TASKS[task_id]
        print(f"AUDIT_TASK_START: {task_id} :: {spec['task']}", flush=True)
        harness.reset()
        payload = await harness.run(spec["task"], list(spec.get("attach", [])))
        out = ROOT / f"_audit_res_{task_id}.json"
        out.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        print(
            f"AUDIT_TASK_END: {task_id} :: answer={payload.get('final_answer')!r} "
            f"verified={payload.get('final_answer_verified')} "
            f"phase={payload.get('phase')} err={payload.get('tool_error')!r}",
            flush=True,
        )


asyncio.run(_main())