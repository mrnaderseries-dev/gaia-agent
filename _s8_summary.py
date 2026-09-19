"""Session-8 evidence extractor for _gaia20_results.json records.

Usage: python _s8_summary.py <output_file> [index ...]
Writes UTF-8 (the shell's `>` redirect is UTF-16 on this host).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / "_gaia20_results.json").read_text(encoding="utf-8"))

output_path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "_s8_detail.txt")
wanted = sys.argv[2:] if len(sys.argv) > 2 else None

sys.stdout = io.TextIOWrapper(
    open(output_path, "wb"),
    encoding="utf-8",
    errors="replace",
)

for record in results:
    index = str(record.get("index"))
    if wanted and index not in wanted:
        continue

    state = record.get("state", {})
    print("=" * 90)
    print(
        f"Q{index} {record.get('task_id')} expected={record.get('expected')!r} "
        f"answer={state.get('final_answer')!r} correct={record.get('correct')} "
        f"verified={state.get('final_answer_verified')} elapsed={record.get('elapsed_s')}s "
        f"error={record.get('error')!r}"
    )
    print(
        f"  phase={state.get('phase')} termination={state.get('termination_reason')} "
        f"replan={state.get('replan_count')} retry={state.get('retry_count')} "
        f"iter={state.get('iteration')} verif_attempts={state.get('verification_attempts')} "
        f"fatal={state.get('fatal_error')}"
    )
    print(f"  transitions={state.get('transitions')}")
    for classification in record.get("classifications", []):
        print(
            f"  classify: intent={classification.get('intent')} "
            f"needs_external={classification.get('needs_external_info')} "
            f"recommended={classification.get('recommended_first_tool')} "
            f"forbidden={classification.get('forbidden_tools')}"
        )
    for plan in record.get("plans", []):
        described = plan.get("plan") or {}
        steps = described.get("steps")
        if steps is None:
            print(
                f"  plan[{plan.get('method')}]: type={described.get('type')} "
                f"repr={str(described.get('repr'))[:300]}"
            )
            continue
        rendered = [
            (
                step.get("step_type"),
                step.get("tool"),
                str(step.get("arguments"))[:160],
                step.get("is_final_answer"),
            )
            for step in steps
        ]
        print(f"  plan[{plan.get('method')}]: {rendered}")
    for execution in state.get("executions", []):
        print(
            f"  exec step={execution.get('step_id')} tool={execution.get('tool')} "
            f"success={execution.get('success')} blocked={execution.get('blocked')} "
            f"output={str(execution.get('output'))[:200]!r} "
            f"error={str(execution.get('error'))[:160]!r}"
        )
    for evidence in state.get("evidence", []):
        print(
            f"  evidence tool={evidence.get('tool')} succeeded={evidence.get('succeeded')} "
            f"result={str(evidence.get('result'))[:200]!r}"
        )
    for verification in record.get("verifications", []):
        print(
            f"  verify status={verification.get('status')} seconds={verification.get('seconds')} "
            f"candidate={verification.get('candidate_answer')!r} "
            f"crash={verification.get('crash')} reason={str(verification.get('reason'))[:220]!r}"
        )
        print(f"    raw_data={verification.get('raw_data')}")
    for call in record.get("llm_calls", []):
        print(
            f"  llm prompt_chars={call.get('prompt_chars')} schema={call.get('output_schema')} "
            f"seconds={call.get('seconds')} error={call.get('error')}"
        )
    print(f"  answer_context_chars={record.get('answer_context_chars')}")
    for context in record.get("answer_contexts", []):
        print(f"    context[{len(context)}]: {context[:1200]}")
