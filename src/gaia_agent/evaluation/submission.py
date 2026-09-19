from __future__ import annotations

"""Submission writer for the official GAIA evaluation contract.

The official GAIA leaderboard (huggingface.co/spaces/gaia-benchmark/leaderboard,
see its content.py SUBMISSION_TEXT and scorer.py::question_scorer) expects a
JSON-Lines file with one object per question and these two mandatory fields:

    {"task_id": "<task id>", "model_answer": "<final answer>"}

`reasoning_trace` is optional. Scoring is a quasi exact match performed by
question_scorer(model_answer, ground_truth): numeric ground truths are compared
via float(model_answer) after stripping "$", "%" and ",", list ground truths are
split on ","/";", and string ground truths are compared after removing all
whitespace and punctuation and lowercasing. The answers produced here must
therefore already be minimal (the Orchestrator extracts the FINAL ANSWER value).
"""

import json
from pathlib import Path
from typing import Any, Iterable


def build_submission_rows(
    results: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Map internal evaluation results to official submission rows."""

    rows: list[dict[str, str]] = []

    for result in results:
        task_id = result.get("task_id")
        answer = result.get("answer")

        if task_id is None or answer is None:
            # The official file requires both fields; a question the agent
            # could not answer is emitted with an empty model_answer rather
            # than being dropped, so the row count still matches the split.
            rows.append(
                {
                    "task_id": str(task_id),
                    "model_answer": "",
                }
            )
            continue

        row = {
            "task_id": str(task_id),
            "model_answer": str(answer),
        }

        reasoning = result.get("reasoning_trace")

        if reasoning:
            row["reasoning_trace"] = str(reasoning)

        rows.append(row)

    return rows


def write_submission(
    results: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write the official JSON-Lines submission file and return its path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for row in build_submission_rows(results):
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return output


def convert_results_file(
    results_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert an evaluation_results.jsonl log into a submission file."""

    source = Path(results_path)
    rows: list[dict[str, Any]] = []

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return write_submission(rows, output_path)


# ---------------------------------------------------------------------------
# Official Agents-Course Unit-4 scoring API rows (POST /submit)
#
# Live-verified contract (2026-09-18, see analysis.md): the API expects
# AnswerItem{task_id: str, submitted_answer: str|int|float}. This is a
# different contract from the leaderboard JSONL above; both writers are kept.
# The submitted value must be the bare extracted answer — never
# "FINAL ANSWER: ..." text.
# ---------------------------------------------------------------------------


def build_course_rows(
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map internal results to official AnswerItem rows for POST /submit.

    An unanswered question is emitted with an empty submitted_answer so the
    row count still matches the split (the API scores every submitted row).
    Non-string answers (int/float) are preserved as-is because the official
    AnswerItem schema accepts str, int and float.
    """
    rows: list[dict[str, Any]] = []

    for result in results:
        task_id = result.get("task_id")
        if task_id is None:
            raise ValueError(
                "Every course submission row requires a task_id."
            )

        answer = result.get("answer")
        row: dict[str, Any] = {"task_id": str(task_id)}
        if answer is None or answer == "":
            row["submitted_answer"] = ""
        elif isinstance(answer, bool):
            # The AnswerItem schema accepts str|int|float; never emit JSON
            # booleans (they would be rejected server-side).
            row["submitted_answer"] = str(answer)
        elif isinstance(answer, (int, float)):
            row["submitted_answer"] = answer
        else:
            row["submitted_answer"] = str(answer)

        rows.append(row)

    return rows


def build_course_submission(
    *,
    username: str,
    agent_code: str,
    results: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full POST /submit payload (Submission schema)."""
    if not username or not agent_code:
        raise ValueError(
            "username and agent_code are mandatory fields of the official "
            "Submission schema; supply them explicitly."
        )

    return {
        "username": username,
        "agent_code": agent_code,
        "answers": build_course_rows(results),
    }

