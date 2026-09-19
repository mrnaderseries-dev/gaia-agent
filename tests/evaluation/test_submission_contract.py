"""Contract tests for the official GAIA submission writer.

The official GAIA submission contract (gaia-benchmark/leaderboard
`content.py::SUBMISSION_TEXT`) is a JSON-Lines file with one object per
question and the two mandatory fields `task_id` and `model_answer`
(`reasoning_trace` optional). These tests pin that shape, including the
empty-answer row, so the writer cannot silently drop questions or leak extra
fields into the submitted file.
"""
from __future__ import annotations

import json

from gaia_agent.evaluation.submission import (
    build_submission_rows,
    convert_results_file,
    write_submission,
)


def test_rows_use_the_official_field_names() -> None:
    rows = build_submission_rows(
        [
            {"task_id": "t-1", "answer": "4"},
            {"task_id": "t-2", "answer": "FunkMonk", "reasoning_trace": "web"},
        ]
    )

    assert rows == [
        {"task_id": "t-1", "model_answer": "4"},
        {"task_id": "t-2", "model_answer": "FunkMonk", "reasoning_trace": "web"},
    ]


def test_unanswered_question_is_kept_as_an_empty_row() -> None:
    rows = build_submission_rows(
        [
            {"task_id": "t-1", "answer": None},
            {"task_id": "t-2", "answer": ""},
            {"task_id": "t-3", "answer": "3"},
        ]
    )

    assert [row["task_id"] for row in rows] == ["t-1", "t-2", "t-3"]
    assert rows[0]["model_answer"] == ""
    assert rows[1]["model_answer"] == ""
    assert rows[2]["model_answer"] == "3"


def test_written_file_is_json_lines_with_mandatory_fields(tmp_path) -> None:
    output = write_submission(
        [
            {"task_id": "t-1", "answer": "4"},
            {"task_id": "t-2", "answer": None},
        ],
        tmp_path / "submission.jsonl",
    )

    lines = output.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2

    for line in lines:
        payload = json.loads(line)

        assert set(payload) <= {"task_id", "model_answer", "reasoning_trace"}
        assert "task_id" in payload
        assert "model_answer" in payload


def test_convert_results_file_round_trips_the_runner_log(tmp_path) -> None:
    source = tmp_path / "evaluation_results.jsonl"

    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"task_id": "t-1", "question": "q1", "answer": "468"},
                {"task_id": "t-2", "question": "q2", "answer": None},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    output = convert_results_file(source, tmp_path / "submission.jsonl")
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").strip().splitlines()
    ]

    assert rows == [
        {"task_id": "t-1", "model_answer": "468"},
        {"task_id": "t-2", "model_answer": ""},
    ]
