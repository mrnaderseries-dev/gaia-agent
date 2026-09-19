"""Contract tests for the official GAIA evaluation path (Agents Course Unit 4).

The live scoring API contract was verified with read-only GET probes
(analysis.md "Current Technical Diagnosis" section 5):

- GET /questions           -> list of questions (Level/file_name/question/task_id)
- GET /files/{task_id}     -> attachment bytes (may 404)
- POST /submit             -> Submission{username, agent_code, answers}
  with answers as AnswerItem{task_id, submitted_answer (str|int|float)}
  -> ScoreResponse{username, score, correct_count, total_attempted, message,
  timestamp}

These tests pin the AnswerItem row shape, the client behavior against a mocked
transport, and the sequential error-containing evaluation loop. No test ever
calls the real /submit endpoint.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from gaia_agent.evaluation.evaluator import evaluate_questions
from gaia_agent.evaluation.gaia import (
    GaiaEvaluationClient,
    GaiaEvaluationError,
    PACKAGED_EVALUATION_FILES_DIR,
    load_official_questions,
    stage_attachments,
)
from gaia_agent.evaluation.submission import (
    build_course_rows,
    build_course_submission,
)


# ---------------------------------------------------------------------------
# AnswerItem rows (POST /submit answers)
# ---------------------------------------------------------------------------


def test_course_rows_use_the_official_answeritem_fields() -> None:
    rows = build_course_rows(
        [
            {"task_id": "t-1", "answer": "4"},
            {"task_id": "t-2", "answer": "FunkMonk"},
        ]
    )

    assert rows == [
        {"task_id": "t-1", "submitted_answer": "4"},
        {"task_id": "t-2", "submitted_answer": "FunkMonk"},
    ]


def test_course_rows_keep_numbers_and_stringify_booleans() -> None:
    rows = build_course_rows(
        [
            {"task_id": "t-1", "answer": 42},
            {"task_id": "t-2", "answer": 3.5},
            {"task_id": "t-3", "answer": True},
        ]
    )

    assert rows[0]["submitted_answer"] == 42
    assert rows[1]["submitted_answer"] == 3.5
    assert rows[2]["submitted_answer"] == "True"
    assert isinstance(rows[0]["submitted_answer"], int)


def test_course_rows_keep_unanswered_questions_with_empty_answer() -> None:
    rows = build_course_rows(
        [
            {"task_id": "t-1", "answer": None},
            {"task_id": "t-2", "answer": ""},
        ]
    )

    assert [row["submitted_answer"] for row in rows] == ["", ""]


def test_course_rows_require_a_task_id() -> None:
    with pytest.raises(ValueError):
        build_course_rows([{"answer": "4"}])


def test_course_submission_payload_matches_the_submission_schema() -> None:
    payload = build_course_submission(
        username="student",
        agent_code="https://huggingface.co/spaces/student/agent/tree/main",
        results=[{"task_id": "t-1", "answer": "4"}],
    )

    assert set(payload) == {"username", "agent_code", "answers"}
    assert payload["answers"] == [{"task_id": "t-1", "submitted_answer": "4"}]


def test_course_submission_requires_explicit_credentials() -> None:
    with pytest.raises(ValueError):
        build_course_submission(
            username="",
            agent_code="",
            results=[{"task_id": "t-1", "answer": "4"}],
        )


# ---------------------------------------------------------------------------
# Scoring API client (mocked transport — never the real /submit)
# ---------------------------------------------------------------------------


def _client(handler) -> GaiaEvaluationClient:
    return GaiaEvaluationClient(
        base_url="https://scoring.test",
        transport=httpx.MockTransport(handler),
    )


def test_fetch_questions_returns_the_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/questions"
        return httpx.Response(
            200,
            json=[{"task_id": "t-1", "Level": "1", "question": "q?"}],
        )

    client = _client(handler)
    questions = asyncio.run(client.fetch_questions())

    assert questions == [{"task_id": "t-1", "Level": "1", "question": "q?"}]


def test_fetch_questions_rejects_an_invalid_payload() -> None:
    client = _client(lambda request: httpx.Response(200, json={"nope": 1}))

    with pytest.raises(GaiaEvaluationError):
        asyncio.run(client.fetch_questions())


def test_fetch_question_file_stages_the_download(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/files/t-1"
        return httpx.Response(
            200,
            content=b"attachment-bytes",
            headers={
                "content-disposition": 'attachment; filename="notes.xlsx"'
            },
        )

    client = _client(handler)
    path = asyncio.run(client.fetch_question_file("t-1", tmp_path))

    assert path is not None
    assert path.name == "notes.xlsx"
    assert path.read_bytes() == b"attachment-bytes"


def test_fetch_question_file_returns_none_on_404(tmp_path) -> None:
    client = _client(lambda request: httpx.Response(404))
    path = asyncio.run(client.fetch_question_file("t-1", tmp_path))

    assert path is None


def test_stage_attachments_augments_every_row(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files/t-1":
            return httpx.Response(200, content=b"data")
        return httpx.Response(404)

    client = _client(handler)
    rows = asyncio.run(
        stage_attachments(
            client,
            [
                {"task_id": "t-1", "file_name": "a.xlsx"},
                {"task_id": "t-2", "file_name": None},
                {"task_id": "t-3", "file_name": "missing.xlsx"},
            ],
            tmp_path,
        )
    )

    assert rows[0]["file_path"] is not None
    assert rows[1]["file_path"] is None
    assert rows[2]["file_path"] is None


# ---------------------------------------------------------------------------
# Attachment staging regression (observed 2026-09-19)
#
# GET /files/{task_id} answered 404 for every task while the default
# local_fallback_dir ("evaluation_files", CWD-relative) resolved to an EMPTY
# directory: the official run then executed all five attachment questions
# with file_path=None (questions_snapshot.json evidence). These tests lock
# the fixed contract.
# ---------------------------------------------------------------------------


def test_load_official_questions_stages_attachments_from_the_packaged_fallback(
    tmp_path,
) -> None:
    if not PACKAGED_EVALUATION_FILES_DIR.is_dir():
        pytest.skip("packaged evaluation_files not present in this checkout")

    def handler(request: httpx.Request) -> httpx.Response:
        # Live evidence 2026-09-18/19: /questions works, /files/{task_id} 404s.
        if request.url.path == "/questions":
            return httpx.Response(
                200,
                json=[
                    {
                        "task_id": "7bd855d8-463d-4ed5-93ca-5fe35145f733",
                        "Level": "1",
                        "question": "Total food sales?",
                        "file_name": (
                            "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
                        ),
                    }
                ],
            )
        return httpx.Response(404)

    client = _client(handler)

    rows = asyncio.run(
        load_official_questions(
            tmp_path,
            client=client,
        )
    )

    assert len(rows) == 1
    file_path = rows[0]["file_path"]
    assert file_path is not None, (
        "attachment must be staged from the packaged fallback"
    )
    assert Path(file_path).is_file()
    assert Path(file_path).name == (
        "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
    )


def test_load_official_questions_can_disable_the_fallback(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/questions":
            return httpx.Response(
                200,
                json=[
                    {
                        "task_id": "t-1",
                        "Level": "1",
                        "question": "q?",
                        "file_name": "missing.xlsx",
                    }
                ],
            )
        return httpx.Response(404)

    client = _client(handler)

    rows = asyncio.run(
        load_official_questions(
            tmp_path,
            client=client,
            local_fallback_dir=None,
        )
    )

    assert rows[0]["file_path"] is None


def test_submit_posts_the_submission_and_returns_the_score_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "username": "student",
                "score": 0.35,
                "correct_count": 7,
                "total_attempted": 20,
                "message": "submitted",
                "timestamp": "2026-09-18T00:00:00",
            },
        )

    client = _client(handler)
    response = asyncio.run(
        client.submit(
            username="student",
            agent_code="code",
            answers=[{"task_id": "t-1", "submitted_answer": "4"}],
        )
    )

    assert captured["path"] == "/submit"
    assert captured["payload"]["username"] == "student"
    assert response["score"] == 0.35
    assert response["total_attempted"] == 20


def test_submit_rejects_missing_credentials_and_incomplete_scores() -> None:
    client = _client(lambda request: httpx.Response(200, json={"score": 1.0}))

    with pytest.raises(ValueError):
        asyncio.run(
            client.submit(username="", agent_code="code", answers=[{"a": 1}])
        )

    with pytest.raises(GaiaEvaluationError):
        asyncio.run(
            client.submit(
                username="student",
                agent_code="code",
                answers=[{"task_id": "t-1", "submitted_answer": "4"}],
            )
        )


# ---------------------------------------------------------------------------
# Evaluation loop (fake agent — no LLM, no network)
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Sequential run recorder: one shared instance proves agent reuse."""

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.run_requests: list[str] = []
        self.fail_on = fail_on or set()

    async def run(self, state) -> None:
        self.run_requests.append(state.user_request)
        if state.user_request in self.fail_on:
            raise RuntimeError("boom")
        if state.user_request.startswith("SLOW"):
            await asyncio.sleep(10)
        state.final_answer = "468"
        state.final_answer_verified = True


def test_evaluate_questions_runs_sequentially_and_collects_answers() -> None:
    agent = _FakeAgent()
    questions = [
        {"task_id": "t-1", "question": "q1", "Level": "1"},
        {"task_id": "t-2", "question": "q2", "Level": "1"},
    ]

    results = asyncio.run(evaluate_questions(questions, agent=agent))

    assert agent.run_requests == ["q1", "q2"]
    assert [r["answer"] for r in results] == ["468", "468"]
    assert all(r["error"] is None for r in results)
    assert all(r["verified"] for r in results)


def test_evaluate_questions_contains_failures_without_stopping() -> None:
    agent = _FakeAgent(fail_on={"q1"})
    questions = [
        {"task_id": "t-1", "question": "q1", "Level": "1"},
        {"task_id": "t-2", "question": "q2", "Level": "1"},
    ]

    results = asyncio.run(evaluate_questions(questions, agent=agent))

    assert results[0]["error"] is not None
    assert results[0]["answer"] is None
    assert results[1]["error"] is None
    assert results[1]["answer"] == "468"


def test_evaluate_questions_reports_timeout_and_salvages_state() -> None:
    agent = _FakeAgent()
    questions = [{"task_id": "t-1", "question": "SLOW question", "Level": "1"}]

    results = asyncio.run(
        evaluate_questions(questions, agent=agent, timeout_per_question=0.05)
    )

    assert results[0]["error"] is not None
    assert "TIMEOUT" in results[0]["error"]


def test_evaluate_questions_stages_the_attachment_path() -> None:
    agent = _FakeAgent()
    questions = [
        {
            "task_id": "t-1",
            "question": "q?",
            "Level": "1",
            "file_path": "files/notes.xlsx",
            "file_name": "notes.xlsx",
        }
    ]

    results = asyncio.run(evaluate_questions(questions, agent=agent))

    assert results[0]["file_name"] == "notes.xlsx"
