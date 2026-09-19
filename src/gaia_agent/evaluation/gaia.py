from __future__ import annotations

"""Official GAIA Level-1 evaluation client (Hugging Face Agents Course Unit 4).

Live-verified contract (2026-09-18, GET-only probe against the real scoring
API, see analysis.md "Current Technical Diagnosis" section 5):

- Base: https://agents-course-unit4-scoring.hf.space
- GET /questions          -> list of 20 questions
                            (fields: Level, file_name, question, task_id)
- GET /random-question    -> single question
- GET /files/{task_id}    -> the attachment for a task (may be absent)
- POST /submit            -> Submission{username, agent_code, answers}
                            where answers are AnswerItem{task_id,
                            submitted_answer (str|int|float)}
                            -> ScoreResponse{username, score, correct_count,
                            total_attempted, message, timestamp}

`agent_code` semantics: the API schema describes it as "The Python class code
for the agent" while the course hands-on shows the public Space `.../tree/main`
URL. The exact accepted value must be confirmed with ONE controlled submission
before scoring; therefore the client requires it as an explicit caller-supplied
parameter and never invents it.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_TIMEOUT = 60.0

# Already-staged copies of the official GAIA validation attachments live next
# to the package (src/gaia_agent/evaluation_files/<task_id>/<file_name>).
# Live evidence 2026-09-19: GET /files/{task_id} answers 404 ("No file path
# associated with task_id ...") for every task, and the previous CWD-relative
# default ("evaluation_files") resolved to an EMPTY directory, so all five
# attachment questions of the official run silently executed without their
# files (file_path=None in the run snapshot). The default fallback must
# therefore be the packaged ABSOLUTE directory, not a CWD-relative guess.
PACKAGED_EVALUATION_FILES_DIR = (
    Path(__file__).resolve().parents[1] / "evaluation_files"
)


class GaiaEvaluationError(RuntimeError):
    """Raised when the official evaluation API returns an unusable response."""


class GaiaEvaluationClient:
    """Read-only question/file access plus the controlled submit call."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    async def fetch_questions(self) -> list[dict[str, Any]]:
        """GET /questions — the full filtered evaluation set."""
        response = await self._request("GET", "/questions")
        questions = response.json()
        if not isinstance(questions, list) or not questions:
            raise GaiaEvaluationError(
                "GET /questions did not return a non-empty list."
            )
        return questions

    async def fetch_question_file(
        self,
        task_id: str,
        destination_dir: str | Path,
    ) -> Path | None:
        """GET /files/{task_id} — stage the attachment; None when absent."""
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
        ) as client:
            response = await client.get(
                f"{self.base_url}/files/{task_id}",
                follow_redirects=True,
            )

        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise GaiaEvaluationError(
                f"GET /files/{task_id} failed with HTTP "
                f"{response.status_code}."
            )

        content_disposition = response.headers.get("content-disposition", "")
        filename = _filename_from_disposition(content_disposition)
        target = destination / filename
        target.write_bytes(response.content)
        return target

    async def submit(
        self,
        *,
        username: str,
        agent_code: str,
        answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """POST /submit — controlled, caller-initiated; returns ScoreResponse.

        `answers` must already be official AnswerItem-shaped rows
        ({task_id, submitted_answer}) — build them with
        gaia_agent.evaluation.submission.build_course_rows.
        """
        if not username or not agent_code:
            raise ValueError(
                "username and agent_code are required by the official "
                "submission contract; they must be supplied explicitly."
            )
        if not isinstance(answers, list) or not answers:
            raise ValueError("answers must be a non-empty list of rows.")

        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
        ) as client:
            response = await client.post(
                f"{self.base_url}/submit",
                json={
                    "username": username,
                    "agent_code": agent_code,
                    "answers": answers,
                },
            )

        if response.status_code >= 400:
            raise GaiaEvaluationError(
                f"POST /submit failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        payload = response.json()
        required = {
            "username",
            "score",
            "correct_count",
            "total_attempted",
            "message",
            "timestamp",
        }
        missing = required - set(payload)
        if missing:
            raise GaiaEvaluationError(
                f"ScoreResponse is missing fields: {sorted(missing)}"
            )
        return payload

    async def _request(self, method: str, path: str) -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
        ) as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                follow_redirects=True,
            )
        if response.status_code >= 400:
            raise GaiaEvaluationError(
                f"{method} {path} failed with HTTP "
                f"{response.status_code}."
            )
        return response


def _filename_from_disposition(content_disposition: str) -> str:
    """Extract a filename from Content-Disposition; fall back to a safe name."""
    for part in content_disposition.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            filename = part.split("=", 1)[1].strip().strip('"')
            if filename:
                return filename
    return "attachment.bin"


async def stage_attachments(
    client: GaiaEvaluationClient,
    questions: list[dict[str, Any]],
    staging_dir: str | Path,
    *,
    local_fallback_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Download attachments for every question that declares one.

    Primary source is the official GET /files/{task_id} endpoint. If it
    returns 404 (live evidence 2026-09-18: the scoring API currently answers
    "No file path associated with task_id ..." for every task) and
    `local_fallback_dir` contains an already-staged copy of the official GAIA
    validation attachment (evaluation_files/<task_id>/<file_name>), that copy
    is used so the official 20-question run stays possible.

    Returns the questions augmented with a `file_path` key (None when no
    attachment is available).
    """
    staged: list[dict[str, Any]] = []
    for question in questions:
        row = dict(question)
        row["file_path"] = None
        file_name = row.get("file_name")
        if file_name:
            path = await client.fetch_question_file(
                str(row["task_id"]),
                staging_dir,
            )
            if path is None and local_fallback_dir is not None:
                path = _find_local_attachment(
                    local_fallback_dir,
                    str(row["task_id"]),
                    str(file_name),
                )
            if path is None:
                # A declared attachment that cannot be staged must never fail
                # silently: the question would otherwise run without its data
                # and produce an unanswerable record (observed 2026-09-19).
                logger.warning(
                    "Attachment for task %s (%s) could not be staged: "
                    "GET /files/%s returned 404 and no local copy was "
                    "found under %s.",
                    row["task_id"],
                    file_name,
                    row["task_id"],
                    local_fallback_dir,
                )
            row["file_path"] = str(path) if path is not None else None
        staged.append(row)
    return staged


def _find_local_attachment(
    local_dir: str | Path,
    task_id: str,
    file_name: str,
) -> Path | None:
    """Locate an already-staged official attachment for a task."""
    base = Path(local_dir) / task_id
    candidate = base / file_name
    if candidate.is_file():
        return candidate
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            if entry.is_file():
                return entry
    return None


_USE_PACKAGED_FALLBACK = object()  # sentinel: resolve to the packaged dir


async def load_official_questions(
    staging_dir: str | Path = "evaluation_files",
    *,
    client: GaiaEvaluationClient | None = None,
    local_fallback_dir: str | Path | None = _USE_PACKAGED_FALLBACK,
) -> list[dict[str, Any]]:
    """Fetch the official 20 questions and stage their attachments.

    `local_fallback_dir` holds already-staged copies of the official GAIA
    validation attachments (used when GET /files/{task_id} 404s). The default
    sentinel resolves to the packaged ABSOLUTE directory
    (`PACKAGED_EVALUATION_FILES_DIR`, i.e. src/gaia_agent/evaluation_files)
    when it exists — a CWD-relative "evaluation_files" default resolved to an
    empty directory and silently stripped every attachment from the official
    run (observed 2026-09-19). Pass None to disable the fallback entirely.
    """
    client = client or GaiaEvaluationClient()
    questions = await client.fetch_questions()
    if local_fallback_dir is _USE_PACKAGED_FALLBACK:
        fallback_dir: str | Path | None = (
            PACKAGED_EVALUATION_FILES_DIR
            if PACKAGED_EVALUATION_FILES_DIR.is_dir()
            else None
        )
    else:
        fallback_dir = local_fallback_dir
    return await stage_attachments(
        client,
        questions,
        staging_dir,
        local_fallback_dir=fallback_dir,
    )


def write_questions_snapshot(
    questions: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Persist the fetched questions for offline diagnostics (no answers)."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def run_offline_probe() -> None:
    """Fetch the official questions, stage files, print a summary (GETs only)."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def _main() -> None:
        client = GaiaEvaluationClient()
        questions = await load_official_questions()
        staged = sum(1 for row in questions if row.get("file_path"))
        print(f"[questions] {len(questions)} (attachments staged: {staged})")
        for row in questions:
            print(
                f"  {row['task_id']} level={row.get('Level')!r} "
                f"file={row.get('file_name') or '-'} "
                f"question={row.get('question', '')[:60]!r}"
            )

    import asyncio

    asyncio.run(_main())


if __name__ == "__main__":
    run_offline_probe()

