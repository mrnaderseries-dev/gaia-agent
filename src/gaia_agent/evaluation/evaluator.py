from __future__ import annotations

"""Official GAIA Level-1 evaluation runner.

Pipeline (proven by the diagnostic harness `_gaia20_runner.py`, reused here
without duplicating its instrumentation):

    GET /questions -> load 20 Level-1 questions
    -> stage attachments (GET /files/{task_id}) when required
    -> reuse ONE agent (gaia_agent.main.create_agent)
    -> run questions sequentially (never concurrent)
    -> collect the bare extracted answer from the agent state
    -> build official AnswerItem rows
    -> optional/controlled POST /submit (explicit caller credentials only)

The runner never exposes expected answers to the agent, never submits
"FINAL ANSWER: ..." text (the Orchestrator already extracts the bare value),
and never raises away a whole run because one question failed: failures are
recorded per question as an empty submitted_answer.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

from gaia_agent.evaluation.gaia import (
    GaiaEvaluationClient,
    load_official_questions,
    write_questions_snapshot,
)
from gaia_agent.evaluation.submission import (
    build_course_rows,
    write_submission,
)
from gaia_agent.core.agent_state import AgentState
from gaia_agent.context.attachments import Attachment
from gaia_agent.main import create_agent

DEFAULT_QUESTION_TIMEOUT = 600.0


async def evaluate_questions(
    questions: list[dict[str, Any]],
    *,
    agent: Any = None,
    timeout_per_question: float = DEFAULT_QUESTION_TIMEOUT,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every question through the real agent; return per-question records.

    Each record: {task_id, question, level, file_name, file_path, answer,
    elapsed_s, error, verified}. `answer` is the bare final answer (or None).
    """
    if agent is None:
        agent = await create_agent()

    results: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        record = await evaluate_single_question(
            agent,
            question,
            timeout=timeout_per_question,
        )
        record["index"] = index
        results.append(record)
        print(
            f"[{index}/{len(questions)}] {record['task_id']} "
            f"answer={record['answer']!r} elapsed={record['elapsed_s']}s "
            f"error={record['error']!r}",
            flush=True,
        )
        if on_result is not None:
            on_result(record)

    return results


async def evaluate_single_question(
    agent: Any,
    question: dict[str, Any],
    *,
    timeout: float = DEFAULT_QUESTION_TIMEOUT,
) -> dict[str, Any]:
    """Run one question; contain any failure inside the record."""
    task_id = str(question["task_id"])
    attachments = []
    file_path = question.get("file_path")
    if file_path:
        attachments.append(
            Attachment(
                attachment_id=f"att-{task_id[:8]}",
                filename=Path(file_path).name,
                path=file_path,
            )
        )

    state = AgentState(
        user_request=question["question"],
        attachments=attachments,
    )

    started = time.time()
    error: str | None = None
    try:
        await asyncio.wait_for(agent.run(state), timeout=timeout)
    except asyncio.TimeoutError:
        error = f"TIMEOUT after {timeout}s (state salvaged)"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.time() - started, 1)
    return {
        "task_id": task_id,
        "question": question.get("question", ""),
        "level": question.get("Level") or question.get("level"),
        "file_name": question.get("file_name"),
        "file_path": file_path,
        "answer": state.final_answer,
        "elapsed_s": elapsed,
        "error": error,
        "verified": bool(state.final_answer_verified),
    }


async def run_official_evaluation(
    *,
    questions_file: str | Path | None = None,
    output_dir: str | Path = "evaluation_runs",
    limit: int | None = None,
    timeout_per_question: float = DEFAULT_QUESTION_TIMEOUT,
    submit: bool = False,
    username: str | None = None,
    agent_code: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run the official evaluation and optionally submit.

    Questions come from the live API (staged attachments included) unless
    `questions_file` is given for an offline rerun. Submission happens ONLY
    when `submit=True` with explicit username/agent_code.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if questions_file is not None:
        questions = json.loads(
            Path(questions_file).read_text(encoding="utf-8")
        )
        print(f"[cfg] offline questions file: {questions_file}")
    else:
        print("[cfg] fetching official questions from the scoring API ...")
        questions = await load_official_questions(
            output / "attachments",
        )
        write_questions_snapshot(
            questions,
            output / "questions_snapshot.json",
        )
        print(f"[cfg] fetched {len(questions)} official questions")

    if limit is not None:
        questions = questions[:limit]
        print(f"[cfg] limited to the first {limit} question(s)")

    results = await evaluate_questions(
        questions,
        timeout_per_question=timeout_per_question,
    )

    results_path = output / "evaluation_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    course_rows = build_course_rows(results)
    course_path = output / "course_submission.json"
    course_path.write_text(
        json.dumps(course_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_submission(results, output / "submission.jsonl")

    answered = sum(1 for row in results if row["answer"] is not None)
    print(
        f"[summary] answered={answered}/{len(results)} "
        f"results={results_path} course_rows={course_path}"
    )

    score_response: dict[str, Any] | None = None
    if submit:
        if not username or not agent_code:
            raise SystemExit(
                "--submit requires --username and --agent-code (the official "
                "Submission schema requires them explicitly)."
            )
        client = GaiaEvaluationClient()
        score_response = await client.submit(
            username=username,
            agent_code=agent_code,
            answers=course_rows,
        )
        (output / "score_response.json").write_text(
            json.dumps(score_response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[score] {json.dumps(score_response, ensure_ascii=False)}")

    return results, score_response


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official GAIA Level-1 evaluation (20 questions) and "
            "optionally submit to the official scoring API."
        )
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Offline JSON questions file (default: fetch from the live API).",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_runs",
        help="Directory for results, snapshots and submission files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions (validation).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_QUESTION_TIMEOUT,
        help="Per-question wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="POST the answers to the official scoring API (controlled).",
    )
    parser.add_argument("--username", default=None)
    parser.add_argument("--agent-code", default=None)
    args = parser.parse_args()

    asyncio.run(
        run_official_evaluation(
            questions_file=args.questions_file,
            output_dir=args.output_dir,
            limit=args.limit,
            timeout_per_question=args.timeout,
            submit=args.submit,
            username=args.username,
            agent_code=args.agent_code,
        )
    )


if __name__ == "__main__":
    main()

