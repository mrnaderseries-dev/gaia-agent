from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

from gaia_agent.main import create_agent
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentState

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_FILE = BASE_DIR / "evaluation_results.jsonl"


def load_questions() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print("[ERROR] The 'datasets' package is not installed.")
        print("Install it with:")
        print("    pip install datasets")
        raise exc

    try:
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_level1",
            split="validation",
        )
    except Exception as exc:
        print("[ERROR] Failed to load GAIA dataset:")
        print(f"        {type(exc).__name__}: {exc}")
        return []

    questions: list[dict[str, Any]] = []

    for row in dataset:
        questions.append(dict(row))

    return questions


def save_result(result: dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


async def run_question(
    agent: AgentLoop,
    question: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    question_id = question.get(
        "task_id",
        index,
    )

    prompt = question.get("Question")

    if not isinstance(prompt, str):
        prompt = str(prompt or "")

    print()
    print("=" * 80)
    print(f"[QUESTION {index}] {question_id}")
    print("=" * 80)
    print(prompt)
    print()

    state = AgentState(
        user_id=1,
        user_request=prompt,
    )

    try:
        result = await agent.run(state)

        output = {
            "task_id": question_id,
            "question": prompt,
            "answer": result.final_answer,
            "final_answer_ready": result.final_answer_ready,
            "final_answer_verified": result.final_answer_verified,
            "tool_error": result.tool_error,
            "fatal_error": result.fatal_error,
            "termination_reason": (
                str(result.termination_reason)
                if result.termination_reason is not None
                else None
            ),
            "status": "completed",
        }

        print("[ANSWER]", result.final_answer)
        print(
            "[VERIFIED]",
            result.final_answer_verified,
        )

        return output

    except Exception as exc:
        error_text = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"[ERROR] Question {question_id}: "
            f"{error_text}"
        )

        return {
            "task_id": question_id,
            "question": prompt,
            "answer": None,
            "final_answer_ready": False,
            "final_answer_verified": False,
            "status": "error",
            "error": error_text,
        }


async def main() -> None:
    print("--- STARTING OFFICIAL GAIA AGENT EVALUATION ---")

    print()
    print(f"[INFO] Base directory: {BASE_DIR}")
    print(f"[INFO] Results file: {RESULTS_FILE}")
    print()

    try:
        agent = await create_agent()
    except Exception as exc:
        print("[ERROR] Failed to create agent:")
        print(f"        {type(exc).__name__}: {exc}")
        return

    print()
    print("[INFO] Agent created successfully.")

    questions = load_questions()

    if not questions:
        print("[ERROR] No GAIA questions loaded.")
        return

    print()
    print(f"[INFO] Loaded {len(questions)} official GAIA questions.")

    completed = 0
    failed = 0
    verified = 0

    for index, question in enumerate(
        questions,
        start=1,
    ):
        result = await run_question(
            agent,
            question,
            index,
        )

        save_result(result)

        if result["status"] == "completed":
            completed += 1

            if result.get(
                "final_answer_verified",
                False,
            ):
                verified += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print("GAIA EVALUATION FINISHED")
    print("=" * 80)

    print(f"Total questions         : {len(questions)}")
    print(f"Completed               : {completed}")
    print(f"Failed                  : {failed}")
    print(f"Final answers verified  : {verified}")
    print(f"Results file            : {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())