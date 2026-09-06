"""
Hugging Face Unit 4 Diagnostic Evaluation.

Runs exactly the first 20 questions returned by the
Hugging Face Agents Course Unit 4 scoring API.

This runner:
- Uses the HF scoring API questions.
- Runs exactly 20 questions by default.
- Supports ONLY_INDICES for targeted reruns.
- Downloads/stages attachments when available.
- Saves each completed result immediately.
- Prints detailed execution information to stdout.
- stdout/stderr can be captured with PowerShell Tee-Object.

Example:

    python src\\_diag_eval.py 2>&1 |
        Tee-Object -FilePath ".\\monitoring321.txt"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------
# UTF-8 terminal support
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Agent imports
# ---------------------------------------------------------------------

from gaia_agent import main as user_agent
from gaia_agent.core.agent_state import AgentState


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

API_URL = "https://agents-course-unit4-scoring.hf.space"

# IMPORTANT:
# These are the Hugging Face Unit 4 evaluation questions.
MAX_QUESTIONS = 20

# Maximum wall-clock time for one question.
PER_QUESTION_TIMEOUT_S = 420

BASE_DIR = Path(__file__).resolve().parent

RESULTS_PATH = (
    BASE_DIR
    / "_eval_results_hf20.json"
)

STAGE_DIR = (
    BASE_DIR
    / "evaluation_files"
)

# Optional targeted rerun:
#
# PowerShell:
#
# $env:ONLY_INDICES="1,3,7"
# python src\_diag_eval.py
#
# Empty means run all 20.
ONLY_INDICES = {
    int(i.strip())
    for i in os.environ.get(
        "ONLY_INDICES",
        "",
    ).split(",")
    if i.strip()
}


# ---------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------

def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(flush=True)


def print_subsection(title: str) -> None:
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)
    print(flush=True)


def safe_repr(value: Any, max_chars: int = 12000) -> str:
    """
    Convert arbitrary state values into printable text.

    We intentionally cap extremely large values so one giant tool
    result does not make the diagnostic log unusable.
    """
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"

    if len(text) > max_chars:
        return (
            text[:max_chars]
            + f"\n... [TRUNCATED; original length={len(text)}]"
        )

    return text


def print_state_snapshot(
    state: AgentState,
    *,
    label: str,
) -> None:
    """
    Print the complete AgentState fields.

    This is intentionally generic and uses dataclass fields rather
    than hardcoding only selected fields. Therefore if AgentState
    gains another field later, it will automatically appear here.
    """

    print_subsection(label)

    try:
        from dataclasses import fields

        for field_info in fields(state):
            name = field_info.name

            try:
                value = getattr(state, name)
            except Exception as exc:
                value = (
                    f"<attribute read failed: "
                    f"{type(exc).__name__}: {exc}>"
                )

            print(
                f"[STATE] {name} = "
                f"{safe_repr(value)}"
            )

    except Exception as exc:
        print(
            "[STATE SNAPSHOT ERROR]",
            f"{type(exc).__name__}: {exc}",
        )

    print(flush=True)


# ---------------------------------------------------------------------
# Attachment handling
# ---------------------------------------------------------------------

def stage_attachment(
    task_id: str | None,
    file_name: str,
) -> str | None:
    """
    Download the task attachment if the HF scoring API serves it.
    """

    if not task_id:
        return None

    if not file_name:
        return None

    try:
        response = requests.get(
            f"{API_URL}/files/{task_id}",
            timeout=30,
        )

        if response.status_code != 200:
            print(
                "[ATTACHMENT] HTTP status:",
                response.status_code,
                flush=True,
            )
            return None

        if len(response.content) < 10:
            print(
                "[ATTACHMENT] Response too small.",
                flush=True,
            )
            return None

        dest_dir = (
            STAGE_DIR
            / str(task_id)
        )

        dest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dest = (
            dest_dir
            / Path(file_name).name
        )

        with dest.open("wb") as handle:
            handle.write(response.content)

        print(
            f"[ATTACHMENT] Saved to: {dest}",
            flush=True,
        )

        return str(dest.resolve())

    except Exception as exc:
        print(
            "[ATTACHMENT ERROR]",
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------
# AgentState construction
# ---------------------------------------------------------------------

def build_state(
    question_text: str,
    attachment_path: str | None,
) -> AgentState:

    state = AgentState(
        user_id=1,
        user_request=question_text,
    )

    # Keep all compatible aliases synchronized.
    aliases = (
        "user_question",
        "prompt",
        "query",
        "question",
        "input",
        "task",
        "text",
    )

    for attr in aliases:
        if hasattr(state, attr) and question_text:
            try:
                setattr(
                    state,
                    attr,
                    question_text,
                )
            except Exception:
                pass

    if attachment_path:
        full_request = (
            f"{question_text}\n\n"
            "The attached file is available at the "
            f"absolute path: {attachment_path}"
        )

        state.user_request = full_request

        if hasattr(state, "user_question"):
            try:
                state.user_question = full_request
            except Exception:
                pass

    return state


# ---------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------

def extract_result(
    state: AgentState,
    elapsed: float,
    error: str | None,
) -> dict[str, Any]:

    def get(
        attr: str,
        default: Any = None,
    ) -> Any:
        return getattr(
            state,
            attr,
            default,
        )

    answer = get("final_answer")

    if hasattr(answer, "final_answer"):
        try:
            answer = answer.final_answer
        except Exception:
            pass

    return {
        "answer": (
            str(answer).strip()
            if answer is not None
            else "0"
        ),
        "elapsed_s": round(
            elapsed,
            1,
        ),
        "termination_reason": str(
            get("termination_reason")
        ),
        "final_answer_ready": bool(
            get(
                "final_answer_ready",
                False,
            )
        ),
        "final_answer_verified": bool(
            get(
                "final_answer_verified",
                False,
            )
        ),
        "task_completed": bool(
            get(
                "task_completed",
                False,
            )
        ),
        "execution_success": bool(
            get(
                "execution_success",
                False,
            )
        ),
        "blocked": bool(
            get(
                "blocked",
                False,
            )
        ),
        "iterations": get(
            "iteration",
            0,
        ),
        "last_tool_error": (
            str(get("tool_error"))[:1000]
            if get("tool_error")
            else None
        ),
        "replan_count": get(
            "replan_count",
            0,
        ),
        "retry_count": get(
            "retry_count",
            0,
        ),
        "recovery_attempted": bool(
            get(
                "recovery_attempted",
                False,
            )
        ),
        "fatal_error": bool(
            get(
                "fatal_error",
                False,
            )
        ),
        "error": error,
    }


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

async def main() -> None:

    print_section(
        "HUGGING FACE UNIT 4 - 20 QUESTION DIAGNOSTIC RUN"
    )

    print(
        "[CONFIG] API_URL:",
        API_URL,
        flush=True,
    )

    print(
        "[CONFIG] MAX_QUESTIONS:",
        MAX_QUESTIONS,
        flush=True,
    )

    print(
        "[CONFIG] RESULTS_PATH:",
        RESULTS_PATH,
        flush=True,
    )

    print(
        "[CONFIG] ONLY_INDICES:",
        (
            sorted(ONLY_INDICES)
            if ONLY_INDICES
            else "ALL 20"
        ),
        flush=True,
    )

    # -------------------------------------------------------------
    # Create Agent
    # -------------------------------------------------------------

    print_section(
        "INITIALIZING AGENT"
    )

    try:
        agent_instance = (
            await user_agent.create_agent()
        )

        print(
            "[AGENT] Agent initialized successfully.",
            flush=True,
        )

    except Exception as exc:
        print(
            "[FATAL] Failed to initialize agent:",
            flush=True,
        )
        traceback.print_exc()
        return

    # -------------------------------------------------------------
    # Fetch HF questions
    # -------------------------------------------------------------

    print_section(
        "FETCHING HUGGING FACE QUESTIONS"
    )

    try:
        response = requests.get(
            f"{API_URL}/questions",
            timeout=60,
        )

        response.raise_for_status()

        questions = response.json()

    except Exception as exc:
        print(
            "[FATAL] Failed to fetch questions:",
            flush=True,
        )
        traceback.print_exc()
        return

    if not isinstance(
        questions,
        list,
    ):
        print(
            "[FATAL] /questions did not return a list.",
            flush=True,
        )
        return

    subset = questions[:MAX_QUESTIONS]

    print(
        f"[HF] API returned {len(questions)} questions.",
        flush=True,
    )

    print(
        f"[HF] This run will use exactly "
        f"{len(subset)} questions.",
        flush=True,
    )

    if len(subset) < MAX_QUESTIONS:
        print(
            "[WARNING] HF API returned fewer than 20 questions.",
            flush=True,
        )

    results: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Question loop
    # -------------------------------------------------------------

    for idx, question in enumerate(
        subset,
        start=1,
    ):

        if (
            ONLY_INDICES
            and idx not in ONLY_INDICES
        ):
            continue

        task_id = question.get(
            "task_id"
        )

        question_text = (
            question.get("question")
            or question.get("Question")
            or ""
        ).strip()

        file_name = (
            question.get("file_name")
            or ""
        ).strip()

        print_section(
            f"QUESTION {idx}/{len(subset)}"
        )

        print(
            "[TASK ID]",
            task_id,
            flush=True,
        )

        print(
            "[FILE]",
            file_name or "-",
            flush=True,
        )

        print(
            "[QUESTION]",
            flush=True,
        )

        print(
            question_text,
            flush=True,
        )

        # ---------------------------------------------------------
        # Attachment
        # ---------------------------------------------------------

        attachment = stage_attachment(
            task_id,
            file_name,
        )

        if (
            file_name
            and not attachment
        ):
            print(
                "[WARNING] Attachment could not be staged.",
                flush=True,
            )

        # ---------------------------------------------------------
        # Build state
        # ---------------------------------------------------------

        state = build_state(
            question_text,
            attachment,
        )

        print_state_snapshot(
            state,
            label="INITIAL AGENT STATE",
        )

        # ---------------------------------------------------------
        # Execute
        # ---------------------------------------------------------

        print_section(
            f"START EXECUTION - QUESTION {idx}"
        )

        t0 = time.time()

        error: str | None = None

        try:

            await asyncio.wait_for(
                agent_instance.run(state),
                timeout=PER_QUESTION_TIMEOUT_S,
            )

        except asyncio.TimeoutError:

            state.timed_out = True

            error = (
                "TIMEOUT after "
                f"{PER_QUESTION_TIMEOUT_S}s"
            )

            print(
                "[TIMEOUT]",
                error,
                flush=True,
            )

            print_state_snapshot(
                state,
                label="STATE AT TIMEOUT",
            )

        except KeyboardInterrupt:

            print(
                "[INTERRUPTED] User stopped the run.",
                flush=True,
            )

            print_state_snapshot(
                state,
                label="STATE AT INTERRUPTION",
            )

            raise

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "[QUESTION ERROR]",
                error,
                flush=True,
            )

            traceback.print_exc()

            print_state_snapshot(
                state,
                label="STATE AT EXCEPTION",
            )

        elapsed = (
            time.time()
            - t0
        )

        # ---------------------------------------------------------
        # Final state
        # ---------------------------------------------------------

        print_section(
            f"FINAL STATE - QUESTION {idx}"
        )

        print_state_snapshot(
            state,
            label="FINAL AGENT STATE",
        )

        rec = extract_result(
            state,
            elapsed,
            error,
        )

        rec.update(
            {
                "task_id": task_id,
                "index": idx,
                "question": question_text,
                "file_name": file_name,
                "attachment_staged": bool(
                    attachment
                ),
            }
        )

        results.append(rec)

        # ---------------------------------------------------------
        # Immediate persistence
        # ---------------------------------------------------------

        try:
            with RESULTS_PATH.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    results,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

        except Exception as exc:

            print(
                "[SAVE ERROR]",
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------

        print_section(
            f"QUESTION {idx} RESULT"
        )

        print(
            "[ANSWER]",
            rec["answer"],
            flush=True,
        )

        print(
            "[VERIFIED]",
            rec["final_answer_verified"],
            flush=True,
        )

        print(
            "[COMPLETED]",
            rec["task_completed"],
            flush=True,
        )

        print(
            "[EXECUTION SUCCESS]",
            rec["execution_success"],
            flush=True,
        )

        print(
            "[ITERATIONS]",
            rec["iterations"],
            flush=True,
        )

        print(
            "[REPLAN COUNT]",
            rec["replan_count"],
            flush=True,
        )

        print(
            "[RETRY COUNT]",
            rec["retry_count"],
            flush=True,
        )

        print(
            "[TERMINATION]",
            rec["termination_reason"],
            flush=True,
        )

        print(
            "[ELAPSED]",
            rec["elapsed_s"],
            "seconds",
            flush=True,
        )

        if rec["error"]:
            print(
                "[ERROR]",
                rec["error"],
                flush=True,
            )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print_section(
        "HUGGING FACE 20 QUESTION RUN FINISHED"
    )

    print(
        "[TOTAL SELECTED]",
        len(subset),
        flush=True,
    )

    print(
        "[TOTAL EXECUTED]",
        len(results),
        flush=True,
    )

    verified = sum(
        bool(
            r.get(
                "final_answer_verified",
                False,
            )
        )
        for r in results
    )

    completed = sum(
        bool(
            r.get(
                "task_completed",
                False,
            )
        )
        for r in results
    )

    failed = sum(
        bool(
            r.get("error")
        )
        for r in results
    )

    print(
        "[VERIFIED]",
        verified,
        flush=True,
    )

    print(
        "[TASK COMPLETED]",
        completed,
        flush=True,
    )

    print(
        "[ERRORS]",
        failed,
        flush=True,
    )

    print(
        "[RESULTS FILE]",
        RESULTS_PATH,
        flush=True,
    )

    print()
    print(
        "FULL TERMINAL TRACE CAN BE SAVED WITH:"
    )
    print(
        'python src\\_diag_eval.py 2>&1 | '
        'Tee-Object -FilePath ".\\monitoring321.txt"'
    )
    print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())