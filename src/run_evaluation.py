import asyncio
import json
import os
import sys
from typing import Any

# ----------------------------------------------------------------------
# ROOT-CAUSE FIX (Windows console crash):
# The evaluation previously died with
#   UnicodeEncodeError: 'charmap' codec can't encode character '\u014d'
# because Python's stdout/stderr used the Windows ANSI code page
# (cp1252/cp1256) when output was piped or captured. Any non-ASCII
# character (task questions contain accents, IPA, CJK, ...) crashed the
# print() call, aborted the whole evaluation run, and made individual
# tasks submit the literal answer "Error".
# Reconfiguring the streams to UTF-8 with errors="replace" fixes the
# encoder itself instead of hiding the data.
# ----------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from huggingface_hub import HfApi

from gaia_agent.agents.answer_sanitizer import AnswerSanitizer
from gaia_agent.agents.verifier import VerifierAgent

from gaia_agent.context.ContextBuilder import ContextBuilder
from gaia_agent.context.ContextBudget import ContextBudget
from gaia_agent.context.ContextCompressor import ContextCompressor
from gaia_agent.context.ContextPolicy import ContextPolicy
from gaia_agent.context.ContextValidator import ContextValidator

from gaia_agent.context.sources.conversation import ConversationSource
from gaia_agent.context.sources.history import HistorySource
from gaia_agent.context.sources.memory import MemorySource
from gaia_agent.context.sources.runtime import RuntimeSource

from gaia_agent.core.agent_execution import AgentExecution
from gaia_agent.core.agent_loop import AgentLoop
from gaia_agent.core.agent_state import AgentState
from gaia_agent.core.llm_executor import LLMExecutor

from gaia_agent.core.orchestration.orchestrator import Orchestrator

from gaia_agent.core.policies.approval import ApprovalPolicy
from gaia_agent.core.policies.execution import ExecutionPolicy
from gaia_agent.core.policies.termination import TerminationPolicy

from gaia_agent.core.risk.analyzer import RiskAnalyzer
from gaia_agent.core.risk.assessor import RiskAssessor
from gaia_agent.core.risk.rules import RiskRules

from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient

from gaia_agent.memory.providers.in_memory import InMemoryMemoryRepository
from gaia_agent.memory.retrieval.dense_retriver import CandidateRetriever
from gaia_agent.memory.retrieval.embedding import OllamaEmbeddingProvider
from gaia_agent.memory.retrieval.lexical_retriever import LexicalRetriever
from gaia_agent.memory.retrieval.retriever import MemoryRetriever

from gaia_agent.observability.logger import EventLogger
from gaia_agent.observability.metrics import Metrics
from gaia_agent.observability.token_tracker import TokenTracker
from gaia_agent.observability.tracer import Tracer

from gaia_agent.planner.planner import Planner

from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.loop_detector import LoopDetector
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.retry import Retry

from gaia_agent.reliability.policies.recovery_policy import RecoveryPolicy
from gaia_agent.reliability.policies.retry_policy import RetryPolicy

from gaia_agent.tools.registry import ToolRegistry
from gaia_agent.tools.path_utils import list_available_files

TARGET_RESULTS_FILE = "evaluation_results.jsonl"
SCORING_API_BASE = "https://agents-course-unit4-scoring.hf.space"
HF_DATASET_REPO = "Nadoura/gaia-agent-results"
HF_AGENT_CODE = "https://huggingface.co/spaces/Nadoura/gaia-agent-eval"

# Real task attachments are downloaded here so the planner and the
# file/image/excel tools always work with EXISTING files instead of
# letting the LLM invent paths like "<file_path_to_sales_file>" or
# "kuznetzov_taxonomist_data.xlsx".
ATTACHMENTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gaia_attachments",
)


def download_attachment(
    task_id: str,
    file_name: str,
) -> str | None:
    """
    Download the GAIA task attachment from the official scoring API
    (GET {SCORING_API_BASE}/file/{task_id}) into
    ``gaia_attachments/{task_id}/{file_name}``.

    Returns the file NAME relative to the attachment directory (which
    the planner may pass to file tools), or None when there is no
    attachment or the download fails. The failure is reported but never
    faked: a missing file stays missing.
    """
    if not file_name or not str(file_name).strip():
        return None

    file_name = os.path.basename(str(file_name).strip())

    target_dir = os.path.join(ATTACHMENTS_DIR, task_id)
    target_path = os.path.join(target_dir, file_name)

    if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
        return file_name

    url = f"{SCORING_API_BASE}/file/{task_id}"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(
            f"[WARN] Could not download attachment for task "
            f"{task_id} from {url}: "
            f"{type(error).__name__}: {error}"
        )
        return None

    content = response.content or b""

    if not content.strip():
        print(
            f"[WARN] Attachment endpoint returned an empty body "
            f"for task {task_id} (HTTP {response.status_code}, "
            f"content-type: "
            f"{response.headers.get('content-type', 'unknown')})."
        )
        return None

    os.makedirs(target_dir, exist_ok=True)

    with open(target_path, "wb") as handle:
        handle.write(content)

    print(
        f"Downloaded attachment '{file_name}' "
        f"({len(content)} bytes) for task {task_id}."
    )

    return file_name

def make_json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    return str(value)

async def create_agent() -> AgentLoop:
    llm_client = OllamaClient(
        base_url="http://localhost:11434"
    )

    llm_model = LLMModel(
        provider="ollama",
        model="qwen2.5:3b",
        max_tokens=768,
        temperature=0.2,
    )

    event_logger = EventLogger()
    metrics = Metrics()
    tracer = Tracer()
    token_tracker = TokenTracker()

    tool_registry = ToolRegistry(
        base_dir=".",
        model=llm_model,
        stt_backend=None,
    )

    available_tools = {
        spec.name: spec
        for spec in tool_registry.get_tool_specs()
    }

    execution_policy = ExecutionPolicy()

    risk_rules = RiskRules()

    risk_analyzer = RiskAnalyzer(
        client=llm_client,
        model=llm_model,
    )

    risk_assessor = RiskAssessor(
        rules=risk_rules,
        analyzer=risk_analyzer,
    )

    approval_policy = ApprovalPolicy()

    termination_policy = TerminationPolicy(
        max_iterations=20
    )

    context_policy = ContextPolicy(
        include_memory=False,
        include_conversation=True,
        include_history=True,
        include_runtime=True,
    )

    context_budget = ContextBudget(
        max_tokens=8000
    )

    context_validator = ContextValidator(
        budget=context_budget
    )

    context_compressor = ContextCompressor(
        client=llm_client,
        model=llm_model,
        budget=context_budget,
        policy=context_policy,
    )

    memory_repository = InMemoryMemoryRepository()

    memory_retriever = MemoryRetriever(
        candidate_retriever=CandidateRetriever(
            embedding_provider=OllamaEmbeddingProvider()
        ),
        lexical_retriever=LexicalRetriever(),
    )

    context_builder = ContextBuilder(
        policy=context_policy,
        budget=context_budget,
        validator=context_validator,
        compressor=context_compressor,
        conversation_source=ConversationSource(),
        history_source=HistorySource(),
        memory_source=MemorySource(
            retriever=memory_retriever,
            repository=memory_repository,
        ),
        runtime_source=RuntimeSource(),
    )

    loop_detector = LoopDetector(
        max_history=50,
        max_sequence_length=10,
        exact_repetition_threshold=3,
        sequence_repetition_threshold=3,
    )

    planner = Planner(
        client=llm_client,
        model=llm_model,
        available_tools=available_tools,
        loop_detector=loop_detector,
        available_files=list_available_files("."),
        base_dir=".",
    )

    reliability_engine = ReliabilityEngine(
        error_handler=ErrorHandler(),
        failure_classifier=FailureClassifier(),
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
        ),
        recovery_policy=RecoveryPolicy(
            allow_replan=True
        ),
        retry=Retry(),
        recovery=Recovery(),
    )

    llm_executor = LLMExecutor(
        client=llm_client,
        model=llm_model,
        context_builder=context_builder,
    )

    agent_execution = AgentExecution(
        tool_registry=tool_registry,
        execution_policy=execution_policy,
        risk_assessor=risk_assessor,
        approval_policy=approval_policy,
        llm_executor=llm_executor,
        event_logger=event_logger,
        metrics=metrics,
        tracer=tracer,
        token_tracker=token_tracker,
    )

    verifier = VerifierAgent(
        client=llm_client,
        model=llm_model,
    )

    answer_sanitizer = AnswerSanitizer()

    orchestrator = Orchestrator(
        context_builder=context_builder,
        planner=planner,
        agent_execution=agent_execution,
        error_handler=ErrorHandler(),
        reliability_engine=reliability_engine,
        loop_detector=loop_detector,
        verifier=verifier,
        answer_sanitizer=answer_sanitizer,
        event_logger=event_logger,
        metrics=metrics,
        tracer=tracer,
    )

    return AgentLoop(
        orchestrator=orchestrator,
        termination_policy=termination_policy,
    )

def save_result_line(
    result_data: dict,
    file_path: str = TARGET_RESULTS_FILE,
) -> bool:
    try:
        safe_result = make_json_safe(result_data)

        with open(
            file_path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    safe_result,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"Saved result for task: "
            f"{result_data.get('task_id')}"
        )

        return True

    except Exception as error:
        print(
            f"[ERROR] Could not save result: "
            f"{type(error).__name__}: {error}"
        )

        return False

def upload_results_to_huggingface(
    local_file_path: str = TARGET_RESULTS_FILE,
) -> bool:
    if not os.path.isfile(local_file_path):
        print(
            f"[WARNING] Results file "
            f"'{local_file_path}' not found."
        )
        return False

    try:
        api = HfApi()

        print(
            f"Uploading results to "
            f"{HF_DATASET_REPO}..."
        )

        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo="results.jsonl",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
        )

        print(
            "Results successfully uploaded "
            "to Hugging Face."
        )

        return True

    except Exception as error:
        print(
            f"[ERROR] Failed to upload results: "
            f"{type(error).__name__}: {error}"
        )

        return False

def fetch_questions() -> list[dict]:
    url = f"{SCORING_API_BASE}/questions"

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            "Hugging Face /questions did not return a list."
        )

    return data

def submit_answers(
    username: str,
    agent_code: str,
    answers: list[dict],
) -> dict:
    url = f"{SCORING_API_BASE}/submit"

    payload = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=60,
    )

    print("\n" + "=" * 60)
    print("HUGGING FACE SUBMISSION RESPONSE")
    print("=" * 60)
    print(f"HTTP Status Code: {response.status_code}")
    print(
        "Content-Type: "
        f"{response.headers.get('content-type', 'unknown')}"
    )
    print("\n----- Response Body -----")
    print(response.text[:10000])
    print("-------------------------")

    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Hugging Face returned a successful HTTP response "
            "but the body was not valid JSON."
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            "Hugging Face submission response is not a JSON object."
        )

    return data

async def run_official_evaluation() -> int:
    print(
        "--- STARTING OFFICIAL GAIA AGENT EVALUATION ---"
    )

    if os.path.exists(TARGET_RESULTS_FILE):
        try:
            os.remove(TARGET_RESULTS_FILE)
        except OSError as error:
            print(
                f"[ERROR] Could not clear "
                f"{TARGET_RESULTS_FILE}: {error}"
            )
            return 1

    try:
        agent = await create_agent()
    except Exception as error:
        print(
            f"[ERROR] Failed to create agent: "
            f"{type(error).__name__}: {error}"
        )
        return 1

    print(
        "\nFetching official evaluation questions "
        "from Hugging Face..."
    )

    try:
        questions = await asyncio.to_thread(
            fetch_questions
        )

    except requests.exceptions.Timeout as error:
        print(
            f"[ERROR] Timeout while fetching questions: {error}"
        )
        return 1

    except requests.exceptions.ConnectionError as error:
        print(
            "[ERROR] Could not connect to "
            f"Hugging Face: {error}"
        )
        return 1

    except requests.exceptions.HTTPError as error:
        print(
            f"[ERROR] Hugging Face returned "
            f"HTTP {error.response.status_code if error.response else 'unknown'}: "
            f"{error}"
        )

        if error.response is not None:
            print(
                error.response.text[:10000]
            )

        return 1

    except Exception as error:
        print(
            f"[ERROR] Failed to fetch questions: "
            f"{type(error).__name__}: {error}"
        )
        return 1

    print(
        f"Successfully fetched "
        f"{len(questions)} questions."
    )

    answers_payload: list[dict] = []

    # Global snapshot of pre-existing data files; per-task attachments
    # are added on top for each task (and reset for the next one).
    base_available_files = list_available_files(".")

    for index, item in enumerate(
        questions,
        start=1,
    ):
        task_id = item.get("task_id")
        question = item.get("question")
        file_name = item.get("file_name")

        print("\n" + "-" * 60)
        print(
            f"[QUESTION {index}/{len(questions)}]"
        )
        print(f"Task ID: {task_id}")
        print(f"Question: {question}")
        print("-" * 60)

        if not task_id:
            print(
                "[ERROR] Question has no task_id."
            )
            continue

        if not isinstance(question, str) or not question.strip():
            print(
                "[ERROR] Question text is missing."
            )

            error_answer = "Error"

            answers_payload.append(
                {
                    "task_id": task_id,
                    "submitted_answer": error_answer,
                }
            )

            save_result_line(
                {
                    "task_id": task_id,
                    "question": question,
                    "agent_answer": error_answer,
                    "submitted_answer": error_answer,
                    "termination_reason": "INVALID_QUESTION",
                }
            )

            continue

        state = AgentState(
            user_id=1,
            user_request=question.strip(),
        )

        # ----------------------------------------------------------
        # ROOT-CAUSE FIX (file handling): task -> identify attachment
        # -> resolve the REAL file -> let the planner/detector pick
        # the correct reader. The planner is re-pointed at the real
        # files for THIS task only, so the LLM can never invent file
        # paths like <file_path_to_sales_file>.
        # ----------------------------------------------------------
        attachment_name = download_attachment(
            task_id,
            file_name,
        )

        planner_files = list(base_available_files)

        if attachment_name:
            planner_files.append(attachment_name)

        agent.orchestrator.planner.available_files = sorted(
            set(planner_files)
        )

        try:
            result = await agent.run(state)

            raw_answer = getattr(
                result,
                "final_answer",
                None,
            )

            if raw_answer is None:
                agent_answer = ""
                print(
                    "Agent returned final_answer=None."
                )
            else:
                agent_answer = str(
                    raw_answer
                ).strip()

            termination_reason = make_json_safe(
                getattr(
                    result,
                    "termination_reason",
                    None,
                )
            )

            print(
                f"Agent Answer: {agent_answer}"
            )

            print(
                f"Termination reason: "
                f"{termination_reason}"
            )

            result_data = {
                "task_id": task_id,
                "question": question,
                "agent_answer": agent_answer,
                "submitted_answer": agent_answer,
                "termination_reason": termination_reason,
                # ROOT-CAUSE FIX (misleading success semantics):
                # execution_success / task_completed / answer_verified
                # are separate concepts. final_answer=None means the
                # task was NOT solved, regardless of step success.
                "execution_success": bool(
                    getattr(result, "execution_success", False)
                ),
                "task_completed": bool(
                    getattr(result, "task_completed", False)
                ),
                "final_answer_verified": bool(
                    getattr(result, "final_answer_verified", False)
                ),
            }

            save_result_line(
                result_data
            )

            answers_payload.append(
                {
                    "task_id": task_id,
                    "submitted_answer": agent_answer,
                }
            )

        except Exception as error:
            print(
                f"[ERROR] Failed to process task "
                f"{task_id}: "
                f"{type(error).__name__}: {error}"
            )

            error_answer = "Error"

            save_result_line(
                {
                    "task_id": task_id,
                    "question": question,
                    "agent_answer": error_answer,
                    "submitted_answer": error_answer,
                    "termination_reason": (
                        f"EXCEPTION: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

            answers_payload.append(
                {
                    "task_id": task_id,
                    "submitted_answer": error_answer,
                }
            )

    if not answers_payload:
        print(
            "[ERROR] No answers were generated."
        )
        return 1

    print("\n" + "=" * 60)
    print("LOCAL EVALUATION COMPLETED")
    print("=" * 60)
    print(
        f"Answers prepared: "
        f"{len(answers_payload)}/{len(questions)}"
    )

    upload_success = await asyncio.to_thread(
        upload_results_to_huggingface,
        TARGET_RESULTS_FILE,
    )

    username = "Nadoura"

    submission_data = {
        "username": username,
        "agent_code": HF_AGENT_CODE,
        "answers": answers_payload,
    }

    print("\n" + "=" * 60)
    print("SUBMITTING TO HUGGING FACE")
    print("=" * 60)
    print(
        f"Username: {username}"
    )
    print(
        f"Agent code: {HF_AGENT_CODE}"
    )
    print(
        f"Answers: {len(answers_payload)}"
    )

    submission_success = False
    scoring_result: dict | None = None

    try:
        scoring_result = await asyncio.to_thread(
            submit_answers,
            username,
            HF_AGENT_CODE,
            answers_payload,
        )

        submission_success = True

        print("\n" + "=" * 60)
        print("OFFICIAL GAIA SUBMISSION COMPLETED")
        print("=" * 60)

        print(
            f"Username: "
            f"{scoring_result.get('username', username)}"
        )

        print(
            f"Overall Score: "
            f"{scoring_result.get('score', 'N/A')}%"
        )

        print(
            f"Correct Count: "
            f"{scoring_result.get('correct_count', '?')}/"
            f"{scoring_result.get('total_attempted', '?')}"
        )

        print(
            f"Message: "
            f"{scoring_result.get('message', '')}"
        )

    except requests.exceptions.HTTPError as error:
        status = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )

        print("\n" + "=" * 60)
        print("HUGGING FACE SUBMISSION FAILED")
        print("=" * 60)
        print(f"HTTP Status: {status}")
        print(
            f"Error: {error}"
        )

        if error.response is not None:
            print(
                "\nServer Response:"
            )
            print(
                error.response.text[:10000]
            )

    except requests.exceptions.Timeout as error:
        print(
            "\n[ERROR] Hugging Face submission timed out:"
        )
        print(error)

    except requests.exceptions.ConnectionError as error:
        print(
            "\n[ERROR] Could not connect to "
            "Hugging Face scoring endpoint:"
        )
        print(error)

    except requests.exceptions.RequestException as error:
        print(
            "\n[ERROR] Hugging Face request failed:"
        )
        print(
            f"{type(error).__name__}: {error}"
        )

    except Exception as error:
        print(
            "\n[ERROR] Unexpected submission error:"
        )
        print(
            f"{type(error).__name__}: {error}"
        )

    print("\n" + "=" * 60)

    if submission_success and upload_success:
        print(
            "EVALUATION FINISHED SUCCESSFULLY"
        )
        print(
            "Local evaluation: SUCCESS"
        )
        print(
            "Results upload: SUCCESS"
        )
        print(
            "Official submission: SUCCESS"
        )
        return 0

    if submission_success:
        print(
            "EVALUATION FINISHED WITH PARTIAL SUCCESS"
        )
        print(
            "Local evaluation: SUCCESS"
        )
        print(
            "Results upload: FAILED"
        )
        print(
            "Official submission: SUCCESS"
        )
        return 1

    print(
        "EVALUATION FINISHED WITH SUBMISSION FAILURE"
    )
    print(
        "Local evaluation: SUCCESS"
    )
    print(
        f"Results upload: "
        f"{'SUCCESS' if upload_success else 'FAILED'}"
    )
    print(
        "Official submission: FAILED"
    )
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(
        run_official_evaluation()
    )
    raise SystemExit(exit_code)