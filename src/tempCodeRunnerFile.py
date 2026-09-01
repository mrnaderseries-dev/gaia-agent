import asyncio
import json
import os
import sys
from typing import Any


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
    into ``gaia_attachments/{task_id}/{file_name}``.

    Returns the REAL absolute path to the downloaded attachment,
    or None when there is no attachment or the download fails.
    """
    if not file_name or not str(file_name).strip():
        return None

    file_name = os.path.basename(str(file_name).strip())

    target_dir = os.path.join(ATTACHMENTS_DIR, task_id)
    target_path = os.path.join(target_dir, file_name)

    if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
        return os.path.abspath(target_path)

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

    return os.path.abspath(target_path)

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
