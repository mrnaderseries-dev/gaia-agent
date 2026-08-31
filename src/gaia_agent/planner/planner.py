import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

for _stream in (os.sys.stdout, os.sys.stderr):
if hasattr(_stream, "reconfigure"):
try:
_stream.reconfigure(
encoding="utf-8",
errors="replace",
)
except Exception:
pass

from ..reliability.errors import AgentError
from ..reliability.loop_detector import LoopDetector
from ..planner.plan_schema import PlanSchema, PlanStep, StepType
from ..tools.contract_validator import ToolContractValidator
from ..tools.path_utils import is_placeholder_path
from .task_classifier import (
TaskAnalysis,
TaskClassifier,
TaskIntent,
)

logger = logging.getLogger(**name**)

class PlannerRecoveryRequired(Exception):
pass

class Planner:
def **init**(
self,
*,
client: Any,
model: Any,
available_tools: dict[str, Any] | list[Any],
loop_detector: LoopDetector | None = None,
available_files: Sequence[str] | None = None,
base_dir: str = ".",
) -> None:
if client is None:
raise ValueError("client cannot be None.")

```
    if isinstance(model, str):
        if not model.strip():
            raise ValueError("model cannot be empty.")
    elif model is None:
        raise ValueError("model cannot be None.")

    if isinstance(available_tools, dict):
        self.available_tools = dict(available_tools)
    elif isinstance(available_tools, (list, tuple)):
        self.available_tools = {
            str(getattr(tool, "name", tool)): tool
            for tool in available_tools
        }
    else:
        raise TypeError(
            "available_tools must be a dictionary or list."
        )

    self.client = client
    self.model = model
    self.loop_detector = loop_detector or LoopDetector()
    self.base_dir = base_dir
    self.available_files = list(available_files or [])
    self.task_classifier = TaskClassifier()
    self._current_question = ""

async def create_plan(
    self,
    user_question: str,
    context: Sequence[Any] | None = None,
) -> PlanSchema:
    if not user_question or not user_question.strip():
        raise ValueError(
            "user_question cannot be empty."
        )

    analysis = self.task_classifier.classify(
        user_question,
        available_files=self.available_files,
        available_tools=sorted(
            self.available_tools.keys()
        ),
    )

    self._current_question = user_question

    prompt = self._build_initial_prompt(
        user_question=user_question,
        context=context,
        analysis=analysis,
    )

    try:
        plan = await self.client.generate(
            [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=self.model,
            output_schema=PlanSchema,
        )

        self._validate_generated_plan(
            plan,
            analysis=analysis,
        )

        return plan

    except Exception as exc:
        logger.exception(
            "Initial planner generation or validation failed: %s",
            exc,
        )

        fallback = self._emergency_fallback_plan(
            user_question=user_question,
            analysis=analysis,
        )

        self._validate_generated_plan(
            fallback,
            analysis=analysis,
        )

        return fallback

async def generate_plan(
    self,
    user_question: str,
    context: Sequence[Any] | None = None,
) -> PlanSchema:
    return await self.create_plan(
        user_question=user_question,
        context=context,
    )

async def replan_step(
    self,
    *,
    user_question: str,
    context: Sequence[Any] | None,
    failed_step: PlanStep,
    failure: AgentError,
) -> PlanStep:
    plan = await self.replan(
        user_question=user_question,
        context=context,
        failed_step=failed_step,
        failure=failure,
    )

    if not plan.steps:
        raise PlannerRecoveryRequired(
            "Replanned step produced an empty plan."
        )

    return plan.steps[0]

async def replan(
    self,
    user_question: str,
    context: Sequence[Any] | None,
    failed_step: PlanStep,
    failure: AgentError,
) -> PlanSchema:
    if not user_question or not user_question.strip():
        raise ValueError(
            "user_question cannot be empty."
        )

    if failed_step is None:
        raise ValueError(
            "failed_step cannot be None."
        )

    if failure is None:
        raise ValueError(
            "failure cannot be None."
        )

    logger.warning(
        "Triggering full replan after failure at step %s: %s",
        failed_step.step_id,
        failure.message,
    )

    failed_step_fingerprint = self._fingerprint_step(
        failed_step
    )

    analysis = self.task_classifier.classify(
        user_question,
        available_files=self.available_files,
        available_tools=sorted(
            self.available_tools.keys()
        ),
    )

    self._current_question = user_question

    prompt = self._build_replan_prompt(
        user_question=user_question,
        context=context,
        failed_step=failed_step,
        failure=failure,
        analysis=analysis,
    )

    try:
        plan = await self.client.generate(
            [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=self.model,
            output_schema=PlanSchema,
        )

        self._validate_generated_plan(
            plan,
            failed_step_fingerprint=failed_step_fingerprint,
            analysis=analysis,
        )

        return plan

    except Exception as exc:
        logger.exception(
            "Planner generation or validation failed during replan: %s",
            exc,
        )

        fallback = self._emergency_fallback_plan(
            user_question=user_question,
            failed_step=failed_step,
            analysis=analysis,
        )

        fallback_fingerprint = {
            self._fingerprint_step(step)
            for step in fallback.steps
        }

        if failed_step_fingerprint in fallback_fingerprint:
            alternative = self.get_alternative_strategy(
                user_question=user_question,
                failed_step=failed_step,
                analysis=analysis,
            )

            if alternative is not None:
                fallback = PlanSchema(
                    steps=[
                        alternative,
                        PlanStep(
                            step_id=1,
                            action=(
                                "Synthesize the final answer "
                                "using only the evidence obtained"
                            ),
                            step_type=StepType.LLM,
                            tool_name=None,
                            arguments={},
                            is_final_answer=True,
                        ),
                    ]
                )
            else:
                fallback = self._llm_only_plan()

        self._validate_generated_plan(
            fallback,
            failed_step_fingerprint=failed_step_fingerprint,
            analysis=analysis,
        )

        return fallback

def _validate_generated_plan(
    self,
    plan: PlanSchema,
    failed_step_fingerprint: str | None = None,
    analysis: TaskAnalysis | None = None,
) -> None:
    if not isinstance(plan, PlanSchema):
        raise ValueError(
            "Planner output must be a PlanSchema."
        )

    self._validate_plan_structure(plan)

    for step in plan.steps:
        self._validate_step(step)

    self._validate_final_answer_structure(plan)

    if (
        analysis is not None
        and analysis.forbidden_tools
    ):
        forbidden = set(
            analysis.forbidden_tools
        )

        for step in plan.steps:
            if (
                step.step_type == StepType.TOOL
                and step.tool_name in forbidden
            ):
                raise PlannerRecoveryRequired(
                    "Planner used forbidden tool "
                    f"'{step.tool_name}' for a "
                    f"{analysis.intent.value} task."
                )

    plan_loop = self.loop_detector.check_plan(
        plan.steps
    )

    if plan_loop.detected:
        logger.warning(
            "Planner produced a repeated plan: %s",
            plan_loop.message,
        )

        raise PlannerRecoveryRequired(
            "Planner produced a repeated execution plan."
        )

    if failed_step_fingerprint is not None:
        for step in plan.steps:
            current_fingerprint = (
                self._fingerprint_step(step)
            )

            if (
                current_fingerprint
                == failed_step_fingerprint
            ):
                raise PlannerRecoveryRequired(
                    "Replanned strategy repeats the "
                    "previously failed execution."
                )

def _validate_plan_structure(
    self,
    plan: PlanSchema,
) -> None:
    if not plan.steps:
        raise ValueError(
            "Plan must contain at least one step."
        )

    if len(plan.steps) > 20:
        raise ValueError(
            "Plan contains too many steps. "
            "Maximum allowed is 20."
        )

    for expected_id, step in enumerate(plan.steps):
        if step is None:
            raise ValueError(
                f"Plan step {expected_id} cannot be None."
            )

        if step.step_id != expected_id:
            raise ValueError(
                "Step sequence error: "
                f"expected ID {expected_id}, "
                f"got {step.step_id}."
            )

def _validate_step(
    self,
    step: PlanStep,
) -> None:
    if step is None:
        raise ValueError(
            "Plan step cannot be None."
        )

    if not step.action or not step.action.strip():
        raise ValueError(
            "Step action cannot be empty."
        )

    if step.step_type == StepType.TOOL:
        if not step.tool_name:
            raise ValueError(
                "TOOL step must specify tool_name."
            )

        if not isinstance(step.tool_name, str):
            raise ValueError(
                "tool_name must be a string."
            )

        if step.tool_name.lower() == "final_answer":
            logger.info(
                "Converted 'final_answer' tool call "
                "to an LLM final-answer step."
            )

            step.step_type = StepType.LLM
            step.tool_name = None
            step.arguments = {}
            step.is_final_answer = True

            return

        repaired_name = step.tool_name

        if repaired_name not in self.available_tools:
            repaired_name = self._repair_tool_name(
                repaired_name
            )

            if not repaired_name:
                raise ValueError(
                    "Unknown or unavailable tool: "
                    f"{step.tool_name}"
                )

            logger.warning(
                "Repaired unknown tool name '%s' -> '%s'.",
                step.tool_name,
                repaired_name,
            )

            step.tool_name = repaired_name

        if not isinstance(
            step.arguments,
            dict,
        ):
            raise ValueError(
                f"Arguments for tool "
                f"'{step.tool_name}' "
                "must be a dictionary."
            )

        step.arguments = (
            self._filter_arguments_for_tool(
                step.tool_name,
                step.arguments,
            )
        )

        if step.tool_name == "web_search":
            raw_query = str(
                step.arguments.get(
                    "query",
                    "",
                )
                or ""
            ).strip()

            normalized_question = " ".join(
                (self._current_question or "")
                .split()
            ).lower()

            normalized_query = " ".join(
                raw_query.split()
            ).lower()

            too_long = (
                len(raw_query.split()) > 15
            )

            echoes_question = (
                bool(normalized_question)
                and normalized_query
                == normalized_question
            )

            if raw_query and (
                echoes_question
                or too_long
            ):
                repaired_query = (
                    self._build_search_query(
                        self._current_question
                        or raw_query
                    )
                )

                if repaired_query:
                    step.arguments["query"] = (
                        repaired_query
                    )

        step.arguments = (
            self._repair_file_arguments(
                step.tool_name,
                step.arguments,
            )
        )

        ToolContractValidator.validate_step_contract(
            step,
            self.available_tools,
        )

        return

    if step.step_type == StepType.LLM:
        if step.tool_name is not None:
            raise ValueError(
                "LLM step cannot specify tool_name."
            )

        if step.arguments:
            raise ValueError(
                "LLM step cannot contain tool arguments."
            )

        return

    raise ValueError(
        f"Unsupported step type: "
        f"{step.step_type}"
    )

def _validate_final_answer_structure(
    self,
    plan: PlanSchema,
) -> None:
    final_steps = [
        step
        for step in plan.steps
        if step.is_final_answer
    ]

    if len(final_steps) != 1:
        raise ValueError(
            "Plan must contain exactly one "
            "final-answer step."
        )

    final_step = final_steps[0]

    if final_step.step_id != len(plan.steps) - 1:
        raise ValueError(
            "Final-answer step must be the last step."
        )

    if final_step.step_type != StepType.LLM:
        raise ValueError(
            "Final-answer step must be an LLM step."
        )

    if final_step.tool_name is not None:
        raise ValueError(
            "Final-answer step cannot contain a tool."
        )

    if final_step.arguments:
        raise ValueError(
            "Final-answer step cannot contain "
            "tool arguments."
        )

def _repair_tool_name(
    self,
    tool_name: str,
) -> str | None:
    if not tool_name or not isinstance(
        tool_name,
        str,
    ):
        return None

    wanted = set(
        re.findall(
            r"[a-z0-9]+",
            tool_name.lower(),
        )
    )

    if not wanted:
        return None

    best_name: str | None = None
    best_score = 0

    for name in self.available_tools:
        candidates = set(
            re.findall(
                r"[a-z0-9]+",
                name.lower(),
            )
        )

        score = len(
            wanted & candidates
        )

        if score > best_score:
            best_name = name
            best_score = score

    return (
        best_name
        if best_score >= 1
        else None
    )

def _filter_arguments_for_tool(
    self,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    spec = self.available_tools.get(
        tool_name
    )

    if spec is None:
        return arguments

    inputs = (
        getattr(
            spec,
            "arguments_schema",
            None,
        )
        or getattr(
            spec,
            "inputs",
            None,
        )
        or {}
    )

    if not arguments or not inputs:
        return arguments

    aliases: dict[str, str] = {
        "q": "query",
        "search": "query",
        "search_query": "query",
        "search_term": "query",
        "question": "query",
        "link": "url",
        "webpage": "url",
        "page_url": "url",
        "website": "url",
        "file": "file_path",
        "filepath": "file_path",
        "filename": "file_path",
        "path": "file_path",
        "image": "image_path",
        "image_file": "image_path",
        "spreadsheet": "file_path",
        "excel_path": "file_path",
        "python_code": "code",
        "script": "code",
    }

    repaired: dict[str, Any] = {}

    for key, value in arguments.items():
        if key in inputs:
            target = key
        else:
            target = aliases.get(key)

        if (
            target
            and target in inputs
            and target not in repaired
        ):
            repaired[target] = value

    return repaired

_SEARCH_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "did",
        "do",
        "does",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "from",
        "with",
        "about",
        "into",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "please",
        "just",
        "only",
        "all",
        "any",
        "some",
        "one",
        "between",
        "each",
        "other",
        "than",
        "then",
        "so",
        "if",
        "not",
        "no",
        "yes",
        "been",
        "being",
        "have",
        "has",
        "had",
        "also",
        "more",
        "most",
        "very",
        "out",
        "up",
        "down",
        "over",
        "under",
        "there",
        "here",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "shall",
        "question",
        "task",
        "answer",
    }
)

_FILE_PATH_ARGUMENT_TOOLS: dict[
    str,
    tuple[str, ...],
] = {
    "file_reader": ("file_path",),
    "analyze_excel": ("file_path",),
    "analyze_image": ("image_path",),
}

def _build_search_query(
    self,
    user_question: str,
) -> str:
    text = (user_question or "").strip()

    if not text:
        return ""

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    words = re.findall(
        r"[A-Za-z0-9'-]+",
        text,
    )

    keywords = [
        word
        for word in words
        if word.lower()
        not in self._SEARCH_STOPWORDS
    ]

    if not keywords:
        keywords = words

    return " ".join(
        keywords[:10]
    )

def _repair_file_arguments(
    self,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    path_args = (
        self._FILE_PATH_ARGUMENT_TOOLS.get(
            tool_name
        )
    )

    if not path_args:
        return arguments

    repaired = dict(arguments or {})
    files = list(self.available_files)

    for arg_name in path_args:
        value = repaired.get(arg_name)

        if value is None:
            continue

        value_str = (
            str(value)
            .strip()
            .strip("'\"")
        )

        if is_placeholder_path(
            value_str
        ):
            raise ValueError(
                f"File path argument "
                f"'{arg_name}' of tool "
                f"'{tool_name}' is a placeholder "
                f"('{value_str}'). "
                "Use a real file from "
                "AVAILABLE LOCAL FILES."
            )

        if not files:
            continue

        normalized = (
            value_str
            .lower()
            .replace("\\", "/")
        )

        normalized_name = Path(
            normalized
        ).name

        real = next(
            (
                candidate
                for candidate in files
                if candidate.lower()
                .replace("\\", "/")
                == normalized
            ),
            None,
        )

        if real is None:
            real = next(
                (
                    candidate
                    for candidate in files
                    if Path(candidate)
                    .name
                    .lower()
                    == normalized_name
                ),
                None,
            )

        if real is None:
            suffix = Path(
                normalized_name
            ).suffix.lower()

            if suffix:
                real = next(
                    (
                        candidate
                        for candidate in files
                        if Path(candidate)
                        .suffix
                        .lower()
                        == suffix
                    ),
                    None,
                )

        if real is None:
            raise ValueError(
                f"File path '{value_str}' "
                "does not match any real "
                f"attachment. Available files: "
                f"{files}."
            )

        if real != value_str:
            logger.warning(
                "Repaired file path '%s' -> '%s' "
                "for tool '%s'.",
                value_str,
                real,
                tool_name,
            )

        repaired[arg_name] = real

    return repaired

def get_alternative_strategy(
    self,
    *,
    user_question: str,
    failed_step: PlanStep,
    analysis: TaskAnalysis | None = None,
) -> PlanStep | None:
    if analysis is None:
        analysis = self.task_classifier.classify(
            user_question,
            available_files=self.available_files,
            available_tools=sorted(
                self.available_tools.keys()
            ),
        )

    failed_tool = (
        failed_step.tool_name
        if (
            failed_step is not None
            and failed_step.step_type
            == StepType.TOOL
        )
        else None
    )

    has_python = (
        "python_interpreter"
        in self.available_tools
    )

    has_web = (
        "web_search"
        in self.available_tools
    )

    has_visit = (
        "visit_webpage"
        in self.available_tools
    )

    url_match = re.search(
        r"https?://[^\s<>]+",
        user_question or "",
        re.IGNORECASE,
    )

    candidates: list[PlanStep] = []

    if (
        failed_tool == "visit_webpage"
        and has_web
        and "web_search"
        not in analysis.forbidden_tools
    ):
        candidates.append(
            PlanStep(
                step_id=0,
                action=(
                    "Search the web for "
                    "the required fact"
                ),
                step_type=StepType.TOOL,
                tool_name="web_search",
                arguments={
                    "query": (
                        self._build_search_query(
                            user_question
                        )
                    )
                },
                is_final_answer=False,
            )
        )

    if (
        failed_tool == "web_search"
        and has_visit
        and url_match
    ):
        candidates.append(
            PlanStep(
                step_id=0,
                action=(
                    "Visit the URL mentioned "
                    "in the task and extract "
                    "the required information"
                ),
                step_type=StepType.TOOL,
                tool_name="visit_webpage",
                arguments={
                    "url": url_match.group(0)
                },
                is_final_answer=False,
            )
        )

    file_step = self._file_fallback_step(
        user_question=user_question,
        failed_tool=failed_tool,
    )

    if file_step is not None:
        candidates.append(
            PlanStep(
                step_id=0,
                action=file_step["action"],
                step_type=StepType.TOOL,
                tool_name=file_step["tool_name"],
                arguments=file_step["arguments"],
                is_final_answer=False,
            )
        )

    if (
        has_python
        and failed_tool != "python_interpreter"
        and analysis.intent
        in (
            TaskIntent.ARITHMETIC,
            TaskIntent.TEXT_TRANSFORMATION,
        )
    ):
        code = self._deterministic_fallback_code(
            user_question
        )

        if code:
            candidates.append(
                PlanStep(
                    step_id=0,
                    action=(
                        "Compute the exact result "
                        "with Python code"
                    ),
                    step_type=StepType.TOOL,
                    tool_name="python_interpreter",
                    arguments={
                        "code": code
                    },
                    is_final_answer=False,
                )
            )

    failed_fingerprint = (
        self._fingerprint_step(
            failed_step
        )
        if failed_step is not None
        else None
    )

    for candidate in candidates:
        if failed_fingerprint is None:
            return candidate

        if (
            self._fingerprint_step(candidate)
            != failed_fingerprint
        ):
            return candidate

    return None

def _build_initial_prompt(
    self,
    *,
    user_question: str,
    context: Sequence[Any] | None,
    analysis: TaskAnalysis | None = None,
) -> str:
    analysis_text = ""

    if analysis is not None:
        analysis_text = (
            "TASK ANALYSIS "
            "(deterministic guidance):\n"
            f"- intent: "
            f"{analysis.intent.value}\n"
            f"- {analysis.analysis_text}\n"
        )

        if analysis.forbidden_tools:
            analysis_text += (
                "- DO NOT use these tools: "
                f"{', '.join(analysis.forbidden_tools)}\n"
            )

        if (
            analysis.recommended_first_tool
            is not None
        ):
            analysis_text += (
                "- Prefer the tool: "
                f"{analysis.recommended_first_tool}\n"
            )

    return f"""
```

Create an executable PlanSchema for this GAIA task.

USER REQUEST:
{user_question}

CONTEXT:
{self._format_context(context)}

AVAILABLE LOCAL FILES:
{self._format_available_files()}

{analysis_text}

AVAILABLE TOOLS:
{self._format_tools()}

PLANNING REQUIREMENTS:

1. Use only available tools.
2. Use exact tool names.
3. Use only arguments allowed by the tool schema.
4. Never invent file paths, URLs, artifact IDs, or tools.
5. Never put Python code into a web-search query.
6. Use the minimum number of steps required.
7. For multi-hop tasks, gather evidence before answering.
8. For numerical tasks, retrieve exact values and calculate when needed.
9. The final-answer step must be the LAST step.
10. The final-answer step must be an LLM step.
11. There must be exactly one final-answer step.
12. Never create a final-answer step merely to hide a failure.
13. Every TOOL step must contain a valid tool_name.
14. Every TOOL step must satisfy its exact contract.
15. TEXT TRANSFORMATION RULE:
    For reversing, counting, sorting, anagrams,
    encoding/decoding, or arithmetic, use
    python_interpreter whenever available.
16. NO SEARCH BY DEFAULT:
    Do not use web_search when reasoning, Python,
    local files, or supplied context are sufficient.
17. URL RULE:
    If the request contains a URL, visit that exact URL first.
18. WIKIPEDIA RULE:
    If Wikipedia is explicitly requested, use visit_webpage
    on the relevant Wikipedia URL.
19. SEARCH HYGIENE:
    web_search queries must be short keyword queries,
    normally 2-10 key terms, never the raw user question.

Return only a valid PlanSchema.
""".strip()

```
def _build_replan_prompt(
    self,
    *,
    user_question: str,
    context: Sequence[Any] | None,
    failed_step: PlanStep,
    failure: AgentError,
    analysis: TaskAnalysis | None = None,
) -> str:
    failure_type = self._get_failure_type(
        failure
    )

    failed_arguments = json.dumps(
        failed_step.arguments or {},
        ensure_ascii=False,
        default=str,
    )

    loop_guidance = ""

    if failure_type == "LoopDetected":
        loop_guidance = """
```

LOOP CONTEXT:
The previous strategy repeated itself without producing
new information.

If the existing CONTEXT already contains enough evidence,
use ONLY the final-answer LLM step.

Do NOT perform the same search again.

If evidence is insufficient, change the tool AND the
query/strategy substantially.
"""

```
    analysis_guidance = ""

    if analysis is not None:
        analysis_guidance = (
            "TASK ANALYSIS:\n"
            f"- intent: {analysis.intent.value}\n"
            f"- {analysis.analysis_text}\n"
        )

        if analysis.forbidden_tools:
            analysis_guidance += (
                "- DO NOT use these tools: "
                f"{', '.join(analysis.forbidden_tools)}\n"
            )

        if analysis.recommended_first_tool:
            analysis_guidance += (
                "- Prefer the tool: "
                f"{analysis.recommended_first_tool}\n"
            )

    return f"""
```

The previous execution strategy failed.

You MUST create a genuinely different execution strategy.

{loop_guidance}

USER REQUEST:
{user_question}

CONTEXT:
{self._format_context(context)}

{analysis_guidance}

EVIDENCE REUSE RULE:

If the CONTEXT already contains enough evidence to answer
the task, return ONLY the final-answer LLM step.

Do not repeat searches or service calls for information
already present in the CONTEXT.

FAILED STEP:
step_id: {failed_step.step_id}
action: {failed_step.action}
step_type: {failed_step.step_type.value}
tool_name: {failed_step.tool_name}
arguments: {failed_arguments}

FAILURE TYPE:
{failure_type}

FAILURE MESSAGE:
{failure.message}

AVAILABLE TOOLS:
{self._format_tools()}

REPLANNING RULES:

1. Do NOT repeat the failed tool call unchanged.
2. Do NOT repeat the same tool with the same arguments.
3. Do NOT invent tools.
4. Do NOT invent arguments.
5. If arguments were invalid, use the exact tool schema.
6. If the previous tool lacks the capability, use another tool.
7. If search failed, substantially change the search strategy.
8. Never invent files or artifact paths.
9. A failure is NOT evidence that the task is solved.
10. Never turn a failed execution into a final answer
    unless sufficient evidence already exists.
11. The final-answer step must always be LAST.
12. The final-answer step must always be an LLM step.
13. There must be exactly one final-answer step.
14. The final answer must be based on gathered evidence.
15. Prefer a different strategy over an identical retry.

Return only PlanSchema.
""".strip()

```
def _system_prompt(self) -> str:
    return f"""
```

You are the GAIA Benchmark Execution Planner.

Your job is to transform a user request into a precise,
executable PlanSchema.

AVAILABLE TOOLS AND CONTRACTS:
{self._format_tools()}

FIRST DECISION:

Determine whether a tool is actually required.

If the task can be answered using reasoning or supplied
context, use ONLY one final LLM step.

NEVER use web_search by default.

RULES:

1. Use only listed tools.
2. Tool names must match exactly.
3. TOOL steps must contain tool_name.
4. Never hallucinate tool names.
5. Never hallucinate tool arguments.
6. Never invent files or paths.
7. Use only arguments defined by each tool schema.
8. Never assume capabilities not explicitly provided.
9. Step IDs must start at 0 and increment by 1.
10. The final-answer step must be the LAST step.
11. The final-answer step must be an LLM step.
12. There must be exactly one final-answer step.
13. Never use a final-answer step to hide a failure.
14. Keep plans minimal and purposeful.

MULTI-HOP:
retrieve
-> inspect
-> extract
-> calculate if necessary
-> verify
-> final answer

NUMERICAL:
retrieve exact information
-> calculate
-> verify
-> final answer

FILE:
locate real file
-> inspect
-> extract
-> verify
-> final answer

SEARCH:
short keyword search
-> inspect relevant evidence
-> extract
-> final answer

TEXT TRANSFORMATION:
python_interpreter
-> final answer

If the request contains a URL:
visit_webpage that exact URL first.

If Wikipedia is explicitly requested:
visit the relevant Wikipedia page.

When replanning:

* change strategy when necessary
* repair invalid arguments
* select another tool when appropriate
* change search queries after failed searches
* never blindly repeat a failed call
* never fabricate evidence
* never claim success because execution failed

Return only a valid PlanSchema.
""".strip()

```
def _format_available_files(self) -> str:
    if not self.available_files:
        return "No local data files were detected."

    return "\n".join(
        f"- {name}"
        for name in self.available_files
    )

def _format_tools(self) -> str:
    if not self.available_tools:
        return "No tools available."

    lines: list[str] = []

    for tool in self.available_tools.values():
        schema = (
            getattr(
                tool,
                "arguments_schema",
                None,
            )
            or getattr(
                tool,
                "inputs",
                None,
            )
            or {}
        )

        description = getattr(
            tool,
            "description",
            "",
        )

        name = getattr(
            tool,
            "name",
            None,
        )

        if not name:
            continue

        lines.append(
            "\n".join(
                [
                    f"TOOL: {name}",
                    f"DESCRIPTION: {description}",
                    "ARGUMENT SCHEMA:",
                    json.dumps(
                        schema,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                ]
            )
        )

    if not lines:
        return "No valid tools available."

    return "\n\n".join(lines)

def _format_context(
    self,
    context: Sequence[Any] | None,
) -> str:
    if not context:
        return "No additional context."

    items = list(context)[-10:]
    formatted: list[str] = []

    for item in items:
        try:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()

            formatted.append(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,
                )
            )

        except Exception:
            formatted.append(
                str(item)
            )

    return "\n".join(
        f"- {item}"
        for item in formatted
    )

def _fingerprint_step(
    self,
    step: PlanStep,
) -> str:
    normalized_arguments = json.dumps(
        step.arguments or {},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return (
        f"{step.step_type.value}|"
        f"{step.tool_name or ''}|"
        f"{normalized_arguments}"
    )

def _get_failure_type(
    self,
    failure: AgentError,
) -> str:
    failure_type = getattr(
        failure,
        "failure_type",
        None,
    )

    if failure_type is not None:
        return getattr(
            failure_type,
            "value",
            str(failure_type),
        )

    error_type = getattr(
        failure,
        "error_type",
        None,
    )

    if error_type is not None:
        return getattr(
            error_type,
            "value",
            str(error_type),
        )

    return failure.__class__.__name__

def _emergency_fallback_plan(
    self,
    *,
    user_question: str,
    failed_step: PlanStep | None = None,
    analysis: TaskAnalysis | None = None,
) -> PlanSchema:
    failed_tool = (
        failed_step.tool_name
        if failed_step is not None
        else None
    )

    intent = (
        analysis.intent
        if analysis is not None
        else TaskIntent.UNKNOWN
    )

    has_python = (
        "python_interpreter"
        in self.available_tools
    )

    has_web = (
        "web_search"
        in self.available_tools
    )

    has_visit = (
        "visit_webpage"
        in self.available_tools
    )

    if intent in (
        TaskIntent.ARITHMETIC,
        TaskIntent.TEXT_TRANSFORMATION,
    ):
        if (
            has_python
            and failed_tool
            != "python_interpreter"
        ):
            code = (
                self._deterministic_fallback_code(
                    user_question
                )
            )

            if code:
                return self._tool_and_final_plan(
                    action=(
                        "Compute the exact result "
                        "with Python code"
                    ),
                    tool_name="python_interpreter",
                    arguments={
                        "code": code
                    },
                )

        return self._llm_only_plan()

    if intent == TaskIntent.LOCAL_FILE:
        file_step = self._file_fallback_step(
            user_question=user_question,
            failed_tool=failed_tool,
        )

        if file_step is not None:
            return self._tool_and_final_plan(
                action=file_step["action"],
                tool_name=file_step["tool_name"],
                arguments=file_step["arguments"],
            )

        return self._llm_only_plan()

    if intent == TaskIntent.URL_PAGE:
        url_match = re.search(
            r"https?://[^\s<>]+",
            user_question or "",
            re.IGNORECASE,
        )

        if (
            url_match
            and has_visit
            and failed_tool
            != "visit_webpage"
        ):
            return self._tool_and_final_plan(
                action=(
                    "Visit the URL and extract "
                    "the required information"
                ),
                tool_name="visit_webpage",
                arguments={
                    "url": url_match.group(0)
                },
            )

        return self._llm_only_plan()

    if (
        analysis is not None
        and "web_search"
        in (
            analysis.forbidden_tools
            or ()
        )
    ):
        return self._llm_only_plan()

    if (
        has_web
        and failed_tool
        != "web_search"
    ):
        query = self._build_search_query(
            user_question
        )

        if query:
            return self._tool_and_final_plan(
                action=(
                    "Search for relevant evidence"
                ),
                tool_name="web_search",
                arguments={
                    "query": query
                },
            )

    return self._llm_only_plan()

def _llm_only_plan(self) -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action=(
                    "Answer the user request directly "
                    "using reasoning and available context"
                ),
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            )
        ]
    )

def _tool_and_final_plan(
    self,
    *,
    action: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> PlanSchema:
    return PlanSchema(
        steps=[
            PlanStep(
                step_id=0,
                action=action,
                step_type=StepType.TOOL,
                tool_name=tool_name,
                arguments=arguments,
                is_final_answer=False,
            ),
            PlanStep(
                step_id=1,
                action=(
                    "Synthesize the final answer "
                    "using only the evidence obtained"
                ),
                step_type=StepType.LLM,
                tool_name=None,
                arguments={},
                is_final_answer=True,
            ),
        ]
    )

def _deterministic_fallback_code(
    self,
    user_question: str,
) -> str | None:
    ratio = detect_factorial_ratio(
        user_question
    )

    if ratio is not None:
        return (
            "import math\n"
            f"result = math.factorial({ratio[0]}) "
            f"// math.factorial({ratio[1]})"
        )

    expression = detect_simple_operation(
        user_question
    )

    if expression is not None:
        return (
            f"result = {expression}"
        )

    return None

def _file_fallback_step(
    self,
    *,
    user_question: str,
    failed_tool: str | None,
) -> dict[str, Any] | None:
    if not self.available_files:
        return None

    has_excel = (
        "analyze_excel"
        in self.available_tools
    )

    has_reader = (
        "file_reader"
        in self.available_tools
    )

    lowered_question = (
        user_question or ""
    ).lower()

    target: str | None = None

    for name in self.available_files:
        stem = (
            os.path.splitext(name)[0]
            .lower()
        )

        if (
            stem
            and stem in lowered_question
        ):
            target = name
            break

    if target is None:
        target = self.available_files[0]

    suffix = (
        os.path.splitext(target)[1]
        .lower()
    )

    if (
        suffix
        in {
            ".xlsx",
            ".xls",
            ".xlsm",
            ".csv",
        }
        and has_excel
        and failed_tool
        != "analyze_excel"
    ):
        return {
            "action": (
                "Analyze the spreadsheet data"
            ),
            "tool_name": "analyze_excel",
            "arguments": {
                "file_path": target,
                "question": user_question,
            },
        }

    if (
        has_reader
        and failed_tool
        != "file_reader"
    ):
        return {
            "action": (
                "Read the local file"
            ),
            "tool_name": "file_reader",
            "arguments": {
                "file_path": target
            },
        }

    return None


def detect_factorial_ratio(
user_question: str,
) -> tuple[int, int] | None:
text = (
user_question or ""
).lower()


patterns = [
    r"(\d+)\s*!?\s*/\s*(\d+)\s*!",
    r"(\d+)\s*factorial\s*/\s*(\d+)\s*factorial",
]

for pattern in patterns:
    match = re.search(
        pattern,
        text,
    )

    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
        )

return None


def detect_simple_operation(
user_question: str,
) -> str | None:
text = (
user_question or ""
).strip()


patterns = [
    r"(?<!\w)(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)(?!\w)",
    r"(?<!\w)(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?!\w)",
]

for pattern in patterns:
    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        continue

    left = match.group(1)
    operator = match.group(2)
    right = match.group(3)

    if operator.lower() == "x":
        operator = "*"

    return (
        f"{left} {operator} {right}"
    )

return None

