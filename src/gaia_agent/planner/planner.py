
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from ..reliability.errors import AgentError
from ..reliability.loop_detector import LoopDetector
from ..planner.plan_schema import PlanSchema, PlanStep, StepType
from ..tools.contract_validator import ToolContractValidator
from ..tools.path_utils import is_placeholder_path
from .task_classifier import TaskAnalysis, TaskClassifier, TaskIntent

logger = logging.getLogger(__name__)


class PlannerRecoveryRequired(Exception):
    """Raised when a generated plan violates a planner invariant."""


class Planner:
    """GAIA execution planner.

    Responsibilities:
    - classify the task
    - generate/repair a PlanSchema
    - validate tool contracts and plan structure
    - produce a genuinely different replan after non-transient failures

    Not responsible for:
    - tool execution
    - retries
    - recovery budgets
    - verification of the final answer
    - loop execution
    """

    MAX_PLAN_STEPS = 20
    MAX_CONTEXT_ITEMS = 10

    _STRATEGY_FAMILY = {
        "web_search": "WEB_SEARCH",
        "visit_webpage": "DIRECT_WEBPAGE",
        "analyze_image": "VISION",
        "analyze_excel": "FILE_ANALYSIS",
        "file_reader": "FILE_READER",
        "python_interpreter": "PYTHON",
    }

    _FILE_PATH_ARGUMENT_TOOLS = {
        "file_reader": ("file_path",),
        "analyze_excel": ("file_path",),
        "analyze_image": ("image_path",),
    }

    _SEARCH_STOPWORDS = frozenset(
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

    def __init__(
        self,
        *,
        client: Any,
        model: Any,
        available_tools: dict[str, Any] | list[Any],
        loop_detector: LoopDetector | None = None,
        available_files: Sequence[str] | None = None,
        base_dir: str = "."
    ) -> None:
        if client is None:
            raise ValueError("client cannot be None.")

        if model is None or (
            isinstance(model, str) and not model.strip()
        ):
            raise ValueError("model cannot be empty.")

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
        self._validate_question(user_question)
        self._current_question = user_question

        analysis = self._classify(user_question)

        deterministic = self._deterministic_plan(
            user_question,
            analysis,
        )

        if deterministic is not None:
            self._validate_generated_plan(
                deterministic,
                analysis=analysis,
            )
            return deterministic

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
                "Initial planner generation/validation failed: %s",
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
                "Replanned plan is empty."
            )

        return plan.steps[0]

    async def replan(
        self,
        user_question: str,
        context: Sequence[Any] | None,
        failed_step: PlanStep,
        failure: AgentError,
    ) -> PlanSchema:
        self._validate_question(user_question)

        if failed_step is None:
            raise ValueError("failed_step cannot be None.")

        if failure is None:
            raise ValueError("failure cannot be None.")

        self._current_question = user_question

        analysis = self._classify(user_question)

        failed_fp = self._fingerprint_step(failed_step)
        failure_type = self._get_failure_type(failure)

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
                failed_step_fingerprint=failed_fp,
                failed_step=failed_step,
                failure_type=failure_type,
                analysis=analysis,
            )

            return plan

        except Exception as exc:
            logger.exception(
                "Planner replan generation/validation failed: %s",
                exc,
            )

        alternative = self.get_alternative_strategy(
            user_question=user_question,
            failed_step=failed_step,
            analysis=analysis,
            failure_type=failure_type,
        )

        if alternative is not None:
            plan = self._tool_and_final_plan(
                action=alternative.action,
                tool_name=alternative.tool_name or "",
                arguments=alternative.arguments,
            )

            self._validate_generated_plan(
                plan,
                failed_step_fingerprint=failed_fp,
                failed_step=failed_step,
                failure_type=failure_type,
                analysis=analysis,
            )

            return plan

        if self._has_successful_evidence(context):
            plan = self._llm_only_plan()

            self._validate_generated_plan(
                plan,
                failed_step_fingerprint=failed_fp,
                failed_step=failed_step,
                failure_type=failure_type,
                analysis=analysis,
            )

            return plan

        raise PlannerRecoveryRequired(
            "Unable to construct a valid, meaningfully different "
            "recovery plan. "
            f"failed_tool={failed_step.tool_name!r}, "
            f"failure_type={failure_type!r}"
        )

    def _validate_generated_plan(
        self,
        plan: PlanSchema,
        *,
        failed_step_fingerprint: str | None = None,
        failed_step: PlanStep | None = None,
        failure_type: str | None = None,
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

        if analysis is not None and analysis.forbidden_tools:
            forbidden = set(analysis.forbidden_tools)

            for step in plan.steps:
                if (
                    step.step_type == StepType.TOOL
                    and step.tool_name in forbidden
                ):
                    raise PlannerRecoveryRequired(
                        f"Forbidden tool '{step.tool_name}' "
                        f"for {analysis.intent.value} task."
                    )

        fingerprints = [
            self._fingerprint_step(step)
            for step in plan.steps
            if not step.is_final_answer
        ]

        if len(fingerprints) != len(set(fingerprints)):
            raise PlannerRecoveryRequired(
                "Plan contains duplicate tool executions."
            )

        loop_result = self.loop_detector.check_plan(
            plan.steps
        )

        if loop_result.detected:
            raise PlannerRecoveryRequired(
                "Planner produced a repeated execution plan."
            )

        if failed_step_fingerprint is not None:
            if any(
                self._fingerprint_step(step)
                == failed_step_fingerprint
                for step in plan.steps
            ):
                raise PlannerRecoveryRequired(
                    "Replanned plan repeats the failed execution."
                )

        if failed_step is not None:
            self._validate_recovery_strategy(
                plan=plan,
                failed_step=failed_step,
                failure_type=failure_type or "",
            )

    def _validate_plan_structure(
        self,
        plan: PlanSchema,
    ) -> None:
        if not plan.steps:
            raise ValueError(
                "Plan must contain at least one step."
            )

        if len(plan.steps) > self.MAX_PLAN_STEPS:
            raise ValueError(
                "Plan contains too many steps. "
                f"Maximum is {self.MAX_PLAN_STEPS}."
            )

        for expected_id, step in enumerate(plan.steps):
            if step is None:
                raise ValueError(
                    f"Plan step {expected_id} cannot be None."
                )

            if step.step_id != expected_id:
                raise ValueError(
                    "Step IDs must be sequential: "
                    f"expected {expected_id}, got {step.step_id}."
                )

    def _validate_step(self, step: PlanStep) -> None:
        if not step.action or not step.action.strip():
            raise ValueError(
                "Step action cannot be empty."
            )

        if step.step_type == StepType.LLM:
            if step.tool_name is not None:
                raise ValueError(
                    "LLM step cannot specify tool_name."
                )

            if step.arguments:
                raise ValueError(
                    "LLM step cannot contain arguments."
                )

            return

        if step.step_type != StepType.TOOL:
            raise ValueError(
                f"Unsupported step type: {step.step_type}"
            )

        self._validate_tool_step(step)

    def _validate_tool_step(self, step: PlanStep) -> None:
        if (
            not isinstance(step.tool_name, str)
            or not step.tool_name
        ):
            raise ValueError(
                "TOOL step must specify tool_name."
            )

        if step.tool_name.lower() in {
            "final_answer",
            "final",
            "answer",
        }:
            raise PlannerRecoveryRequired(
                "Final-answer pseudo-tool is not allowed."
            )

        if step.tool_name not in self.available_tools:
            raise ValueError(
                f"Unknown or unavailable tool: "
                f"{step.tool_name}"
            )

        if not isinstance(step.arguments, dict):
            raise ValueError(
                f"Arguments for '{step.tool_name}' "
                "must be a dictionary."
            )

        step.arguments = self._normalize_arguments(
            step.tool_name,
            step.arguments,
        )

        if step.tool_name == "web_search":
            self._validate_search_arguments(step)

        step.arguments = self._repair_file_arguments(
            step.tool_name,
            step.arguments,
        )

        ToolContractValidator.validate_step_contract(
            step,
            self.available_tools,
        )

    def _validate_final_answer_structure(
        self,
        plan: PlanSchema,
    ) -> None:
        finals = [
            step
            for step in plan.steps
            if step.is_final_answer
        ]

        if len(finals) != 1:
            raise ValueError(
                "Plan must contain exactly one "
                "final-answer step."
            )

        final = finals[0]

        if final.step_id != len(plan.steps) - 1:
            raise ValueError(
                "Final-answer step must be the last step."
            )

        if final.step_type != StepType.LLM:
            raise ValueError(
                "Final-answer step must be an LLM step."
            )

        if final.tool_name is not None or final.arguments:
            raise ValueError(
                "Final-answer step cannot contain "
                "tool data."
            )

    def _validate_recovery_strategy(
        self,
        *,
        plan: PlanSchema,
        failed_step: PlanStep,
        failure_type: str,
    ) -> None:
        if failed_step.step_type != StepType.TOOL:
            return

        non_final = [
            step
            for step in plan.steps
            if (
                not step.is_final_answer
                and step.step_type == StepType.TOOL
            )
        ]

        if not non_final:
            return

        normalized = (
            failure_type.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        transient = any(
            item in normalized
            for item in (
                "timeout",
                "rate_limit",
                "transient",
                "connection",
            )
        )

        invalid_args = any(
            item in normalized
            for item in (
                "invalid_argument",
                "validation",
                "schema",
                "argument",
            )
        )

        if transient or invalid_args:
            return

        failed_family = self._strategy_family(
            failed_step
        )

        families = {
            self._strategy_family(step)
            for step in non_final
        }

        if families == {failed_family}:
            raise PlannerRecoveryRequired(
                "Recovery did not change strategy family: "
                + failed_family
            )

    def get_alternative_strategy(
        self,
        *,
        user_question: str,
        failed_step: PlanStep,
        analysis: TaskAnalysis | None = None,
        failure_type: str | None = None,
    ) -> PlanStep | None:
        analysis = analysis or self._classify(
            user_question
        )

        failed_tool = (
            failed_step.tool_name
            if failed_step.step_type == StepType.TOOL
            else None
        )

        failed_family = self._strategy_family(
            failed_step
        )

        failure_type = failure_type or ""

        candidates: list[PlanStep] = []

        url = self._extract_url(user_question)

        if (
            failed_tool == "web_search"
            and url
            and "visit_webpage" in self.available_tools
        ):
            candidates.append(
                self._tool_step(
                    action=(
                        "Visit the exact URL provided and "
                        "extract the required information"
                    ),
                    tool_name="visit_webpage",
                    arguments={"url": url},
                )
            )

        if (
            failed_tool == "visit_webpage"
            and "web_search" in self.available_tools
            and "web_search"
            not in (analysis.forbidden_tools or ())
        ):
            query = self._build_search_query(
                user_question
            )

            if query:
                candidates.append(
                    self._tool_step(
                        action=(
                            "Search for independent evidence "
                            "using concise factual keywords"
                        ),
                        tool_name="web_search",
                        arguments={"query": query},
                    )
                )

        file_step = self._file_fallback_step(
            user_question=user_question,
            failed_tool=failed_tool,
        )

        if file_step:
            candidates.append(
                self._tool_step(**file_step)
            )

        if (
            "python_interpreter" in self.available_tools
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
                    self._tool_step(
                        action=(
                            "Compute the exact result "
                            "with Python"
                        ),
                        tool_name="python_interpreter",
                        arguments={"code": code},
                    )
                )

        for candidate in candidates:
            if (
                self._fingerprint_step(candidate)
                == self._fingerprint_step(failed_step)
            ):
                continue

            if (
                self._strategy_family(candidate)
                == failed_family
            ):
                continue

            return candidate

        return None

    def _deterministic_plan(
        self,
        question: str,
        analysis: TaskAnalysis,
    ) -> PlanSchema | None:
        if analysis.intent in (
            TaskIntent.ARITHMETIC,
            TaskIntent.TEXT_TRANSFORMATION,
        ):
            if "python_interpreter" in self.available_tools:
                code = self._deterministic_fallback_code(
                    question
                )

                if code:
                    return self._tool_and_final_plan(
                        action=(
                            "Compute the exact result "
                            "with Python"
                        ),
                        tool_name="python_interpreter",
                        arguments={"code": code},
                    )

        if analysis.intent == TaskIntent.URL_PAGE:
            url = self._extract_url(question)

            if (
                url
                and "visit_webpage"
                in self.available_tools
            ):
                return self._tool_and_final_plan(
                    action=(
                        "Visit the exact URL and extract "
                        "the requested information"
                    ),
                    tool_name="visit_webpage",
                    arguments={"url": url},
                )

        if analysis.intent == TaskIntent.IMAGE:
            if "analyze_image" in self.available_tools:
                target = self._select_file(
                    question,
                    self._image_extensions(),
                )

                if target:
                    return self._tool_and_final_plan(
                        action=(
                            "Analyze the provided image "
                            "and answer the requested question"
                        ),
                        tool_name="analyze_image",
                        arguments={
                            "image_path": target,
                            "question": question,
                        },
                    )

        return None

    def _emergency_fallback_plan(
        self,
        *,
        user_question: str,
        failed_step: PlanStep | None = None,
        analysis: TaskAnalysis | None = None,
    ) -> PlanSchema:
        analysis = analysis or self._classify(
            user_question
        )

        failed_tool = (
            failed_step.tool_name
            if failed_step
            else None
        )

        if analysis.intent in (
            TaskIntent.ARITHMETIC,
            TaskIntent.TEXT_TRANSFORMATION,
        ):
            if (
                "python_interpreter"
                in self.available_tools
                and failed_tool != "python_interpreter"
            ):
                code = self._deterministic_fallback_code(
                    user_question
                )

                if code:
                    return self._tool_and_final_plan(
                        action=(
                            "Compute the exact result "
                            "with Python"
                        ),
                        tool_name="python_interpreter",
                        arguments={"code": code},
                    )

        if analysis.intent == TaskIntent.URL_PAGE:
            url = self._extract_url(user_question)

            if (
                url
                and "visit_webpage"
                in self.available_tools
                and failed_tool != "visit_webpage"
            ):
                return self._tool_and_final_plan(
                    action=(
                        "Visit the exact URL and extract "
                        "the requested information"
                    ),
                    tool_name="visit_webpage",
                    arguments={"url": url},
                )

        file_step = self._file_fallback_step(
            user_question=user_question,
            failed_tool=failed_tool,
        )

        if file_step:
            return self._tool_and_final_plan(
                **file_step
            )

        if (
            "web_search" in self.available_tools
            and failed_tool != "web_search"
            and "web_search"
            not in (analysis.forbidden_tools or ())
        ):
            query = self._build_search_query(
                user_question
            )

            if query:
                return self._tool_and_final_plan(
                    action=(
                        "Search for relevant "
                        "independent evidence"
                    ),
                    tool_name="web_search",
                    arguments={"query": query},
                )

        return self._llm_only_plan()

    def _build_initial_prompt(
        self,
        *,
        user_question: str,
        context: Sequence[Any] | None,
        analysis: TaskAnalysis,
    ) -> str:
        return f"""
Create an executable PlanSchema for this GAIA task.

USER REQUEST:

{user_question}

TASK ANALYSIS:

- intent: {analysis.intent.value}
- needs_external_info: {analysis.needs_external_info}
- recommended_first_tool: {analysis.recommended_first_tool}
- analysis: {analysis.analysis_text}
- forbidden_tools: {', '.join(analysis.forbidden_tools) or 'none'}

CURRENT CONTEXT:

{self._format_context(context)}

AVAILABLE LOCAL FILES:

{self._format_available_files()}

AVAILABLE TOOLS:

{self._format_tools()}

RULES:

1. Use only listed tools and exact tool names.
2. Use only arguments allowed by each tool schema.
3. Never invent files, URLs, artifact IDs, tools, or arguments.
4. Do not use web_search when supplied context, reasoning, Python, or a real local artifact is sufficient.
5. Web queries must be concise factual keywords, not the raw question and never Python code.
6. For a direct URL, visit that exact URL first when visit_webpage is available.
7. For image/file tasks, use only a real file from AVAILABLE LOCAL FILES.
8. Never select an arbitrary local file merely because one exists.
9. For arithmetic/text transformations, prefer python_interpreter when available.
10. Multi-hop: retrieve -> inspect/extract -> calculate if necessary -> verify -> final.
11. Keep the plan minimal and purposeful.
12. Exactly one final-answer step.
13. Final-answer step is an LLM step and MUST be last.
14. Never use a final step to hide a failed tool call.
15. Return only a valid PlanSchema.
""".strip()

    def _build_replan_prompt(
        self,
        *,
        user_question: str,
        context: Sequence[Any] | None,
        failed_step: PlanStep,
        failure: AgentError,
        analysis: TaskAnalysis,
    ) -> str:
        failure_type = self._get_failure_type(
            failure
        )

        failed_arguments = json.dumps(
            failed_step.arguments or {},
            ensure_ascii=False,
            default=str,
        )

        failed_family = self._strategy_family(
            failed_step
        )

        return f"""
Create a replacement PlanSchema after an execution failure.

USER REQUEST:

{user_question}

TASK ANALYSIS:

- intent: {analysis.intent.value}
- needs_external_info: {analysis.needs_external_info}
- recommended_first_tool: {analysis.recommended_first_tool}
- analysis: {analysis.analysis_text}
- forbidden_tools: {', '.join(analysis.forbidden_tools) or 'none'}

CURRENT CONTEXT / EVIDENCE:

{self._format_context(context)}

FAILED STEP:

- step_id: {failed_step.step_id}
- action: {failed_step.action}
- step_type: {failed_step.step_type.value}
- tool_name: {failed_step.tool_name}
- strategy_family: {failed_family}
- arguments: {failed_arguments}

FAILURE TYPE:

{failure_type}

FAILURE MESSAGE:

{getattr(failure, "message", str(failure))}

AVAILABLE LOCAL FILES:

{self._format_available_files()}

AVAILABLE TOOLS:

{self._format_tools()}

REPLANNING RULES:

1. Do not repeat the failed execution or its exact arguments.
2. A different query using the same failed capability is NOT automatically a different strategy.
3. For capability, access, blocked-source, or loop failures, change strategy family.
4. For invalid-argument/schema failures, the same tool is allowed ONLY after correcting its contract.
5. Do not retry transient failures here; ReliabilityEngine owns transient retry policy.
6. Never invent a tool, argument, URL, artifact ID, or file path.
7. Never select an arbitrary local file.
8. A failure is never evidence that the answer is known.
9. Reuse successful evidence already in context.
10. If sufficient evidence already exists, use only the final LLM step.
11. If web_search failed and a direct URL exists, visit the exact URL when applicable.
12. If visit_webpage failed, use independent search when applicable.
13. If a required artifact cannot be resolved, fail cleanly rather than fabricate a path.
14. Exactly one final-answer step; it must be last and must be an LLM step.
15. Return only a valid PlanSchema.
""".strip()

    def _system_prompt(self) -> str:
        return f"""
You are the GAIA Benchmark Execution Planner.

Your output is a PlanSchema, not an answer.

AVAILABLE TOOLS AND CONTRACTS:

{self._format_tools()}

CORE RULES:

- Use only available tools.
- Exact tool names only.
- Exact argument schemas only.
- Never invent files, URLs, artifact IDs, tools, or evidence.
- Never use a final-answer pseudo-tool.
- Exactly one final-answer LLM step, and it must be last.
- Keep plans minimal.
- Never use web_search by default.
- Web queries are short factual keywords.
- Never put Python code into web-search queries.
- For local artifacts, use only paths explicitly listed as available.
- For image tasks, do not invent an image filename.
- For replanning, change capability/strategy after a non-transient capability failure.
- Invalid arguments may be repaired only according to the exact tool schema.
- Never turn a failed execution into evidence or a final answer.

MULTI-HOP:

retrieve -> inspect/extract -> calculate if necessary -> verify -> final

Return only a valid PlanSchema.
""".strip()

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

        blocks = []

        for name, tool in self.available_tools.items():
            schema = (
                getattr(tool, "arguments_schema", None)
                or getattr(tool, "inputs", None)
                or {}
            )

            description = getattr(
                tool,
                "description",
                "",
            )

            blocks.append(
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

        return (
            "\n\n".join(blocks)
            or "No valid tools available."
        )

    def _format_context(
        self,
        context: Sequence[Any] | None,
    ) -> str:
        if not context:
            return "No additional context."

        formatted = []

        for item in list(context)[
            -self.MAX_CONTEXT_ITEMS:
        ]:
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
                formatted.append(str(item))

        return "\n".join(
            f"- {item}"
            for item in formatted
        )

    def _normalize_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        spec = self.available_tools.get(tool_name)

        inputs = (
            getattr(spec, "arguments_schema", None)
            or getattr(spec, "inputs", None)
            or {}
        )

        if not isinstance(inputs, dict):
            return dict(arguments)

        aliases = {
            "q": "query",
            "search": "query",
            "search_query": "query",
            "search_term": "query",
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

        out = dict(arguments)

        for key, value in list(arguments.items()):
            target = (
                key
                if key in inputs
                else aliases.get(key)
            )

            if (
                target in inputs
                and target not in out
            ):
                out[target] = value

        return out

    def _validate_search_arguments(
        self,
        step: PlanStep,
    ) -> None:
        query = str(
            (step.arguments or {}).get(
                "query",
                "",
            )
            or ""
        ).strip()

        if not query:
            raise ValueError(
                "web_search requires a non-empty query."
            )

        if len(query.split()) > 15:
            raise ValueError(
                "web_search query is too long; "
                "use concise factual keywords."
            )

        code_pattern = re.compile(
            r"(?:"
            r"\bimport\s+\w+"
            r"|\bfrom\s+\w+\s+import\b"
            r"|\bprint\s*\("
            r"|\bmath\.\w+"
            r"|\bresult\s*="
            r"|\bpython\b"
            r")",
            re.IGNORECASE,
        )

        if code_pattern.search(query):
            raise PlannerRecoveryRequired(
                "Python/code detected in "
                "web-search query."
            )

    def _extract_url(
        self,
        text: str,
    ) -> str | None:
        match = re.search(
            r"https?://[^\s<>\"']+",
            text or "",
            re.IGNORECASE,
        )

        if not match:
            return None

        return match.group(0).rstrip(
            ".,;:!?)]}"
        )

    def _build_search_query(
        self,
        user_question: str,
    ) -> str:
        text = re.sub(
            r"https?://[^\s<>\"']+",
            " ",
            user_question or "",
            flags=re.IGNORECASE,
        )

        words = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_'/-]*",
            text,
        )

        keywords = [
            word
            for word in words
            if word.lower()
            not in self._SEARCH_STOPWORDS
        ]

        return " ".join(
            keywords[:10] or words[:10]
        )

    def _validate_question(
        self,
        user_question: str,
    ) -> None:
        if not isinstance(user_question, str):
            raise TypeError(
                "user_question must be a string."
            )

        if not user_question.strip():
            raise ValueError(
                "user_question cannot be empty."
            )

    def _classify(
        self,
        user_question: str,
    ) -> TaskAnalysis:
        return self.task_classifier.classify(
            user_question
        )

    def _fingerprint_step(
        self,
        step: PlanStep,
    ) -> str:
        arguments = json.dumps(
            step.arguments or {},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        return "|".join(
            [
                str(step.step_type),
                str(step.tool_name or ""),
                step.action.strip().lower(),
                arguments,
            ]
        )

    def _strategy_family(
        self,
        step: PlanStep,
    ) -> str:
        if step.step_type == StepType.LLM:
            return "LLM"

        return self._STRATEGY_FAMILY.get(
            step.tool_name or "",
            f"TOOL:{step.tool_name or 'UNKNOWN'}",
        )

    def _get_failure_type(
        self,
        failure: AgentError,
    ) -> str:
        for attribute in (
            "failure_type",
            "error_type",
            "code",
            "reason",
        ):
            value = getattr(
                failure,
                attribute,
                None,
            )

            if value:
                return str(value)

        return type(failure).__name__

    def _has_successful_evidence(
        self,
        context: Sequence[Any] | None,
    ) -> bool:
        if not context:
            return False

        for item in context:
            text = str(item).strip().lower()

            if not text:
                continue

            failure_markers = (
                "error",
                "failed",
                "failure",
                "exception",
                "timeout",
                "blocked",
            )

            if not any(
                marker in text
                for marker in failure_markers
            ):
                return True

        return False

    def _tool_step(
        self,
        *,
        action: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PlanStep:
        return PlanStep(
            step_id=0,
            action=action,
            step_type=StepType.TOOL,
            tool_name=tool_name,
            arguments=arguments,
            is_final_answer=False,
        )

    def _tool_and_final_plan(
        self,
        *,
        action: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PlanSchema:
        tool_step = PlanStep(
            step_id=0,
            action=action,
            step_type=StepType.TOOL,
            tool_name=tool_name,
            arguments=arguments,
            is_final_answer=False,
        )

        final_step = PlanStep(
            step_id=1,
            action=(
                "Use the verified tool result and "
                "produce the final answer"
            ),
            step_type=StepType.LLM,
            tool_name=None,
            arguments={},
            is_final_answer=True,
        )

        return PlanSchema(
            steps=[
                tool_step,
                final_step,
            ]
        )

    def _llm_only_plan(self) -> PlanSchema:
        return PlanSchema(
            steps=[
                PlanStep(
                    step_id=0,
                    action=(
                        "Use the available context and "
                        "produce the final answer"
                    ),
                    step_type=StepType.LLM,
                    tool_name=None,
                    arguments={},
                    is_final_answer=True,
                )
            ]
        )

    def _image_extensions(self) -> tuple[str, ...]:
        return (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".bmp",
            ".tiff",
            ".tif",
        )

    def _select_file(
        self,
        question: str,
        extensions: Sequence[str],
    ) -> str | None:
        if not self.available_files:
            return None

        question_lower = question.lower()

        candidates = []

        for file_name in self.available_files:
            path = Path(str(file_name))

            if path.suffix.lower() not in extensions:
                continue

            candidates.append(path)

        if not candidates:
            return None

        question_tokens = set(
            re.findall(
                r"[a-zA-Z0-9_-]+",
                question_lower,
            )
        )

        scored: list[tuple[int, Path]] = []

        for path in candidates:
            name_lower = path.name.lower()

            score = sum(
                1
                for token in question_tokens
                if token and token in name_lower
            )

            scored.append(
                (score, path)
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_path = scored[0]

        if best_score <= 0 and len(scored) > 1:
            return None

        return str(best_path)

    def _file_fallback_step(
        self,
        *,
        user_question: str,
        failed_tool: str | None,
    ) -> dict[str, Any] | None:
        if not self.available_files:
            return None

        question_lower = user_question.lower()

        image_requested = any(
            word in question_lower
            for word in (
                "image",
                "picture",
                "photo",
                "screenshot",
            )
        )

        spreadsheet_requested = any(
            word in question_lower
            for word in (
                "excel",
                "spreadsheet",
                "xlsx",
                "csv",
            )
        )

        if image_requested:
            if (
                "analyze_image"
                in self.available_tools
                and failed_tool != "analyze_image"
            ):
                target = self._select_file(
                    user_question,
                    self._image_extensions(),
                )

                if target:
                    return {
                        "action": (
                            "Analyze the provided image "
                            "and extract the required information"
                        ),
                        "tool_name": "analyze_image",
                        "arguments": {
                            "image_path": target,
                            "question": user_question,
                        },
                    }

        if spreadsheet_requested:
            if (
                "analyze_excel"
                in self.available_tools
                and failed_tool != "analyze_excel"
            ):
                target = self._select_file(
                    user_question,
                    (
                        ".xlsx",
                        ".xls",
                        ".csv",
                    ),
                )

                if target:
                    return {
                        "action": (
                            "Analyze the provided spreadsheet "
                            "and extract the required information"
                        ),
                        "tool_name": "analyze_excel",
                        "arguments": {
                            "file_path": target,
                            "question": user_question,
                        },
                    }

        if (
            "file_reader" in self.available_tools
            and failed_tool != "file_reader"
        ):
            target = self._select_file(
                user_question,
                (
                    ".txt",
                    ".md",
                    ".json",
                    ".pdf",
                    ".docx",
                    ".csv",
                    ".xlsx",
                    ".xls",
                ),
            )

            if target:
                return {
                    "action": (
                        "Read the relevant provided "
                        "file and extract the required information"
                    ),
                    "tool_name": "file_reader",
                    "arguments": {
                        "file_path": target,
                    },
                }

        return None

    def _repair_file_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        required_keys = self._FILE_PATH_ARGUMENT_TOOLS.get(
            tool_name
        )

        if not required_keys:
            return arguments

        repaired = dict(arguments)

        for key in required_keys:
            value = repaired.get(key)

            if value is None:
                continue

            if not isinstance(value, str):
                continue

            if is_placeholder_path(value):
                raise PlannerRecoveryRequired(
                    f"Placeholder path is not allowed "
                    f"for '{tool_name}': {value!r}"
                )

            path = Path(value)

            if path.is_absolute():
                continue

            base_path = Path(self.base_dir)

            candidate = base_path / path

            if candidate.exists():
                repaired[key] = str(candidate)

        return repaired

    def _deterministic_fallback_code(
        self,
        question: str,
    ) -> str | None:
        factorial_ratio = detect_factorial_ratio(
            question
        )

        if factorial_ratio:
            left, right = factorial_ratio

            return (
                "import math\n"
                f"left = math.factorial({left})\n"
                f"right = math.factorial({right})\n"
                "print(left / right)"
            )

        operation = detect_simple_operation(
            question
        )

        if operation:
            return (
                f"result = {operation}\n"
                "print(result)"
            )

        return None


def detect_factorial_ratio(
    user_question: str,
) -> tuple[int, int] | None:
    text = (user_question or "").lower()

    patterns = (
        r"(\d+)\s*!\s*/\s*(\d+)\s*!",
        r"(\d+)\s+factorial\s*/\s*(\d+)\s+factorial",
    )

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
    text = (user_question or "").strip()

    patterns = (
        r"(?<!\w)"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"([+\-*/])"
        r"\s*"
        r"(\d+(?:\.\d+)?)"
        r"(?!\w)",
        r"(?<!\w)"
        r"(\d+(?:\.\d+)?)"
        r"\s*x\s*"
        r"(\d+(?:\.\d+)?)"
        r"(?!\w)",
    )

    for index, pattern in enumerate(patterns):
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if not match:
            continue

        if index == 0:
            op = match.group(2)
            left = match.group(1)
            right = match.group(3)
        else:
            op = "*"
            left = match.group(1)
            right = match.group(2)

        return f"{left} {op} {right}"

    return None

