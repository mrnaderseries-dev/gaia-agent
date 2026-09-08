from __future__ import annotations

from pathlib import Path
from typing import Sequence

from plan_schema import PlanSchema, PlanStep, StepType
from strategy_selector import StrategyDecision, StrategyFamily
from task_classifier import TaskAnalysis, TaskIntent


class SemanticPlanError(ValueError):
    """Raised when a structurally valid plan is semantically wrong."""


class SemanticPlanValidator:
    """Validates whether a PlanSchema matches the task's selected strategy."""

    def validate(
        self,
        plan: PlanSchema,
        *,
        analysis: TaskAnalysis,
        strategy: StrategyDecision,
        available_files: Sequence[str] = (),
        failed_strategy: str | None = None,
        failure_type: str | None = None,
    ) -> None:
        for step in plan.steps:
            if step.step_type != StepType.TOOL or step.is_final_answer:
                continue
            self._validate_tool_step(
                step,
                analysis=analysis,
                strategy=strategy,
                available_files=available_files,
            )

        if failed_strategy and failure_type:
            self._validate_recovery_family(
                plan,
                failed_strategy=failed_strategy,
                failure_type=failure_type,
            )

    def _validate_tool_step(
        self,
        step: PlanStep,
        *,
        analysis: TaskAnalysis,
        strategy: StrategyDecision,
        available_files: Sequence[str],
    ) -> None:
        tool = step.tool_name or ""

        if tool in analysis.forbidden_tools:
            raise SemanticPlanError(
                f"Tool '{tool}' is forbidden for {analysis.intent.value} tasks."
            )

        if analysis.intent == TaskIntent.ARITHMETIC and tool == "web_search":
            raise SemanticPlanError("ARITHMETIC tasks cannot use web_search.")

        if analysis.intent == TaskIntent.TEXT_TRANSFORMATION and tool == "web_search":
            raise SemanticPlanError("TEXT_TRANSFORMATION tasks cannot use web_search.")

        if analysis.intent == TaskIntent.URL_PAGE and tool == "web_search":
            raise SemanticPlanError(
                "A direct URL must use visit_webpage first when that capability is available."
            )

        if analysis.intent == TaskIntent.IMAGE and self._has_image(available_files):
            if tool == "web_search":
                raise SemanticPlanError(
                    "A real local image exists; IMAGE tasks must use analyze_image instead of web_search."
                )

        if analysis.intent == TaskIntent.LOCAL_FILE and self._has_spreadsheet(available_files):
            if tool == "web_search":
                raise SemanticPlanError(
                    "A real spreadsheet artifact exists; analyze the local artifact instead of searching the web."
                )

        if strategy.primary_tool and strategy.deterministic:
            expected = strategy.primary_tool
            if analysis.intent in {
                TaskIntent.ARITHMETIC,
                TaskIntent.TEXT_TRANSFORMATION,
                TaskIntent.URL_PAGE,
                TaskIntent.IMAGE,
            } and tool != expected:
                raise SemanticPlanError(
                    f"{analysis.intent.value} requires strategy tool '{expected}', got '{tool}'."
                )

    @staticmethod
    def _has_image(files: Sequence[str]) -> bool:
        return any(
            Path(name).suffix.lower()
            in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
            for name in files
        )

    @staticmethod
    def _has_spreadsheet(files: Sequence[str]) -> bool:
        return any(
            Path(name).suffix.lower()
            in {".xlsx", ".xls", ".xlsm", ".csv"}
            for name in files
        )

    @staticmethod
    def _validate_recovery_family(
        plan: PlanSchema,
        *,
        failed_strategy: str,
        failure_type: str,
    ) -> None:
        normalized = failure_type.lower().replace("-", "_").replace(" ", "_")
        invalid_args = any(
            marker in normalized
            for marker in ("invalid_argument", "validation", "schema", "argument", "contract")
        )
        if invalid_args:
            return

        non_final = [
            step
            for step in plan.steps
            if step.step_type == StepType.TOOL and not step.is_final_answer
        ]
        if not non_final:
            return

        families = {
            SemanticPlanValidator.strategy_family(step)
            for step in non_final
        }
        if families == {failed_strategy}:
            raise SemanticPlanError(
                f"Recovery kept the failed strategy family '{failed_strategy}'."
            )

    @staticmethod
    def strategy_family(step: PlanStep) -> str:
        if step.step_type == StepType.LLM:
            return StrategyFamily.LLM_ONLY.value
        return {
            "web_search": StrategyFamily.WEB_RETRIEVAL.value,
            "visit_webpage": StrategyFamily.DIRECT_URL.value,
            "analyze_image": StrategyFamily.VISION.value,
            "analyze_excel": StrategyFamily.FILE_ANALYSIS.value,
            "file_reader": StrategyFamily.FILE_READING.value,
            "python_interpreter": StrategyFamily.LOCAL_COMPUTATION.value,
        }.get(step.tool_name or "", f"TOOL:{step.tool_name or 'UNKNOWN'}")
