from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .plan_schema import PlanSchema, PlanStep, StepType
from .strategy_selector import StrategyDecision
from .task_classifier import TaskAnalysis, TaskIntent


class SemanticPlanError(ValueError):

class SemanticPlanValidator:
  

    def validate(
        self,
        plan: PlanSchema,
        *,
        analysis: TaskAnalysis,
        strategy: StrategyDecision,
        available_files: Sequence[str] = (),
        failed_strategy: str | None = None,
        failure_type: str | None = None,
        strategy_family_resolver: Callable[[PlanStep], str] | None = None,
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


        if (
            analysis.intent == TaskIntent.AUDIO_VIDEO
            and strategy.deterministic
            and strategy.primary_tool
        ):
            self._require_media_grounding(
                plan,
                expected=strategy.primary_tool,
            )

        if failed_strategy and failure_type:
            if strategy_family_resolver is None:
                raise SemanticPlanError(
                    "strategy_family_resolver is required for recovery validation."
                )
            self._validate_recovery_family(
                plan,
                failed_strategy=failed_strategy,
                failure_type=failure_type,
                strategy_family_resolver=strategy_family_resolver,
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
    def _require_media_grounding(
        plan: PlanSchema,
        *,
        expected: str,
    ) -> None:
        used = {
            step.tool_name
            for step in plan.steps
            if step.step_type == StepType.TOOL and not step.is_final_answer
        }

        if expected not in used:
            raise SemanticPlanError(
                "AUDIO_VIDEO tasks must ground the answer in the media "
                f"itself with '{expected}'; this plan never calls it."
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
        strategy_family_resolver: Callable[[PlanStep], str],
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
            strategy_family_resolver(step)
            for step in non_final
        }
        if families == {failed_strategy}:
            raise SemanticPlanError(
                f"Recovery kept the failed strategy family '{failed_strategy}'."
            )
