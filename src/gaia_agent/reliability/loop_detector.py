from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Sequence

from gaia_agent.planner.plan_schema import PlanStep, StepType


class LoopType(str, Enum):
    NONE = "none"
    EXACT = "exact"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass(frozen=True, slots=True)
class LoopDetection:
    detected: bool
    loop_type: LoopType = LoopType.NONE
    similarity: float = 0.0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StepSignature:
    exact: str
    structural: str
    semantic: str


class LoopDetector:
    """
    Detects repeated execution attempts at three levels:

    1. EXACT
       Same step type, tool and arguments.

    2. STRUCTURAL
       Same execution structure and strategy family.

    3. SEMANTIC
       Same underlying execution objective even when:
         - action wording changes
         - tool changes
         - strategy family changes
         - argument wording changes

    The detector is intentionally deterministic.
    It does not own planning, recovery or retry decisions.
    """

    def __init__(
        self,
        *,
        semantic_threshold: float = 0.88,
        max_history: int = 50,
    ) -> None:
        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError(
                "semantic_threshold must be between 0 and 1."
            )

        if max_history <= 0:
            raise ValueError(
                "max_history must be greater than zero."
            )

        self.semantic_threshold = semantic_threshold
        self.max_history = max_history
        self._history: list[StepSignature] = []
    def check(
        self,
        step: PlanStep,
        *,
        strategy_family: str,
    ) -> LoopDetection:
        """
        Check whether a step repeats something already executed.

        Does not mutate detector history.
        """

        candidate = self.signature(
            step,
            strategy_family=strategy_family,
        )
        for previous in reversed(self._history):
            if candidate.exact == previous.exact:
                return LoopDetection(
                    detected=True,
                    loop_type=LoopType.EXACT,
                    similarity=1.0,
                    reason=(
                        "The exact execution step was repeated."
                    ),
                )
        for previous in reversed(self._history):
            if candidate.structural == previous.structural:
                return LoopDetection(
                    detected=True,
                    loop_type=LoopType.STRUCTURAL,
                    similarity=1.0,
                    reason=(
                        "The same execution structure and "
                        "strategy were repeated."
                    ),
                )


        for previous in reversed(self._history):
            similarity = SequenceMatcher(
                None,
                candidate.semantic,
                previous.semantic,
            ).ratio()

            if similarity >= self.semantic_threshold:
                return LoopDetection(
                    detected=True,
                    loop_type=LoopType.SEMANTIC,
                    similarity=similarity,
                    reason=(
                        "The proposed execution is semantically "
                        "equivalent to a previous execution."
                    ),
                )

        return LoopDetection(detected=False)

    def record(
        self,
        step: PlanStep,
        *,
        strategy_family: str,
    ) -> None:

        self._history.append(
            self.signature(
                step,
                strategy_family=strategy_family,
            )
        )

        if len(self._history) > self.max_history:
            self._history = self._history[
                -self.max_history:
            ]

    def check_and_record(
        self,
        step: PlanStep,
        *,
        strategy_family: str,
    ) -> LoopDetection:

        result = self.check(
            step,
            strategy_family=strategy_family,
        )

        if not result.detected:
            self.record(
                step,
                strategy_family=strategy_family,
            )

        return result

    def check_plan(
        self,
        steps: Sequence[PlanStep],
        *,
        strategy_family_resolver: Callable[[PlanStep], str],
    ) -> LoopDetection:
        """
        Detect repeated execution inside one generated plan.

        The resolver belongs to the Planner/StrategySelector layer.
        LoopDetector does not know how strategy families are calculated.
        """

        seen: list[StepSignature] = []

        for step in steps:
            if step.is_final_answer:
                continue

            strategy_family = strategy_family_resolver(step)

            current = self.signature(
                step,
                strategy_family=strategy_family,
            )

            for previous in seen:
                if current.exact == previous.exact:
                    return LoopDetection(
                        detected=True,
                        loop_type=LoopType.EXACT,
                        similarity=1.0,
                        reason=(
                            "Plan contains repeated execution."
                        ),
                    )

                if current.structural == previous.structural:
                    return LoopDetection(
                        detected=True,
                        loop_type=LoopType.STRUCTURAL,
                        similarity=1.0,
                        reason=(
                            "Plan contains repeated execution "
                            "with the same strategy."
                        ),
                    )

                similarity = SequenceMatcher(
                    None,
                    current.semantic,
                    previous.semantic,
                ).ratio()

                if similarity >= self.semantic_threshold:
                    return LoopDetection(
                        detected=True,
                        loop_type=LoopType.SEMANTIC,
                        similarity=similarity,
                        reason=(
                            "Plan contains semantically repeated "
                            "execution."
                        ),
                    )

            seen.append(current)

        return LoopDetection(detected=False)

    def reset(self) -> None:
        """Clear execution history."""

        self._history.clear()

    def signature(
        self,
        step: PlanStep,
        *,
        strategy_family: str,
    ) -> StepSignature:
        """
        Build all three signatures for a step.
        """

        normalized_action = self._normalize_text(
            step.action
        )

        normalized_tool = self._normalize_text(
            step.tool_name or ""
        )

        normalized_arguments = self._normalize_arguments(
            step.arguments or {}
        )
        exact_payload = {
            "step_type": step.step_type.value,
            "tool": normalized_tool,
            "arguments": normalized_arguments,
        }

      

        structural_payload = {
            "step_type": step.step_type.value,
            "strategy_family": self._normalize_text(
                strategy_family
            ),
            "tool": normalized_tool,
            "argument_keys": sorted(
                normalized_arguments.keys()
            ),
        }

        semantic_payload = {
            "action": self._semantic_text(
                normalized_action
            ),
            "arguments": self._semantic_arguments(
                normalized_arguments
            ),
        }

        return StepSignature(
            exact=self._serialize(exact_payload),
            structural=self._serialize(
                structural_payload
            ),
            semantic=self._serialize(
                semantic_payload
            ),
        )

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"[^\w\s:/.-]",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _normalize_arguments(
        cls,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            str(key).lower(): cls._normalize_value(
                value
            )
            for key, value in sorted(
                arguments.items()
            )
        }

    @classmethod
    def _normalize_value(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, dict):
            return {
                str(key).lower(): cls._normalize_value(
                    item
                )
                for key, item in sorted(
                    value.items()
                )
            }

        if isinstance(value, (list, tuple)):
            return [
                cls._normalize_value(item)
                for item in value
            ]

        if isinstance(value, str):
            return re.sub(
                r"\s+",
                " ",
                value.strip().lower(),
            )

        return value
    @classmethod
    def _semantic_text(
        cls,
        text: str,
    ) -> str:
        words = re.findall(
            r"[a-z0-9_:/.-]+",
            text.lower(),
        )

        stop_words = {
            "a",
            "an",
            "the",
            "to",
            "of",
            "for",
            "and",
            "with",
            "using",
            "use",
            "please",
            "find",
            "search",
            "look",
            "lookup",
            "get",
            "retrieve",
            "obtain",
            "execute",
            "perform",
            "run",
            "do",
            "make",
            "try",
            "attempt",
            "relevant",
            "alternative",
            "another",
            "method",
            "way",
        }

        meaningful = [
            word
            for word in words
            if word not in stop_words
        ]

        return " ".join(
            sorted(meaningful)
        )

    @classmethod
    def _semantic_arguments(
        cls,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key, value in arguments.items():
            if isinstance(value, str):
                result[key] = cls._semantic_text(
                    value
                )

            elif isinstance(value, dict):
                result[key] = cls._semantic_arguments(
                    value
                )

            elif isinstance(value, list):
                result[key] = [
                    cls._semantic_text(item)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]

            else:
                result[key] = value

        return result