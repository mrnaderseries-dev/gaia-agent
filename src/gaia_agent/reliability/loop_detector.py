from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any


class LoopType(str, Enum):
    PLAN = "plan"
    EXACT = "exact"
    SEQUENCE = "sequence"


@dataclass(frozen=True, slots=True)
class LoopDetectionResult:
    detected: bool
    loop_type: LoopType | None
    repetition_count: int
    sequence_length: int
    fingerprint: str | None
    message: str = ""


class LoopDetector:

    def __init__(
        self,
        *,
        max_history: int = 50,
        max_sequence_length: int = 10,
        exact_repetition_threshold: int = 3,
        sequence_repetition_threshold: int = 3,
    ) -> None:

        if max_history <= 0:
            raise ValueError(
                "max_history must be greater than 0."
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be greater than 0."
            )

        if exact_repetition_threshold <= 1:
            raise ValueError(
                "exact_repetition_threshold must be greater than 1."
            )

        if sequence_repetition_threshold <= 1:
            raise ValueError(
                "sequence_repetition_threshold must be greater than 1."
            )

        self.max_history = max_history
        self.max_sequence_length = max_sequence_length

        self.exact_repetition_threshold = (
            exact_repetition_threshold
        )

        self.sequence_repetition_threshold = (
            sequence_repetition_threshold
        )

        # History of execution signatures.
        self._history: deque[str] = deque(
            maxlen=max_history
        )

        # History of plans.
        self._plan_history: deque[str] = deque(
            maxlen=max_history
        )

        # Number of exact executions.
        self._exact_counts: Counter[str] = Counter()

        # Number of exact plans.
        self._plan_counts: Counter[str] = Counter()

    def check(
        self,
        *,
        action: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> LoopDetectionResult:

        execution = {
            "action": action,
            "tool_name": tool_name,
            "arguments": arguments,
        }

        fingerprint = self._fingerprint(
            execution
        )

        self._history.append(
            fingerprint
        )

        self._exact_counts[fingerprint] += 1

        exact_count = self._exact_counts[
            fingerprint
        ]

        
        if (
            exact_count
            >= self.exact_repetition_threshold
        ):

            return LoopDetectionResult(
                detected=True,
                loop_type=LoopType.EXACT,
                repetition_count=exact_count,
                sequence_length=1,
                fingerprint=fingerprint,
                message=(
                    "The same execution "
                    "was repeated repeatedly."
                ),
            )

        sequence_result = (
            self._detect_repeating_sequence()
        )

        if sequence_result is not None:
            return sequence_result
        return LoopDetectionResult(
            detected=False,
            loop_type=None,
            repetition_count=exact_count,
            sequence_length=1,
            fingerprint=fingerprint,
            message="No execution loop detected.",
        )

   
    def check_plan(
        self,
        plan: list[Any],
    ) -> LoopDetectionResult:

        fingerprint = self._fingerprint(
            plan
        )

        self._plan_history.append(
            fingerprint
        )

        self._plan_counts[fingerprint] += 1

        count = self._plan_counts[
            fingerprint
        ]

        if (
            count
            >= self.exact_repetition_threshold
        ):

            return LoopDetectionResult(
                detected=True,
                loop_type=LoopType.PLAN,
                repetition_count=count,
                sequence_length=1,
                fingerprint=fingerprint,
                message=(
                    "The same execution plan "
                    "was produced repeatedly."
                ),
            )

        return LoopDetectionResult(
            detected=False,
            loop_type=None,
            repetition_count=count,
            sequence_length=1,
            fingerprint=fingerprint,
            message="No plan loop detected.",
        )
    
    def _detect_repeating_sequence(
        self,
    ) -> LoopDetectionResult | None:

        history = list(
            self._history
        )

        history_size = len(history)

     
        max_length = min(
            self.max_sequence_length,
            history_size // 2,
        )

        for sequence_length in range(
            1,
            max_length + 1,
        ):

            pattern = history[
                -sequence_length:
            ]

            repetitions = 1

            index = (
                history_size
                - (sequence_length * 2)
            )

            while index >= 0:

                previous = history[
                    index:
                    index + sequence_length
                ]

                if previous != pattern:
                    break

                repetitions += 1

                index -= sequence_length

            if (
                repetitions
                >= self.sequence_repetition_threshold
            ):

                fingerprint = self._fingerprint(
                    pattern
                )

                return LoopDetectionResult(
                    detected=True,
                    loop_type=LoopType.SEQUENCE,
                    repetition_count=repetitions,
                    sequence_length=sequence_length,
                    fingerprint=fingerprint,
                    message=(
                        "A repeating execution "
                        "sequence was detected."
                    ),
                )

        return None

   
    def reset(self) -> None:

        self._history.clear()

        self._plan_history.clear()

        self._exact_counts.clear()

        self._plan_counts.clear()

    def _fingerprint(
        self,
        value: Any,
    ) -> str:

        normalized = self._normalize(
            value
        )

        serialized = repr(
            normalized
        )

        return sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _normalize(
        self,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        
        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

  
        if isinstance(value, dict):

            return tuple(
                sorted(
                    (
                        str(key),
                        self._normalize(item),
                    )
                    for key, item in value.items()
                )
            )

       
        if isinstance(
            value,
            (list, tuple),
        ):

            return tuple(
                self._normalize(item)
                for item in value
            )

   
        if isinstance(value, set):

            items = [
                self._normalize(item)
                for item in value
            ]

            return tuple(
                sorted(
                    items,
                    key=repr,
                )
            )

      
        if hasattr(
            value,
            "model_dump",
        ):

            return self._normalize(
                value.model_dump()
            )

      
        if hasattr(
            value,
            "dict",
        ):

            return self._normalize(
                value.dict()
            )

        if hasattr(
            value,
            "dict",
        ):

            return self._normalize(
                vars(value)
            )

        return str(value)