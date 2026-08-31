from __future__ import annotations

import re
from typing import Any


class AnswerSanitizer:

    _PREFIXES = (
        "final answer:",
        "final answer",
        "answer:",
        "answer",
        "response:",
        "response",
    )

    def sanitize(
        self,
        answer: Any,
    ) -> str:

        if answer is None:
            raise ValueError(
                "Cannot sanitize a None answer."
            )

        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "Final answer cannot be empty."
            )

        answer = self._remove_code_fences(answer)

        answer = self._remove_answer_prefix(
            answer
        )

        answer = self._normalize_whitespace(
            answer
        )

        if not answer:
            raise ValueError(
                "Final answer became empty after sanitization."
            )

        return answer

    @staticmethod
    def _remove_code_fences(
        answer: str,
    ) -> str:

        if answer.startswith("`") and answer.endswith("```"):

            lines = answer.splitlines()

            if len(lines) >= 2:

                lines = lines[1:-1]

                return "\n".join(lines).strip()

        return answer

    @classmethod
    def _remove_answer_prefix(
        cls,
        answer: str,
    ) -> str:

        normalized = answer.strip()

        for prefix in cls._PREFIXES:

            pattern = rf"^{re.escape(prefix)}\s*"

            normalized = re.sub(
                pattern,
                "",
                normalized,
                count=1,
                flags=re.IGNORECASE,
            )

        return normalized.strip()

    @staticmethod
    def _normalize_whitespace(
        answer: str,
    ) -> str:

        lines = [
            line.strip()
            for line in answer.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)