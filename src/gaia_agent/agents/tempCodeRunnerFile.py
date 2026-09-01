from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel


class VerificationResult(BaseModel):
    verified: bool = Field(
        description=(
            "Whether the candidate answer is supported "
            "by the available data."
        )
    )
    reason: str = Field(
        description=(
            "Brief explanation for the verification decision."
        )
    )


class VerificationInput(BaseModel):
    question: str
    candidate_answer: str
    raw_data: list[Any] = Field(
        default_factory=list
    )


class VerificationStatus(str):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


# Tool names whose results are treated as strong computation/data
# evidence. A single numeric value produced by these tools that
# contradicts the candidate answer is a determinable failure.
_STRONG_TOOL_NAMES = frozenset(
    {
        "python_interpreter",
        "analyze_excel",
        "file_reader",
        "analyze_image",
        "web_search",
    }
)

_NUMBER_RE = re.compile(
    r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


def _extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    cleaned = re.sub(r"(?<=\d)[,](?=\d{3}(?!\d))", "", text)
    return [
        float(match.group(0))
        for match in _NUMBER_RE.finditer(cleaned)
    ]


def _iter_strong_evidence(raw_data: list[Any]) -> list[str]:
    """Collect result text from strong tool evidence records."""
    texts: list[str] = []

    for item in raw_data or []:
        tool_name = getattr(item, "tool_name", None)
        if tool_name is None:
            continue
        if tool_name == "llm":
            continue
        if tool_name not in _STRONG_TOOL_NAMES:
            continue
        succeeded = getattr(item, "succeeded", True)
        if not succeeded:
            continue
        result = getattr(item, "result", None)
        if result is None:
            continue
        texts.append(str(result))

    return texts


def evidence_supports_candidate(
    candidate_answer: str,
    raw_data: list[Any],
) -> bool | None:
    """
    Deterministic support gate applied AFTER the LLM judge says
    "verified" (orchestrator STEP 8 hard gate).

    The LLM must never be able to simply declare its own answer
    verified when the gathered tool evidence does not contain it.

    Returns:

    - True  : the candidate is explicitly present in strong tool
              evidence (verbatim text or exact numeric match).
    - False : strong tool evidence exists but the candidate does NOT
              appear in it -> the LLM verdict must NOT be trusted.
    - None  : no strong tool evidence exists (or the candidate is a
              long natural-language answer that cannot be matched
              verbatim); nothing deterministic to check.
    """
    candidate = str(candidate_answer or "").strip()

    if not candidate:
        return False

    evidence_texts = _iter_strong_evidence(list(raw_data or []))

    if not evidence_texts:
        return None

    candidate_numbers = _extract_numbers(candidate)

    numeric_like = bool(
        re.fullmatch(r"[\d\s.,+-]+", candidate)
    )

    # Long natural-language answers are paraphrased by design; only
    # short, specific answers (numbers, codes, names) can be matched
    # against evidence deterministically.
    if not numeric_like and len(candidate.split()) > 3:
        return None

    # A purely numeric candidate must be matched numerically, never as
    # a substring ("2" would otherwise match any evidence text).
    if not numeric_like:
        lowered_candidate = candidate.lower()
        for text in evidence_texts:
            if lowered_candidate in text.lower():
                return True

    if len(candidate_numbers) == 1:
        value = candidate_numbers[0]
        for text in evidence_texts:
            for number in _extract_numbers(text):
                if number == value:
                    return True

    return False


def deterministic_verification(
    candidate_answer: str,
    raw_data: list[Any],
) -> tuple[VerificationStatus, str]:
    """
    Independent, deterministic evidence check (STEP 7).

    The candidate answer is checked against the actual tool evidence
    BEFORE the LLM judge is consulted.

    Returns PASS / FAIL / UNCERTAIN:

    - PASS      : the candidate is explicitly present in strong
                  evidence (verbatim text or an exact numeric match).
    - FAIL      : a computation/data tool produced exactly one distinct
                  numeric value and the candidate is a number that
                  differs from it (e.g. evidence = 3, answer = 4).
    - UNCERTAIN : no conflict detected, but no direct match either;
                  the LLM judge still has to decide.
    """
    if candidate_answer is None:
        return VerificationStatus.FAIL, (
            "Candidate answer is missing."
        )

    candidate = str(candidate_answer).strip()

    if not candidate:
        return VerificationStatus.FAIL, (
            "Candidate answer is empty."
        )

    evidence_texts = _iter_strong_evidence(raw_data)

    if not evidence_texts:
        return VerificationStatus.UNCERTAIN, (
            "No strong tool evidence is available for "
            "an independent deterministic check."
        )

    joined = "\n".join(evidence_texts)

    candidate_numbers = _extract_numbers(candidate)
    evidence_numbers = _extract_numbers(joined)

    # ------------------------------------------------------------------
    # Single-number candidate: strict numeric conflict detection.
    # ------------------------------------------------------------------
    if len(candidate_numbers) == 1:
        candidate_value = candidate_numbers[0]

        if evidence_numbers:
            distinct = sorted(set(evidence_numbers))
            if len(distinct) == 1:
                evidence_value = distinct[0]
                if abs(candidate_value - evidence_value) < 1e-9:
                    return VerificationStatus.PASS, (
                        "The candidate number matches the single "
                        "numeric value in the tool evidence."
                    )
                return VerificationStatus.FAIL, (
                    f"Numeric evidence conflict detected: the "
                    f"tool/computation evidence contains "
                    f"'{distinct[0]:g}', but the candidate answer is "
                    f"'{candidate_value:g}'."
                )

            if any(
                abs(candidate_value - value) < 1e-9
                for value in distinct
            ):
                return VerificationStatus.PASS, (
                    "The candidate number appears verbatim in "
                    "the tool evidence."
                )

        return VerificationStatus.UNCERTAIN, (
            "The evidence contains multiple or unrelated numbers; "
            "the numeric check is inconclusive."
        )

    # ------------------------------------------------------------------
    # Non-numeric candidate: verbatim containment check.
    # ------------------------------------------------------------------
    normalized_candidate = re.sub(
        r"\s+", " ", candidate.lower()
    ).strip()
    normalized_evidence = re.sub(
        r"\s+", " ", joined.lower()
    )

    if (
        normalized_candidate
        and normalized_candidate in normalized_evidence
    ):
        return VerificationStatus.PASS, (
            "The candidate answer appears verbatim in the "
            "tool evidence."
        )

    return VerificationStatus.UNCERTAIN, (
        "No direct deterministic match or conflict was found; "
        "the evidence-based judge must decide."
    )


class VerifierAgent:
    """
    Verifies whether a candidate final answer is supported by
    the information gathered during agent execution.

    The verifier is a judge only.

    It does NOT:
    - generate a replacement answer
    - modify the candidate answer
    - execute tools
    - perform recovery
    - replan

    It only returns a VerificationResult.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        model: LLMModel,
    ) -> None:

        self.client = client
        self.model = model

    async def verify(
        self,
        data: VerificationInput,
    ) -> VerificationResult:
        """
        Verify a candidate answer against the available raw data.
        """

        messages = self._build_messages(
            data
        )

        result = await self.client.generate(
            messages=messages,
            model=self.model,
            output_schema=VerificationResult,
        )

        if not isinstance(
            result,
            VerificationResult,
        ):
            raise TypeError(
                "LLMClient.generate() returned an invalid "
                "verification result."
            )

        return result

    def _build_messages(
        self,
        data: VerificationInput,
    ) -> list[dict[str, str]]:
        """
        Build the messages sent to the verification LLM.
        """

        return [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_prompt(
                    data
                ),
            },
        ]

    @staticmethod
    def _system_prompt() -> str:
        """
        System instructions for the verifier.

        The verifier must judge factual support only and must
        not invent information.
        """

        return (
            "You are a strict answer verification agent.\n\n"

            "Your task is to determine whether the candidate "
            "answer is supported by the provided information.\n\n"

            "Rules:\n"

            "1. Verify the answer against the provided raw data.\n"

            "2. Do not use unsupported assumptions.\n"

            "3. Do not invent facts.\n"

            "4. Do not rewrite or improve the candidate answer.\n"

            "5. Return verified=true only when the candidate "
            "answer is adequately supported by the available "
            "information.\n"

            "6. Return verified=false when the answer is "
            "unsupported, contradicted, incomplete in a materially "
            "important way, or otherwise unreliable.\n"

            "7. Give a concise reason for the decision.\n"
            "8. Single-word or exact-number answers are valid as long as they are factually supported by the raw data."
        )

    @staticmethod
    def _build_prompt(
        data: VerificationInput,
    ) -> str:
        """
        Build the user prompt containing the question,
        candidate answer, and gathered data.
        """

        raw_data = "\n".join(
            VerifierAgent._format_raw_item(
                item
            )
            for item in data.raw_data
        )

        if not raw_data:
            raw_data = "(No raw data was provided.)"

        return (
            "Verify the following candidate answer.\n\n"

            f"Question:\n{data.question}\n\n"

            f"Candidate answer:\n{data.candidate_answer}\n\n"

            "Available raw data:\n"
            f"{raw_data}\n\n"

            "Determine whether the candidate answer is supported "
            "by the available raw data."
        )

    @staticmethod
    def _format_raw_item(
        item: Any,
    ) -> str:
        """
        Convert an arbitrary raw-data item into a safe textual
        representation for the verifier prompt.
        """

        if item is None:
            return "None"

        if isinstance(item, str):
            return item

        if isinstance(item, BaseModel):
            return item.model_dump_json(indent=2)

        if isinstance(item, (dict, list, tuple, set)):
            try:
                return json.dumps(item, default=str, indent=2)
            except Exception:
                return str(item)

        return str(item)