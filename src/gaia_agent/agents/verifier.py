from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from gaia_agent.llm.client import LLMClient
from gaia_agent.llm.model import LLMModel


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNSUPPORTED = "unsupported"


class VerificationResult(BaseModel):
    status: VerificationStatus
    reason: str = Field(
        description="Brief explanation for the verification decision."
    )


class VerificationInput(BaseModel):
    question: str
    candidate_answer: str
    raw_data: list[Any] = Field(default_factory=list)
    task_type: str | None = None


_STRONG_TOOL_NAMES = frozenset(
    {
        "python_interpreter",
        "analyze_excel",
        "file_reader",
        "analyze_image",
    }
)

_NUMBER_RE = re.compile(
    r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


def _extract_numbers(text: str) -> list[float]:
    if not text:
        return []

    cleaned = re.sub(
        r"(?<=\d),(?=\d{3}(?!\d))",
        "",
        text,
    )

    return [
        float(match.group(0))
        for match in _NUMBER_RE.finditer(cleaned)
    ]


def _get_tool_name(item: Any) -> str | None:
    if isinstance(item, dict):
        value = item.get("tool_name")
    else:
        value = getattr(item, "tool_name", None)

    if value is None:
        return None

    return str(value)


def _get_result(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("result")

    return getattr(item, "result", None)


def _get_succeeded(item: Any) -> bool:
    if isinstance(item, dict):
        return bool(item.get("succeeded", True))

    return bool(getattr(item, "succeeded", True))


def _iter_successful_evidence(
    raw_data: list[Any],
) -> list[tuple[str, Any]]:
    evidence: list[tuple[str, Any]] = []

    for item in raw_data or []:
        tool_name = _get_tool_name(item)

        if not tool_name:
            continue

        if tool_name == "llm":
            continue

        if not _get_succeeded(item):
            continue

        result = _get_result(item)

        if result is None:
            continue

        evidence.append(
            (
                tool_name,
                result,
            )
        )

    return evidence


def _iter_strong_evidence(
    raw_data: list[Any],
) -> list[tuple[str, Any]]:
    return [
        (tool_name, result)
        for tool_name, result in _iter_successful_evidence(raw_data)
        if tool_name in _STRONG_TOOL_NAMES
    ]


def _normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).casefold(),
    ).strip()


def _deterministic_verification(
    candidate_answer: str,
    raw_data: list[Any],
) -> tuple[VerificationStatus | None, str]:

    candidate = str(candidate_answer or "").strip()

    if not candidate:
        return (
            VerificationStatus.INVALID,
            "Candidate answer is missing or empty.",
        )

    strong_evidence = _iter_strong_evidence(raw_data)

    if not strong_evidence:
        return (
            None,
            "No strong deterministic evidence is available.",
        )

    candidate_numbers = _extract_numbers(candidate)

    if len(candidate_numbers) == 1:
        candidate_value = candidate_numbers[0]

        evidence_numbers: list[float] = []

        for _, result in strong_evidence:
            evidence_numbers.extend(
                _extract_numbers(str(result))
            )

        if not evidence_numbers:
            return (
                None,
                "Candidate is numeric but the evidence contains no "
                "numeric value.",
            )

        distinct_numbers = list(
            dict.fromkeys(evidence_numbers)
        )

        if len(distinct_numbers) == 1:
            evidence_value = distinct_numbers[0]

            if abs(candidate_value - evidence_value) < 1e-9:
                return (
                    VerificationStatus.VERIFIED,
                    (
                        "The candidate number matches the "
                        "single numeric value found in "
                        "deterministic evidence."
                    ),
                )

            return (
                VerificationStatus.INVALID,
                (
                    "Deterministic evidence contradicts the "
                    f"candidate: evidence={evidence_value:g}, "
                    f"candidate={candidate_value:g}."
                ),
            )

        return (
            None,
            (
                "Deterministic evidence contains multiple "
                "numeric values, so numeric matching alone "
                "cannot determine the answer."
            ),
        )

    normalized_candidate = _normalize_text(candidate)

    if not normalized_candidate:
        return (
            VerificationStatus.INVALID,
            "Candidate answer is empty after normalization.",
        )

    for _, result in strong_evidence:
        normalized_result = _normalize_text(str(result))

        if not normalized_result:
            continue

        if normalized_candidate in normalized_result:
            return (
                VerificationStatus.VERIFIED,
                (
                    "The candidate answer appears directly in "
                    "deterministic evidence."
                ),
            )

    return (
        None,
        "No deterministic textual match was found.",
    )


class VerifierAgent:
    """
    Strict final-answer verifier.

    Responsibilities:
        - evaluate candidate answers
        - evaluate evidence
        - distinguish verification failure types

    Does NOT:
        - execute tools
        - generate replacement answers
        - replan
        - retry
        - recover
        - modify AgentState
        - decide termination
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
        candidate = str(
            data.candidate_answer or ""
        ).strip()

        if not candidate:
            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason="Candidate answer is missing or empty.",
            )

        evidence = _iter_successful_evidence(
            data.raw_data
        )

        if not evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason=(
                    "No successful evidence is available "
                    "to verify the candidate answer."
                ),
            )

        unsupported = self._check_source_support(
            data
        )

        if unsupported is not None:
            return unsupported

        (
            deterministic_status,
            deterministic_reason,
        ) = _deterministic_verification(
            data.candidate_answer,
            data.raw_data,
        )

        if deterministic_status is not None:
            return VerificationResult(
                status=deterministic_status,
                reason=deterministic_reason,
            )

        messages = self._build_messages(data)

        result = await self.client.generate(
            messages=messages,
            model=self.model,
            output_schema=VerificationResult,
        )

        if not isinstance(result, VerificationResult):
            raise TypeError(
                "LLMClient.generate() returned an invalid "
                "VerificationResult."
            )

        return self._validate_llm_result(result)

    @staticmethod
    def _check_source_support(
        data: VerificationInput,
    ) -> VerificationResult | None:
        task_type = (
            str(data.task_type or "")
            .strip()
            .upper()
        )

        tool_names = {
            tool_name
            for tool_name, _ in _iter_successful_evidence(
                data.raw_data
            )
        }

        if task_type in {"IMAGE", "VISION"}:
            if "analyze_image" not in tool_names:
                return VerificationResult(
                    status=VerificationStatus.UNSUPPORTED,
                    reason=(
                        "The task requires image evidence, "
                        "but no successful image-analysis "
                        "evidence is available."
                    ),
                )

        if task_type in {"AUDIO"}:
            if not (
                "audio_reader" in tool_names
                or "transcribe_audio" in tool_names
            ):
                return VerificationResult(
                    status=VerificationStatus.UNSUPPORTED,
                    reason=(
                        "The task requires audio evidence, "
                        "but no successful audio evidence "
                        "is available."
                    ),
                )

        if task_type in {"VIDEO"}:
            if not (
                "video_reader" in tool_names
                or "analyze_video" in tool_names
                or "youtube_transcript" in tool_names
            ):
                return VerificationResult(
                    status=VerificationStatus.UNSUPPORTED,
                    reason=(
                        "The task requires video evidence, "
                        "but no successful video-related "
                        "evidence is available."
                    ),
                )

        return None

    @staticmethod
    def _validate_llm_result(
        result: VerificationResult,
    ) -> VerificationResult:
        """
        Apply safety rules to the semantic verifier result.
        """

        if result.status == VerificationStatus.VERIFIED:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                reason=(
                    "Candidate was semantically verified against "
                    f"the available evidence: {result.reason}"
                ),
            )

        if result.status == VerificationStatus.INVALID:
            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason=(
                    "The evidence does not support the candidate: "
                    f"{result.reason}"
                ),
            )

        if result.status == VerificationStatus.CONFLICTING_EVIDENCE:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "The available evidence contains conflicting "
                    f"information: {result.reason}"
                ),
            )

        if result.status == VerificationStatus.UNSUPPORTED:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason=(
                    "The available evidence does not support "
                    f"verification of this task: {result.reason}"
                ),
            )

        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason=(
                "The available evidence is insufficient to "
                f"verify the candidate: {result.reason}"
            ),
        )

    def _build_messages(
        self,
        data: VerificationInput,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_prompt(data),
            },
        ]

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a strict factual answer verification agent.\n\n"
            "Your ONLY task is to determine whether the candidate "
            "answer is supported by the provided evidence.\n\n"
            "CRITICAL RULES:\n"
            "1. Evaluate the evidence against the EXACT question.\n"
            "2. The evidence must actually support the answer "
            "to the question.\n"
            "3. The mere presence of the candidate value in "
            "the evidence is NOT sufficient.\n"
            "4. Do not use outside knowledge.\n"
            "5. Do not invent missing evidence.\n"
            "6. Do not rewrite the candidate answer.\n"
            "7. If the evidence contradicts the candidate, "
            "return INVALID.\n"
            "8. If two or more pieces of evidence contradict "
            "each other in a material way, return "
            "CONFLICTING_EVIDENCE.\n"
            "9. If the evidence is relevant but insufficient "
            "to establish the answer, return "
            "INSUFFICIENT_EVIDENCE.\n"
            "10. If the required source/modality is not actually "
            "represented by the evidence, return UNSUPPORTED.\n"
            "11. Return VERIFIED only when the evidence provides "
            "a direct and reasonable factual basis for the "
            "candidate answer.\n"
            "12. For web evidence, verify semantic relevance "
            "to the exact question. Matching words or numbers "
            "alone are not sufficient.\n"
            "13. For files, images, audio, and video, verify "
            "that the evidence corresponds to the requested "
            "source.\n"
            "14. When uncertain, do NOT guess. Prefer "
            "INSUFFICIENT_EVIDENCE.\n\n"
            "Return only the structured verification result."
        )

    @staticmethod
    def _build_prompt(
        data: VerificationInput,
    ) -> str:
        evidence_items = _iter_successful_evidence(
            data.raw_data
        )

        if not evidence_items:
            evidence_text = (
                "(No successful evidence was provided.)"
            )
        else:
            chunks: list[str] = []

            for index, (tool_name, result) in enumerate(
                evidence_items,
                start=1,
            ):
                chunks.append(
                    f"Evidence {index} "
                    f"(source tool: {tool_name}):\n"
                    f"{result}"
                )

            evidence_text = "\n\n".join(chunks)

        return (
            "Verify the following candidate answer.\n\n"
            f"QUESTION:\n{data.question}\n\n"
            f"TASK TYPE:\n{data.task_type or 'unknown'}\n\n"
            f"CANDIDATE ANSWER:\n{data.candidate_answer}\n\n"
            "EVIDENCE:\n"
            f"{evidence_text}\n\n"
            "Choose exactly one status:\n"
            "- VERIFIED: evidence directly supports the answer.\n"
            "- INVALID: evidence contradicts the answer.\n"
            "- INSUFFICIENT_EVIDENCE: evidence is relevant but "
            "not sufficient to establish the answer.\n"
            "- CONFLICTING_EVIDENCE: relevant evidence materially "
            "contradicts other relevant evidence.\n"
            "- UNSUPPORTED: the evidence does not represent the "
            "source or modality required by the question.\n\n"
            "Do not use outside knowledge.\n"
            "Do not guess.\n"
            "Return a concise reason."
        )

    @staticmethod
    def _format_raw_item(
        item: Any,
    ) -> str:
      

        if item is None:
            return "None"

        if isinstance(item, str):
            return item

        if isinstance(item, BaseModel):
            return item.model_dump_json(indent=2)

        if isinstance(item, (dict, list, tuple, set)):
            try:
                return json.dumps(
                    item,
                    default=str,
                    indent=2,
                )
            except Exception:
                return str(item)

        return str(item)