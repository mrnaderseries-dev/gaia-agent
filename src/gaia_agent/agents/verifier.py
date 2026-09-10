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


class EvidenceItem(BaseModel):
    tool_name: str
    result: Any
    artifact_id: str | None = None
    source_type: str | None = None
    succeeded: bool = True


_STRONG_TOOL_NAMES = frozenset(
    {
        "python_interpreter",
        "analyze_excel",
        "file_reader",
        "analyze_image",
        "audio_reader",
        "transcribe_audio",
        "video_reader",
        "analyze_video",
        "youtube_transcript",
    }
)


_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    -?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    (?:[eE][+-]?\d+)?
    (?![\w.])
    """,
    re.VERBOSE,
)


def _extract_numbers(text: str) -> list[float]:
    if not text:
        return []

    matches = _NUMBER_RE.finditer(str(text))
    numbers: list[float] = []

    for match in matches:
        value = match.group(0).replace(",", "")
        try:
            numbers.append(float(value))
        except ValueError:
            continue

    return numbers


def _normalize_text(text: str) -> str:
    text = str(text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_field(
    item: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _get_tool_name(item: Any) -> str | None:
    value = _get_field(item, "tool_name")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _get_result(item: Any) -> Any:
    return _get_field(item, "result")


def _get_succeeded(item: Any) -> bool:
    return bool(_get_field(item, "succeeded", True))


def _get_artifact_id(item: Any) -> str | None:
    value = _get_field(item, "artifact_id")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _get_source_type(item: Any) -> str | None:
    value = _get_field(item, "source_type")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _iter_successful_evidence(
    raw_data: list[Any],
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []

    for item in raw_data or []:
        tool_name = _get_tool_name(item)

        if not tool_name:
            continue
        if tool_name.casefold() == "llm":
            continue
        if not _get_succeeded(item):
            continue

        result = _get_result(item)
        if result is None:
            continue

        evidence.append(
            EvidenceItem(
                tool_name=tool_name,
                result=result,
                artifact_id=_get_artifact_id(item),
                source_type=_get_source_type(item),
                succeeded=True,
            )
        )

    return evidence


def _iter_strong_evidence(
    raw_data: list[Any],
) -> list[EvidenceItem]:
    evidence = _iter_successful_evidence(raw_data)
    return [
        item
        for item in evidence
        if item.tool_name in _STRONG_TOOL_NAMES
    ]


def _distinct_numbers(
    numbers: list[float],
) -> list[float]:
    result: list[float] = []

    for number in numbers:
        if any(
            abs(number - existing) < 1e-9
            for existing in result
        ):
            continue
        result.append(number)

    return result


def _candidate_is_single_number(
    candidate: str,
) -> tuple[bool, float | None]:
    candidate = candidate.strip()
    if not candidate:
        return False, None

    match = _NUMBER_RE.fullmatch(candidate)
    if not match:
        return False, None

    try:
        return (
            True,
            float(match.group(0).replace(",", "")),
        )
    except ValueError:
        return False, None


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

    is_numeric, candidate_value = _candidate_is_single_number(candidate)

    if is_numeric and candidate_value is not None:
        evidence_numbers: list[float] = []

        for evidence in strong_evidence:
            evidence_numbers.extend(
                _extract_numbers(str(evidence.result))
            )

        if not evidence_numbers:
            return (
                None,
                (
                    "The candidate is numeric, but the "
                    "deterministic evidence contains no "
                    "numeric value."
                ),
            )

        distinct_numbers = _distinct_numbers(evidence_numbers)

        if len(distinct_numbers) == 1:
            evidence_value = distinct_numbers[0]

            if abs(candidate_value - evidence_value) < 1e-9:
                return (
                    VerificationStatus.VERIFIED,
                    (
                        "The candidate exactly matches the "
                        "single distinct numeric value found "
                        "in deterministic evidence."
                    ),
                )

            return (
                VerificationStatus.INVALID,
                (
                    "Deterministic evidence contains a "
                    "different numeric value: "
                    f"evidence={evidence_value:g}, "
                    f"candidate={candidate_value:g}."
                ),
            )

        return (
            None,
            (
                "Deterministic evidence contains multiple "
                "distinct numeric values. Numeric matching "
                "alone cannot determine which value answers "
                "the question."
            ),
        )

    normalized_candidate = _normalize_text(candidate)

    if not normalized_candidate:
        return (
            VerificationStatus.INVALID,
            "Candidate answer is empty after normalization.",
        )

    return (
        None,
        (
            "The candidate is textual and deterministic "
            "substring matching is unsafe. Semantic evidence "
            "verification is required."
        ),
    )


def _normalize_task_type(
    task_type: str | None,
) -> str:
    if not task_type:
        return ""
    return str(task_type).strip().upper()


def _check_source_support(
    data: VerificationInput,
) -> VerificationResult | None:
    task_type = _normalize_task_type(data.task_type)
    evidence = _iter_successful_evidence(data.raw_data)

    tool_names = {item.tool_name for item in evidence}

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

    if task_type == "AUDIO":
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

    if task_type == "VIDEO":
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


class VerifierAgent:

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
        candidate = str(data.candidate_answer or "").strip()

        if not candidate:
            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason="Candidate answer is missing or empty.",
            )

        evidence = _iter_successful_evidence(data.raw_data)

        if not evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason=(
                    "No successful evidence is available "
                    "to verify the candidate answer."
                ),
            )

        unsupported = _check_source_support(data)
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
    def _validate_llm_result(
        result: VerificationResult,
    ) -> VerificationResult:
        if result.status == VerificationStatus.VERIFIED:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                reason=(
                    "Candidate was semantically verified "
                    "against the available evidence: "
                    f"{result.reason}"
                ),
            )

        if result.status == VerificationStatus.INVALID:
            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason=(
                    "The evidence contradicts or fails to "
                    "support the candidate: "
                    f"{result.reason}"
                ),
            )

        if result.status == VerificationStatus.CONFLICTING_EVIDENCE:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "The available evidence contains "
                    "materially conflicting information: "
                    f"{result.reason}"
                ),
            )

        if result.status == VerificationStatus.UNSUPPORTED:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason=(
                    "The available evidence does not "
                    "represent the source or modality "
                    "required for verification: "
                    f"{result.reason}"
                ),
            )

        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason=(
                "The available evidence is insufficient "
                "to verify the candidate: "
                f"{result.reason}"
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
        return """
You are a strict factual answer verification agent.

Your ONLY responsibility is to determine whether the
candidate answer is supported by the provided evidence.

You are NOT an answer generator.
You are NOT allowed to solve the task independently.
You are NOT allowed to use outside knowledge.
You must reason only from the supplied QUESTION,
CANDIDATE ANSWER, TASK TYPE, and EVIDENCE.

CRITICAL RULES:
1. Evaluate the evidence against the EXACT question.
2. The evidence must actually answer or support the question being asked.
3. The mere presence of the candidate text, number, name, or value inside evidence is NOT sufficient.
4. Never use outside knowledge.
5. Never invent missing evidence.
6. Never generate a replacement answer.
7. If the evidence explicitly contradicts the candidate, return INVALID.
8. If relevant evidence contains material contradictions between sources, return CONFLICTING_EVIDENCE.
9. If evidence is relevant but does not establish the candidate with sufficient confidence, return INSUFFICIENT_EVIDENCE.
10. If the required source or modality is not represented by the evidence, return UNSUPPORTED.
11. Return VERIFIED only when the evidence provides a direct and reasonable factual basis for the candidate.
12. For web evidence, verify semantic relevance to the exact question. Matching words, numbers, titles, or names alone are not sufficient.
13. For files, images, audio, and video, verify that the evidence actually corresponds to the requested source.
14. Do not infer that a source supports the candidate merely because the candidate appears somewhere in the source.
15. Pay attention to negation.
16. Pay attention to temporal context.
17. Pay attention to relationships between values and entities.
18. When multiple numbers occur in evidence, do not assume that the candidate is correct merely because its number appears in the evidence.
19. When uncertain, prefer INSUFFICIENT_EVIDENCE.
20. The evidence is the only source of truth available to you.

Return ONLY the structured VerificationResult.
""".strip()

    @staticmethod
    def _format_evidence(
        evidence: list[EvidenceItem],
    ) -> str:
        if not evidence:
            return "(No successful evidence was provided.)"

        chunks: list[str] = []

        for index, item in enumerate(evidence, start=1):
            metadata: list[str] = [f"source_tool={item.tool_name}"]

            if item.source_type:
                metadata.append(f"source_type={item.source_type}")

            if item.artifact_id:
                metadata.append(f"artifact_id={item.artifact_id}")

            metadata_text = ", ".join(metadata)

            chunks.append(
                f"Evidence {index} ({metadata_text}):\n{item.result}"
            )

        return "\n\n".join(chunks)

    def _build_prompt(
        self,
        data: VerificationInput,
    ) -> str:
        evidence = _iter_successful_evidence(data.raw_data)
        evidence_text = self._format_evidence(evidence)

        return f"""
Verify the candidate answer using ONLY the provided evidence.

QUESTION:
{data.question}

TASK TYPE:
{data.task_type or "unknown"}

CANDIDATE ANSWER:
{data.candidate_answer}

EVIDENCE:
{evidence_text}

Your decision must be based on whether the evidence actually
supports the candidate answer to the exact question.

Choose exactly ONE status:

VERIFIED
The evidence directly and sufficiently supports the candidate.

INVALID
The evidence contradicts the candidate.

INSUFFICIENT_EVIDENCE
The evidence is relevant but insufficient to establish whether
the candidate is correct.

CONFLICTING_EVIDENCE
Two or more relevant evidence sources materially contradict
each other.

UNSUPPORTED
The required source or modality is not represented by the
available evidence.

IMPORTANT:
- Do not use outside knowledge.
- Do not guess.
- Do not solve the question independently.
- Do not treat a matching number or string as sufficient.
- Check relationships, context, entity, time, and negation.
- If uncertain, return INSUFFICIENT_EVIDENCE.

Return a concise reason.
""".strip()

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