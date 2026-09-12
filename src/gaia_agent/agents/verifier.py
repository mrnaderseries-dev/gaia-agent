from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator
from gaia_agent.llm.client import LLMClient


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNSUPPORTED = "unsupported"

    PASS = "verified"
    FAIL = "invalid"
    UNCERTAIN = "insufficient_evidence"


class VerificationResult(BaseModel):
    status: VerificationStatus | None = None
    verified: bool | None = None
    reason: str = ""

    @model_validator(mode="after")
    def normalize(self) -> "VerificationResult":
        if self.status is not None:
            self.verified = self.status == VerificationStatus.VERIFIED
        return self


class VerificationInput(BaseModel):
    question: str
    candidate_answer: str
    raw_data: list[Any] = Field(default_factory=list)
    task_type: str


class EvidenceItem(BaseModel):
    tool_name: str
    result: Any = None
    artifact_id: str | None = None
    source_type: str | None = None
    succeeded: bool = True
    step_id: str | None = None
    execution_id: str | None = None
    attempt_id: str | None = None
    run_id: str | None = None
    plan_version: int | None = None
    step_status: str | None = None
    relevant: bool | None = None


_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)

_STRONG_TOOL_NAMES = {
    "python_interpreter", "analyze_excel", "file_reader", "analyze_image",
    "audio_reader", "transcribe_audio", "video_reader", "analyze_video",
    "youtube_transcript",
}

_SEMANTIC_TOOL_NAMES = _STRONG_TOOL_NAMES | {
    "web_search", "browser", "http", "http_get", "github",
}


def _get_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _get_tool_name(item: Any) -> str | None:
    value = _get_field(item, "tool_name")
    return None if value is None else str(value).strip().lower()


def _get_result(item: Any) -> Any:
    value = _get_field(item, "result", None)
    return value if value is not None else _get_field(item, "output", None)


def _get_succeeded(item: Any) -> bool:
    value = _get_field(item, "succeeded", None)
    if value is not None:
        return bool(value)
    value = _get_field(item, "success", None)
    return True if value is None else bool(value)


def _get_str(item: Any, name: str) -> str | None:
    value = _get_field(item, name)
    return None if value is None else str(value)


def _get_plan_version(item: Any) -> int | None:
    value = _get_field(item, "plan_version")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_relevant(item: Any) -> bool | None:
    value = _get_field(item, "relevant")
    return None if value is None else bool(value)


def _iter_successful_evidence(raw_data: list[Any]):
    """Exclude LLM output, failed execution and explicitly irrelevant data."""
    for item in raw_data:
        tool = _get_tool_name(item)
        if not tool or tool == "llm":
            continue
        if not _get_succeeded(item):
            continue
        if _get_relevant(item) is False:
            continue
        if _get_result(item) is None:
            continue
        yield item


def _filter_current_evidence(raw_data: list[Any]) -> list[Any]:
    """Drop stale evidence when execution/replan identity is available."""
    evidence = list(_iter_successful_evidence(raw_data))
    if not evidence:
        return []

    # Keep the newest coherent identity while allowing legacy evidence that has
    # no identity metadata at all. This makes the verifier backward compatible.
    for field in ("run_id", "execution_id", "attempt_id"):
        values = [_get_str(x, field) for x in evidence]
        values = [x for x in values if x]
        if values:
            newest = values[-1]
            evidence = [x for x in evidence if _get_str(x, field) in (None, newest)]

    versions = [_get_plan_version(x) for x in evidence]
    versions = [x for x in versions if x is not None]
    if versions:
        newest = max(versions)
        evidence = [x for x in evidence if _get_plan_version(x) in (None, newest)]

    # A failed step must never leak back in through a success alias.
    valid = {"", "success", "succeeded", "completed", "complete", "ok", "verified"}
    return [
        x for x in evidence
        if (_get_str(x, "step_status") or "").strip().lower() in valid
    ]


def _iter_strong_evidence(raw_data: list[Any]):
    for item in _filter_current_evidence(raw_data):
        if _get_tool_name(item) in _STRONG_TOOL_NAMES:
            yield item


def _extract_numbers(value: Any) -> list[str]:
    return [] if value is None else _NUMBER_RE.findall(str(value))


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value).strip().lower())


def _distinct_numbers(numbers: list[str]) -> list[str]:
    result: list[str] = []
    for raw in numbers:
        try:
            normalized = format(float(raw), ".15g")
        except (TypeError, ValueError):
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _numbers_equal(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _candidate_is_single_number(candidate: str) -> bool:
    numbers = _extract_numbers(candidate)
    return len(numbers) == 1 and _normalize_text(candidate) == _normalize_text(numbers[0])


def _find_explicit_labeled_numbers(text: str) -> list[str]:
    patterns = [
        r"\b(?:final\s+)?(?:answer|result|score|value|total|output)\s*(?:is|=|:)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        r"\b(?:calculated|computed)\s+(?:answer|result|value|score)\s*(?:is|=|:)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        r"\bcalculation\s+[A-Za-z0-9_-]+\s*(?:is|=|:)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        r'"(?:answer|final_answer|result|output)"\s*:\s*"?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"?',
    ]
    out: list[str] = []
    for pattern in patterns:
        out.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return out


def _is_intermediate(text: str, number: str) -> bool:
    escaped = re.escape(number)
    patterns = (
        rf"(?:intermediate|partial|subtotal|sub-result|subresult|step)\s*.{{0,100}}{escaped}",
        rf"{escaped}.{{0,100}}(?:intermediate|partial|subtotal|sub-result|subresult|step)",
    )
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def _check_source_support(data: VerificationInput, evidence: list[Any]) -> bool:
    task_type = data.task_type.strip().upper()
    if task_type in {"TEXT", "GENERAL", "UNKNOWN"}:
        return True
    required = {
        "IMAGE": {"analyze_image"},
        "VISION": {"analyze_image"},
        "AUDIO": {"audio_reader", "transcribe_audio"},
        "VIDEO": {"video_reader", "analyze_video", "youtube_transcript"},
    }.get(task_type)
    return True if required is None else any(_get_tool_name(x) in required for x in evidence)


def deterministic_verification(candidate_answer: str, raw_data: list[Any]) -> VerificationResult:
    """Conservative proof layer. Ambiguous semantics go to the LLM verifier."""
    candidate = candidate_answer.strip()
    if not candidate:
        return VerificationResult(status=VerificationStatus.INVALID, reason="The candidate answer is empty.")

    strong = list(_iter_strong_evidence(raw_data))
    if not strong:
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason="No successful current strong evidence is available for deterministic verification.",
        )

    if _candidate_is_single_number(candidate):
        candidate_number = _extract_numbers(candidate)[0]
        explicit: list[str] = []
        appears = False
        intermediate = False

        for item in strong:
            text = str(_get_result(item))
            nums = _extract_numbers(text)
            appears |= any(_numbers_equal(candidate_number, n) for n in nums)
            intermediate |= _is_intermediate(text, candidate_number)
            explicit.extend(_find_explicit_labeled_numbers(text))

        normalized = _distinct_numbers(explicit)
        if not normalized:
            if intermediate:
                return VerificationResult(
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    reason="The candidate appears as an intermediate/partial value, not an established final answer.",
                )
            if appears:
                return VerificationResult(
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    reason="The candidate appears in strong evidence but is not explicitly established as the final result.",
                )
            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason="The current strong evidence does not contain the candidate value.",
            )

        if len(normalized) > 1:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason="Multiple explicitly labelled result values require semantic resolution.",
            )

        if _numbers_equal(candidate_number, normalized[0]):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                reason="The candidate is explicitly supported by a labelled result in current strong evidence.",
            )
        return VerificationResult(
            status=VerificationStatus.INVALID,
            reason="The candidate is contradicted by the explicitly labelled result in current strong evidence.",
        )

    normalized_candidate = _normalize_text(candidate)
    exact = False
    appears = False
    for item in strong:
        text = _normalize_text(_get_result(item))
        exact |= text == normalized_candidate
        appears |= bool(re.search(rf"\b{re.escape(normalized_candidate)}\b", text))

    if exact:
        return VerificationResult(status=VerificationStatus.VERIFIED, reason="The candidate exactly matches strong evidence.")
    if appears:
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason="The candidate appears in evidence, but occurrence alone does not establish semantic support.",
        )
    return VerificationResult(
        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        reason="The deterministic verifier cannot establish semantic support for the textual candidate.",
    )


def evidence_supports_candidate(candidate_answer: str, raw_data: list[Any]) -> bool | None:
    result = deterministic_verification(candidate_answer, raw_data)
    if result.status == VerificationStatus.VERIFIED:
        return True
    if result.status == VerificationStatus.INVALID:
        return False
    return None


class VerifierAgent:
    """Two-stage verifier: deterministic proof first, semantic verification second."""

    def __init__(self, client: LLMClient, model: str) -> None:
        self.client = client
        self.model = model

    async def verify(self, data: VerificationInput) -> VerificationResult:
        candidate = data.candidate_answer.strip()
        if not candidate:
            return VerificationResult(status=VerificationStatus.INVALID, reason="The candidate answer is empty.")

        evidence = _filter_current_evidence(data.raw_data)
        if not evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason="No successful current independent evidence is available for verification.",
            )
        if not _check_source_support(data, evidence):
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason="The available evidence is incompatible with the requested task modality.",
            )

        deterministic = deterministic_verification(candidate, evidence)
        if deterministic.status in {VerificationStatus.VERIFIED, VerificationStatus.INVALID}:
            return deterministic
        return await self._semantic_verify(data, evidence, deterministic)

    async def _semantic_verify(
        self,
        data: VerificationInput,
        evidence: list[Any],
        deterministic_result: VerificationResult,
    ) -> VerificationResult:
        result = await self.client.generate(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._build_prompt(data, evidence, deterministic_result)},
            ],
            output_schema=VerificationResult,
        )
        return self._validate_llm_result(result, deterministic_result, evidence)

    def _validate_llm_result(
        self,
        result: Any,
        deterministic_result: VerificationResult,
        evidence: list[Any],
    ) -> VerificationResult:
        if not evidence:
            return VerificationResult(status=VerificationStatus.INSUFFICIENT_EVIDENCE, reason="No usable evidence.")
        if not isinstance(result, VerificationResult) or result.status is None:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason="The semantic verifier returned an invalid or incomplete verification result.",
            )

        reason = result.reason.strip()
        if result.status == VerificationStatus.VERIFIED and not reason:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason="The verifier returned VERIFIED without an evidence-grounded explanation.",
            )

        if deterministic_result.status == VerificationStatus.CONFLICTING_EVIDENCE:
            if result.status == VerificationStatus.VERIFIED and not self._reason_resolves_conflict(reason):
                return VerificationResult(
                    status=VerificationStatus.CONFLICTING_EVIDENCE,
                    reason="The verifier did not explain how the detected evidence conflict was resolved.",
                )

        if result.status in {
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.CONFLICTING_EVIDENCE,
            VerificationStatus.UNSUPPORTED,
        }:
            result.verified = False
        return result

    @staticmethod
    def _reason_resolves_conflict(reason: str) -> bool:
        markers = (
            "intermediate", "partial", "final", "different question", "different entity",
            "unrelated", "earlier state", "stale", "different unit", "more direct",
            "directly answers", "multi-step", "calculation", "artifact",
        )
        text = reason.lower()
        return any(marker in text for marker in markers)

    @staticmethod
    def _system_prompt() -> str:
        return """
You are a strict factual verification agent for a GAIA-style system.

Your ONLY source of truth is CURRENT EVIDENCE supplied in the prompt.
Never use outside knowledge. Never invent facts, calculations, citations, or evidence.
LLM output is not evidence. Failed steps are not evidence. Explicitly irrelevant
sources are not evidence. Evidence from an obsolete run/replan/attempt is not evidence.

A candidate is VERIFIED only when the evidence itself establishes the answer to the
EXACT QUESTION. A matching number or word is never sufficient by itself.

You MUST handle these adversarial cases:
- multiple numbers in one artifact
- intermediate/partial/subtotal vs final result
- multi-step calculations with missing steps
- conflicting tools or sources
- weak vs strong evidence
- hallucinated candidate answers
- verifier knowledge outside the evidence
- unrelated web evidence
- multiple artifacts from unrelated tasks
- failed-step evidence
- stale evidence after a replan
- correct answer with insufficient proof
- wrong answer whose number happens to occur in evidence
- structured artifact fields such as answer/final_answer/result/output

For conflicts, determine whether values belong to different entities, questions,
units, dates, attempts, intermediate steps, or genuinely conflict. If genuinely
unresolved, return CONFLICTING_EVIDENCE.

If the candidate is plausible but the evidence does not establish it, return
INSUFFICIENT_EVIDENCE. Do not reward correctness by coincidence.

Return VERIFIED only when the supplied evidence establishes the candidate.
Return INVALID when current evidence establishes a different answer or directly
contradicts the candidate.
Return UNSUPPORTED when the evidence modality cannot support the requested task.
Return a concise reason grounded only in the supplied evidence.
""".strip()

    def _build_prompt(
        self,
        data: VerificationInput,
        evidence: list[Any],
        deterministic_result: VerificationResult,
    ) -> str:
        return f"""
QUESTION:
{data.question}

CANDIDATE ANSWER:
{data.candidate_answer}

TASK TYPE:
{data.task_type}

DETERMINISTIC FINDING:
status={deterministic_result.status.value if deterministic_result.status else 'unknown'}
reason={deterministic_result.reason}

CURRENT EVIDENCE:
{self._format_evidence(evidence)}

Before returning VERIFIED, check:
1. Is every piece of support relevant to this exact question?
2. Is the candidate final rather than intermediate?
3. Are there multiple numbers/entities/artifacts that could be confused?
4. Do all required steps of a calculation exist in evidence?
5. Are any sources weak, unrelated, failed, stale, or from an older replan?
6. Does the candidate merely appear somewhere without being the requested answer?
7. Can the conclusion be justified without any outside knowledge?

If any required support is missing, return INSUFFICIENT_EVIDENCE.
""".strip()

    @staticmethod
    def _format_raw_item(item: Any) -> str:
        if isinstance(item, BaseModel):
            try:
                return json.dumps(item.model_dump(), ensure_ascii=False, default=str)
            except Exception:
                return str(item)
        if isinstance(item, dict):
            try:
                return json.dumps(item, ensure_ascii=False, default=str)
            except Exception:
                return str(item)
        return str(item)

    def _format_evidence(self, evidence: list[Any]) -> str:
        chunks: list[str] = []
        for index, item in enumerate(evidence, start=1):
            metadata = [
                f"tool={_get_tool_name(item) or 'unknown'}",
                f"strength={self._evidence_strength(item)}",
            ]
            for field in (
                "artifact_id", "step_id", "execution_id", "attempt_id", "run_id",
                "step_status",
            ):
                value = _get_str(item, field)
                if value:
                    metadata.append(f"{field}={value}")
            version = _get_plan_version(item)
            if version is not None:
                metadata.append(f"plan_version={version}")
            relevant = _get_relevant(item)
            metadata.append(f"relevant={relevant if relevant is not None else 'unknown'}")
            chunks.append(
                f"Evidence {index} ({', '.join(metadata)}):\n"
                f"{self._format_raw_item(_get_result(item))}"
            )
        return "\n\n".join(chunks)

    @staticmethod
    def _evidence_strength(item: Any) -> str:
        tool = _get_tool_name(item)
        if tool in _STRONG_TOOL_NAMES:
            return "strong"
        if tool in _SEMANTIC_TOOL_NAMES:
            return "semantic"
        return "weak"
