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
    r"(?<![\w.])"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?![\w.])"
)


_STRONG_TOOL_NAMES = {
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


_SEMANTIC_TOOL_NAMES = _STRONG_TOOL_NAMES | {
    "web_search",
    "browser",
    "http",
    "http_get",
    "github",
}


_STRUCTURED_ANSWER_KEYS = (
    "answer",
    "final_answer",
    "result",
    "output",
)


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
    return str(value).strip().lower()


def _get_result(item: Any) -> Any:
    value = _get_field(item, "result", None)
    if value is not None:
        return value
    return _get_field(item, "output", None)


def _get_succeeded(item: Any) -> bool:
    value = _get_field(item, "succeeded", None)
    if value is not None:
        return bool(value)
    value = _get_field(item, "success", None)
    if value is None:
        return True
    return bool(value)


def _get_str(
    item: Any,
    name: str,
) -> str | None:
    value = _get_field(item, name)
    if value is None:
        return None
    return str(value)


def _get_plan_version(
    item: Any,
) -> int | None:
    value = _get_field(item, "plan_version")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_relevant(
    item: Any,
) -> bool | None:
    value = _get_field(item, "relevant")
    if value is None:
        return None
    return bool(value)


def _iter_successful_evidence(
    raw_data: list[Any],
):
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


def _filter_current_evidence(
    raw_data: list[Any],
) -> list[Any]:
    evidence = list(
        _iter_successful_evidence(raw_data)
    )
    if not evidence:
        return []

    for field in (
        "run_id",
        "execution_id",
        "attempt_id",
    ):
        values = [
            _get_str(item, field)
            for item in evidence
        ]
        values = [
            value
            for value in values
            if value
        ]
        if values:
            newest = values[-1]
            evidence = [
                item
                for item in evidence
                if _get_str(item, field)
                in (None, newest)
            ]

    versions = [
        _get_plan_version(item)
        for item in evidence
    ]
    versions = [
        version
        for version in versions
        if version is not None
    ]
    if versions:
        newest = max(versions)
        evidence = [
            item
            for item in evidence
            if _get_plan_version(item)
            in (None, newest)
        ]

    valid_step_statuses = {
        "",
        "success",
        "succeeded",
        "completed",
        "complete",
        "ok",
        "verified",
    }

    return [
        item
        for item in evidence
        if (
            _get_str(item, "step_status")
            or ""
        ).strip().lower()
        in valid_step_statuses
    ]


def _iter_strong_evidence(
    raw_data: list[Any],
):
    for item in _filter_current_evidence(raw_data):
        if _get_tool_name(item) in _STRONG_TOOL_NAMES:
            yield item


def _extract_numbers(
    value: Any,
) -> list[str]:
    if value is None:
        return []
    return _NUMBER_RE.findall(
        str(value)
    )


def _normalize_text(
    value: Any,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        ""
        if value is None
        else str(value).strip().lower(),
    )


def _distinct_numbers(
    numbers: list[str],
) -> list[str]:
    result: list[str] = []
    for raw in numbers:
        try:
            normalized = format(
                float(raw),
                ".15g",
            )
        except (TypeError, ValueError):
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _numbers_equal(
    left: str,
    right: str,
) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _candidate_is_single_number(
    candidate: str,
) -> bool:
    numbers = _extract_numbers(candidate)
    return (
        len(numbers) == 1
        and _normalize_text(candidate)
        == _normalize_text(numbers[0])
    )


def _parse_structured_object(
    value: Any,
) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_structured_answer(
    value: Any,
) -> tuple[bool, Any]:
    parsed = _parse_structured_object(value)
    if parsed is None:
        return False, None
    for key in _STRUCTURED_ANSWER_KEYS:
        if key not in parsed:
            continue
        answer = parsed[key]
        if isinstance(
            answer,
            (dict, list, tuple, set),
        ):
            continue
        return True, answer
    return False, None


def _extract_structured_answers(
    strong: list[Any],
) -> list[Any]:
    answers: list[Any] = []
    for item in strong:
        found, answer = _extract_structured_answer(
            _get_result(item)
        )
        if found:
            answers.append(answer)
    return answers


def _extract_explicit_final_numbers(
    text: str,
) -> list[str]:
    values: list[str] = []
    patterns = (
        r"\bfinal\s+answer"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bfinal\s+result"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bcomputed\s+"
        r"(?:answer|result|value|score)"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bcalculated\s+"
        r"(?:answer|result|value|score)"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bscore\s+is\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\btotal\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
    )
    for pattern in patterns:
        values.extend(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )
    return values


def _extract_incidental_numbers(
    text: str,
) -> list[str]:
    patterns = (
        r"\binput(?:\s+\w+)?"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bpopulation"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\btemperature"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bid"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\byear"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bprevious\s+answer"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bexample\s+answer"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
        r"\bscore\s+of\s+another\s+item"
        r"\s*(?:is|=|:)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)",
    )
    values: list[str] = []
    for pattern in patterns:
        values.extend(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )
    return values


def _has_intermediate_marker(
    text: str,
) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(?:intermediate|partial|subtotal|"
            r"sub-result|subresult)\b",
            lowered,
        )
        or re.search(
            r"\bstep\s+\d+\b",
            lowered,
        )
    )


def _is_intermediate(
    text: str,
    number: str,
) -> bool:
    escaped = re.escape(number)
    patterns = (
        rf"(?:intermediate|partial|subtotal|"
        rf"sub-result|subresult|step)"
        rf".{{0,100}}{escaped}",
        rf"{escaped}.{{0,100}}"
        rf"(?:intermediate|partial|subtotal|"
        rf"sub-result|subresult|step)",
    )
    lowered = text.lower()
    return any(
        re.search(
            pattern,
            lowered,
        )
        for pattern in patterns
    )


def _check_source_support(
    data: VerificationInput,
    evidence: list[Any],
) -> bool:
    task_type = (
        data.task_type.strip().upper()
    )
    if task_type in {
        "TEXT",
        "GENERAL",
        "UNKNOWN",
    }:
        return True

    required = {
        "IMAGE": {
            "analyze_image",
        },
        "VISION": {
            "analyze_image",
        },
        "AUDIO": {
            "audio_reader",
            "transcribe_audio",
        },
        "VIDEO": {
            "video_reader",
            "analyze_video",
            "youtube_transcript",
        },
    }.get(task_type)

    if required is None:
        return True

    return any(
        _get_tool_name(item) in required
        for item in evidence
    )


def deterministic_verification(
    candidate_answer: str,
    raw_data: list[Any],
) -> VerificationResult:
    candidate = candidate_answer.strip()
    if not candidate:
        return VerificationResult(
            status=VerificationStatus.INVALID,
            reason="The candidate answer is empty.",
        )

    strong = list(
        _iter_strong_evidence(raw_data)
    )
    if not strong:
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason=(
                "No successful current strong evidence is "
                "available for deterministic verification."
            ),
        )

    structured_answers = _extract_structured_answers(
        strong
    )
    if structured_answers:
        if _candidate_is_single_number(candidate):
            candidate_number = _extract_numbers(
                candidate
            )[0]
            numeric_answers: list[str] = []

            for answer in structured_answers:
                numbers = _extract_numbers(answer)
                if (
                    len(numbers) == 1
                    and _candidate_is_single_number(
                        str(answer)
                    )
                ):
                    numeric_answers.append(
                        numbers[0]
                    )
                else:
                    return VerificationResult(
                        status=VerificationStatus.INVALID,
                        reason=(
                            "The authoritative structured answer "
                            "is not the numeric candidate."
                        ),
                    )

            numeric_answers = _distinct_numbers(
                numeric_answers
            )
            if len(numeric_answers) > 1:
                return VerificationResult(
                    status=VerificationStatus.CONFLICTING_EVIDENCE,
                    reason=(
                        "Multiple current structured artifacts "
                        "establish different authoritative answer values."
                    ),
                )

            if _numbers_equal(
                candidate_number,
                numeric_answers[0],
            ):
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    reason=(
                        "The candidate matches the "
                        "authoritative structured answer field."
                    ),
                )

            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason=(
                    "The authoritative structured answer "
                    "contradicts the candidate."
                ),
            )

        normalized_candidate = _normalize_text(
            candidate
        )
        normalized_answers = [
            _normalize_text(answer)
            for answer in structured_answers
        ]
        normalized_answers = list(
            dict.fromkeys(normalized_answers)
        )
        if len(normalized_answers) > 1:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "Multiple current structured artifacts "
                    "establish different authoritative answers."
                ),
            )

        if normalized_candidate == normalized_answers[0]:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                reason=(
                    "The candidate exactly matches the "
                    "authoritative structured answer field."
                ),
            )

        return VerificationResult(
            status=VerificationStatus.INVALID,
            reason=(
                "The authoritative structured answer "
                "contradicts the candidate."
            ),
        )

    if _candidate_is_single_number(candidate):
        candidate_number = _extract_numbers(
            candidate
        )[0]
        explicit: list[str] = []
        contextual: list[str] = []
        appears = False
        any_intermediate = False
        candidate_is_intermediate = False

        for item in strong:
            text = str(
                _get_result(item)
            )
            numbers = _extract_numbers(text)
            if any(
                _numbers_equal(
                    candidate_number,
                    number,
                )
                for number in numbers
            ):
                appears = True

            if _has_intermediate_marker(text):
                any_intermediate = True

            if _is_intermediate(
                text,
                candidate_number,
            ):
                candidate_is_intermediate = True

            explicit.extend(
                _extract_explicit_final_numbers(
                    text
                )
            )
            contextual.extend(
                _extract_incidental_numbers(
                    text
                )
            )

        explicit = _distinct_numbers(
            explicit
        )
        contextual = _distinct_numbers(
            contextual
        )

        if not explicit:
            if any_intermediate:
                return VerificationResult(
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    reason=(
                        "The evidence contains intermediate or "
                        "multi-step values but does not explicitly "
                        "establish a final numeric answer."
                    ),
                )

            if appears:
                return VerificationResult(
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    reason=(
                        "The candidate appears in strong evidence "
                        "but is not explicitly established as the "
                        "final result."
                    ),
                )

            if contextual:
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    reason=(
                        "The candidate is absent from current strong "
                        "evidence, which instead contains a different "
                        "explicit non-answer value."
                    ),
                )

            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason=(
                    "The evidence does not explicitly establish "
                    "a final numeric answer."
                ),
            )

        if len(explicit) > 1:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "Multiple explicitly labelled result values "
                    "require semantic resolution."
                ),
            )

        final_number = explicit[0]

        if any_intermediate:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "Intermediate/multi-step values coexist with "
                    "an explicit final result; semantic resolution "
                    "is required."
                ),
            )

        if _numbers_equal(
            candidate_number,
            final_number,
        ):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                reason=(
                    "The candidate is explicitly supported by a "
                    "labelled final result in current strong evidence."
                ),
            )

        if any(
            _numbers_equal(
                candidate_number,
                number,
            )
            for number in contextual
        ):
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "The candidate appears in a non-answer field "
                    "while current evidence establishes a different "
                    "final result."
                ),
            )

        if appears:
            return VerificationResult(
                status=VerificationStatus.CONFLICTING_EVIDENCE,
                reason=(
                    "The candidate appears in current evidence, "
                    "but a different explicitly labelled final "
                    "result exists."
                ),
            )

        return VerificationResult(
            status=VerificationStatus.INVALID,
            reason=(
                "The candidate is contradicted by the explicitly "
                "labelled final result in current strong evidence."
            ),
        )

    normalized_candidate = _normalize_text(
        candidate
    )
    exact = False
    appears = False

    for item in strong:
        text = _normalize_text(
            _get_result(item)
        )
        if text == normalized_candidate:
            exact = True
        if re.search(
            rf"\b{re.escape(normalized_candidate)}\b",
            text,
        ):
            appears = True

    if exact:
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason=(
                "The candidate exactly matches strong evidence."
            ),
        )

    if appears:
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason=(
                "The candidate appears in evidence, but occurrence "
                "alone does not establish semantic support."
            ),
        )

    return VerificationResult(
        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        reason=(
            "The deterministic verifier cannot establish semantic "
            "support for the textual candidate."
        ),
    )


def evidence_supports_candidate(
    candidate_answer: str,
    raw_data: list[Any],
) -> bool | None:
    result = deterministic_verification(
        candidate_answer,
        raw_data,
    )
    if result.status == VerificationStatus.VERIFIED:
        return True
    if result.status == VerificationStatus.INVALID:
        return False
    return None


class VerifierAgent:
    def __init__(
        self,
        client: LLMClient,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    async def verify(
        self,
        data: VerificationInput,
    ) -> VerificationResult:
        candidate = (
            data.candidate_answer.strip()
        )
        if not candidate:
            return VerificationResult(
                status=VerificationStatus.INVALID,
                reason="The candidate answer is empty.",
            )

        evidence = _filter_current_evidence(
            data.raw_data
        )
        if not evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason=(
                    "No successful current independent evidence "
                    "is available for verification."
                ),
            )

        if not self._check_source_support(
            data,
            evidence,
        ):
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason=(
                    "The available evidence is incompatible with "
                    "the requested task modality."
                ),
            )

        deterministic = deterministic_verification(
            candidate,
            evidence,
        )
        if deterministic.status in {
            VerificationStatus.VERIFIED,
            VerificationStatus.INVALID,
        }:
            return deterministic

        return await self._semantic_verify(
            data,
            evidence,
            deterministic,
        )

    async def _semantic_verify(
        self,
        data: VerificationInput,
        evidence: list[Any],
        deterministic_result: VerificationResult,
    ) -> VerificationResult:
        result = await self.client.generate(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(
                        data,
                        evidence,
                        deterministic_result,
                    ),
                },
            ],
            output_schema=VerificationResult,
        )
        return self._validate_llm_result(
            result,
            deterministic_result,
            evidence,
        )

    def _validate_llm_result(
        self,
        result: Any,
        deterministic_result: VerificationResult,
        evidence: list[Any],
    ) -> VerificationResult:
        if not evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason="No usable evidence.",
            )

        if (
            not isinstance(
                result,
                VerificationResult,
            )
            or result.status is None
        ):
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason=(
                    "The semantic verifier returned an invalid "
                    "or incomplete verification result."
                ),
            )

        reason = result.reason.strip()
        if (
            result.status
            == VerificationStatus.VERIFIED
            and not reason
        ):
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason=(
                    "The verifier returned VERIFIED without an "
                    "evidence-grounded explanation."
                ),
            )

        if (
            deterministic_result.status
            == VerificationStatus.CONFLICTING_EVIDENCE
        ):
            if (
                result.status
                == VerificationStatus.VERIFIED
                and not self._reason_resolves_conflict(
                    reason
                )
            ):
                return VerificationResult(
                    status=VerificationStatus.CONFLICTING_EVIDENCE,
                    reason=(
                        "The verifier did not explain how the "
                        "detected evidence conflict was resolved."
                    ),
                )

        if result.status in {
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.CONFLICTING_EVIDENCE,
            VerificationStatus.UNSUPPORTED,
        }:
            result.verified = False

        return result

    @staticmethod
    def _reason_resolves_conflict(
        reason: str,
    ) -> bool:
        markers = (
            "intermediate",
            "partial",
            "final",
            "different question",
            "different entity",
            "unrelated",
            "earlier state",
            "stale",
            "different unit",
            "more direct",
            "directly answers",
            "multi-step",
            "calculation",
            "artifact",
        )
        text = reason.lower()
        return any(
            marker in text
            for marker in markers
        )

    @staticmethod
    def _system_prompt() -> str:
        return """
You are a strict factual verification agent for a GAIA-style system.

Your ONLY source of truth is CURRENT EVIDENCE supplied in the prompt.

Never use outside knowledge.
Never invent facts, calculations, citations, or evidence.

LLM output is NOT evidence.
Failed steps are NOT evidence.
Explicitly irrelevant sources are NOT evidence.
Evidence from an obsolete run, replan, attempt, or plan version
is NOT evidence.

A candidate is VERIFIED only when the evidence itself establishes
the answer to the EXACT QUESTION.

A matching number or word is never sufficient by itself.

You MUST handle:
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

For structured artifacts:
    answer
    final_answer
    result
    output
are authoritative answer fields.

Metadata such as IDs, years, question IDs,
unrelated scores, and other incidental values
must NOT be treated as the answer.

For conflicts, determine whether values belong to:
- different entities
- different questions
- different units
- different dates
- intermediate steps
- earlier attempts
- genuinely conflicting final answers

If genuinely unresolved, return CONFLICTING_EVIDENCE.

If the candidate is plausible but the evidence does not establish it,
return INSUFFICIENT_EVIDENCE.

Do not reward correctness by coincidence.

Return VERIFIED only when the supplied evidence establishes the candidate.
Return INVALID when current evidence establishes a different answer
or directly contradicts the candidate.
Return UNSUPPORTED when the evidence modality cannot support
the requested task.

Return a concise reason grounded only in supplied evidence.
""".strip()

    def _build_prompt(
        self,
        data: VerificationInput,
        evidence: list[Any],
        deterministic_result: VerificationResult,
    ) -> str:
        status = (
            deterministic_result.status.value
            if deterministic_result.status
            else "unknown"
        )
        return f"""
QUESTION:
{data.question}

CANDIDATE ANSWER:
{data.candidate_answer}

TASK TYPE:
{data.task_type}

DETERMINISTIC FINDING:
status={status}
reason={deterministic_result.reason}

CURRENT EVIDENCE:
{self._format_evidence(evidence)}

Before returning VERIFIED, check:
1. Is every piece of support relevant to this exact question?
2. Is the candidate final rather than intermediate?
3. Are there multiple numbers, entities, or artifacts that could be confused?
4. Do all required steps of a calculation exist in evidence?
5. Are any sources weak, unrelated, failed, stale, or from an older replan?
6. Does the candidate merely appear somewhere without being the requested answer?
7. If the evidence is structured, does the candidate match the
   authoritative answer/final_answer/result/output field?
8. Can the conclusion be justified without outside knowledge?

If any required support is missing, return INSUFFICIENT_EVIDENCE.
""".strip()

    @staticmethod
    def _format_raw_item(
        item: Any,
    ) -> str:
        if isinstance(item, BaseModel):
            try:
                return json.dumps(
                    item.model_dump(),
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:
                return str(item)

        if isinstance(item, dict):
            try:
                return json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:
                return str(item)

        return str(item)

    def _format_evidence(
        self,
        evidence: list[Any],
    ) -> str:
        chunks: list[str] = []
        for index, item in enumerate(
            evidence,
            start=1,
        ):
            metadata = [
                f"tool={_get_tool_name(item) or 'unknown'}",
                f"strength={self._evidence_strength(item)}",
            ]
            for field in (
                "artifact_id",
                "step_id",
                "execution_id",
                "attempt_id",
                "run_id",
                "step_status",
            ):
                value = _get_str(
                    item,
                    field,
                )
                if value:
                    metadata.append(
                        f"{field}={value}"
                    )

            version = _get_plan_version(item)
            if version is not None:
                metadata.append(
                    f"plan_version={version}"
                )

            relevant = _get_relevant(item)
            metadata.append(
                "relevant="
                + (
                    str(relevant)
                    if relevant is not None
                    else "unknown"
                )
            )

            chunks.append(
                f"Evidence {index} "
                f"({', '.join(metadata)}):\n"
                f"{self._format_raw_item(_get_result(item))}"
            )

        return "\n\n".join(chunks)

    @staticmethod
    def _evidence_strength(
        item: Any,
    ) -> str:
        tool = _get_tool_name(item)
        if tool in _STRONG_TOOL_NAMES:
            return "strong"
        if tool in _SEMANTIC_TOOL_NAMES:
            return "semantic"
        return "weak"

    def _check_source_support(
        self,
        data: VerificationInput,
        evidence: list[Any],
    ) -> bool:
        return _check_source_support(
            data,
            evidence,
        )