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
            "Whether the candidate answer is directly and "
            "adequately supported by evidence relevant to the question."
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
    raw_data: list[Any] = Field(default_factory=list)


class VerificationStatus(str):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"

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
        r"(?<=\d)[,](?=\d{3}(?!\d))",
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


def _iter_strong_evidence(raw_data: list[Any]) -> list[Any]:
    items: list[Any] = []

    for item in raw_data or []:
        tool_name = _get_tool_name(item)

        if not tool_name:
            continue

        if tool_name == "llm":
            continue

        if tool_name not in _STRONG_TOOL_NAMES:
            continue

        if not _get_succeeded(item):
            continue

        result = _get_result(item)

        if result is None:
            continue

        items.append(item)

    return items


def _iter_all_successful_evidence(
    raw_data: list[Any],
) -> list[tuple[str, str]]:
    evidence: list[tuple[str, str]] = []

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
                str(result),
            )
        )

    return evidence


def evidence_supports_candidate(
    candidate_answer: str,
    raw_data: list[Any],
) -> bool | None:
    """
    Determine whether strong deterministic evidence supports the
    candidate answer.

    Returns:
        True:
            Strong evidence clearly supports the candidate.

        False:
            Strong evidence clearly contradicts the candidate.

        None:
            Evidence is missing, ambiguous, or insufficient.
    """
    if candidate_answer is None:
        return None

    candidate = str(candidate_answer).strip()

    if not candidate:
        return None

    strong_items = list(_iter_strong_evidence(raw_data))

    if not strong_items:
        return None

    # ------------------------------------------------------------
    # Numeric candidate
    # ------------------------------------------------------------
    candidate_numbers = _extract_numbers(candidate)

    if candidate_numbers:
        candidate_number = candidate_numbers[0]

        all_numbers: list[float] = []

        for item in strong_items:
            result = _get_result(item)

            if result is None:
                continue

            all_numbers.extend(
                _extract_numbers(str(result))
            )

        # Remove duplicates while preserving order.
        distinct_numbers = list(dict.fromkeys(all_numbers))

        # No numeric evidence.
        if not distinct_numbers:
            return None

        # More than one distinct number means that we cannot
        # deterministically identify which number is the answer.
        if len(distinct_numbers) != 1:
            return None

        # Exactly one numeric value exists in the strong evidence.
        return abs(candidate_number - distinct_numbers[0]) < 1e-9

    # ------------------------------------------------------------
    # Text candidate
    # ------------------------------------------------------------
    normalized_candidate = candidate.casefold()

    for item in strong_items:
        result = _get_result(item)

        if result is None:
            continue

        text = str(result).casefold().strip()

        if not text:
            continue

        if normalized_candidate in text:
            return True

    return None


def deterministic_verification(
    candidate_answer: str,
    raw_data: list[Any],
) -> tuple[VerificationStatus, str]:

    if candidate_answer is None:
        return (
            VerificationStatus.FAIL,
            "Candidate answer is missing.",
        )

    candidate = str(candidate_answer).strip()

    if not candidate:
        return (
            VerificationStatus.FAIL,
            "Candidate answer is empty.",
        )

    strong_evidence = _iter_strong_evidence(
        list(raw_data or [])
    )

    if not strong_evidence:
        return (
            VerificationStatus.UNCERTAIN,
            (
                "No strong deterministic tool evidence is "
                "available for independent verification."
            ),
        )

    evidence_texts = [str(_get_result(item)) for item in strong_evidence]
    joined = "\n".join(evidence_texts)

    candidate_numbers = _extract_numbers(candidate)
    evidence_numbers = _extract_numbers(joined)
    if len(candidate_numbers) == 1:
        candidate_value = candidate_numbers[0]

        if not evidence_numbers:
            return (
                VerificationStatus.UNCERTAIN,
                (
                    "The candidate is numeric, but the strong "
                    "tool evidence contains no numeric value."
                ),
            )

        distinct = sorted(set(evidence_numbers))

        if len(distinct) == 1:
            evidence_value = distinct[0]

            if abs(candidate_value - evidence_value) < 1e-9:
                return (
                    VerificationStatus.PASS,
                    (
                        "The candidate number matches the single "
                        "numeric value in deterministic tool evidence."
                    ),
                )

            return (
                VerificationStatus.FAIL,
                (
                    "Deterministic numeric evidence contradicts "
                    f"the candidate: evidence={evidence_value:g}, "
                    f"candidate={candidate_value:g}."
                ),
            )

        return (
            VerificationStatus.UNCERTAIN,
            (
                "The deterministic evidence contains multiple "
                "numeric values, so the correct answer cannot be "
                "identified safely by numeric matching alone."
            ),
        )

    normalized_candidate = re.sub(
        r"\s+",
        " ",
        candidate.lower(),
    ).strip()

    normalized_evidence = re.sub(
        r"\s+",
        " ",
        joined.lower(),
    )

    if (
        normalized_candidate
        and normalized_candidate in normalized_evidence
    ):
        return (
            VerificationStatus.PASS,
            (
                "The candidate answer appears directly in "
                "deterministic tool evidence."
            ),
        )

    return (
        VerificationStatus.UNCERTAIN,
        (
            "No direct deterministic match or contradiction "
            "was found."
        ),
    )


class VerifierAgent:
    """
    Strict final-answer verifier.

    The verifier:
    - does not generate a replacement answer
    - does not execute tools
    - does not replan
    - does not perform recovery
    - evaluates whether evidence actually supports the candidate
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
        Verify the candidate answer.

        Verification pipeline:

            1. Check whether evidence exists.
            2. Run deterministic verification.
            3. If deterministic PASS -> accept.
            4. If deterministic FAIL -> reject.
            5. If UNCERTAIN -> ask the LLM to judge relevance.
            6. For web evidence, the LLM is allowed to make the
               semantic relevance decision.
            7. Never allow the LLM to verify an answer when there is
               no successful evidence.
        """
        # ------------------------------------------------------------
        # 1. Evidence must exist before we ask the LLM anything.
        # ------------------------------------------------------------
        successful_evidence = list(
            _iter_all_successful_evidence(data.raw_data)
        )

        if not successful_evidence:
            return VerificationResult(
                verified=False,
                reason=(
                    "Verification failed because no successful "
                    "evidence is available."
                ),
            )

        # ------------------------------------------------------------
        # 2. Deterministic verification
        # ------------------------------------------------------------
        deterministic_status, deterministic_reason = (
            deterministic_verification(
                data.candidate_answer,
                data.raw_data,
            )
        )

        # ------------------------------------------------------------
        # 3. Deterministic contradiction always wins.
        # ------------------------------------------------------------
        if deterministic_status == VerificationStatus.FAIL:
            return VerificationResult(
                verified=False,
                reason=(
                    "Deterministic verification rejected the "
                    f"candidate: {deterministic_reason}"
                ),
            )

        # ------------------------------------------------------------
        # 4. Strong deterministic evidence is sufficient.
        # ------------------------------------------------------------
        if deterministic_status == VerificationStatus.PASS:
            return VerificationResult(
                verified=True,
                reason=deterministic_reason,
            )

        # ------------------------------------------------------------
        # 5. Deterministic verification is uncertain.
        #
        #    Now the LLM judges whether the QUESTION, CANDIDATE and
        #    EVIDENCE actually correspond.
        #
        #    This is especially important for web_search evidence.
        # ------------------------------------------------------------
        messages = self._build_messages(data)

        result = await self.client.generate(
            messages=messages,
            model=self.model,
            output_schema=VerificationResult,
        )

        # ------------------------------------------------------------
        # 6. Validate LLM response.
        # ------------------------------------------------------------
        if not isinstance(result, VerificationResult):
            raise TypeError(
                "LLMClient.generate() returned an invalid "
                "verification result."
            )

        # ------------------------------------------------------------
        # 7. LLM rejection means rejection.
        # ------------------------------------------------------------
        if not result.verified:
            return VerificationResult(
                verified=False,
                reason=(
                    "LLM verification rejected the candidate: "
                    f"{result.reason}"
                ),
            )

        # ------------------------------------------------------------
        # 8. LLM accepted the evidence.
        #
        #    Web evidence is semantically verified by the LLM.
        # ------------------------------------------------------------
        return VerificationResult(
            verified=True,
            reason=(
                "Candidate was verified by the LLM using the "
                "available evidence: "
                f"{result.reason}"
            ),
        )

    def _build_messages(
        self,
        data: VerificationInput,
    ) -> list[dict[str, str]]:
        """Build the verifier messages."""
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
        """
        Strict verification policy.

        The critical rule is that evidence must answer the question,
        not merely contain the candidate value.
        """
        return (
            "You are a strict factual answer verification agent.\n\n"

            "Your ONLY task is to determine whether the candidate "
            "answer is supported by the provided evidence.\n\n"

            "CRITICAL RULES:\n"
            "1. Evaluate the evidence against the EXACT question.\n"
            "2. The evidence must actually support the answer to "
            "the question.\n"
            "3. The mere presence of the candidate value in the "
            "evidence is NOT sufficient.\n"
            "4. A number appearing in an unrelated web-search "
            "result is NOT evidence for that number being the "
            "answer.\n"
            "5. Do not assume that the first, largest, smallest, "
            "or most visible number in a source is the answer.\n"
            "6. Do not infer facts that are not supported by the "
            "provided evidence.\n"
            "7. Do not use outside knowledge.\n"
            "8. Do not rewrite or improve the candidate answer.\n"
            "9. If the evidence is irrelevant, ambiguous, "
            "insufficient, or contradictory, return verified=false.\n"
            "10. If the question refers to a specific file, image, "
            "audio recording, video, URL, table, or document, "
            "evidence must correspond to that specific source.\n"
            "11. For web-search evidence, verify that the retrieved "
            "content actually addresses the question rather than "
            "merely containing matching words or numbers.\n"
            "12. Return verified=true ONLY when the evidence provides "
            "a reasonable and direct factual basis for the candidate.\n"
            "13. Single-word and exact-number answers are valid when "
            "the evidence directly supports them.\n"
            "14. When uncertain, prefer verified=false.\n\n"

            "Return a concise reason explaining the decision."
        )

    @staticmethod
    def _build_prompt(
        data: VerificationInput,
    ) -> str:
        """Build the user prompt for the verification judge."""
        evidence_items = _iter_all_successful_evidence(
            data.raw_data
        )

        if not evidence_items:
            raw_data = "(No successful evidence was provided.)"
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

            raw_data = "\n\n".join(chunks)

        return (
            "Verify the following candidate answer.\n\n"
            f"QUESTION:\n{data.question}\n\n"
            f"CANDIDATE ANSWER:\n{data.candidate_answer}\n\n"
            "EVIDENCE:\n"
            f"{raw_data}\n\n"
            "Decision requirements:\n"
            "- Does the evidence actually answer the question?\n"
            "- Is the candidate supported by that evidence?\n"
            "- Is the evidence about the specific source referenced "
            "by the question?\n"
            "- Is there any contradiction or material uncertainty?\n\n"
            "Return verified=true only if the evidence directly "
            "supports the candidate answer."
        )

    @staticmethod
    def _format_raw_item(
        item: Any,
    ) -> str:
        """
        Convert arbitrary raw data to safe text.

        Kept as a compatibility helper for existing callers/tests.
        """
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