from __future__ import annotations

import pytest

from gaia_agent.agents.verifier import (
    EvidenceItem,
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
)


# ============================================================================
# Fake LLM
# ============================================================================


class FakeLLMClient:
    """
    Configurable semantic verifier.

    Each test decides exactly what the semantic layer should return.
    """

    def __init__(self, result: VerificationResult):
        self.result = result
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


# ============================================================================
# Helpers
# ============================================================================


def evidence(
    tool_name: str,
    result,
    *,
    succeeded: bool = True,
    artifact_id: str | None = None,
    source_type: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        tool_name=tool_name,
        result=result,
        succeeded=succeeded,
        artifact_id=artifact_id,
        source_type=source_type,
    )


def make_input(
    *,
    question: str = "What is the result?",
    candidate_answer: str = "42",
    raw_data: list[EvidenceItem] | None = None,
    task_type: str = "general",
) -> VerificationInput:
    return VerificationInput(
        question=question,
        candidate_answer=candidate_answer,
        raw_data=raw_data or [],
        task_type=task_type,
    )


def verified_result(
    reason: str = "The evidence semantically supports the candidate.",
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.VERIFIED,
        verified=True,
        reason=reason,
    )


def invalid_result(
    reason: str = "The candidate is not supported by the evidence.",
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.INVALID,
        verified=False,
        reason=reason,
    )


def insufficient_result(
    reason: str = "The evidence is insufficient.",
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        verified=False,
        reason=reason,
    )


def conflicting_result(
    reason: str = "The evidence contains unresolved conflicting values.",
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.CONFLICTING_EVIDENCE,
        verified=False,
        reason=reason,
    )


def unsupported_result(
    reason: str = "The evidence does not support the candidate.",
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.UNSUPPORTED,
        verified=False,
        reason=reason,
    )


def build_verifier(
    llm_result: VerificationResult,
) -> tuple[VerifierAgent, FakeLLMClient]:
    llm = FakeLLMClient(llm_result)

    verifier = VerifierAgent(
        client=llm,
        model="test-model",
    )

    return verifier, llm


# ============================================================================
# 1. Empty candidate
# ============================================================================


@pytest.mark.asyncio
async def test_empty_candidate_is_invalid_without_llm():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 42",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INVALID
    assert result.verified is False
    assert len(llm.calls) == 0


# ============================================================================
# 2. No evidence
# ============================================================================


@pytest.mark.asyncio
async def test_no_successful_evidence_is_insufficient():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="42",
        raw_data=[],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 0


# ============================================================================
# 3. Failed evidence
# ============================================================================


@pytest.mark.asyncio
async def test_failed_evidence_is_ignored():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 999",
                succeeded=False,
            ),
            evidence(
                "python_interpreter",
                "Calculation failed",
                succeeded=False,
            ),
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 0


# ============================================================================
# 4. LLM evidence is not strong evidence
# ============================================================================


@pytest.mark.asyncio
async def test_llm_evidence_is_not_used_as_strong_tool_evidence():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "llm",
                "Final answer = 42",
            ),
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 0


# ============================================================================
# 5. Deterministic numeric verification
# ============================================================================


@pytest.mark.asyncio
async def test_numeric_candidate_is_verified_from_explicit_final_result():
    verifier, llm = build_verifier(
        invalid_result(
            "This should never be used.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 42",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 0


# ============================================================================
# 6. Numeric mismatch
# ============================================================================


@pytest.mark.asyncio
async def test_numeric_candidate_mismatch_is_invalid():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 43",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INVALID
    assert result.verified is False
    assert len(llm.calls) == 0


# ============================================================================
# 7. Explicit conflict reaches LLM
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_explicit_results_create_conflict_and_reach_llm():
    verifier, llm = build_verifier(
        conflicting_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Calculation A = 42; Calculation B = 43",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1


# ============================================================================
# 8. LLM resolves conflict
# ============================================================================


@pytest.mark.asyncio
async def test_llm_can_resolve_conflicting_evidence():
    verifier, llm = build_verifier(
        verified_result(
            "42 is the final result; 43 is an intermediate calculation.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                """
                Calculation A = 42
                Calculation B = 43
                Final result = 42
                """,
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 1

    prompt_text = str(llm.calls[0])

    assert "42" in prompt_text
    assert "43" in prompt_text


# ============================================================================
# 9. LLM refuses conflict
# ============================================================================


@pytest.mark.asyncio
async def test_llm_can_keep_conflict_unresolved():
    verifier, llm = build_verifier(
        conflicting_result(
            "Both 42 and 43 are presented as possible final results.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Calculation A = 42; Calculation B = 43",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1


# ============================================================================
# 10. Insufficient evidence reaches semantic verification
# ============================================================================


@pytest.mark.asyncio
async def test_insufficient_evidence_reaches_llm():
    verifier, llm = build_verifier(
        insufficient_result(
            "The evidence mentions the candidate but does not establish it.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "The calculation produced values including 42.",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1


# ============================================================================
# 11. Semantic verification of ambiguous evidence
# ============================================================================


@pytest.mark.asyncio
async def test_llm_can_upgrade_ambiguous_evidence_to_verified():
    verifier, llm = build_verifier(
        verified_result(
            "The evidence clearly supports 42 as the requested answer.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "The process log contains: intermediate calculation 42, "
                "followed by additional processing.",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True

    # This evidence intentionally does not contain an explicit
    # result/value/score/total/output label, reaching the semantic LLM.
    assert len(llm.calls) == 1


# ============================================================================
# 12. Unsupported candidate
# ============================================================================


@pytest.mark.asyncio
async def test_unsupported_candidate_is_not_semantically_rescued():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="999",
        raw_data=[
            evidence(
                "python_interpreter",
                "Temperature = 42",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.verified is False
    assert result.status in {
        VerificationStatus.INVALID,
        VerificationStatus.UNSUPPORTED,
    }

    assert len(llm.calls) == 0


# ============================================================================
# 13. Text exact match
# ============================================================================


@pytest.mark.asyncio
async def test_text_candidate_exact_match_follows_current_contract():
    verifier, llm = build_verifier(
        invalid_result(),
    )

    request = make_input(
        candidate_answer="Paris",
        raw_data=[
            evidence(
                "web_search",
                "Paris is the capital of France.",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status in {
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.INVALID,
    }

    assert result.verified is False


# ============================================================================
# 14. Text candidate can be semantically verified
# ============================================================================


@pytest.mark.asyncio
async def test_text_candidate_can_be_semantically_verified():
    verifier, llm = build_verifier(
        verified_result(
            "The evidence clearly supports Paris as the answer.",
        ),
    )

    request = make_input(
        candidate_answer="Paris",
        raw_data=[
            evidence(
                "web_search",
                "The capital city is Paris.",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 1


# ============================================================================
# 15. Text candidate in unrelated evidence
# ============================================================================


@pytest.mark.asyncio
async def test_completely_unsupported_text_is_not_verified():
    verifier, llm = build_verifier(
        insufficient_result(),
    )

    request = make_input(
        candidate_answer="Tokyo",
        raw_data=[
            evidence(
                "web_search",
                "The capital mentioned in the source is Paris.",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.verified is False
    assert result.status in {
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.INVALID,
        VerificationStatus.UNSUPPORTED,
    }


# ============================================================================
# 16. Artifact ID reaches semantic prompt
# ============================================================================


@pytest.mark.asyncio
async def test_artifact_id_reaches_semantic_verification():
    verifier, llm = build_verifier(
        conflicting_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Calculation A = 42; Calculation B = 43",
                artifact_id="artifact-123",
                source_type="computed",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert len(llm.calls) == 1

    prompt_text = str(llm.calls[0])

    assert "artifact-123" in prompt_text


# ============================================================================
# 17. Alternate output/success fields
# ============================================================================


@pytest.mark.asyncio
async def test_evidence_supports_output_and_success_aliases():
    verifier, llm = build_verifier(
        invalid_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            {
                "tool_name": "python_interpreter",
                "output": "Final result = 42",
                "success": True,
            }
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 0


# ============================================================================
# 18. None result ignored
# ============================================================================


@pytest.mark.asyncio
async def test_evidence_with_none_result_is_ignored():
    verifier, llm = build_verifier(verified_result())

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                None,
            ),
            evidence(
                "web_search",
                None,
            ),
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 0


# ============================================================================
# 19. Weak LLM evidence cannot override strong tool evidence
# ============================================================================


@pytest.mark.asyncio
async def test_weak_llm_evidence_does_not_override_strong_tool_evidence():
    verifier, llm = build_verifier(
        invalid_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "llm",
                "The answer is 999.",
            ),
            evidence(
                "python_interpreter",
                "Final result = 42",
            ),
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 0


# ============================================================================
# 20. Multiple strong sources
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_strong_sources_with_explicit_conflict_reach_llm():
    verifier, llm = build_verifier(
        conflicting_result(
            "Two independent sources disagree.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Calculation A = 42; Calculation B = 43",
            ),
            evidence(
                "calculator",
                "Calculation A = 42; Calculation B = 43",
            ),
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1


# ============================================================================
# 21. Numeric candidate inside natural language
# ============================================================================


@pytest.mark.asyncio
async def test_numeric_candidate_inside_natural_language():
    verifier, llm = build_verifier(
        verified_result(
            "The candidate is semantically supported.",
        ),
    )

    request = make_input(
        candidate_answer="The final answer is 42.",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 42",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 1


# ============================================================================
# 22. Duplicate identical explicit results
# ============================================================================


@pytest.mark.asyncio
async def test_duplicate_identical_results_are_not_conflicting():
    verifier, llm = build_verifier(
        invalid_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 42",
            ),
            evidence(
                "calculator",
                "Final result = 42",
            ),
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 0


# ============================================================================
# 23. Deterministic status reaches semantic prompt
# ============================================================================


@pytest.mark.asyncio
async def test_semantic_prompt_contains_deterministic_status():
    verifier, llm = build_verifier(
        conflicting_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Calculation A = 42; Calculation B = 43",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert len(llm.calls) == 1

    prompt_text = str(llm.calls[0])

    assert "CONFLICTING_EVIDENCE" in prompt_text


# ============================================================================
# 24. Deterministic reason is preserved
# ============================================================================


@pytest.mark.asyncio
async def test_deterministic_reason_is_preserved():
    verifier, llm = build_verifier(
        verified_result(
            "This reason should never replace the deterministic reason.",
        ),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 42",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True

    assert result.reason == (
        "The candidate is explicitly supported by a labelled result "
        "in strong evidence."
    )

    assert len(llm.calls) == 0


# ============================================================================
# 25. VerificationResult invariant
# ============================================================================


def test_verification_result_normalizes_verified_flag():
    verified = VerificationResult(
        status=VerificationStatus.VERIFIED,
        verified=False,
        reason="Verified.",
    )

    assert verified.verified is True

    invalid = VerificationResult(
        status=VerificationStatus.INVALID,
        verified=True,
        reason="Invalid.",
    )

    assert invalid.verified is False


# ============================================================================
# 26. Status aliases
# ============================================================================


def test_verification_status_aliases_are_preserved():
    assert VerificationStatus.PASS == VerificationStatus.VERIFIED
    assert VerificationStatus.FAIL == VerificationStatus.INVALID
    assert VerificationStatus.UNCERTAIN == (
        VerificationStatus.INSUFFICIENT_EVIDENCE
    )


# ============================================================================
# 27. Deterministic VERIFIED short-circuits LLM
# ============================================================================


@pytest.mark.asyncio
async def test_deterministic_verified_short_circuits_semantic_verification():
    verifier, llm = build_verifier(
        conflicting_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 42",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert llm.calls == []


# ============================================================================
# 28. Deterministic INVALID short-circuits LLM
# ============================================================================


@pytest.mark.asyncio
async def test_deterministic_invalid_short_circuits_semantic_verification():
    verifier, llm = build_verifier(
        verified_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Final result = 100",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.INVALID
    assert result.verified is False
    assert llm.calls == []


# ============================================================================
# 29. Conflict is not silently converted to invalid
# ============================================================================


@pytest.mark.asyncio
async def test_conflict_is_not_silently_converted_to_invalid():
    verifier, llm = build_verifier(
        conflicting_result(),
    )

    request = make_input(
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "Calculation A = 42; Calculation B = 43",
            )
        ],
    )

    result = await verifier.verify(request)

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1


# ============================================================================
# 30. Full end-to-end verifier contract
# ============================================================================


@pytest.mark.asyncio
async def test_verifier_strong_end_to_end_contract():

    # ------------------------------------------------------------------
    # Case 1: deterministic VERIFIED
    # ------------------------------------------------------------------

    verifier, llm = build_verifier(
        invalid_result(),
    )

    result = await verifier.verify(
        make_input(
            candidate_answer="42",
            raw_data=[
                evidence(
                    "python_interpreter",
                    "Final result = 42",
                )
            ],
        )
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 0

    # ------------------------------------------------------------------
    # Case 2: deterministic INVALID
    # ------------------------------------------------------------------

    verifier, llm = build_verifier(
        verified_result(),
    )

    result = await verifier.verify(
        make_input(
            candidate_answer="42",
            raw_data=[
                evidence(
                    "python_interpreter",
                    "Final result = 100",
                )
            ],
        )
    )

    assert result.status == VerificationStatus.INVALID
    assert result.verified is False
    assert len(llm.calls) == 0

    # ------------------------------------------------------------------
    # Case 3: conflict -> semantic verifier
    # ------------------------------------------------------------------

    verifier, llm = build_verifier(
        conflicting_result(
            "The evidence contains two unresolved values.",
        ),
    )

    result = await verifier.verify(
        make_input(
            candidate_answer="42",
            raw_data=[
                evidence(
                    "python_interpreter",
                    "Calculation A = 42; Calculation B = 43",
                )
            ],
        )
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1

    # ------------------------------------------------------------------
    # Case 4: semantic verifier resolves conflict
    # ------------------------------------------------------------------

    verifier, llm = build_verifier(
        verified_result(
            "42 is explicitly marked as the final result.",
        ),
    )

    result = await verifier.verify(
        make_input(
            candidate_answer="42",
            raw_data=[
                evidence(
                    "python_interpreter",
                    """
                    Calculation A = 42
                    Calculation B = 43
                    Final result = 42
                    """,
                )
            ],
        )
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert len(llm.calls) == 1

    # ------------------------------------------------------------------
    # Case 5: semantic verifier refuses conflict
    # ------------------------------------------------------------------

    verifier, llm = build_verifier(
        conflicting_result(
            "There is no reliable basis to choose between 42 and 43.",
        ),
    )

    result = await verifier.verify(
        make_input(
            candidate_answer="42",
            raw_data=[
                evidence(
                    "python_interpreter",
                    "Calculation A = 42; Calculation B = 43",
                )
            ],
        )
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE
    assert result.verified is False
    assert len(llm.calls) == 1