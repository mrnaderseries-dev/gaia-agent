from __future__ import annotations

from types import SimpleNamespace

import pytest

from gaia_agent.evaluation.verification import (
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
    deterministic_verification,
    evidence_supports_candidate,
)


# ============================================================================
# Helpers
# ============================================================================


def ev(
    tool_name: str,
    result,
    *,
    succeeded: bool = True,
    artifact_id: str | None = None,
    source_type: str | None = None,
    step_id: str | None = None,
    execution_id: str | None = None,
    attempt_id: str | None = None,
    run_id: str | None = None,
    plan_version: int | None = None,
    step_status: str | None = None,
    relevant: bool | None = None,
):
    return SimpleNamespace(
        tool_name=tool_name,
        result=result,
        succeeded=succeeded,
        artifact_id=artifact_id,
        source_type=source_type,
        step_id=step_id,
        execution_id=execution_id,
        attempt_id=attempt_id,
        run_id=run_id,
        plan_version=plan_version,
        step_status=step_status,
        relevant=relevant,
    )


def assert_status(
    candidate: str,
    evidence: list,
    expected: VerificationStatus,
):
    result = deterministic_verification(
        candidate_answer=candidate,
        raw_data=evidence,
    )

    assert result.status == expected, (
        f"\nExpected: {expected}"
        f"\nActual:   {result.status}"
        f"\nReason:   {result.reason}"
    )


def make_agent() -> VerifierAgent:
    return object.new(VerifierAgent)


# ============================================================================
# 1. MULTIPLE NUMBERS
# ============================================================================


def test_multiple_numbers_candidate_appears_but_is_not_result():
    evidence = [
        ev(
            "python_interpreter",
            "Input A = 10, input B = 32, unrelated value = 42.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_multiple_numbers_with_explicit_final_result():
    evidence = [
        ev(
            "python_interpreter",
            "Input A = 10. Input B = 32. Final result: 42.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_candidate_matches_wrong_number_in_artifact():
    evidence = [
        ev(
            "python_interpreter",
            (
                "Population = 42. "
                "Temperature = 17. "
                "Final result = 25."
            ),
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.CONFLICTING_EVIDENCE,
    )


# ============================================================================
# 2. INTERMEDIATE VS FINAL RESULT
# ============================================================================


def test_intermediate_and_final_are_detected_as_conflict():
    evidence = [
        ev(
            "python_interpreter",
            "Intermediate result: 42. Final result: 84.",
        )
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


def test_final_value_with_intermediate_value_is_delegated():
    evidence = [
        ev(
            "python_interpreter",
            "Intermediate result: 42. Final result: 84.",
        )
    ]

    result = deterministic_verification(
        "84",
        evidence,
    )

    # Deterministic verifier deliberately refuses to guess which labelled
    # value is semantically final when multiple result values exist.
    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


def test_intermediate_value_without_final_is_not_verified():
    evidence = [
        ev(
            "python_interpreter",
            "Intermediate result: 42.",
        )
    ]
    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


# ============================================================================
# 3. MULTI-STEP CALCULATION
# ============================================================================


def test_multistep_without_explicit_final_is_insufficient():
    evidence = [
        ev(
            "python_interpreter",
            (
                "Step 1 result: 20. "
                "Step 2 result: 22."
            ),
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_multistep_with_final_result():
    evidence = [
        ev(
            "python_interpreter",
            (
                "Step 1 result: 20. "
                "Step 2 result: 22. "
                "Final result: 42."
            ),
        )
    ]

    # Multiple labelled values exist, so deterministic layer must delegate.
    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


# ============================================================================
# 4. WRONG ANSWER DESPITE MATCHING NUMBER
# ============================================================================


def test_matching_number_does_not_prove_answer():
    evidence = [
        ev(
            "python_interpreter",
            "Input number = 42. Final result = 84.",
        )
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


def test_number_only_in_context_is_insufficient():
    evidence = [
        ev(
            "python_interpreter",
            (
                "The document contains "
                "10, 20, 30, and 42."
            ),
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


# ============================================================================
# 5. FAILED STEP EVIDENCE
# ============================================================================


def test_failed_step_cannot_verify_answer():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            succeeded=False,
            step_status="failed",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_successful_step_can_verify_answer():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            succeeded=True,
            step_status="completed",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


# ============================================================================
# 6. LLM MUST NEVER BE EVIDENCE
# ============================================================================


def test_llm_output_is_excluded():
    evidence = [
        ev(
            "llm",
            "Final result: 42.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_llm_cannot_verify_itself():
    evidence = [
        ev(
            "llm",
            "The answer is definitely 99.",
        ),
        ev(
            "python_interpreter",
            "Input value = 10.",
        ),
    ]

    assert_status(
        "99",
        evidence,
        VerificationStatus.INVALID,
    )


# ============================================================================
# 7. WEAK VS STRONG EVIDENCE
# ============================================================================


def test_web_search_alone_is_not_strong_deterministic_evidence():
    evidence = [
        ev(
            "web_search",
            "A webpage mentions the number 42.",
        )
    ]
    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_file_reader_can_be_strong_evidence():
    evidence = [
        ev(
            "file_reader",
            "Final result: 42.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_python_interpreter_can_be_strong_evidence():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


# ============================================================================
# 8. UNRELATED WEB EVIDENCE
# ============================================================================


def test_irrelevant_web_evidence_is_removed():
    evidence = [
        ev(
            "web_search",
            "Completely unrelated article. Final result: 42.",
            relevant=False,
        ),
        ev(
            "python_interpreter",
            "No final result available.",
        ),
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_web_result_with_matching_number_is_not_enough():
    evidence = [
        ev(
            "web_search",
            (
                "An article about another entity "
                "contains Score: 42."
            ),
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


# ============================================================================
# 9. MULTIPLE ARTIFACTS
# ============================================================================


def test_same_number_in_multiple_artifacts_is_not_proof():
    evidence = [
        ev(
            "file_reader",
            "Artifact A: input = 42.",
            artifact_id="artifact-a",
        ),
        ev(
            "file_reader",
            "Artifact B: score of another item = 42.",
            artifact_id="artifact-b",
        ),
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


def test_final_answer_artifact_is_extractable():
    evidence = [
        ev(
            "file_reader",
            '{"answer": 42}',
            artifact_id="answer-artifact",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_final_answer_artifact_can_contain_other_numbers():
    evidence = [
        ev(
            "file_reader",
            (
                '{"question_id": 123, '
                '"metadata": 999, '
                '"answer": 42}'
            ),
            artifact_id="answer-artifact",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


# ============================================================================
# 10. REPLANNED EXECUTION
# ============================================================================


def test_old_execution_is_removed_after_replan():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            execution_id="execution-old",
        ),
        ev(
            "python_interpreter",
            "Final result: 84.",
            execution_id="execution-new",
        ),
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INVALID,
    )

    assert_status(
        "84",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_old_attempt_is_removed_after_new_attempt():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            attempt_id="attempt-1",
        ),
        ev(
            "python_interpreter",
            "Final result: 84.",
            attempt_id="attempt-2",
        ),
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INVALID,
    )
    assert_status(
        "84",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_old_plan_version_is_removed_after_replan():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            plan_version=1,
        ),
        ev(
            "python_interpreter",
            "Final result: 84.",
            plan_version=2,
        ),
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INVALID,
    )

    assert_status(
        "84",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_same_execution_can_have_multiple_steps():
    evidence = [
        ev(
            "python_interpreter",
            "Step 1 result: 20.",
            execution_id="execution-1",
            step_id="step-1",
        ),
        ev(
            "python_interpreter",
            "Final result: 42.",
            execution_id="execution-1",
            step_id="step-2",
        ),
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status in {
        VerificationStatus.VERIFIED,
        VerificationStatus.CONFLICTING_EVIDENCE,
    }


# ============================================================================
# 11. CONFLICTING TOOLS
# ============================================================================


def test_two_tools_with_different_final_values_create_conflict():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
        ),
        ev(
            "file_reader",
            "Final result: 43.",
        ),
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


def test_conflicting_tools_do_not_auto_verify_second_value():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
        ),
        ev(
            "file_reader",
            "Final result: 43.",
        ),
    ]

    result = deterministic_verification(
        "43",
        evidence,
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


# ============================================================================
# 12. CORRECT ANSWER BUT INSUFFICIENT EVIDENCE
# ============================================================================


def test_correct_by_coincidence_is_not_verified():
    evidence = [
        ev(
            "file_reader",
            "The source contains 10, 20, 42 and 100.",
        )
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE


# ============================================================================
# 13. SOURCE MODALITY
# ============================================================================


def test_image_task_requires_image_capable_source():
    agent = make_agent()

    data = VerificationInput(
        question="What number is visible in the image?",
        candidate_answer="42",
        raw_data=[
            ev(
                "python_interpreter",
                "Final result: 42.",
            )
        ],
        task_type="IMAGE",
    )

    assert (
        agent._check_source_support(
            data,
            data.raw_data,
        )
        is False
    )


def test_image_task_accepts_image_evidence():
    agent = make_agent()

    data = VerificationInput(
        question="What number is visible in the image?",
        candidate_answer="42",
        raw_data=[
            ev(
                "analyze_image",
                "Final result: 42.",
            )
        ],
        task_type="IMAGE",
    )

    assert (
        agent._check_source_support(
            data,
            data.raw_data,
        )
        is True
    )


# ============================================================================
# 14. EMPTY / MISSING EVIDENCE
# ============================================================================


def test_empty_candidate_is_invalid():
    result = deterministic_verification(
        "",
        [
            ev(
                "python_interpreter",
                "Final result: 42.",
            )
        ],
    )

    assert result.status == VerificationStatus.INVALID


def test_no_evidence_is_insufficient():
    result = deterministic_verification(
        "42",
        [],
    )

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE


# ============================================================================
# 15. NUMERIC NORMALIZATION
# ============================================================================


def test_integer_and_float_are_equal():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.0.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_scientific_notation_is_equal():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 4.2e1.",
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.VERIFIED,
    )


# ============================================================================
# 16. TEXT ANSWERS
# ============================================================================


def test_exact_text_match_is_verified():
    evidence = [
        ev(
            "file_reader",
            "Paris",
        )
    ]

    assert_status(
        "Paris",
        evidence,
        VerificationStatus.VERIFIED,
    )


def test_text_appearing_inside_longer_text_is_insufficient():
    evidence = [
        ev(
            "file_reader",
            "Paris is one city among many.",
        )
    ]

    assert_status(
        "Paris",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


# ============================================================================
# 17. TRI-STATE API
# ============================================================================


def test_evidence_supports_candidate_true():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
        )
    ]

    assert (
        evidence_supports_candidate(
            "42",
            evidence,
        )
        is True
    )


def test_evidence_supports_candidate_false():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 84.",
        )
    ]

    assert (
        evidence_supports_candidate(
            "42",
            evidence,
        )
        is False
    )


def test_evidence_supports_candidate_none():
    evidence = [
        ev(
            "python_interpreter",
            "The values 10, 20, and 42 appear.",
        )
    ]

    assert (
        evidence_supports_candidate(
            "42",
            evidence,
        )
        is None
    )


# ============================================================================
# 18. LLM RESULT VALIDATION
# ============================================================================


def test_llm_verified_without_reason_is_rejected():
    agent = make_agent()

    result = agent._validate_llm_result(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason="",
        ),
        VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason="ambiguous",
        ),
        [
            ev(
                "python_interpreter",
                "The value 42 appears.",
            )
        ],
    )

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_llm_cannot_resolve_conflict_without_explanation():
    agent = make_agent()

    result = agent._validate_llm_result(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason="42 is correct.",
        ),
        VerificationResult(
            status=VerificationStatus.CONFLICTING_EVIDENCE,
            reason="42 and 43 are both final.",
        ),
        [
            ev(
                "python_interpreter",
                "Final result: 42.",
            ),
            ev(
                "file_reader",
                "Final result: 43.",
            ),
        ],
    )

    assert result.status == VerificationStatus.CONFLICTING_EVIDENCE


def test_llm_can_resolve_conflict_with_intermediate_explanation():
    agent = make_agent()

    result = agent._validate_llm_result(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            reason=(
                "42 is the intermediate value and 84 is the "
                "final result."
            ),
        ),
        VerificationResult(
            status=VerificationStatus.CONFLICTING_EVIDENCE,
            reason="Multiple labelled values detected.",
        ),
        [
            ev(
                "python_interpreter",
                "Intermediate result: 42. Final result: 84.",
            )
        ],
    )

    assert result.status == VerificationStatus.VERIFIED


def test_llm_insufficient_evidence_forces_verified_false():
    agent = make_agent()

    result = agent._validate_llm_result(
        VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason="Not enough evidence.",
            verified=True,
        ),
        VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            reason="ambiguous",
        ),
        [
            ev(
                "python_interpreter",
                "42 appears.",
            )
        ],
    )

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.verified is False


# ============================================================================
# 19. BACKWARD COMPATIBILITY
# ============================================================================


def test_verification_status_aliases_are_preserved():
    assert VerificationStatus.PASS == VerificationStatus.VERIFIED
    assert VerificationStatus.FAIL == VerificationStatus.INVALID
    assert (
        VerificationStatus.UNCERTAIN
        == VerificationStatus.INSUFFICIENT_EVIDENCE
    )


# ============================================================================
# 20. DETERMINISTIC SHORT-CIRCUIT
# ============================================================================


@pytest.mark.asyncio
async def test_deterministic_verified_short_circuits_llm():
    class ExplodingClient:
        async def generate(self, **kwargs):
            raise AssertionError(
                "LLM should NOT be called for deterministic VERIFIED."
            )

    agent = VerifierAgent(
        client=ExplodingClient(),
        model="test-model",
    )

    result = await agent.verify(
        VerificationInput(
            question="What is the result?",
            candidate_answer="42",
            raw_data=[
                ev(
                    "python_interpreter",
                    "Final result: 42.",
                )
            ],
            task_type="GENERAL",
        )
    )

    assert result.status == VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_deterministic_invalid_short_circuits_llm():
    class ExplodingClient:
        async def generate(self, **kwargs):
            raise AssertionError(
                "LLM should NOT be called for deterministic INVALID."
            )

    agent = VerifierAgent(
        client=ExplodingClient(),
        model="test-model",
    )

    result = await agent.verify(
        VerificationInput(
            question="What is the result?",
            candidate_answer="42",
            raw_data=[
                ev(
                    "python_interpreter",
                    "Final result: 84.",
                )
            ],
            task_type="GENERAL",
        )
    )

    assert result.status == VerificationStatus.INVALID


# ============================================================================
# 21. SEMANTIC VERIFIER MUST RECEIVE ONLY CURRENT EVIDENCE
# ============================================================================


@pytest.mark.asyncio
async def test_semantic_verifier_does_not_receive_failed_old_evidence():
    captured = {}

    class FakeClient:
        async def generate(self, **kwargs):
            captured["messages"] = kwargs["messages"]

            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                reason="Not enough evidence.",
            )

    agent = VerifierAgent(
        client=FakeClient(),
        model="test-model",
    )

    result = await agent.verify(
        VerificationInput(
            question="What is the final value?",
            candidate_answer="42",
            raw_data=[
                ev(
                    "python_interpreter",
                    "Final result: 42.",
                    succeeded=False,
                    execution_id="old-execution",
                ),
                ev(
                    "python_interpreter",
                    "Intermediate value: 42.",
                    execution_id="new-execution",
                ),
            ],
            task_type="GENERAL",
        )
    )

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE

    prompt = captured["messages"][1]["content"]

    assert "old-execution" not in prompt
    assert "succeeded=False" not in prompt


# ============================================================================
# 22. EXPLICIT IRRELEVANCE
# ============================================================================


def test_explicitly_irrelevant_strong_evidence_is_removed():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            relevant=False,
        )
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    )


# ============================================================================
# 23. RUN ID / EXECUTION ID
# ============================================================================


def test_new_run_replaces_old_run():
    evidence = [
        ev(
            "python_interpreter",
            "Final result: 42.",
            run_id="run-1",
            execution_id="exec-1",
        ),
        ev(
            "python_interpreter",
            "Final result: 84.",
            run_id="run-2",
            execution_id="exec-2",
        ),
    ]

    assert_status(
        "42",
        evidence,
        VerificationStatus.INVALID,
    )

    assert_status(
        "84",
        evidence,
        VerificationStatus.VERIFIED,
    )


# ============================================================================
# 24. ARTIFACT ANSWER EXTRACTION
# ============================================================================


@pytest.mark.parametrize(
    "artifact",
    [
        '{"answer": 42}',
        '{"final_answer": 42}',
        '{"result": 42}',
        '{"output": 42}',
    ],
)
def test_structured_artifact_answer_is_verified(artifact):
    evidence = [
        ev(
            "file_reader",
            artifact,
            artifact_id="artifact-1",
        )
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.VERIFIED


def test_artifact_with_wrong_answer_is_invalid():
    evidence = [
        ev(
            "file_reader",
            '{"answer": 43, "metadata": 42}',
            artifact_id="artifact-1",
        )
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status == VerificationStatus.INVALID


# ============================================================================
# 25. INCIDENTAL NUMBERS MUST NEVER ACCIDENTALLY VERIFY
# ============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "42",
        "The document contains 42.",
        "Input = 42.",
        "Previous answer = 42.",
        "Example answer = 42.",
        "Score of another item = 42.",
        "ID = 42.",
        "Year = 42.",
    ],
)
def test_incidental_number_never_becomes_verified(text):
    evidence = [
        ev(
            "python_interpreter",
            text,
        )
    ]

    result = deterministic_verification(
        "42",
        evidence,
    )

    assert result.status != VerificationStatus.VERIFIED


# ============================================================================
# 26. POSITIVE FINAL ANSWER REGRESSION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "candidate, evidence_text",
    [
        ("42", "Final answer: 42."),
        ("42", "Final result = 42."),
        ("42", "Computed result: 42."),
        ("42", "Score is 42."),
        ("42", "Total: 42."),
        ("42", '{"answer": 42}'),
        ("42", '{"final_answer": 42}'),
        ("42", '{"result": 42}'),
    ],
)
def test_explicit_final_answer_forms_are_verified(
    candidate: str,
    evidence_text: str,
):
    evidence = [
        ev(
            "python_interpreter",
            evidence_text,
        )
    ]

    result = deterministic_verification(
        candidate,
        evidence,
    )

    assert result.status == VerificationStatus.VERIFIED


# ============================================================================
# 27. NEGATIVE FINAL ANSWER REGRESSION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "candidate, evidence_text",
    [
        ("42", "Final answer: 43."),
        ("42", "Final result = 43."),
        ("42", "Computed result: 43."),
        ("42", "Score is 43."),
        ("42", '{"answer": 43}'),
        ("42", '{"final_answer": 43}'),
        ("42", '{"result": 43}'),
    ],
)
def test_explicitly_wrong_final_answer_is_invalid(
    candidate: str,
    evidence_text: str,
):
    evidence = [
        ev(
            "python_interpreter",
            evidence_text,
        )
    ]

    result = deterministic_verification(
        candidate,
        evidence,
    )

    assert result.status == VerificationStatus.INVALID
    