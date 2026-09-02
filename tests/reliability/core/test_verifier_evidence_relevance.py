from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gaia_agent.agents.verifier import (
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
    deterministic_verification,
    evidence_supports_candidate,
)


def evidence(
    tool_name: str,
    result: str,
    succeeded: bool = True,
):
    """Create a minimal evidence record for verifier tests."""
    return SimpleNamespace(
        tool_name=tool_name,
        result=result,
        succeeded=succeeded,
    )


# ============================================================================
# deterministic_verification()
# ============================================================================


def test_deterministic_numeric_match_passes():
    raw_data = [
        evidence(
            "python_interpreter",
            "The calculated result is 42.",
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="42",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.PASS
    assert "matches" in reason.lower()


def test_deterministic_numeric_conflict_fails():
    raw_data = [
        evidence(
            "python_interpreter",
            "The calculated result is 42.",
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="43",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.FAIL
    assert "contradict" in reason.lower()


def test_multiple_numeric_values_are_uncertain():
    raw_data = [
        evidence(
            "python_interpreter",
            "Input values: 10 and 20. "
            "The calculation uses both values.",
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="20",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.UNCERTAIN
    assert "multiple" in reason.lower()


def test_exact_text_match_passes():
    raw_data = [
        evidence(
            "file_reader",
            "The final answer is reverse.",
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="reverse",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.PASS
    assert "directly" in reason.lower()


def test_no_strong_evidence_is_uncertain():
    raw_data = [
        evidence(
            "web_search",
            "The search result mentions 21.",
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="21",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.UNCERTAIN
    assert "no strong" in reason.lower()


def test_web_search_number_alone_is_not_strong_deterministic_evidence():
    raw_data = [
        SimpleNamespace(
            tool_name="web_search",
            succeeded=True,
            result=(
                "Unrelated search result. "
                "This page mentions the number 21, "
                "but it does not answer the question."
            ),
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="21",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.UNCERTAIN
    assert "strong deterministic tool evidence" in reason.lower()


def test_failed_tool_evidence_is_ignored():
    raw_data = [
        evidence(
            "python_interpreter",
            "42",
            succeeded=False,
        )
    ]

    status, reason = deterministic_verification(
        candidate_answer="42",
        raw_data=raw_data,
    )

    assert status == VerificationStatus.UNCERTAIN


def test_empty_candidate_fails():
    status, reason = deterministic_verification(
        candidate_answer="",
        raw_data=[
            evidence(
                "python_interpreter",
                "42",
            )
        ],
    )

    assert status == VerificationStatus.FAIL
    assert "empty" in reason.lower()


def test_missing_candidate_fails():
    status, reason = deterministic_verification(
        candidate_answer=None,
        raw_data=[
            evidence(
                "python_interpreter",
                "42",
            )
        ],
    )

    assert status == VerificationStatus.FAIL
    assert "missing" in reason.lower()


# ============================================================================
# evidence_supports_candidate()
# ============================================================================


def test_evidence_supports_numeric_candidate():
    raw_data = [
        evidence(
            "python_interpreter",
            "42",
        )
    ]

    result = evidence_supports_candidate(
        candidate_answer="42",
        raw_data=raw_data,
    )

    assert result is True


def test_evidence_rejects_numeric_conflict():
    raw_data = [
        evidence(
            "python_interpreter",
            "42",
        )
    ]

    result = evidence_supports_candidate(
        candidate_answer="43",
        raw_data=raw_data,
    )

    assert result is False


def test_web_search_does_not_support_candidate_deterministically():
    raw_data = [
        evidence(
            "web_search",
            "An unrelated page contains 42.",
        )
    ]

    result = evidence_supports_candidate(
        candidate_answer="42",
        raw_data=raw_data,
    )

    assert result is None


def test_multiple_numbers_do_not_prove_candidate():
    raw_data = [
        evidence(
            "python_interpreter",
            "Input=10, output=20, threshold=30.",
        )
    ]

    result = evidence_supports_candidate(
        candidate_answer="20",
        raw_data=raw_data,
    )

    assert result is None


def test_no_evidence_returns_none():
    result = evidence_supports_candidate(
        candidate_answer="42",
        raw_data=[],
    )

    assert result is None


# ============================================================================
# VerifierAgent.verify()
# ============================================================================


@pytest.mark.asyncio
async def test_verifier_accepts_deterministically_verified_answer():
    client = AsyncMock()

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="What is 6 × 7?",
        candidate_answer="42",
        raw_data=[
            evidence(
                "python_interpreter",
                "42",
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is True
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_rejects_deterministic_conflict():
    client = AsyncMock()

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="What is the result?",
        candidate_answer="43",
        raw_data=[
            evidence(
                "python_interpreter",
                "42",
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is False
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_rejects_when_no_evidence_exists():
    client = AsyncMock()

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="What is the answer?",
        candidate_answer="42",
        raw_data=[],
    )

    result = await verifier.verify(data)

    assert result.verified is False
    assert "no successful evidence" in result.reason.lower()
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_uses_llm_for_web_evidence():
    client = AsyncMock()
    client.generate.return_value = VerificationResult(
        verified=True,
        reason="The retrieved source directly answers the question.",
    )

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="What is the answer?",
        candidate_answer="reverse",
        raw_data=[
            evidence(
                "web_search",
                "The source directly states that the answer is reverse.",
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is True
    client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_verifier_rejects_irrelevant_web_evidence():
    client = AsyncMock()
    client.generate.return_value = VerificationResult(
        verified=False,
        reason="The search result is unrelated to the question.",
    )

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question=(
            "What is the highest number of bird species "
            "on camera simultaneously in the specified video?"
        ),
        candidate_answer="21",
        raw_data=[
            evidence(
                "web_search",
                (
                    "Unrelated article. "
                    "The number 21 appears in this article, "
                    "but it discusses a different subject."
                ),
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is False
    client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_rejection_is_preserved():
    client = AsyncMock()
    client.generate.return_value = VerificationResult(
        verified=False,
        reason="The evidence does not support the candidate.",
    )

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="What is the answer?",
        candidate_answer="wrong",
        raw_data=[
            evidence(
                "web_search",
                "This source does not contain the answer.",
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is False
    assert "does not support" in result.reason.lower()


@pytest.mark.asyncio
async def test_llm_can_verify_relevant_web_evidence():
    client = AsyncMock()
    client.generate.return_value = VerificationResult(
        verified=True,
        reason="The source directly provides the requested information.",
    )

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="Which opening is given?",
        candidate_answer="e7-e5",
        raw_data=[
            evidence(
                "web_search",
                "The referenced chess game begins with 1. e4 e7-e5.",
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is True


@pytest.mark.asyncio
async def test_llm_true_requires_successful_evidence():
    client = AsyncMock()
    client.generate.return_value = VerificationResult(
        verified=True,
        reason="Looks correct.",
    )

    verifier = VerifierAgent(
        client=client,
        model="test-model",
    )

    data = VerificationInput(
        question="What is the answer?",
        candidate_answer="42",
        raw_data=[
            evidence(
                "web_search",
                "Search failed.",
                succeeded=False,
            )
        ],
    )

    result = await verifier.verify(data)

    assert result.verified is False


# ============================================================================
# Prompt contract
# ============================================================================


def test_verifier_prompt_requires_question_specific_support():
    prompt = VerifierAgent._system_prompt().lower()

    assert "exact question" in prompt
    assert "actually support" in prompt
    assert "unrelated web-search" in prompt
    assert "specific source" in prompt
    assert "verified=false" in prompt