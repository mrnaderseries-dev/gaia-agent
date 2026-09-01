from types import SimpleNamespace

from gaia_agent.agents.verifier import (
    VerificationStatus,
    deterministic_verification,
    evidence_supports_candidate,
)


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
    assert "strong tool evidence" in reason.lower()

    support = evidence_supports_candidate(
        candidate_answer="21",
        raw_data=raw_data,
    )

    assert support is None
