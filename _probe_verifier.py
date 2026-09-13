"""Probe: adapt test_verifier_evidence_relevance intent to current contract."""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, "src")

from gaia_agent.agents.verifier import (
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
    deterministic_verification,
    evidence_supports_candidate,
)


def evidence(tool_name, result, succeeded=True):
    return SimpleNamespace(tool_name=tool_name, result=result, succeeded=succeeded)


def status_of(candidate, raw):
    return deterministic_verification(candidate_answer=candidate, raw_data=raw).status


def check():
    results = []

    # 1 numeric match passes
    s = status_of("42", [evidence("python_interpreter", "The calculated result is 42.")])
    results.append(("numeric_match_verified", s == VerificationStatus.VERIFIED, s))

    # 2 numeric conflict fails
    s = status_of("43", [evidence("python_interpreter", "The calculated result is 42.")])
    results.append(("numeric_conflict_invalid", s == VerificationStatus.INVALID, s))

    # 3 multiple values not explicit final -> insufficient
    s = status_of("20", [evidence("python_interpreter", "Input values: 10 and 20. The calculation uses both.")])
    results.append(("multiple_values_insufficient", s == VerificationStatus.INSUFFICIENT_EVIDENCE, s))

    # 4 exact text match passes
    s = status_of("reverse", [evidence("file_reader", "The final answer is reverse.")])
    results.append(("exact_text_verified", s == VerificationStatus.VERIFIED, s))

    # 5 no strong evidence -> insufficient
    s = status_of("21", [evidence("web_search", "The search result mentions 21.")])
    results.append(("no_strong_insufficient", s == VerificationStatus.INSUFFICIENT_EVIDENCE, s))

    # 6 failed tool ignored
    s = status_of("42", [evidence("python_interpreter", "42", succeeded=False)])
    results.append(("failed_ignored_insufficient", s == VerificationStatus.INSUFFICIENT_EVIDENCE, s))

    # 7 empty candidate invalid
    s = status_of("", [evidence("python_interpreter", "42")])
    results.append(("empty_invalid", s == VerificationStatus.INVALID, s))

    # 8 evidence_supports numeric candidate
    r = evidence_supports_candidate("42", [evidence("python_interpreter", "42")])
    results.append(("supports_numeric", r is True, r))

    # 9 evidence_rejects conflict
    r = evidence_supports_candidate("43", [evidence("python_interpreter", "42")])
    results.append(("rejects_conflict", r is False, r))

    # 10 llm used for web evidence
    client = AsyncMock()
    client.generate.return_value = VerificationResult(verified=True, reason="The retrieved source directly answers the question.")
    verifier = VerifierAgent(client=client, model="test-model")
    async def run11():
        return await verifier.verify(VerificationInput(
            question="What is the answer?", candidate_answer="reverse",
            raw_data=[evidence("web_search", "The source directly states that the answer is reverse.")],
            task_type="TEXT"))
    res = asyncio.get_event_loop().run_until_complete(run11())
    results.append(("llm_web_verified", res.verified is True and client.generate.await_count == 1, (res.status, client.generate.await_count)))

    # 11 irrelevant web evidence rejected via llm
    client2 = AsyncMock()
    client2.generate.return_value = VerificationResult(verified=False, reason="The search result is unrelated to the question.")
    v2 = VerifierAgent(client=client2, model="test-model")
    async def run12():
        return await v2.verify(VerificationInput(
            question="Q", candidate_answer="21",
            raw_data=[evidence("web_search", "Unrelated article. The number 21 appears but it discusses a different subject.")],
            task_type="TEXT"))
    res = asyncio.get_event_loop().run_until_complete(run12())
    results.append(("llm_rejects_irrelevant", res.verified is False and client2.generate.await_count == 1, (res.status, client2.generate.await_count)))

    # 12 llm true requires successful evidence
    client3 = AsyncMock()
    client3.generate.return_value = VerificationResult(verified=True, reason="Looks correct.")
    v3 = VerifierAgent(client=client3, model="test-model")
    async def run13():
        return await v3.verify(VerificationInput(
            question="Q", candidate_answer="42",
            raw_data=[evidence("web_search", "Search failed.", succeeded=False)],
            task_type="TEXT"))
    res = asyncio.get_event_loop().run_until_complete(run13())
    results.append(("llm_true_requires_successful_evidence", res.verified is False and client3.generate.await_count == 0, (res.status, client3.generate.await_count)))

    ok = True
    for name, passed, detail in results:
        ok = ok and passed
        print(("PASS" if passed else "FAIL"), name, detail)
    print("ALL_OK" if ok else "SOME_FAILED")


check()