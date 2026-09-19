"""Regression tests: structured-output boundary repair (GAIA Q1-Q7 triage).

Live Ollama 0.34.1 + qwen2.5:3b evidence showed that format=json_schema
constrained outputs still fail PlanSchema's cross-field invariant
(step_ids sequential from 0) and that VerificationResult status can be
legally omitted/null while the sibling `verified` bool carries the decision.
These tests lock the lossless repair behavior and prove the strict public
contracts are unchanged.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gaia_agent.agents.verifier import (
    VerificationResult,
    VerificationStatus,
)
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.planner.plan_schema import PlanSchema


def _plan_step(**overrides) -> dict:
    step = {
        "step_id": 0,
        "action": "search the web",
        "step_type": "tool",
        "tool_name": "web_search",
        "arguments": {"query": "albums"},
        "is_final_answer": False,
    }
    step.update(overrides)
    return step


# ---------------------------------------------------------------------------
# PlanSchema: lossless normalization
# ---------------------------------------------------------------------------


def test_plan_schema_accepts_bare_top_level_list() -> None:
    plan = PlanSchema.model_validate(
        [
            _plan_step(),
            {
                "step_id": 1,
                "action": "answer",
                "step_type": "llm",
                "is_final_answer": True,
            },
        ]
    )
    assert [s.step_id for s in plan.steps] == [0, 1]
    assert plan.steps[-1].is_final_answer is True


def test_plan_schema_rekeys_one_based_step_ids() -> None:
    raw = {
        "steps": [
            _plan_step(step_id=1),
            {
                "step_id": 2,
                "action": "answer",
                "step_type": "llm",
                "is_final_answer": True,
            },
        ]
    }
    plan = PlanSchema.model_validate(raw)
    assert [s.step_id for s in plan.steps] == [0, 1]
    # order and content preserved
    assert plan.steps[0].tool_name == "web_search"
    assert plan.steps[1].is_final_answer is True


def test_plan_schema_leaves_zero_based_ids_unchanged() -> None:
    raw = {
        "steps": [
            _plan_step(step_id=0),
            {
                "step_id": 1,
                "action": "answer",
                "step_type": "llm",
                "is_final_answer": True,
            },
        ]
    }
    plan = PlanSchema.model_validate(raw)
    assert [s.step_id for s in plan.steps] == [0, 1]


def test_plan_schema_still_rejects_duplicate_step_ids() -> None:
    raw = {
        "steps": [
            _plan_step(step_id=0),
            {
                "step_id": 0,
                "action": "answer",
                "step_type": "llm",
                "is_final_answer": True,
            },
        ]
    }
    with pytest.raises(ValidationError):
        PlanSchema.model_validate(raw)


def test_plan_schema_still_requires_final_llm_step_last() -> None:
    # tool-only plan (the live probe output) must still be rejected
    with pytest.raises(ValidationError):
        PlanSchema.model_validate({"steps": [_plan_step()]})

    # final step not last must still be rejected
    raw = {
        "steps": [
            {
                "step_id": 0,
                "action": "answer",
                "step_type": "llm",
                "is_final_answer": True,
            },
            _plan_step(step_id=1),
        ]
    }
    with pytest.raises(ValidationError):
        PlanSchema.model_validate(raw)

    # final LLM step carrying arguments must still be rejected
    raw = {
        "steps": [
            _plan_step(step_id=0),
            {
                "step_id": 1,
                "action": "answer",
                "step_type": "llm",
                "is_final_answer": True,
                "arguments": {"x": 1},
            },
        ]
    }
    with pytest.raises(ValidationError):
        PlanSchema.model_validate(raw)


# ---------------------------------------------------------------------------
# VerificationStatus / VerificationResult: lossless label repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("VERIFIED", VerificationStatus.VERIFIED),
        ("Verified", VerificationStatus.VERIFIED),
        ("pass", VerificationStatus.VERIFIED),
        ("correct", VerificationStatus.VERIFIED),
        ("FAIL", VerificationStatus.INVALID),
        ("incorrect", VerificationStatus.INVALID),
        ("wrong", VerificationStatus.INVALID),
        ("uncertain", VerificationStatus.INSUFFICIENT_EVIDENCE),
        ("UNKNOWN", VerificationStatus.INSUFFICIENT_EVIDENCE),
        ("insufficient_evidence", VerificationStatus.INSUFFICIENT_EVIDENCE),
        ("conflicting", VerificationStatus.CONFLICTING_EVIDENCE),
    ],
)
def test_verification_status_synonyms_resolve(label, expected) -> None:
    assert (
        VerificationResult.model_validate({"status": label}).status == expected
    )


def test_verification_status_unknown_label_still_rejected() -> None:
    with pytest.raises(ValidationError):
        VerificationResult.model_validate({"status": "definitely_gibberish"})


def test_verification_result_salvages_verified_bool() -> None:
    result = VerificationResult.model_validate(
        {"verified": True, "reason": "evidence supports it"}
    )
    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True


def test_verification_result_salvages_false_bool_as_invalid() -> None:
    result = VerificationResult.model_validate({"verified": False})
    assert result.status == VerificationStatus.INVALID


def test_verification_result_with_null_status_and_bool_is_salvaged() -> None:
    result = VerificationResult.model_validate(
        {"status": None, "verified": True, "reason": "r"}
    )
    assert result.status == VerificationStatus.VERIFIED


def test_verification_result_with_neither_field_stays_incomplete() -> None:
    # never silently accepted: status stays None and downstream validation
    # (_validate_llm_result) returns INSUFFICIENT_EVIDENCE for it.
    result = VerificationResult.model_validate({"reason": "thinking out loud"})
    assert result.status is None
    assert result.verified is None


def test_verified_flag_always_derived_from_status() -> None:
    result = VerificationResult.model_validate(
        {"status": "verified", "verified": False, "reason": "conflicting bool"}
    )
    assert result.verified is True


# ---------------------------------------------------------------------------
# OllamaClient payload parsing: fence tolerance
# ---------------------------------------------------------------------------


def _client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", timeout=10.0)


def test_parse_plain_json_unchanged() -> None:
    payload = json.dumps({"verified": True, "reason": "ok"})
    parsed = _client()._parse_structured_output(payload, VerificationResult)
    assert parsed.status == VerificationStatus.VERIFIED


def test_parse_fenced_json_is_tolerated() -> None:
    payload = (
        "```json\n"
        + json.dumps({"status": "invalid", "reason": "no"})
        + "\n```"
    )
    parsed = _client()._parse_structured_output(payload, VerificationResult)
    assert parsed.status == VerificationStatus.INVALID


def test_parse_garbage_still_raises_invalid_json_error() -> None:
    with pytest.raises(Exception, match="invalid JSON"):
        _client()._parse_structured_output(
            "not json at all", VerificationResult
        )
