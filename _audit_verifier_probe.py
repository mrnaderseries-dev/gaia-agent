from __future__ import annotations

"""Verifier-level controlled probe: conflicting / invalid / insufficient evidence.

Uses the REAL VerifierAgent from the real composition root. All cases are
decided by the deterministic branch (no LLM), so the probe is fast and stable.
"""

import asyncio
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, r"C:\Users\user\gaia-agent\src")

from gaia_agent.main import create_agent
from gaia_agent.agents.verifier import VerificationInput


def ev(code: str, result: str, succeeded: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        tool_name="python_interpreter",
        result=result,
        succeeded=succeeded,
        arguments={"code": code},
        evidence_type="tool_output",
        source="python_interpreter",
        step_id="0",
    )


CASES = {
    "conflicting_two_authoritative": VerificationInput(
        question="What is the value?",
        candidate_answer="468",
        raw_data=[
            ev("result = 25 * 17 + 43\n", "468"),
            ev("result = 17 * 23\n", "391"),
        ],
        task_type="arithmetic",
    ),
    "invalid_single_contradiction": VerificationInput(
        question="What is the value?",
        candidate_answer="500",
        raw_data=[
            ev("result = 25 * 17 + 43\n", "468"),
        ],
        task_type="arithmetic",
    ),
    "verified_authoritative_match": VerificationInput(
        question="What is the value?",
        candidate_answer="468",
        raw_data=[
            ev("result = 25 * 17 + 43\n", "468"),
        ],
        task_type="arithmetic",
    ),
    "insufficient_no_evidence": VerificationInput(
        question="What is the value?",
        candidate_answer="88",
        raw_data=[],
        task_type="arithmetic",
    ),
    "insufficient_failed_tool_only": VerificationInput(
        question="What is the value?",
        candidate_answer="468",
        raw_data=[
            ev("result = 25 * 17 + 43\n", "468", succeeded=False),
        ],
        task_type="arithmetic",
    ),
    "conflicting_candidate_chooses_first": VerificationInput(
        question="What is the value?",
        candidate_answer="391",
        raw_data=[
            ev("result = 25 * 17 + 43\n", "468"),
            ev("result = 17 * 23\n", "391"),
        ],
        task_type="arithmetic",
    ),
}


async def main() -> None:
    agent = await create_agent()
    verifier = agent.orchestrator.verifier
    results = []
    for name, payload in CASES.items():
        result = await verifier.verify(payload)
        results.append(
            {
                "case": name,
                "status": str(getattr(result.status, "value", result.status)),
                "verified": result.verified,
                "reason": result.reason,
            }
        )
    print("VERIFIER_PROBE_START")
    print(json.dumps(results, indent=2))
    print("VERIFIER_PROBE_END")


asyncio.run(main())