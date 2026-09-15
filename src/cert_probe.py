from __future__ import annotations

import asyncio
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gaia_agent.main import create_agent
from gaia_agent.core.agent_state import AgentState


async def run_one(agent, label: str, request: str, timeout: float = 600.0):
    state = AgentState(user_request=request)
    t0 = time.time()
    run = None
    try:
        async def _drive():
            loop = agent.orchestrator
            run_holder: dict = {}
            run_holder["run"] = await loop.start(state)
            while True:
                if state.is_terminal:
                    break
                outcome = await loop.step(state, run_holder["run"])
                if outcome is not None and getattr(outcome, "terminal", False):
                    break
            return run_holder["run"]
        agent._state = state
        try:
            run = await asyncio.wait_for(_drive(), timeout=timeout)
        finally:
            agent._state = None
            try:
                agent._run = None
            except Exception:
                pass
        result = state
        dt = time.time() - t0
        plan = None
        ev = []
        vhist = []
        if run is not None:
            plan = (
                [(s.step_id, s.tool_name, s.arguments) for s in run.plan_runtime.plan.steps]
                if run.plan_runtime and run.plan_runtime.plan
                else None
            )
            ev = [
                (e.tool_name, e.arguments, e.result, e.succeeded)
                for e in run.verification_evidence()
            ]
            vhist = [(v.answer, str(v.result.status), v.result.reason) for v in run.verification_history]
        print(f"[{label}] ELAPSED={dt:.1f}s PHASE={result.phase} "
              f"ANSWER={result.final_answer!r} VERIFIED={result.final_answer_verified} "
              f"REPLANS={result.replan_count} VATTEMPTS={result.verification_attempts} "
              f"TOOLERR={result.tool_error!r} FATAL={result.fatal_error}")
        print(f"[{label}] PLAN={plan}")
        print(f"[{label}] EV={ev}")
        print(f"[{label}] VHIST={vhist}")
    except asyncio.TimeoutError:
        print(f"[{label}] TIMEOUT after {timeout}s")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] EXC {type(exc).__name__}: {exc}")


async def main() -> None:
    which = sys.argv[1:] or ["ALL"]
    if which == ["ALL"]:
        which = ["T1", "T2"]
    tasks = {
        "T1": "Calculate 2 + 2.",
        "T2": "Calculate 25 * 17 + 43.",
        "T3": 'Reverse the string "architecture".',
        "T3B": 'Reverse the string "hello".',
        "T3C": 'Convert the string "gaia" to uppercase.',
        "REC": "Calculate 7 * 6.",
        "WEB": "What is the current population of Paris?",
        "FILE": "Read the file NOTES_SAMPLE.txt and report its second line.",
        "MH": "Calculate 12 * 8, then add 100 to that result.",
    }
    agent = None
    for label in which:
        if label in tasks:
            agent = await create_agent()
            await run_one(agent, label, tasks[label])
            agent = None


if __name__ == "__main__":
    asyncio.run(main())
