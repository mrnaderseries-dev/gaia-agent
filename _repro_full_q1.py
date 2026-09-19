"""Full instrumented reproduction of Q1 (Mercedes Sosa) through the real agent."""
from __future__ import annotations
import asyncio, json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from gaia_agent.main import create_agent
from gaia_agent.core.agent_state import AgentState

Q1 = ("How many studio albums were published by Mercedes Sosa between "
      "2000 and 2009 (included)? You can use the latest 2022 version of "
      "english wikipedia.")
GEN_TIMEOUT = 590.0

async def main():
    print("Creating agent ...", flush=True)
    agent = await create_agent()
    planner = agent.orchestrator.planner
    client = planner.client
    calls = []
    plans = []
    orig_generate = client.generate

    async def wrapped_generate(messages, **kwargs):
        t0 = time.time(); schema = kwargs.get("output_schema")
        sn = schema.__name__ if schema else None
        try:
            result = await orig_generate(messages, **kwargs)
        except Exception as exc:
            dt = round(time.time() - t0, 2)
            raw = getattr(exc, "raw_content", None)
            calls.append({"schema": sn, "seconds": dt, "error": f"{type(exc).__name__}: {exc}",
                          "raw_content_len": len(raw) if raw else None,
                          "raw_content": (raw[:2000] if raw else None)})
            print(f"[GEN] schema={sn} FAILED {dt}s: {type(exc).__name__}", flush=True)
            if raw: print("[GEN RAW]:", raw[:2000], flush=True)
            raise
        dt = round(time.time() - t0, 2)
        calls.append({"schema": sn, "seconds": dt, "error": None,
                      "response_chars": len(str(result)),
                      "response_snippet": str(result)[:400]})
        print(f"[GEN] schema={sn} OK {dt}s chars={len(str(result))}", flush=True)
        print("[GEN RESP]:", str(result)[:400], flush=True)
        return result
    client.generate = wrapped_generate

    def steps_of(result):
        try:
            return [(s.step_type.value, s.tool_name, s.is_final_answer) for s in result.plan.steps]
        except Exception:
            return "RAISED"
    def make_wrapper(name, orig):
        async def wrapper(*a, **k):
            t0 = time.time()
            try:
                r = await orig(*a, **k); dt = round(time.time() - t0, 2)
                plans.append({"phase": name, "seconds": dt, "steps": steps_of(r), "raised": False})
                print(f"[PLANNER] {name} OK {dt}s steps={steps_of(r)}", flush=True)
                return r
            except Exception as exc:
                dt = round(time.time() - t0, 2)
                plans.append({"phase": name, "seconds": dt, "steps": "RAISED", "raised": True,
                              "error": f"{type(exc).__name__}: {exc}"})
                print(f"[PLANNER] {name} RAISED {dt}s: {type(exc).__name__}: {exc}", flush=True)
                raise
        return wrapper
    planner.create_plan = make_wrapper("create_plan", planner.create_plan)
    planner.replan = make_wrapper("replan", planner.replan)

    state = AgentState(user_request=Q1)
    print("=== agent.run(Q1) ===", flush=True)
    t0 = time.time(); timed_out = False
    try:
        await asyncio.wait_for(agent.run(state), timeout=GEN_TIMEOUT)
    except asyncio.TimeoutError:
        timed_out = True
        print(f"[RUN] TIMED OUT after {GEN_TIMEOUT}s", flush=True)
    except Exception as exc:
        print(f"[RUN] RAISED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    elapsed = round(time.time() - t0, 2)
    print("\n=== SUMMARY ===", flush=True)
    print(f"elapsed={elapsed}s timed_out={timed_out}", flush=True)
    print(f"phase={state.phase} final_answer={state.final_answer!r} "
          f"verified={state.final_answer_verified} task_completed={state.task_completed}", flush=True)
    print(f"tool_error={state.tool_error!r} fatal={state.fatal_error} "
          f"termination={state.termination_reason!r}", flush=True)
    print(f"replan_count={state.replan_count} iteration={state.iteration} "
          f"verification_attempts={state.verification_attempts} retry_count={state.retry_count}", flush=True)
    print("execution_history:", flush=True)
    for e in state.execution_results:
        print(f"  step={e.step_id} tool={getattr(e,'tool_name',None)!r} "
              f"success={getattr(e,'success',None)} "
              f"output={str(getattr(e,'output',None))[:200]!r}", flush=True)
    print("\n=== PLANNER CALLS ===", flush=True)
    for p in plans: print(json.dumps(p, ensure_ascii=False, default=str), flush=True)
    print("\n=== LLM CALLS ===", flush=True)
    for c in calls: print(json.dumps(c, ensure_ascii=False, default=str), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
