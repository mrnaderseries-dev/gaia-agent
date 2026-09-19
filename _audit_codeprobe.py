from __future__ import annotations

"""Offline probe: which task texts make the deterministic planner emit code
that RAISES at runtime (genuine execution failure for the recovery probe)?"""

import faulthandler
import sys

faulthandler.dump_traceback_later(25, exit=True)

sys.path.insert(0, r"C:\Users\user\gaia-agent\src")

print("importing planner...", flush=True)
from gaia_agent.planner.planner import Planner

print("imported", flush=True)

planner = Planner(
    client=object(),
    model="probe",
    available_tools={"python_interpreter": object()},
)

CANDIDATES = [
    "Calculate 25 * 17 + 43.",
    "Calculate 8 / 0.",
    "Calculate 10 / (5 - 5).",
    "Calculate 7 // (3 - 3).",
    "Calculate 10 % (4 - 4).",
    "Calculate 10 % 0.",
    "Calculate 5 // 0.",
    "Calculate 1 / 0.",
    "Calculate int('abc').",
]

for text in CANDIDATES:
    code = planner._deterministic_fallback_code(text)
    outcome = ""
    if code:
        try:
            exec(code, {"__builtins__": {}}, {})
            outcome = "OK"
        except BaseException as exc:  # noqa: BLE001
            outcome = f"RAISES {type(exc).__name__}: {exc}"
    print(f"{text!r:45} -> code={code!r:40} {outcome}", flush=True)