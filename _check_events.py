import re
import pathlib
import subprocess

# Working tree comparison
orch = pathlib.Path("src/gaia_agent/core/orchestration/orchestrator.py").read_text(encoding="utf-8")
events = pathlib.Path("src/gaia_agent/observability/events.py").read_text(encoding="utf-8")
used = set(re.findall(r"EventType\.([A-Z_]+)", orch))
defined = set(re.findall(r"^\s+([A-Z_]+)\s*=\s*\"", events, re.M))
print("WORKING TREE USED BUT NOT DEFINED:", sorted(used - defined))

# HEAD comparison
out = subprocess.run(
    ["git", "show", "HEAD:src/gaia_agent/observability/events.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
head_defined = set(re.findall(r"^\s+([A-Z_]+)\s*=\s*\"", out.stdout, re.M))
head_orch = subprocess.run(
    ["git", "show", "HEAD:src/gaia_agent/core/orchestration/orchestrator.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
).stdout
head_used = set(re.findall(r"EventType\.([A-Z_]+)", head_orch))
print("HEAD USED BUT NOT DEFINED in HEAD events:", sorted(head_used - head_defined))
print("HEAD events defined:", sorted(head_defined))
out2 = subprocess.run(
    ["git", "show", "HEAD:src/gaia_agent/observability/events.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
print("HEAD has PLANNING_STARTED:", "PLANNING_STARTED" in out2.stdout)
print("HEAD has PLAN_GENERATED:", "PLAN_GENERATED" in out2.stdout)
print("HEAD has PLAN_REJECTED:", "PLAN_REJECTED" in out2.stdout)
print("HEAD has LOOP_DETECTED:", "LOOP_DETECTED" in out2.stdout)