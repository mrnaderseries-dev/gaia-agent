"""Remove a question record from _gaia20_results.json so the runner re-runs it fresh.

Usage: python _s8c_clear_q.py 3 [7 ...]   (1-based indices)
"""
import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "_gaia20_results.json"
indices = {int(a) for a in sys.argv[1:]}
records = json.loads(PATH.read_text(encoding="utf-8"))
kept = [r for r in records if r.get("index") not in indices]
removed = [r.get("index") for r in records if r.get("index") in indices]
PATH.write_text(json.dumps(kept, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("removed:", removed, "remaining:", len(kept))
