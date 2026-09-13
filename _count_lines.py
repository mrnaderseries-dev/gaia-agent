import pathlib, sys

base = pathlib.Path(sys.argv[1])
skip = {".venv", "site-packages", "node_modules", "evaluation_runs"}
for p in sorted(base.rglob("*.py")):
    if any(part in skip for part in p.parts):
        continue
    if p.name == "tempCodeRunnerFile.py":
        continue
    n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
    print(f"{n:5d}  {p}")