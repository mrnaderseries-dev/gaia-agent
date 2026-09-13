import pathlib

patterns = ["Recovery(", ".init(", "error_handler=error_handler", "max_recoveries", "max_total_executions", "allow_replan", "allow_replanning"]

for base in (pathlib.Path("src"), pathlib.Path("tests")):
    for p in sorted(base.rglob("*.py")):
        if ".venv" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        hits = [pat for pat in patterns if pat in text]
        if hits:
            print(p, "->", hits)