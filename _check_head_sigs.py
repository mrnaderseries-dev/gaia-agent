import subprocess
import sys

for path in [
    "src/gaia_agent/main.py",
    "src/gaia_agent/reliability/loop_detector.py",
    "src/gaia_agent/core/orchestration/orchestrator.py",
    "src/gaia_agent/core/llm_executor.py",
    "src/gaia_agent/reliability/recovery.py",
]:
    out = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print("=" * 80)
    print("FILE:", path, "HEAD lines:", len(out.stdout.splitlines()))
    print("=" * 80)
    # Print only signature-relevant lines
    for line in out.stdout.splitlines():
        if any(
            token in line
            for token in ("def __init__", "max_execution_attempts", "max_verification_attempts", "event_logger", "metrics", "tracer", "observability", "config", "LoopDetector(", "max_sequence_length", "exact_repetition_threshold", "sequence_repetition_threshold", "Recovery(", "error_handler", "context_builder", "LLMExecutor(")
        ):
            print(line)