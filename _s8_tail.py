"""Read the tail of a log file that another process is still appending to."""

from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = int(sys.argv[2]) if len(sys.argv) > 2 else 12

text = path.read_text(encoding="utf-8", errors="replace")
tail = text.splitlines()[-lines:]
print("\n".join(tail))
print(f"--- [{path.name}] total_lines={len(text.splitlines())} bytes={len(text)} ---")
