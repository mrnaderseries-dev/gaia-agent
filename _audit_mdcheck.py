from __future__ import annotations

from pathlib import Path

path = Path(r"C:\Users\user\gaia-agent\src\gaia_agent\analysis.md")
text = path.read_text(encoding="utf-8")
text = text.replace("د8 triage", "Section 8 triage")
path.write_text(text, encoding="utf-8")
print("bytes:", len(text.encode("utf-8")))
print("lines:", text.count("\n") + 1)
print("--- H1/H2 sections ---")
for line in text.splitlines():
    if line.startswith("#"):
        print(line)