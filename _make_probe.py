import shutil

src = "tests/integration/test_real_orchestration_flow.py"
dst = "tests/integration/_probe_real_orchestration_flow.py"

text = open(src, encoding="utf-8-sig").read()

old = "loop_detector.check.return_value = False"
new = (
    "loop_detector.check.return_value = SimpleNamespace(\n"
    "        detected=False, loop_type=None, similarity=0.0, reason='',\n"
    "    )"
)
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new)

open(dst, "w", encoding="utf-8").write(text)
print("probe written")