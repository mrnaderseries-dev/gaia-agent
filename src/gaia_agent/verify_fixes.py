#!/usr/bin/env python3
"""Verify the fixes in planner/planner.py"""
import ast
import sys

# Read the file
with open('planner/planner.py', 'r') as f:
    content = f.read()

# Check for syntax errors
try:
    ast.parse(content)
    print("✓ Syntax OK")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

# Check that strategy_family is public
if 'def strategy_family(' in content:
    print("✓ strategy_family is public")
else:
    print("✗ strategy_family is NOT public")

# Check that _strategy_family is a wrapper (not the main implementation)
lines = content.split('\n')
in_wrapper = False
for i, line in enumerate(lines):
    if 'def _strategy_family(' in line:
        # Check if next few lines just call self.strategy_family
        next_lines = '\n'.join(lines[i:i+10])
        if 'return self.strategy_family(step)' in next_lines:
            print("✓ _strategy_family is a backward-compatible wrapper")
        else:
            print("✗ _strategy_family is NOT a wrapper")
        break

# Check that print(result) is NOT in _deterministic_fallback_code
if 'print(result)' in content:
    print("✗ print(result) still present in code")
else:
    print("✓ print(result) removed")

# Check that check_plan receives strategy_family_resolver
if 'strategy_family_resolver=self.strategy_family' in content:
    print("✓ check_plan receives strategy_family_resolver")
else:
    print("✗ check_plan missing strategy_family_resolver")

# Check all call sites
import re
check_plan_calls = re.findall(r'self\.loop_detector\.check_plan\([^)]+\)', content)
print(f"\nFound {len(check_plan_calls)} check_plan call(s):")
for call in check_plan_calls:
    has_resolver = 'strategy_family_resolver' in call
    status = "✓" if has_resolver else "✗"
    print(f"  {status} {call[:100]}...")

print("\n=== All checks completed ===")
