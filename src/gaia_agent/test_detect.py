#!/usr/bin/env python3
"""Test the deterministic fallback code generation."""
import sys
sys.path.insert(0, '.')

# Read planner.py and extract the detect_simple_operation function
with open('planner/planner.py', 'r') as f:
    content = f.read()

# Find and execute the detect_simple_operation function
start = content.find('def detect_simple_operation(')
if start == -1:
    print("ERROR: detect_simple_operation not found")
    sys.exit(1)

exec_globals = {'re': __import__('re')}
exec(content[start:], exec_globals)

detect_simple_operation = exec_globals['detect_simple_operation']

# Test cases
test_cases = [
    "Calculate 2 + 2.",
    "25 * 17 + 43",
    "What is 10 - 3?",
    "Calculate 25 * 17 + 43",
]

for question in test_cases:
    result = detect_simple_operation(question)
    print(f"'{question}' -> {result!r}")
