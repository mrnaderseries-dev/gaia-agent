import re

with open('planner/task_classifier.py', 'r') as f:
    content = f.read()

# Extract just the _VERBAL_OPERATION_RE pattern
match = re.search(r'_VERBAL_OPERATION_RE\s*=\s*re\.compile\(.*?\),', content, re.DOTALL)
if match:
    print('_VERBAL_OPERATION_RE:')
    print(match.group(0))
