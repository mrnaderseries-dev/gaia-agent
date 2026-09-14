# Debugging Report: gaia-agent Runtime Issues

**Date**: 2026-09-14
**Context**: Debugging runtime failures in `python -m gaia_agent.main`

---

## Issue 1: `LoopDetector.check_plan()` Missing Required Argument

### Problem
Running `python -m gaia_agent.main` with "Calculate 2 + 2." produces:

```
FATAL ERROR: True
TOOL ERROR: "LoopDetector.check_plan() missing 1 required keyword-only argument:
'strategy_family_resolver'"
```

The original error also included:
```
Tool 'python_interpreter' execution failed:
Undefined name: name 'print' is not defined.
```

### Evidence
- **Error location**: `planner/planner.py:363-365`
- **Call site**: `self.loop_detector.check_plan(plan.steps)`
- **Required signature**: `check_plan(self, steps, *, strategy_family_resolver: Callable[[PlanStep], str])`
- **Actual call**: Missing `strategy_family_resolver` keyword argument

### Root Cause
`PlanValidator._validate_generated_plan()` calls `check_plan()` without the required `strategy_family_resolver` argument. The `LoopDetector.check_plan()` method signature requires this keyword-only argument because the detector intentionally does not know how strategy families are calculated - that belongs to the Planner/StrategySelector layer.

Additionally, `Orchestrator._execute_step()` at line 464-466 calls `self.planner.strategy_family(step)` but `Planner` only has a private `_strategy_family()` method, not a public `strategy_family()` method.

---

## Issue 2: Python Code Generation Including print()

### Problem
The deterministic fallback code generator was producing:
```python
result = 2 + 2
print(result)
```

But the Python sandbox restricts builtins and doesn't include `print`.

### Evidence
- **Tool error**: `Undefined name: name 'print' is not defined.`
- **Tool contract**: Python sandbox excludes `print` from `__builtins__`

### Root Cause
`_deterministic_fallback_code()` was appending `print(result)` to generated code.

