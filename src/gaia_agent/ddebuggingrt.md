# Debugging Report: gaia-agent Runtime Issues

**Date**: 2026-09-14
**Agent Version**: Latest from main branch
**Task**: Fix runtime failures in `python -m gaia_agent.main`

---

## Issue 1: `LoopDetector.check_plan()` Missing Required Argument

### Problem
Running `python -m gaia_agent.main` with "Calculate 2 + 2." produces:

```
FATAL ERROR: True
TOOL ERROR: "LoopDetector.check_plan() missing 1 required keyword-only argument:
'strategy_family_resolver'"
```

### Evidence
- **Error location**: `planner/planner.py:363-365`
- **Call site**: `self.loop_detector.check_plan(plan.steps)`
- **Required signature**: `check_plan(self, steps, *, strategy_family_resolver: Callable[[PlanStep], str])`
- **Actual call**: Missing `strategy_family_resolver` keyword argument

### Root Cause
`PlanValidator._validate_generated_plan()` calls `check_plan()` without the required `strategy_family_resolver` argument. The `LoopDetector.check_plan()` method signature requires this keyword-only argument because the detector intentionally does not know how strategy families are calculated - that belongs to the Planner/StrategySelector layer.

### Contract Involved
- **LoopDetector**: `check_plan(steps, *, strategy_family_resolver)` - requires a resolver function
- **Planner**: Has `_strategy_family(step)` private method
- **Orchestrator**: Expected a public `strategy_family()` method on Planner

### Fix
**File: `planner/planner.py`**

1. Changed `_validate_generated_plan()` to pass `strategy_family_resolver=self._strategy_family`
2. Renamed `_strategy_family` to `strategy_family` (public method)
3. Updated all internal callers

### Why This Fix
The `strategy_family_resolver` is a core part of the LoopDetector's contract. Making the existing method public correctly wires the abstractions.

---

## Files Changed

1. `planner/planner.py`:
   - Fixed `_validate_generated_plan()` to pass `strategy_family_resolver`
   - Renamed `_strategy_family` to `strategy_family` (public API)

---

## Architecture Preservation

No architectural redesign was performed. The fixes:
- Preserve LoopDetector's contract (requires resolver function)
- Preserve Planner's strategy abstraction
- Preserve Orchestrator's layer responsibilities
