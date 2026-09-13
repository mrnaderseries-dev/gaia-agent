from __future__ import annotations

from dataclasses import dataclass

from .plan_schema import PlanSchema
from .task_classifier import TaskAnalysis


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """
    Immutable result produced by the planning layer.

    The Planner owns both:
    - task analysis
    - executable plan generation

    Downstream layers consume both values from this single explicit
    contract instead of reading mutable Planner state.
    """

    plan: PlanSchema
    task_analysis: TaskAnalysis