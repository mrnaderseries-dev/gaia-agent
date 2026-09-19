from __future__ import annotations

from dataclasses import dataclass

from .plan_schema import PlanSchema
from .task_classifier import TaskAnalysis


@dataclass(frozen=True, slots=True)
class PlanningResult:
    plan: PlanSchema
    task_analysis: TaskAnalysis