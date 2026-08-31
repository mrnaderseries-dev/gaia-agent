from __future__ import annotations

from dataclasses import dataclass, field

from gaia_agent.core.agent_state import AgentState
from gaia_agent.planner.plan_schema import PlanStep, StepType
from .base import ContextSource


@dataclass(slots=True)
class HistoryContext:
    plan: list[PlanStep]
    current_step: int
    completed_steps: list[int]
    current_action: str | None
    step_type: StepType | None


class HistorySource(ContextSource):

    async def get(
        self,
        state: AgentState,
    ) -> list[HistoryContext]:
        return [
            HistoryContext(
                plan=list(state.plan),
                current_step=state.current_step,
                completed_steps=list(state.completed_steps),
                current_action=state.current_action,
                step_type=state.step_type,
            )
        ]

    def is_available(self, state: AgentState) -> bool:
        return (
            bool(state.plan)
            or bool(state.completed_steps)
            or state.current_action is not None
        )