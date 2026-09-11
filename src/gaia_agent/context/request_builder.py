from __future__ import annotations

from gaia_agent.context.models import ContextRequest
from gaia_agent.core.agent_state import AgentState


class ContextRequestBuilder:
    @staticmethod
    def from_state(
        state: AgentState,
    ) -> ContextRequest:
        return ContextRequest(
            user_request=state.user_request,
            plan=list(state.plan),
            current_step=state.current_step,
            completed_steps=list(
                state.completed_steps
            ),
            current_action=state.current_action,
            step_type=state.step_type,
            iteration=state.iteration,
            tool_name=state.tool_name,
            blocked=state.blocked,
            tool_result=state.tool_result,
            tool_error=state.tool_error,
        )