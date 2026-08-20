"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Select the next missing artifact in the deterministic pipeline."""
        if state.final_answer:
            route = "done"
        elif not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        else:
            route = "writer"
        state.record_route(route)
        state.add_trace_event("supervisor", {"next": route, "iteration": state.iteration})
        return state
