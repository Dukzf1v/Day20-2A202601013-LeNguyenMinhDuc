"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate that the answer exists and contains source references."""
        if not state.final_answer:
            state.errors.append("Critic found no final answer")
        elif state.sources and not any("[" in state.final_answer for _ in state.sources):
            state.errors.append("Critic found no citation markers")
        state.add_trace_event("critic", {"errors": len(state.errors)})
        return state
