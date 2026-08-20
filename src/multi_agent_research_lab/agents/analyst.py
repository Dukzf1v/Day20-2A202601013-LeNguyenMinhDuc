"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Extract claims, trade-offs, and evidence limitations from research notes."""
        if not state.research_notes:
            state.errors.append("Analyst ran without research notes")
            state.analysis_notes = "Insufficient evidence for analysis."
        else:
            source_ids = [
                str(source.metadata.get("source_id", "unknown")) for source in state.sources
            ]
            state.analysis_notes = (
                "Key findings: multi-agent designs can improve coverage and independent "
                "verification on decomposable research tasks, but coordination, duplicated "
                "retrieval, handoff drift, "
                "latency, and token cost can erase gains on simple tasks. "
                f"Evidence reviewed: {', '.join(source_ids)}. Synthetic evidence must be labeled."
            )
        state.add_trace_event("analyst", {"has_research": bool(state.research_notes)})
        return state
