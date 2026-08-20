"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Retrieve ranked evidence and create a traceable research packet."""
        state.sources = self.search_client.search(state.request.query, state.request.max_sources)
        if not state.sources:
            state.errors.append("No matching offline sources found")
            state.research_notes = "No matching sources were found in the offline corpus."
        else:
            state.research_notes = "\n".join(
                f"[{source.metadata.get('source_id', 'unknown')}] {source.title}: {source.snippet}"
                for source in state.sources
            )
        state.add_trace_event("researcher", {"sources": len(state.sources)})
        return state
