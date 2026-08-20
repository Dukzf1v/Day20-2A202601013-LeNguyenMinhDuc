"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Write a concise evidence-grounded answer with source identifiers."""
        citations = " ".join(
            f"[{source.metadata.get('source_id', 'unknown')}]" for source in state.sources
        )
        evidence = state.research_notes or "No evidence available."
        response = self.llm_client.complete(
            "Write a concise research answer. Preserve source IDs and distinguish "
            "synthetic evidence.",
            f"QUESTION: {state.request.query}\nANALYSIS: {state.analysis_notes}\n"
            f"EVIDENCE:\n{evidence}",
        )
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.final_answer = (
            f"Research question: {state.request.query}\n\n{response.content}\n\n"
            f"Conclusion: use multi-agent orchestration for decomposable work with independent "
            f"verification; prefer a single agent for short, linear tasks. Sources: "
            f"{citations or '[none]'}."
        )
        state.add_trace_event("writer", {"answer_length": len(state.final_answer)})
        return state
