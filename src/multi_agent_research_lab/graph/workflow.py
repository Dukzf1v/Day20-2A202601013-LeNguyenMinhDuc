"""Deterministic multi-agent workflow with optional LangGraph integration."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Describe the nodes and entrypoint used by the workflow."""

        return {"nodes": ["supervisor", "researcher", "analyst", "writer"], "entry": "supervisor"}

    def run(self, state: ResearchState) -> ResearchState:
        """Execute supervisor and workers until a final answer or iteration limit."""
        settings = get_settings()
        agents = {
            "supervisor": SupervisorAgent(),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }
        while state.iteration < settings.max_iterations and not state.final_answer:
            with trace_span("supervisor", {"iteration": state.iteration}) as span:
                state = agents["supervisor"].run(state)
            state.trace.append(span)
            route = state.route_history[-1]
            if route == "done":
                break
            with trace_span(route, {"query": state.request.query}) as span:
                state = agents[route].run(state)
            state.trace.append(span)
        if not state.final_answer and state.analysis_notes:
            with trace_span("writer_fallback") as span:
                state = agents["writer"].run(state)
            state.trace.append(span)
        if not state.final_answer:
            state.errors.append("Workflow stopped before producing a final answer")
        state.add_trace_event("workflow_complete", {"routes": state.route_history})
        return state
