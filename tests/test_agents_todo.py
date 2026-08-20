from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_by_missing_artifact() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()

    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [SourceDocument(title="Source", snippet="Evidence")]
    state.research_notes = "Evidence"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Analysis"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "Answer"
    supervisor.run(state)
    assert state.route_history[-1] == "done"
