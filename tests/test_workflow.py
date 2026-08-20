from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def test_workflow_runs_end_to_end(monkeypatch) -> None:
    def fake_search(self: SearchClient, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="Test source",
                snippet="Structured roles improve verification but add coordination cost.",
                metadata={"source_id": "test-source"},
            )
        ]

    monkeypatch.setattr(SearchClient, "search", fake_search)
    state = ResearchState(request=ResearchQuery(query="Compare agent architectures"))
    result = MultiAgentWorkflow().run(state)

    assert result.final_answer
    assert result.route_history == ["researcher", "analyst", "writer"]
    assert "[test-source]" in result.final_answer
    assert result.analysis_notes
