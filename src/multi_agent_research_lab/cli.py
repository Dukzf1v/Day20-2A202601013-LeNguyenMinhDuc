"""Command-line entrypoint for the lab starter."""

import json
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_baseline(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    with trace_span("single_agent_baseline", {"query": query}) as span:
        state.sources = SearchClient().search(query, state.request.max_sources)
        evidence = "\n".join(
            f"[{source.metadata.get('source_id', 'unknown')}] {source.snippet}"
            for source in state.sources
        )
        response = LLMClient().complete(
            "Answer directly using only supplied evidence and source IDs.",
            f"QUESTION: {query}\nEVIDENCE:\n{evidence}",
        )
        state.final_answer = response.content
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
    state.trace.append(span)
    return state


def run_multi_agent(query: str) -> ResearchState:
    return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=query)))


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    _parse_query(query)
    state = run_baseline(query)
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    _parse_query(query)
    result = run_multi_agent(query)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q")] = (
        "Compare single-agent and multi-agent architectures for complex research tasks"
    ),
) -> None:
    """Run both workflows and write report plus local trace evidence."""
    _init()
    baseline_state, baseline_metrics = run_benchmark("single-agent", query, run_baseline)
    multi_state, multi_metrics = run_benchmark("multi-agent", query, run_multi_agent)
    store = LocalArtifactStore()
    report_path = store.write_text(
        "benchmark_report.md", render_markdown_report([baseline_metrics, multi_metrics])
    )
    trace_path = store.write_text(
        "trace_example.json",
        json.dumps(
            {"query": query, "baseline": baseline_state.trace, "multi_agent": multi_state.trace},
            indent=2,
        ),
    )
    console.print(f"Report: {report_path}\nTrace: {trace_path}")


if __name__ == "__main__":
    app()
