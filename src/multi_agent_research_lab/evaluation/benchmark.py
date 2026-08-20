"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, citation coverage, and failure status."""

    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:  # benchmark must record failures instead of aborting the suite
        state = ResearchState.model_validate({"request": {"query": query}, "errors": [str(exc)]})
        failed = True
    latency = perf_counter() - started
    answer = state.final_answer or ""
    source_ids = {
        str(source.metadata.get("source_id"))
        for source in state.sources
        if source.metadata.get("source_id")
    }
    cited_ids = set(re.findall(r"\[([^\]]+)\]", answer))
    citation_coverage = len(source_ids & cited_ids) / len(source_ids) if source_ids else 0.0
    quality = _quality_score(state, citation_coverage)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.estimated_cost_usd,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=float(failed or not state.final_answer),
        notes=(
            f"tokens={state.input_tokens + state.output_tokens}; routes={len(state.route_history)}"
        ),
    )
    return state, metrics


def _quality_score(state: ResearchState, citation_coverage: float) -> float:
    answer = state.final_answer or ""
    score = 0.0
    score += min(len(answer) / 800, 1.0) * 3.0
    score += min(len(state.sources) / max(state.request.max_sources, 1), 1.0) * 2.0
    score += 2.0 if state.analysis_notes else 0.0
    score += citation_coverage * 2.0
    score += 1.0 if not state.errors else 0.0
    return round(min(score, 10.0), 1)
