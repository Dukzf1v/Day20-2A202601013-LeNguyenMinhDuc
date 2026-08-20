"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render comparable benchmark metrics and the observed failure analysis."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Failure mode and fix",
            "",
            "The first smoke run used a GraphRAG query that had no exact corpus keyword match. "
            "The supervisor repeatedly selected Researcher because no sources were returned. "
            "The fix was to provide ranked fallback sources, route from persisted artifacts, "
            "record every route, "
            "and retain `MAX_ITERATIONS` as a final guardrail.",
            "",
            "## Interpretation",
            "",
            "Multi-agent execution adds handoffs and usually costs more, so it is justified only "
            "when specialized research and analysis improve evidence coverage or answer quality.",
        ]
    )
    return "\n".join(lines) + "\n"
