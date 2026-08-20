# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 0.17 | 0.0000 | 6.5 | 40% | 0% | tokens=1470; routes=0 |
| multi-agent | 0.15 | 0.0000 | 10.0 | 100% | 0% | tokens=1671; routes=3 |

## Failure mode and fix

The first smoke run used a GraphRAG query that had no exact corpus keyword match. The supervisor repeatedly selected Researcher because no sources were returned. The fix was to provide ranked fallback sources, route from persisted artifacts, record every route, and retain `MAX_ITERATIONS` as a final guardrail.

## Interpretation

Multi-agent execution adds handoffs and usually costs more, so it is justified only when specialized research and analysis improve evidence coverage or answer quality.
