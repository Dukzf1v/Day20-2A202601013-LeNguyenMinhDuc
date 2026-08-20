"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Create a local timed span and mirror it to LangSmith when configured."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    langsmith_context: Any = None
    try:
        if os.getenv("LANGSMITH_API_KEY"):
            try:
                from langsmith.run_helpers import trace

                langsmith_context = trace(name, run_type="chain", inputs=attributes or {})
                langsmith_context.__enter__()
            except ImportError:
                langsmith_context = None
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        if langsmith_context is not None:
            langsmith_context.__exit__(None, None, None)
