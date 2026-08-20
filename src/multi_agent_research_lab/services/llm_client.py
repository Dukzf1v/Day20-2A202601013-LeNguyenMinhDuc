"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from re import split

from multi_agent_research_lab.core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """OpenAI client with a deterministic offline fallback for the lab corpus."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion and provider usage when available."""

        if self.settings.openai_api_key:
            return self._complete_openai(system_prompt, user_prompt)
        return self._complete_offline(user_prompt)

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install the 'llm' extra to use OPENAI_API_KEY") from exc

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
        )
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    def _complete_offline(self, user_prompt: str) -> LLMResponse:
        evidence = user_prompt.split("EVIDENCE:", maxsplit=1)[-1].strip()
        sentences = [item.strip() for item in split(r"(?<=[.!?])\s+", evidence) if item.strip()]
        selected = sentences[:6]
        if not selected:
            selected = ["No relevant evidence was supplied for this query."]
        content = "\n\n".join(selected)
        input_tokens = max(1, len(user_prompt) // 4)
        output_tokens = max(1, len(content) // 4)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
        )

    @staticmethod
    def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None and output_tokens is None:
            return None
        # Configurable provider billing is outside this lab; use gpt-4o-mini list-price defaults.
        return (input_tokens or 0) * 0.15 / 1_000_000 + (output_tokens or 0) * 0.60 / 1_000_000
