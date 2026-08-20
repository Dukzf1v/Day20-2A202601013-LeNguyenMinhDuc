"""Offline corpus search used by ResearcherAgent."""

import json
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Search the self-contained JSON benchmark corpus without network access."""

    def __init__(self, corpus_root: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.corpus_root = corpus_root or project_root / "ai_agent_offline_research_corpus_v2"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return ranked source summaries from the bundled corpus."""

        terms = self._terms(query)
        candidates: list[tuple[int, SourceDocument]] = []
        for path in sorted((self.corpus_root / "topics").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            topic = payload.get("topic", {})
            topic_text = " ".join(
                [topic.get("name", ""), topic.get("research_question", ""), *topic.get("tags", [])]
            )
            topic_score = self._score(terms, topic_text)
            knowledge_base: dict[str, Any] = payload.get("knowledge_base", {})
            for document in knowledge_base.get("source_documents", []):
                text = " ".join(
                    [
                        document.get("title", ""),
                        document.get("full_text", ""),
                        " ".join(document.get("key_takeaways", [])),
                    ]
                )
                score = topic_score * 3 + self._score(terms, text)
                source_id = document.get("document_id") or document.get("source_id") or path.stem
                candidates.append(
                    (
                        score,
                        SourceDocument(
                            title=document.get("title", source_id),
                            url=document.get("provenance_url"),
                            snippet=self._snippet(document),
                            metadata={
                                "source_id": source_id,
                                "topic": topic.get("name", path.stem),
                                "is_synthetic": bool(document.get("is_synthetic", False)),
                            },
                        ),
                    )
                )
        candidates.sort(key=lambda item: (-item[0], item[1].title))
        return [document for _, document in candidates[:max_results]]

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop_words = {"and", "for", "the", "with", "what", "when", "write", "research"}
        return {word for word in re.findall(r"[a-z0-9-]+", text.lower()) if word not in stop_words}

    @classmethod
    def _score(cls, terms: set[str], text: str) -> int:
        haystack = cls._terms(text)
        return len(terms & haystack)

    @staticmethod
    def _snippet(document: dict[str, Any]) -> str:
        takeaways = document.get("key_takeaways", [])
        content = " ".join(takeaways) if takeaways else document.get("full_text", "")
        return content[:1600].strip()
