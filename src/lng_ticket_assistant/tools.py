"""Tools exposed to the Deep Agents support assistant."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .knowledge_base import (
    DEFAULT_RESULT_LIMIT,
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeBaseMatch,
)


def _citation(match: KnowledgeBaseMatch) -> str:
    if match.page_start is None:
        return match.source
    if match.page_end is None or match.page_end == match.page_start:
        return f"{match.source}, page {match.page_start}"
    return f"{match.source}, pages {match.page_start}-{match.page_end}"


def make_search_knowledge_base_tool(
    database_path: str | Path | None = None,
    *,
    knowledge_base: KnowledgeBase | None = None,
) -> Callable[[str, int], str]:
    """Build a search tool bound to one LanceDB knowledge base."""
    retriever = (
        knowledge_base
        if knowledge_base is not None
        else KnowledgeBase(database_path)
    )

    def search_knowledge_base(
        query: str,
        k: int = DEFAULT_RESULT_LIMIT,
    ) -> str:
        """Search support articles using semantic and keyword retrieval.

        Use this before answering any substantive support question. Phrase
        ``query`` as the user's issue, including exact error messages and
        product names. ``k`` is the number of evidence chunks to retrieve
        (default 5, maximum 10). Results contain article text and citations.
        """
        try:
            matches = retriever.search(query, k=k)
        except (KnowledgeBaseError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)})

        results = [
            {
                "evidence_id": f"chunk-{match.chunk_id}",
                "chunk_id": match.chunk_id,
                "source": match.source,
                "page_start": match.page_start,
                "page_end": match.page_end,
                "citation": _citation(match),
                "content": match.text,
            }
            for match in matches
        ]
        return json.dumps(
            {
                "query": query,
                "result_count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )

    return search_knowledge_base


def get_tools(
    database_path: str | Path | None = None,
    *,
    knowledge_base: KnowledgeBase | None = None,
) -> list[Callable[..., str]]:
    """Return every tool available to the support agent."""
    return [
        make_search_knowledge_base_tool(
            database_path,
            knowledge_base=knowledge_base,
        )
    ]
