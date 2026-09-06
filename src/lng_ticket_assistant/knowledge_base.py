"""LanceDB access and hybrid retrieval for the support knowledge base."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lancedb

from .embeddings import EMBEDDING_DIMENSION, JinaRetrievalEmbedder

TABLE_NAME = "knowledge_base"
TEXT_COLUMN = "text"
VECTOR_COLUMN = "embedding"
FTS_INDEX_NAME = "text_idx"
DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 10
REQUIRED_COLUMNS = {
    "chunk_id",
    "source",
    "page_start",
    "page_end",
    TEXT_COLUMN,
    "n_chars",
    VECTOR_COLUMN,
}


class KnowledgeBaseError(RuntimeError):
    """Raised when the local knowledge base cannot be opened or searched."""


@dataclass(frozen=True, slots=True)
class KnowledgeBaseMatch:
    """One grounded knowledge-base search result."""

    chunk_id: int
    source: str
    page_start: int | None
    page_end: int | None
    text: str
    relevance_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def default_database_path() -> Path:
    """Resolve the configured database, preferring a local project checkout."""
    configured_path = os.getenv("KB_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    checkout_path = Path(__file__).resolve().parents[2] / "kb_db"
    working_directory_path = Path.cwd() / "kb_db"
    if working_directory_path.exists():
        return working_directory_path.resolve()
    return checkout_path.resolve()


def ensure_fts_index(table: Any) -> None:
    """Create the text index required by LanceDB hybrid search if absent."""
    indices = table.list_indices()
    has_text_fts = any(
        str(getattr(index, "index_type", "")).upper() == "FTS"
        and TEXT_COLUMN in getattr(index, "columns", [])
        for index in indices
    )
    if not has_text_fts:
        table.create_fts_index(TEXT_COLUMN, name=FTS_INDEX_NAME)


class KnowledgeBase:
    """Open and search the local support knowledge base."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        embedder: JinaRetrievalEmbedder | None = None,
        ensure_fts: bool = True,
    ) -> None:
        self.database_path = Path(database_path or default_database_path()).resolve()
        if not self.database_path.exists():
            raise KnowledgeBaseError(
                f"Knowledge-base database not found at {self.database_path}."
            )

        try:
            self.database = lancedb.connect(self.database_path)
            self.table = self.database.open_table(TABLE_NAME)
            self._validate_schema()
            if ensure_fts:
                ensure_fts_index(self.table)
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Could not open knowledge base at {self.database_path}: {exc}"
            ) from exc

        self.embedder = embedder or JinaRetrievalEmbedder()

    def _validate_schema(self) -> None:
        schema = self.table.schema
        missing_columns = REQUIRED_COLUMNS.difference(schema.names)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise KnowledgeBaseError(
                f"Knowledge-base table is missing required columns: {missing}."
            )

        vector_type = schema.field(VECTOR_COLUMN).type
        vector_dimension = getattr(vector_type, "list_size", None)
        if vector_dimension != EMBEDDING_DIMENSION:
            raise KnowledgeBaseError(
                "Knowledge-base embedding dimension is "
                f"{vector_dimension}; expected {EMBEDDING_DIMENSION}."
            )

    def search(
        self,
        query: str,
        k: int = DEFAULT_RESULT_LIMIT,
    ) -> list[KnowledgeBaseMatch]:
        """Run cosine vector and full-text search, fused by LanceDB."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The search query cannot be empty.")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")
        result_limit = min(max(k, 1), MAX_RESULT_LIMIT)

        query_embedding = self.embedder.encode_query(clean_query)
        try:
            rows = (
                self.table.search(query_type="hybrid")
                .vector(query_embedding)
                .text(clean_query)
                .distance_type("cosine")
                .limit(result_limit)
                .to_list()
            )
        except Exception as exc:
            raise KnowledgeBaseError(f"Knowledge-base search failed: {exc}") from exc

        matches: list[KnowledgeBaseMatch] = []
        for row in rows:
            score = row.get("_relevance_score")
            matches.append(
                KnowledgeBaseMatch(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    page_start=_optional_int(row.get("page_start")),
                    page_end=_optional_int(row.get("page_end")),
                    text=str(row["text"]),
                    relevance_score=float(score) if score is not None else None,
                )
            )
        return matches


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
