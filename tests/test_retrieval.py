from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lng_ticket_assistant.embeddings import (
    DOCUMENT_PROMPT,
    EMBEDDING_DIMENSION,
    QUERY_PROMPT,
    RETRIEVAL_TASK,
    EmbeddingError,
    JinaRetrievalEmbedder,
)
from lng_ticket_assistant.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeBaseMatch,
    ensure_fts_index,
)
from lng_ticket_assistant.preprocess import CHUNK_SCHEMA
from lng_ticket_assistant.rebuild_embeddings import (
    KNOWLEDGE_BASE_SCHEMA,
    load_chunks,
    rebuild_embeddings,
)
from lng_ticket_assistant.tools import make_search_knowledge_base_tool


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, **kwargs: object) -> np.ndarray:
        self.calls.append(kwargs)
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        return np.ones((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)


class EmbeddingTests(unittest.TestCase):
    def test_uses_distinct_retrieval_prompts(self) -> None:
        model = FakeModel()
        embedder = JinaRetrievalEmbedder(model=model)

        query = embedder.encode_query("VPN error")
        documents = embedder.encode_documents(["VPN troubleshooting", "MFA help"])

        self.assertEqual(query.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(documents.shape, (2, EMBEDDING_DIMENSION))
        self.assertEqual(model.calls[0]["task"], RETRIEVAL_TASK)
        self.assertEqual(model.calls[0]["prompt_name"], QUERY_PROMPT)
        self.assertEqual(model.calls[1]["task"], RETRIEVAL_TASK)
        self.assertEqual(model.calls[1]["prompt_name"], DOCUMENT_PROMPT)

    def test_rejects_wrong_embedding_shape(self) -> None:
        model = Mock()
        model.encode.return_value = np.ones((1, 8), dtype=np.float32)
        embedder = JinaRetrievalEmbedder(model=model)

        with self.assertRaises(EmbeddingError):
            embedder.encode_query("VPN error")


class FakeSearchTable:
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.int64()),
            pa.field("source", pa.string()),
            pa.field("page_start", pa.int64()),
            pa.field("page_end", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("n_chars", pa.int64()),
            pa.field(
                "embedding",
                pa.list_(pa.float32(), EMBEDDING_DIMENSION),
            ),
        ]
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def list_indices(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                index_type="FTS",
                columns=["text"],
                name="text_idx",
            )
        ]

    def search(self, *, query_type: str) -> FakeSearchTable:
        self.calls.append(("search", query_type))
        return self

    def vector(self, value: np.ndarray) -> FakeSearchTable:
        self.calls.append(("vector", value))
        return self

    def text(self, value: str) -> FakeSearchTable:
        self.calls.append(("text", value))
        return self

    def distance_type(self, value: str) -> FakeSearchTable:
        self.calls.append(("distance_type", value))
        return self

    def limit(self, value: int) -> FakeSearchTable:
        self.calls.append(("limit", value))
        return self

    def to_list(self) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": 7,
                "source": "KB Articles.pdf",
                "page_start": 3,
                "page_end": 4,
                "text": "Reconnect the VPN client.",
                "_relevance_score": 0.9,
            }
        ]


class RetrievalTests(unittest.TestCase):
    def test_builds_explicit_hybrid_cosine_search(self) -> None:
        table = FakeSearchTable()
        database = Mock()
        database.open_table.return_value = table
        embedder = Mock()
        embedder.encode_query.return_value = np.zeros(
            EMBEDDING_DIMENSION, dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "lng_ticket_assistant.knowledge_base.lancedb.connect",
                return_value=database,
            ):
                knowledge_base = KnowledgeBase(directory, embedder=embedder)
                matches = knowledge_base.search("VPN E4012", k=99)

        self.assertEqual(matches[0].chunk_id, 7)
        self.assertEqual(
            [(name, value) for name, value in table.calls if name != "vector"],
            [
                ("search", "hybrid"),
                ("text", "VPN E4012"),
                ("distance_type", "cosine"),
                ("limit", 10),
            ],
        )

    def test_creates_missing_fts_index(self) -> None:
        table = Mock()
        table.list_indices.return_value = []

        ensure_fts_index(table)

        table.create_fts_index.assert_called_once_with("text", name="text_idx")

    def test_search_tool_returns_cited_evidence(self) -> None:
        knowledge_base = Mock()
        knowledge_base.search.return_value = [
            KnowledgeBaseMatch(
                chunk_id=2,
                source="KB Articles.pdf",
                page_start=1,
                page_end=2,
                text="Reset the account password.",
                relevance_score=0.8,
            )
        ]
        tool = make_search_knowledge_base_tool(knowledge_base=knowledge_base)

        payload = json.loads(tool("password reset", 3))

        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["citation"], "KB Articles.pdf, pages 1-2")
        self.assertNotIn("relevance_score", payload["results"][0])


class RebuildTests(unittest.TestCase):
    def test_rejects_parquet_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "chunks.parquet"
            pq.write_table(pa.table({"text": ["Password reset"]}), chunks_path)

            with self.assertRaisesRegex(
                KnowledgeBaseError,
                "missing required columns",
            ):
                load_chunks(chunks_path)

    def test_does_not_replace_table_when_document_embedding_fails(self) -> None:
        source = pa.Table.from_pylist(
            [
                {
                    "chunk_id": 1,
                    "source": "KB Articles.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Password reset",
                    "n_chars": 14,
                }
            ],
            schema=CHUNK_SCHEMA,
        )
        embedder = Mock()
        embedder.encode_documents.side_effect = RuntimeError("model failed")

        with tempfile.TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "chunks.parquet"
            pq.write_table(source, chunks_path)
            with patch(
                "lng_ticket_assistant.rebuild_embeddings.lancedb.connect"
            ) as connect:
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    rebuild_embeddings(
                        Path(directory) / "database",
                        chunks_path=chunks_path,
                        embedder=embedder,
                    )

        connect.assert_not_called()

    def test_publishes_complete_table_from_parquet(self) -> None:
        source = pa.Table.from_pylist(
            [
                {
                    "chunk_id": 1,
                    "source": "KB Articles.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Password reset",
                    "n_chars": 14,
                }
            ],
            schema=CHUNK_SCHEMA,
        )
        embedder = Mock()
        embedder.encode_documents.return_value = np.ones(
            (1, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        published_table = Mock()
        published_table.count_rows.return_value = 1
        published_table.list_indices.return_value = []
        database = Mock()
        database.create_table.return_value = published_table

        with tempfile.TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "chunks.parquet"
            pq.write_table(source, chunks_path)
            with patch(
                "lng_ticket_assistant.rebuild_embeddings.lancedb.connect",
                return_value=database,
            ):
                count = rebuild_embeddings(
                    Path(directory) / "database",
                    chunks_path=chunks_path,
                    embedder=embedder,
                )

        self.assertEqual(count, 1)
        published = database.create_table.call_args.kwargs["data"]
        self.assertEqual(published.schema, KNOWLEDGE_BASE_SCHEMA)
        self.assertEqual(published.column("embedding")[0].as_py()[0], 1.0)
        published_table.create_fts_index.assert_called_once_with(
            "text",
            name="text_idx",
        )


if __name__ == "__main__":
    unittest.main()
