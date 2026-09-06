"""Embed preprocessed chunks and publish a complete LanceDB table."""

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq

from .embeddings import EMBEDDING_DIMENSION, JinaRetrievalEmbedder
from .knowledge_base import (
    TABLE_NAME,
    VECTOR_COLUMN,
    KnowledgeBaseError,
    default_database_path,
    ensure_fts_index,
)
from .preprocess import CHUNK_SCHEMA, DEFAULT_CHUNKS_PATH

KNOWLEDGE_BASE_SCHEMA = pa.schema(
    [
        *CHUNK_SCHEMA,
        pa.field(
            VECTOR_COLUMN,
            pa.list_(pa.float32(), EMBEDDING_DIMENSION),
            nullable=False,
        ),
    ]
)


def load_chunks(chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> pa.Table:
    """Read and validate the preprocessed chunk Parquet file."""
    source_path = Path(chunks_path).expanduser().resolve()
    if not source_path.is_file():
        raise KnowledgeBaseError(f"Chunk Parquet file not found at {source_path}.")

    try:
        source = pq.read_table(source_path)
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not read chunk Parquet file {source_path}: {exc}"
        ) from exc

    missing_columns = set(CHUNK_SCHEMA.names).difference(source.schema.names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KnowledgeBaseError(
            f"Chunk Parquet file is missing required columns: {missing}."
        )

    try:
        chunks = source.select(CHUNK_SCHEMA.names).cast(CHUNK_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise KnowledgeBaseError(
            f"Chunk Parquet schema is incompatible with the publisher: {exc}"
        ) from exc
    _validate_chunks(chunks)
    return chunks


def _validate_chunks(chunks: pa.Table) -> None:
    if chunks.num_rows == 0:
        raise KnowledgeBaseError("The chunk Parquet file contains no documents.")

    records = chunks.to_pylist()
    chunk_ids = [record["chunk_id"] for record in records]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise KnowledgeBaseError("Chunk ids must be unique.")

    for record in records:
        chunk_id = record["chunk_id"]
        text = record["text"]
        source = record["source"]
        page_start = record["page_start"]
        page_end = record["page_end"]
        if not text or not text.strip():
            raise KnowledgeBaseError(f"Chunk {chunk_id} contains no text.")
        if not source or not source.strip():
            raise KnowledgeBaseError(f"Chunk {chunk_id} has no source.")
        if record["n_chars"] != len(text):
            raise KnowledgeBaseError(
                f"Chunk {chunk_id} has an incorrect n_chars value."
            )
        if page_start is not None and page_start < 1:
            raise KnowledgeBaseError(f"Chunk {chunk_id} has an invalid start page.")
        if page_end is not None and page_end < 1:
            raise KnowledgeBaseError(f"Chunk {chunk_id} has an invalid end page.")
        if (
            page_start is not None
            and page_end is not None
            and page_start > page_end
        ):
            raise KnowledgeBaseError(
                f"Chunk {chunk_id} ends before its start page."
            )


def rebuild_embeddings(
    database_path: str | Path | None = None,
    *,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    embedder: JinaRetrievalEmbedder | None = None,
) -> int:
    """Publish a complete vector table after a successful embedding pass."""
    chunks = load_chunks(chunks_path)
    texts = [str(text) for text in chunks.column("text").to_pylist()]
    document_embedder = (
        embedder if embedder is not None else JinaRetrievalEmbedder()
    )

    document_embeddings = document_embedder.encode_documents(texts)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if document_embeddings.shape != expected_shape:
        raise KnowledgeBaseError(
            f"Rebuild produced shape {document_embeddings.shape}; "
            f"expected {expected_shape}."
        )

    vector_array = pa.array(
        document_embeddings.tolist(),
        type=KNOWLEDGE_BASE_SCHEMA.field(VECTOR_COLUMN).type,
    )
    replacement_data = pa.Table.from_arrays(
        [
            *(chunks.column(name) for name in CHUNK_SCHEMA.names),
            vector_array,
        ],
        schema=KNOWLEDGE_BASE_SCHEMA,
    )
    if replacement_data.num_rows != chunks.num_rows:
        raise KnowledgeBaseError(
            "Replacement row count changed before the table could be written."
        )

    target_path = Path(database_path or default_database_path()).resolve()
    try:
        database = lancedb.connect(target_path)
        replacement_table = database.create_table(
            TABLE_NAME,
            data=replacement_data,
            mode="overwrite",
        )
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not publish knowledge base at {target_path}: {exc}"
        ) from exc

    if replacement_table.count_rows() != chunks.num_rows:
        raise KnowledgeBaseError(
            "Knowledge-base row count changed during the embedding rebuild."
        )
    try:
        ensure_fts_index(replacement_table)
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not create the knowledge-base text index: {exc}"
        ) from exc
    return replacement_table.count_rows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embed preprocessed Parquet chunks with Jina retrieval document "
            "prompts and publish them to LanceDB."
        )
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=f"Preprocessed chunk Parquet file (default: {DEFAULT_CHUNKS_PATH}).",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help=(
            "Directory containing the LanceDB database "
            f"(default: {default_database_path()})."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        row_count = rebuild_embeddings(
            args.database_path,
            chunks_path=args.chunks_path,
        )
    except (KnowledgeBaseError, ValueError) as exc:
        raise SystemExit(f"Embedding rebuild failed: {exc}") from exc
    print(f"Published {row_count} knowledge-base embeddings and the text index.")


if __name__ == "__main__":
    main()
