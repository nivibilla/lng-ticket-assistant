"""Extract, OCR, and chunk a PDF into a Parquet knowledge-base source."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
import pytesseract
from PIL import Image

DEFAULT_PDF_PATH = Path("KB Articles.pdf")
DEFAULT_CHUNKS_PATH = Path("chunks_df.parquet")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MIN_EXTRACTED_CHARS = 50
DEFAULT_OCR_ZOOM = 2.0

CHUNK_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("page_start", pa.int64()),
        pa.field("page_end", pa.int64()),
        pa.field("text", pa.string(), nullable=False),
        pa.field("n_chars", pa.int64(), nullable=False),
    ]
)


class PreprocessError(RuntimeError):
    """Raised when the PDF cannot be converted into valid chunks."""


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one one-indexed PDF page."""

    page: int
    text: str


@dataclass(frozen=True, slots=True)
class PreprocessSummary:
    """Summary returned after writing a chunk Parquet file."""

    pages: int
    chunks: int
    output_path: Path


def extract_page_text(
    page: pymupdf.Page,
    *,
    min_extracted_chars: int = DEFAULT_MIN_EXTRACTED_CHARS,
    ocr_zoom: float = DEFAULT_OCR_ZOOM,
) -> str:
    """Use native PDF text when sufficient, otherwise OCR the page."""
    text = page.get_text("text").strip()
    if len(text) >= min_extracted_chars:
        return text

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(ocr_zoom, ocr_zoom),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )
    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise PreprocessError(
            "OCR is required for this PDF, but the Tesseract executable "
            "was not found."
        ) from exc


def extract_pdf_pages(
    pdf_path: str | Path,
    *,
    min_extracted_chars: int = DEFAULT_MIN_EXTRACTED_CHARS,
    ocr_zoom: float = DEFAULT_OCR_ZOOM,
) -> list[PageText]:
    """Extract all PDF pages with one-indexed page numbers."""
    source_path = Path(pdf_path).expanduser().resolve()
    if not source_path.is_file():
        raise PreprocessError(f"PDF not found at {source_path}.")
    if source_path.suffix.casefold() != ".pdf":
        raise PreprocessError(f"Input file is not a PDF: {source_path}.")
    if min_extracted_chars < 0:
        raise ValueError("min_extracted_chars cannot be negative.")
    if ocr_zoom <= 0:
        raise ValueError("ocr_zoom must be greater than zero.")

    try:
        with pymupdf.open(source_path) as document:
            if document.page_count == 0:
                raise PreprocessError(f"PDF contains no pages: {source_path}.")
            return [
                PageText(
                    page=page_number,
                    text=extract_page_text(
                        page,
                        min_extracted_chars=min_extracted_chars,
                        ocr_zoom=ocr_zoom,
                    ),
                )
                for page_number, page in enumerate(document, start=1)
            ]
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(f"Could not read PDF {source_path}: {exc}") from exc


def recursive_split(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> list[str]:
    """Recursively split text while retaining bounded contextual overlap."""
    _validate_chunk_config(chunk_size, chunk_overlap)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0] if separators else ""
    remaining = separators[1:] if separators else ()
    parts = text.split(separator) if separator else list(text)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    def current_text() -> str:
        return separator.join(current) if separator else "".join(current)

    def overflows(piece: str) -> bool:
        extra = len(separator) if current and separator else 0
        return current_length + extra + len(piece) > chunk_size

    for part in parts:
        if overflows(part) and current:
            merged = current_text()
            if len(merged) > chunk_size and remaining:
                chunks.extend(
                    recursive_split(
                        merged,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=remaining,
                    )
                )
            else:
                chunks.append(merged)
            while current and current_length > chunk_overlap:
                dropped = current.pop(0)
                current_length -= len(dropped)
                if current:
                    current_length -= len(separator)

        if overflows(part) and not current:
            if remaining:
                chunks.extend(
                    recursive_split(
                        part,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=remaining,
                    )
                )
            else:
                start = 0
                while start < len(part):
                    end = min(start + chunk_size, len(part))
                    chunks.append(part[start:end])
                    if end == len(part):
                        break
                    start = max(end - chunk_overlap, start + 1)
            continue

        if current:
            current_length += len(separator) + len(part)
        else:
            current_length = len(part)
        current.append(part)

    if current:
        merged = current_text()
        if len(merged) > chunk_size and remaining:
            chunks.extend(
                recursive_split(
                    merged,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=remaining,
                )
            )
        else:
            chunks.append(merged)
    return [chunk for chunk in chunks if chunk.strip()]


def chunk_pages(
    pages: Sequence[PageText],
    *,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> pa.Table:
    """Chunk extracted pages and attach source and page-range metadata."""
    _validate_chunk_config(chunk_size, chunk_overlap)
    if not pages:
        raise PreprocessError("No PDF pages were extracted.")

    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    cursor = 0
    for index, page in enumerate(pages):
        if index:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        page_spans.append((page.page, start, cursor))

    full_text = "".join(parts)
    raw_chunks = recursive_split(
        full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not raw_chunks:
        raise PreprocessError("The PDF produced no non-empty text chunks.")

    records: list[dict[str, object]] = []
    search_from = 0
    for chunk_id, text in enumerate(raw_chunks):
        start = full_text.find(text, search_from)
        if start < 0:
            start = full_text.find(text)
        if start < 0:
            raise PreprocessError(
                f"Could not map generated chunk {chunk_id} back to a PDF page."
            )
        end = start + len(text)
        covered_pages = [
            page_number
            for page_number, span_start, span_end in page_spans
            if span_start < end and span_end > start
        ]
        records.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "page_start": min(covered_pages) if covered_pages else None,
                "page_end": max(covered_pages) if covered_pages else None,
                "text": text,
                "n_chars": len(text),
            }
        )
        search_from = max(start + 1, end - chunk_overlap)

    return pa.Table.from_pylist(records, schema=CHUNK_SCHEMA)


def preprocess_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_extracted_chars: int = DEFAULT_MIN_EXTRACTED_CHARS,
    ocr_zoom: float = DEFAULT_OCR_ZOOM,
) -> tuple[list[PageText], pa.Table]:
    """Extract and chunk a PDF without writing output."""
    source_path = Path(pdf_path).expanduser().resolve()
    pages = extract_pdf_pages(
        source_path,
        min_extracted_chars=min_extracted_chars,
        ocr_zoom=ocr_zoom,
    )
    chunks = chunk_pages(
        pages,
        source=source_path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return pages, chunks


def write_parquet_atomic(table: pa.Table, output_path: str | Path) -> Path:
    """Write Parquet via a same-directory temporary file and atomic replace."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def preprocess_to_parquet(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
    output_path: str | Path = DEFAULT_CHUNKS_PATH,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_extracted_chars: int = DEFAULT_MIN_EXTRACTED_CHARS,
    ocr_zoom: float = DEFAULT_OCR_ZOOM,
) -> PreprocessSummary:
    """Extract, chunk, and atomically write one PDF to Parquet."""
    pages, chunks = preprocess_pdf(
        pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_extracted_chars=min_extracted_chars,
        ocr_zoom=ocr_zoom,
    )
    destination = write_parquet_atomic(chunks, output_path)
    return PreprocessSummary(
        pages=len(pages),
        chunks=chunks.num_rows,
        output_path=destination,
    )


def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and chunk a PDF into a Parquet knowledge base."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help=f"Input PDF (default: {DEFAULT_PDF_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=f"Output Parquet file (default: {DEFAULT_CHUNKS_PATH}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(
            "Target overlap between adjacent chunks "
            f"(default: {DEFAULT_CHUNK_OVERLAP})."
        ),
    )
    parser.add_argument(
        "--min-extracted-chars",
        type=int,
        default=DEFAULT_MIN_EXTRACTED_CHARS,
        help=(
            "Use OCR below this native-text length "
            f"(default: {DEFAULT_MIN_EXTRACTED_CHARS})."
        ),
    )
    parser.add_argument(
        "--ocr-zoom",
        type=float,
        default=DEFAULT_OCR_ZOOM,
        help=f"PDF render scale for OCR (default: {DEFAULT_OCR_ZOOM:g}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = preprocess_to_parquet(
            args.pdf,
            args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_extracted_chars=args.min_extracted_chars,
            ocr_zoom=args.ocr_zoom,
        )
    except (OSError, PreprocessError, ValueError) as exc:
        raise SystemExit(f"PDF preprocessing failed: {exc}") from exc

    print(
        f"Wrote {summary.chunks} chunks from {summary.pages} pages "
        f"to {summary.output_path}."
    )


if __name__ == "__main__":
    main()
