from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow.parquet as pq

from lng_ticket_assistant.preprocess import (
    CHUNK_SCHEMA,
    PageText,
    chunk_pages,
    extract_page_text,
    preprocess_to_parquet,
    recursive_split,
)


class ExtractionTests(unittest.TestCase):
    def test_uses_native_text_without_ocr_when_sufficient(self) -> None:
        page = Mock()
        page.get_text.return_value = "Native PDF text"

        with patch(
            "lng_ticket_assistant.preprocess.pytesseract.image_to_string"
        ) as image_to_string:
            text = extract_page_text(page, min_extracted_chars=5)

        self.assertEqual(text, "Native PDF text")
        page.get_pixmap.assert_not_called()
        image_to_string.assert_not_called()

    def test_uses_ocr_when_native_text_is_too_short(self) -> None:
        page = Mock()
        page.get_text.return_value = ""
        page.get_pixmap.return_value = Mock(
            width=1,
            height=1,
            samples=b"\x00\x00\x00",
        )

        with patch(
            "lng_ticket_assistant.preprocess.pytesseract.image_to_string",
            return_value="OCR text\n",
        ) as image_to_string:
            text = extract_page_text(page, min_extracted_chars=5)

        self.assertEqual(text, "OCR text")
        page.get_pixmap.assert_called_once()
        image_to_string.assert_called_once()


class ChunkingTests(unittest.TestCase):
    def test_recursive_split_respects_character_limit(self) -> None:
        chunks = recursive_split(
            "abcdefghijklmnopqrst",
            chunk_size=8,
            chunk_overlap=2,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 8 for chunk in chunks))

    def test_chunk_pages_assigns_page_ranges(self) -> None:
        pages = [
            PageText(page=1, text="alpha beta gamma"),
            PageText(page=2, text="delta epsilon zeta"),
        ]

        separate = chunk_pages(
            pages,
            source="guide.pdf",
            chunk_size=20,
            chunk_overlap=5,
        ).to_pylist()
        combined = chunk_pages(
            pages,
            source="guide.pdf",
            chunk_size=100,
            chunk_overlap=5,
        ).to_pylist()

        self.assertEqual(
            [(row["page_start"], row["page_end"]) for row in separate],
            [(1, 1), (2, 2)],
        )
        self.assertEqual(combined[0]["page_start"], 1)
        self.assertEqual(combined[0]["page_end"], 2)
        self.assertEqual(combined[0]["source"], "guide.pdf")

    def test_rejects_overlap_equal_to_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than"):
            recursive_split("text", chunk_size=10, chunk_overlap=10)


class ParquetTests(unittest.TestCase):
    def test_preprocess_writes_expected_parquet_schema(self) -> None:
        pages = [PageText(page=1, text="Password reset")]
        chunks = chunk_pages(
            pages,
            source="guide.pdf",
            chunk_size=100,
            chunk_overlap=10,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "chunks.parquet"
            with patch(
                "lng_ticket_assistant.preprocess.preprocess_pdf",
                return_value=(pages, chunks),
            ):
                summary = preprocess_to_parquet(
                    "guide.pdf",
                    output_path,
                )
            written = pq.read_table(output_path)

        self.assertEqual(summary.pages, 1)
        self.assertEqual(summary.chunks, 1)
        self.assertEqual(written.schema, CHUNK_SCHEMA)
        self.assertEqual(written.column("text")[0].as_py(), "Password reset")


if __name__ == "__main__":
    unittest.main()
