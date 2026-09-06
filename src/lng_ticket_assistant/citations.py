"""Resolve browser citations from actual retrieval messages, never model claims."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

CITATION_PATTERN = re.compile(r"\[\[cite:(chunk-\d+)\]\]")


class Evidence(BaseModel):
    evidence_id: str = Field(pattern=r"^chunk-\d+$")
    chunk_id: int = Field(ge=0)
    source: str = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    content: str = Field(min_length=1)


def retrieved_evidence(messages: list[Any]) -> dict[str, Evidence]:
    evidence: dict[str, Evidence] = {}
    for message in messages:

        def field(name: str, current: Any = message) -> Any:
            return (
                current.get(name)
                if isinstance(current, dict)
                else getattr(current, name, None)
            )

        if (field("type") or field("role")) != "tool" or field(
            "name"
        ) != "search_knowledge_base":
            continue
        try:
            payload = json.loads(field("content"))
            if not isinstance(payload, dict):
                continue
            for item in payload.get("results", []):
                try:
                    match = Evidence.model_validate(item)
                    if match.evidence_id == f"chunk-{match.chunk_id}":
                        evidence[match.evidence_id] = match
                except ValidationError:
                    continue
        except (TypeError, ValueError):
            continue
    return evidence


class SourceStore:
    """Allowlist known source filenames beneath a configured directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self._names: dict[str, str] = {}

    def describe(self, source: str) -> dict[str, Any]:
        source_id = hashlib.sha256(source.encode()).hexdigest()[:24]
        if Path(source).name == source and Path(source).suffix.lower() == ".pdf":
            self._names[source_id] = source
        path = self.path(source_id)
        return {
            "source_id": source_id,
            "pdf_url": f"/api/sources/{source_id}" if path else None,
        }

    def path(self, source_id: str) -> Path | None:
        name = self._names.get(source_id)
        if name is None:
            return None
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory) or not path.is_file():
            return None
        return path


def resolve_citations(
    text: str, evidence: dict[str, Evidence], sources: SourceStore
) -> list[dict[str, Any]]:
    resolved = []
    seen: set[str] = set()
    for match in CITATION_PATTERN.finditer(text):
        evidence_id = match[1]
        if evidence_id in seen or evidence_id not in evidence:
            continue
        seen.add(evidence_id)
        item = evidence[evidence_id]
        resolved.append(
            {
                **item.model_dump(),
                **sources.describe(item.source),
                "number": len(resolved) + 1,
            }
        )
    return resolved
