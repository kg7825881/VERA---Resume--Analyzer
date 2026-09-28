"""Immutable source bundle for extracted documents.

Extraction produces Markdown before any JD/resume interpretation takes place.
This module gives that exact Markdown a content-addressed source ID so every
downstream record and evidence chunk can point back to one durable source.
"""
from __future__ import annotations

import hashlib
from typing import Any


SOURCE_BUNDLE_VERSION = "1.0"


def normalize_markdown(markdown: Any) -> str:
    """Make equivalent line endings hash identically without changing content."""
    if not isinstance(markdown, str):
        return ""
    return markdown.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def source_hash(markdown: Any) -> str:
    """SHA-256 of normalized extracted Markdown, suitable for deduplication."""
    return hashlib.sha256(normalize_markdown(markdown).encode("utf-8")).hexdigest()


def build_source_bundle(
    *, document_id: str, markdown: Any, file_name: str, extraction_method: str,
    extraction_warnings: list[str] | None = None,
) -> dict:
    """Return the canonical, serialisable Stage 2 source contract.

    ``source_id`` is content-addressed and intentionally stable across
    re-uploads of identical extracted content. ``document_id`` remains the
    per-ingestion identity used by the persistence model.
    """
    content = normalize_markdown(markdown)
    digest = source_hash(content)
    return {
        "bundle_version": SOURCE_BUNDLE_VERSION,
        "document_id": document_id,
        "source_id": f"src_{digest}",
        "source_hash": digest,
        "file_name": file_name or "",
        "extraction_method": extraction_method or "",
        "extraction_warnings": list(extraction_warnings or []),
        "markdown": content,
    }
