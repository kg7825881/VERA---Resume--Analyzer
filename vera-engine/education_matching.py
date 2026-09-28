"""Deterministic recovery of explicit education evidence from resumes.

PDF/DOCX layouts can omit a valid degree from structured extraction.  This
module keeps the parsed entries and safely recovers only degree labels that
are explicitly written in retained resume text; it never infers a degree from
skills or job history.
"""
from __future__ import annotations

import re


_UNDERGRAD_ENGINEERING = re.compile(
    r"\b(?:"
    r"b\s*\.?\s*tech(?:nology)?\b|"
    r"b\s*\.?\s*e\s*\.?\b|"
    r"bachelor(?:\s*['’]\s*s|s)?\s+(?:of|in)\s+(?:technology|engineering)\b|"
    r"bachelor(?:\s*['’]\s*s|s)?\s+(?:degree\s+)?(?:in|of)\s+(?:"
    r"computer\s+science(?:\s+(?:and|&)\s+engineering)?|"
    r"information\s+technology|engineering|technology)\b"
    r")",
    re.IGNORECASE,
)
_MCA = re.compile(
    r"\b(?:m\s*\.?\s*c\s*\.?\s*a\b|master(?:\s*['’]\s*s)?\s+of\s+computer\s+applications)\b",
    re.IGNORECASE,
)


def verified_education_entries(candidate: dict) -> list[dict]:
    """Return parsed education plus explicitly stated degree equivalents."""
    entries = [item for item in (candidate.get("education") or []) if isinstance(item, dict)]
    values = " ".join(f"{item.get('degree_level', '')} {item.get('field', '')}" for item in entries)
    source = candidate.get("source_markdown")
    text = f"{values}\n{source}" if isinstance(source, str) else values
    if _UNDERGRAD_ENGINEERING.search(text):
        entries.append({"degree_level": "B.Tech", "field": "Engineering", "evidence_source": "resume_text"})
    if _MCA.search(text):
        entries.append({"degree_level": "MCA", "field": "Computer Applications", "evidence_source": "resume_text"})
    return entries


def education_keys(candidate: dict) -> set[str]:
    """Map verified education evidence to VERA's JD alternatives."""
    keys: set[str] = set()
    for item in verified_education_entries(candidate):
        clean = "".join(ch for ch in str(item.get("degree_level", "")).casefold() if ch.isalnum())
        field = str(item.get("field", "")).casefold()
        if clean in {"btech", "btechnology", "be", "bengineering"}:
            keys.add("undergraduate-engineering")
        elif clean in {"bachelor", "bachelors", "bachelorsdegree"} and any(
            token in field for token in ("technology", "engineering", "computer science", "information technology")
        ):
            keys.add("undergraduate-engineering")
        elif clean == "mca":
            keys.add("mca")
    return keys
