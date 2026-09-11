"""Section-aware, deterministic Job Description extraction for TalentLens.

Mandatory/preferred classification comes exclusively from source headings.  This is
intentionally independent of Ollama (and therefore repeatable for identical input).
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from jd_skills import DOMAIN_PATTERNS, ROLE_PATTERNS, SKILL_PATTERNS, TARGET_TITLE_PATTERNS, find_catalog_items

logger = logging.getLogger("talentlens.jd_extractor")

_HEADINGS = {
    "responsibilities": re.compile(r"^(?:key )?(?:responsibilities|what you(?:'ll| will) do)\s*:?$", re.I),
    "required": re.compile(r"^(?:required|minimum|must[- ]have)(?:\s+(?:skills?|qualifications?|requirements?|&|and))*\s*:?$", re.I),
    "preferred": re.compile(r"^(?:preferred|nice[- ]to[- ]have|desired)(?:\s+(?:skills?|qualifications?|requirements?|&|and))*\s*:?$", re.I),
    "education": re.compile(r"^(?:education|education requirements?)\s*:?$", re.I),
    "certifications": re.compile(r"^(?:certifications?|licenses?)\s*:?$", re.I),
}
_STOPWORDS = re.compile(r"\b(?:experience|knowledge|understanding|familiarity|skills?|ability|proficiency|expertise)\b", re.I)
# Requirement sections commonly say either "3-5 years of experience", "3-8
# years as a Business Analyst", or "7-10 years of hands-on DevOps experience".
_YEARS = re.compile(r"\b(\d{1,2})\s*(?:[-–—]|to)\s*(\d{1,2})?\s*\+?\s*years?\b|\b(\d{1,2})\s*\+\s*years?\b", re.I)
_DEGREE = re.compile(r"\b(bachelor(?:'s)?|master(?:'s)?|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?s\.?|m\.?s\.?|bca|mca|mba|ph\.?d)\b(?:\s+(?:in|of)\s+([^,;.\n]+))?", re.I)


def _clean_lines(text: str) -> list[str]:
    """Normalise DOCX/PDF text while preserving one logical line per source line.

    pymupdf4llm emits DOCX headings as ``# Heading`` / ``### Heading``.  Those
    markers are presentation syntax, not part of the JD heading, so remove them
    before the deterministic section rules run.
    """
    cleaned = []
    for line in (text or "").replace("\r", "").split("\n"):
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def split_sections(jd_text: str) -> dict[str, list[str]]:
    """Partition text by headings; no section content crosses a later heading."""
    sections = {key: [] for key in _HEADINGS}
    current = "other"
    for line in _clean_lines(jd_text):
        plain = line.strip("•*- \t")
        matched = next((name for name, pattern in _HEADINGS.items() if pattern.match(plain)), None)
        if matched:
            current = matched
        else:
            sections.setdefault(current, []).append(plain)
    return sections


def _section_text(lines: list[str]) -> str:
    return "\n".join(lines)


def _requirement_groups(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    text = _section_text(lines)
    # Emit only atomic, named keywords.  Requirement sentences, degrees, and years
    # belong in their own fields and must never become matchable skill pills.
    technical = find_catalog_items(text, SKILL_PATTERNS)
    domains = find_catalog_items(text, DOMAIN_PATTERNS)
    role = find_catalog_items(text, ROLE_PATTERNS)
    return technical, domains, role


def _dedupe(items: list[str]) -> list[str]:
    seen, result = set(), []
    for item in items:
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _role_title(lines: list[str]) -> str:
    for line in lines[:12]:
        match = re.match(r"(?:job title|position|role)\s*:\s*(.+)", line, re.I)
        if match:
            return match.group(1).strip()
    # A document may put the title in its first heading rather than in a labelled
    # metadata line. Avoid treating generic headings such as "About Company" as one.
    if lines and not re.match(r"(?:about|responsibilities|required|preferred|education|certifications)\b", lines[0], re.I):
        return lines[0].strip()
    return ""


def _experience_years(text: str) -> tuple[int | None, int | None]:
    match = _YEARS.search(text or "")
    if not match:
        return None, None
    low = int(match.group(1) or match.group(3))
    high = int(match.group(2)) if match.group(2) else None
    return low, high


def _education(lines: list[str]) -> list[dict]:
    output = []
    for match in _DEGREE.finditer(_section_text(lines)):
        output.append({"degree_level": match.group(1), "field": (match.group(2) or "").strip(), "required": True})
    return output


def extract_structured_jd(jd_text: str) -> dict:
    """Return a stable schema from raw JD text; this function performs no I/O or LLM calls."""
    sections = split_sections(jd_text)
    mandatory_skills, mandatory_domains, mandatory_role = _requirement_groups(sections["required"])
    preferred_skills, preferred_domains, preferred_role = _requirement_groups(sections["preferred"])
    min_years, max_years = _experience_years(_section_text(sections["required"]) or jd_text)
    result = {
        "role_title": _role_title(_clean_lines(jd_text)),
        "department": "",
        "target_job_titles": _dedupe([_role_title(_clean_lines(jd_text))] + find_catalog_items(_section_text(sections["required"]), TARGET_TITLE_PATTERNS)),
        "mandatory_skills": mandatory_skills,
        "mandatory_domain_requirements": mandatory_domains,
        "mandatory_role_specific_requirements": mandatory_role,
        "preferred_technical_skills": preferred_skills,
        "soft_preferred_skills": preferred_role,
        "industry_keywords": preferred_domains,
        "min_years_experience": min_years,
        "max_years_experience": max_years,
        "education_requirements": _education(sections["education"] + sections["required"]),
        "relevant_certifications": _dedupe(sections["certifications"]),
        "responsibilities": sections["responsibilities"],
    }
    return result


def ingest_jd(file_path: str) -> dict:
    """Extract document text using the existing extractor and attach ingestion metadata."""
    from extractor import extract_text
    from common import new_id, now_iso, slugify
    raw_text, warnings, method = extract_text(file_path)
    structured = extract_structured_jd(raw_text)
    structured.update({"document_id": new_id(), "role_id": slugify(structured["role_title"]),
                       "file_name": os.path.basename(file_path), "extraction_method": method,
                       "extraction_warnings": warnings, "uploaded_at": now_iso()})
    return structured
