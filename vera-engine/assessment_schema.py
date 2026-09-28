"""Small, stable read model for reviewing one JD/candidate pairing.

The persistence records intentionally retain extraction and scoring detail.  API
clients that only need to render a decision, however, should not have to know
about every extractor field or reconstruct a decision from multiple records.
This module projects those records into a compact, versioned assessment shape.
"""
from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "1.0"

_REQUIREMENT_FIELDS = (
    ("mandatory_skills", "mandatory"),
    ("mandatory_domain_requirements", "mandatory"),
    ("mandatory_role_specific_requirements", "mandatory"),
    ("preferred_technical_skills", "preferred"),
    ("soft_preferred_skills", "role_specific"),
    ("industry_keywords", "preferred"),
)


def _strings(values: Any, limit: int | None = None) -> list[str]:
    """Return non-empty, de-duplicated display strings in source order."""
    if not isinstance(values, list):
        return []
    output, seen = [], set()
    for value in values:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        key = clean.casefold()
        if clean and key not in seen:
            output.append(clean)
            seen.add(key)
        if limit is not None and len(output) >= limit:
            break
    return output


def _number(value: Any, default: int | float = 0) -> int | float:
    """Return a JSON numeric value without allowing malformed source data through."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else default
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def compact_jd(jd: dict) -> dict:
    """Project an extracted JD into only the criteria used in an assessment."""
    requirements = []
    for field, kind in _REQUIREMENT_FIELDS:
        for value in _strings(jd.get(field)):
            requirements.append({"name": value, "kind": kind})

    minimum_years = jd.get("min_years_experience")
    if minimum_years is not None:
        years = _number(minimum_years, default=-1)
        if years >= 0:
            requirements.append({"name": f"{years:g}+ years experience", "kind": "experience"})

    for degree in jd.get("education_requirements") or []:
        if not isinstance(degree, dict):
            continue
        level = str(degree.get("degree_level") or "").strip()
        field = str(degree.get("field") or "").strip()
        name = " in ".join(part for part in (level, field) if part)
        if name:
            requirements.append({"name": name, "kind": "education"})

    return {
        "id": jd.get("role_id", ""),
        "title": jd.get("role_title", ""),
        "target_titles": _strings(jd.get("target_job_titles")),
        "requirements": requirements,
    }


def compact_candidate(candidate: dict) -> dict:
    """Return a resume summary, deliberately excluding raw resume text and PII."""
    latest_role = (candidate.get("experience") or [{}])[0] or {}
    return {
        "id": candidate.get("candidate_id", ""),
        "name": candidate.get("candidate_name", ""),
        "headline": candidate.get("current_role_title_from_summary") or latest_role.get("title", ""),
        "years_experience": _number(candidate.get("total_years_experience")),
        # A cap keeps an unusually verbose resume from turning this compact
        # contract into another raw extraction payload.
        "skills": _strings(candidate.get("skills_all_sources") or candidate.get("skills"), limit=30),
    }


def compact_assessment(score: dict) -> dict:
    """Summarise a scoring result without leaking the full evidence payload."""
    dimensions = []
    for key, category in (score.get("category_scores") or {}).items():
        if not isinstance(category, dict) or category.get("not_applicable"):
            continue
        dimensions.append({
            "key": key,
            "score": _number(category.get("score")),
            "matched": _strings(category.get("matched")),
            "missing": _strings(category.get("missing")),
        })
    return {
        "score": _number(score.get("final_score")),
        "eligible": not bool(score.get("hard_gate_failed")),
        "decision_reason": score.get("hard_gate_reason", ""),
        "dimensions": dimensions,
        "scored_at": score.get("scored_at", ""),
    }


def build_assessment(jd: dict, candidate: dict, score: dict) -> dict:
    """Build the public assessment contract for exactly one scored pairing."""
    return {
        "schema_version": SCHEMA_VERSION,
        "jd": compact_jd(jd),
        "candidate": compact_candidate(candidate),
        "assessment": compact_assessment(score),
    }
