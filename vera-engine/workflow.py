"""Deterministic validation and content-addressed cache helpers for analysis runs."""
from __future__ import annotations

import hashlib
import json


# Bump this whenever scoring semantics change so old outcomes cannot be reused.
SCORING_RULES_VERSION = "2026-09-26.2"

_JD_FIELDS = (
    "role_title", "target_job_titles", "mandatory_skills", "mandatory_domain_requirements",
    "mandatory_role_specific_requirements", "preferred_technical_skills", "soft_preferred_skills",
    "industry_keywords", "min_years_experience", "education_requirements",
)
_CANDIDATE_FIELDS = (
    "candidate_id", "skills", "skills_all_sources", "current_role_title_from_summary",
    "total_years_experience", "experience", "education", "certifications", "projects", "skill_evidence",
)


def validate_analysis_request(candidate_ids: list[str] | None, available_ids: set[str]) -> list[str]:
    """Return user-facing validation errors before creating work or calling a model."""
    if candidate_ids is None:
        return []
    clean_ids = [candidate_id for candidate_id in candidate_ids if isinstance(candidate_id, str) and candidate_id.strip()]
    if len(clean_ids) != len(candidate_ids):
        return ["candidate_ids must contain non-empty strings."]
    duplicates = sorted({candidate_id for candidate_id in clean_ids if clean_ids.count(candidate_id) > 1})
    if duplicates:
        return [f"candidate_ids contains duplicates: {', '.join(duplicates)}."]
    unknown = sorted(set(clean_ids) - available_ids)
    if unknown:
        return [f"Unknown candidate_ids: {', '.join(unknown)}."]
    return []


def cache_key(jd: dict, candidate: dict, *, audit_mode: bool = False) -> str:
    """Fingerprint every scoring-relevant input; names and timestamps are excluded."""
    payload = {
        "rules": SCORING_RULES_VERSION,
        "audit_mode": bool(audit_mode),
        "jd": {field: jd.get(field) for field in _JD_FIELDS},
        "candidate": {field: candidate.get(field) for field in _CANDIDATE_FIELDS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
