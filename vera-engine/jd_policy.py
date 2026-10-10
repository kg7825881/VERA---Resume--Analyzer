"""Build stable, recruiter-adjustable JD requirement policy metadata."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from common import Requirement, stable_requirement_id
from role_policies import ROLE_POLICY_VERSION, default_groups_for_role
from skill_aliases import canonical_skill
from capability_groups import CAPABILITY_GROUP_VERSION, capability_group_for_skill


# 2.5 expands the reusable capability taxonomy. A policy with an older
# schema is rebuilt before the next semantic screening run so its visible JD
# skills are assigned to the newest calibrated score groups.
POLICY_SCHEMA_VERSION = "2.5"

_FIELD_SPECS = (
    ("mandatory_skills", "mandatory_skills", "Required qualifications"),
    ("mandatory_domain_requirements", "domain_alignment", "Required qualifications"),
    ("mandatory_role_specific_requirements", "role_alignment", "Required qualifications"),
    ("preferred_technical_skills", "preferred_skills", "Preferred qualifications"),
    ("soft_preferred_skills", "preferred_skills", "Preferred qualifications"),
    ("industry_keywords", "domain_alignment", "Preferred qualifications"),
)


def _clean_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    output, seen = [], set()
    for value in values:
        if isinstance(value, str) and value.strip() and value.casefold() not in seen:
            output.append(value.strip())
            seen.add(value.casefold())
    return output


def _normalize_overrides(overrides: Any) -> dict[str, dict]:
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("recruiter_overrides must be an object keyed by requirement id.")
    normalized = {}
    for requirement_id, value in overrides.items():
        if not isinstance(requirement_id, str) or not requirement_id.strip() or not isinstance(value, dict):
            raise ValueError("Each recruiter override needs a requirement id and object value.")
        allowed = {key: value[key] for key in ("non_negotiable", "priority") if key in value}
        if not allowed or set(value) - set(allowed):
            raise ValueError("Overrides may contain only non_negotiable and priority.")
        if "non_negotiable" in allowed and not isinstance(allowed["non_negotiable"], bool):
            raise ValueError("non_negotiable override must be boolean.")
        if "priority" in allowed and (isinstance(allowed["priority"], bool) or float(allowed["priority"]) <= 0):
            raise ValueError("priority override must be a positive number.")
        if "priority" in allowed:
            allowed["priority"] = float(allowed["priority"])
        normalized[requirement_id.strip()] = allowed
    return normalized


def build_requirement_metadata(jd: dict, recruiter_overrides: dict | None = None) -> dict:
    """Convert legacy extracted JD fields into stable requirement objects."""
    role_id = str(jd.get("role_id") or "").strip()
    if not role_id:
        raise ValueError("JD role_id is required to build requirement metadata.")
    role_title = str(jd.get("role_title") or role_id)
    source_id = str(jd.get("source_id") or "")
    defaults = default_groups_for_role(role_title)
    default_by_label = {canonical_skill(label).casefold(): group for group in defaults for label in group.labels}
    occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)
    requirements: list[Requirement] = []
    for field, category, section in _FIELD_SPECS:
        for text in _clean_strings(jd.get(field)):
            key = (category, text.casefold())
            occurrences[key] += 1
            default = default_by_label.get(canonical_skill(text).casefold())
            # Score related tools as one capability while retaining each tool
            # as visible, traceable JD evidence. Role-specific alternatives
            # take precedence where they are more precise.
            capability_group = capability_group_for_skill(text)
            requirements.append(Requirement(
                # Preserve the JD author's Required/Preferred distinction.
                # The compact role-policy core controls the hard gate only;
                # it does not silently move a JD requirement into Preferred.
                id=stable_requirement_id(role_id, category, text, occurrences[key]), text=text,
                skills=(canonical_skill(text),), category=category,
                non_negotiable=bool(default and default.non_negotiable and category == "mandatory_skills"),
                source_section=section, source_id=source_id,
                alternative_group=(default.alternative_group if default and default.alternative_group else capability_group),
            ))
    # Title fit is a first-class, deterministic requirement.  The matcher
    # expands only approved title families; it never invents broad synonyms.
    title_text = f"Approved job-title family for {role_title}"
    requirements.append(Requirement(
        id=stable_requirement_id(role_id, "role_alignment", title_text), text=title_text, skills=(),
        category="role_alignment", source_section="Role title", source_id=source_id,
    ))
    years = jd.get("min_years_experience")
    if years is not None:
        try:
            years_number = float(years)
        except (TypeError, ValueError):
            years_number = -1
        if years_number >= 0:
            text = f"{years_number:g}+ years relevant experience"
            requirements.append(Requirement(
                id=stable_requirement_id(role_id, "relevant_experience", text), text=text,
                skills=(), category="relevant_experience", source_section="Required qualifications", source_id=source_id,
            ))
    education_options = [item for item in (jd.get("education_requirements") or []) if isinstance(item, dict)]
    education_group = "education-degree-options" if len(education_options) > 1 else ""
    for index, education in enumerate(education_options, start=1):
        if not isinstance(education, dict):
            continue
        parts = (str(education.get("degree_level") or "").strip(), str(education.get("field") or "").strip())
        text = " in ".join(part for part in parts if part)
        if text:
            requirements.append(Requirement(
                id=stable_requirement_id(role_id, "education", text, index), text=text, skills=(),
                category="education", source_section="Education", source_id=source_id,
                alternative_group=education_group,
            ))
    for index, certification in enumerate(_clean_strings(jd.get("relevant_certifications")), start=1):
        requirements.append(Requirement(
            id=stable_requirement_id(role_id, "certifications", certification, index), text=certification,
            skills=(certification,), category="certifications", source_section="Certifications", source_id=source_id,
        ))
    metadata = {
        "schema_version": POLICY_SCHEMA_VERSION, "role_policy_version": ROLE_POLICY_VERSION,
        "capability_group_version": CAPABILITY_GROUP_VERSION,
        "requirements": [requirement.to_dict() for requirement in requirements], "recruiter_overrides": {},
    }
    return apply_recruiter_overrides_to_metadata(metadata, recruiter_overrides, reject_unknown=True)


def refresh_stale_requirement_metadata(jd: dict) -> dict | None:
    """Return rebuilt metadata only when a persisted policy is explicitly stale.

    Test fixtures and integrations may supply requirements without a schema
    version, so those remain untouched.  Persisted, versioned JD policies are
    safe to rebuild from the JD's source fields while retaining valid recruiter
    overrides.
    """
    metadata = jd.get("requirement_metadata")
    if not isinstance(metadata, dict) or not metadata.get("schema_version"):
        return None
    if metadata.get("schema_version") == POLICY_SCHEMA_VERSION:
        return None
    rebuilt = build_requirement_metadata(jd)
    # Requirement IDs can legitimately change when an old policy is repaired.
    # Retain only overrides still associated with a current requirement.
    return apply_recruiter_overrides_to_metadata(
        rebuilt, metadata.get("recruiter_overrides"), reject_unknown=False,
    )


def apply_recruiter_overrides_to_metadata(metadata: dict, recruiter_overrides: dict | None, *, reject_unknown: bool) -> dict:
    """Apply overrides to existing requirement metadata without changing IDs."""
    overrides = _normalize_overrides(recruiter_overrides)
    requirements = metadata.get("requirements") if isinstance(metadata, dict) else None
    if not isinstance(requirements, list):
        raise ValueError("requirement metadata is malformed.")
    valid_ids = {item.get("id") for item in requirements if isinstance(item, dict)}
    unknown = sorted(set(overrides) - valid_ids)
    if unknown and reject_unknown:
        raise ValueError(f"Unknown requirement override ids: {', '.join(unknown)}.")
    retained = {key: value for key, value in overrides.items() if key in valid_ids}
    updated = []
    for item in requirements:
        current = dict(item)
        current.update(retained.get(current.get("id"), {}))
        updated.append(current)
    return {
        "schema_version": metadata.get("schema_version", POLICY_SCHEMA_VERSION),
        "role_policy_version": metadata.get("role_policy_version", ROLE_POLICY_VERSION),
        "requirements": updated, "recruiter_overrides": retained,
    }
