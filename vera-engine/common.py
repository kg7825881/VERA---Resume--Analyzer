"""
common.py — shared utilities for generating document/candidate/role IDs
and timestamps, per the Phase 1 schema.
"""

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar


def new_id() -> str:
    """Generates a new unique id (used for document_id / candidate_id)."""
    return str(uuid.uuid4())


def slugify(text: str) -> str:
    """
    Turns a role title into a stable role_id slug.
    e.g. "AI Engineer" -> "ai-engineer", "AI/ML Engineer II" -> "ai-ml-engineer-ii"
    """
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown-role"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Slice 1 architecture contracts -------------------------------------------------
#
# These contracts deliberately use stdlib dataclasses instead of introducing a
# second runtime validation dependency. They are immutable, JSON-serialisable
# through to_dict(), and coexist with the dictionaries currently returned by
# extraction/scoring modules. Later slices can adopt them one boundary at a
# time without changing existing API response shapes.

REQUIREMENT_CATEGORIES = frozenset({
    "mandatory_skills", "preferred_skills", "relevant_experience", "education",
    "certifications", "role_alignment", "domain_alignment",
})
MATCH_DECISIONS = frozenset({"matched", "weak", "missing"})
MATCH_METHODS = frozenset({"exact", "lexical", "cross_encoder", "webllm", "manual"})
ELIGIBILITY_TIERS = ("ineligible", "needs_review", "strong_match", "excellent_match")


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _bounded_confidence(value: Any, field_name: str = "confidence") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number from 0 through 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number from 0 through 1.") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"{field_name} must be a number from 0 through 1.")
    return parsed


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_requirement_id(role_id: str, category: str, text: str, occurrence: int = 1) -> str:
    """Return a stable, content-addressed identifier for one JD requirement.

    ``occurrence`` only differentiates duplicate wording inside the same JD;
    callers should enumerate duplicates in source order. The short digest keeps
    IDs readable in an HR UI while still making changes to role/category/text
    invalidate downstream matching caches.
    """
    clean_role = _clean_text(role_id, "role_id")
    clean_category = _clean_text(category, "category")
    clean_text = _clean_text(text, "text")
    if occurrence < 1:
        raise ValueError("occurrence must be at least 1.")
    digest = _content_hash("\x1f".join((clean_role, clean_category, clean_text.casefold(), str(occurrence))))[:12]
    return f"req_{slugify(clean_category)}_{digest}"


@dataclass(frozen=True)
class Requirement:
    """A traceable, policy-ready requirement extracted from one JD."""

    id: str
    text: str
    skills: tuple[str, ...]
    category: str
    non_negotiable: bool = False
    priority: float = 1.0
    source_section: str = ""
    source_id: str = ""
    alternative_group: str = ""

    def __post_init__(self):
        object.__setattr__(self, "id", _clean_text(self.id, "requirement id"))
        object.__setattr__(self, "text", _clean_text(self.text, "requirement text"))
        if self.category not in REQUIREMENT_CATEGORIES:
            raise ValueError(f"Unknown requirement category: {self.category!r}.")
        if isinstance(self.priority, bool) or float(self.priority) <= 0:
            raise ValueError("priority must be a positive number.")
        object.__setattr__(self, "priority", float(self.priority))
        cleaned_skills = tuple(dict.fromkeys(_clean_text(skill, "requirement skill") for skill in self.skills))
        object.__setattr__(self, "skills", cleaned_skills)
        if self.alternative_group:
            object.__setattr__(self, "alternative_group", _clean_text(self.alternative_group, "alternative_group"))

    def to_dict(self) -> dict:
        return {**asdict(self), "skills": list(self.skills)}


@dataclass(frozen=True)
class EvidenceChunk:
    """A candidate-isolated, section-aware passage sent to matching models."""

    evidence_id: str
    candidate_id: str
    section: str
    text: str
    page: int | None = None
    source_id: str = ""
    content_hash: str = ""
    source_label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "evidence_id", _clean_text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "candidate_id", _clean_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "section", _clean_text(self.section, "section"))
        object.__setattr__(self, "text", _clean_text(self.text, "evidence text"))
        if self.page is not None and (isinstance(self.page, bool) or int(self.page) < 1):
            raise ValueError("page must be a positive integer when present.")
        if self.page is not None:
            object.__setattr__(self, "page", int(self.page))
        object.__setattr__(self, "content_hash", self.content_hash or _content_hash(self.text))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RequirementMatch:
    """A validated semantic decision for one requirement and one candidate."""

    requirement_id: str
    candidate_id: str
    decision: str
    confidence: float
    evidence_ids: tuple[str, ...] = ()
    method: str = "cross_encoder"
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "requirement_id", _clean_text(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "candidate_id", _clean_text(self.candidate_id, "candidate_id"))
        if self.decision not in MATCH_DECISIONS:
            raise ValueError(f"Unknown match decision: {self.decision!r}.")
        if self.method not in MATCH_METHODS:
            raise ValueError(f"Unknown match method: {self.method!r}.")
        object.__setattr__(self, "confidence", _bounded_confidence(self.confidence))
        ids = tuple(dict.fromkeys(_clean_text(value, "evidence_id") for value in self.evidence_ids))
        if self.decision == "matched" and not ids:
            raise ValueError("A matched requirement must reference at least one evidence_id.")
        object.__setattr__(self, "evidence_ids", ids)

    def to_dict(self) -> dict:
        return {**asdict(self), "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class ScoringProfile:
    """The one versioned category-weight policy shared by every candidate for a JD."""

    profile_id: str
    role_id: str
    weights: dict[str, float]
    rationales: dict[str, str] = field(default_factory=dict)
    policy_version: str = "1"
    model_version: str = ""
    prompt_version: str = ""

    def __post_init__(self):
        object.__setattr__(self, "profile_id", _clean_text(self.profile_id, "profile_id"))
        object.__setattr__(self, "role_id", _clean_text(self.role_id, "role_id"))
        weights = {key: float(value) for key, value in self.weights.items() if float(value) != 0}
        if not weights or any(key not in REQUIREMENT_CATEGORIES for key in weights):
            raise ValueError("weights must contain known scoring categories.")
        if any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 0.0001:
            raise ValueError("weights must be non-negative and sum to 1.0.")
        object.__setattr__(self, "weights", weights)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CategoryScore:
    category: str
    weight: float
    confidence: float
    contribution: float
    matched_requirement_ids: tuple[str, ...] = ()
    weak_requirement_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.category not in REQUIREMENT_CATEGORIES:
            raise ValueError(f"Unknown scoring category: {self.category!r}.")
        object.__setattr__(self, "weight", _bounded_confidence(self.weight, "weight"))
        object.__setattr__(self, "confidence", _bounded_confidence(self.confidence))
        if isinstance(self.contribution, bool) or float(self.contribution) < 0:
            raise ValueError("contribution must be a non-negative number.")
        object.__setattr__(self, "contribution", float(self.contribution))

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("matched_requirement_ids", "weak_requirement_ids", "missing_requirement_ids"):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class EligibilityResult:
    mandatory_coverage: float
    non_negotiable_coverage: float
    tier: str
    eligible: bool
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "mandatory_coverage", _bounded_confidence(self.mandatory_coverage, "mandatory_coverage"))
        object.__setattr__(self, "non_negotiable_coverage", _bounded_confidence(self.non_negotiable_coverage, "non_negotiable_coverage"))
        if self.tier not in ELIGIBILITY_TIERS:
            raise ValueError(f"Unknown eligibility tier: {self.tier!r}.")
        if self.eligible != (self.tier != "ineligible"):
            raise ValueError("eligible must agree with the eligibility tier.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateScoreBreakdown:
    candidate_id: str
    profile_id: str
    final_score: float
    categories: tuple[CategoryScore, ...]
    eligibility: EligibilityResult
    CONTRACT_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self):
        object.__setattr__(self, "candidate_id", _clean_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "profile_id", _clean_text(self.profile_id, "profile_id"))
        if isinstance(self.final_score, bool) or not 0 <= float(self.final_score) <= 100:
            raise ValueError("final_score must be a number from 0 through 100.")
        object.__setattr__(self, "final_score", float(self.final_score))

    def to_dict(self) -> dict:
        return {
            "contract_version": self.CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "profile_id": self.profile_id,
            "final_score": self.final_score,
            "categories": [category.to_dict() for category in self.categories],
            "eligibility": self.eligibility.to_dict(),
        }
