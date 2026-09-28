"""Deterministic scoring, eligibility, and tier-first ranking for semantic matches.

This is deliberately separate from the legacy scorer.  Its inputs are the
versioned Slice 1 contracts plus validated matching outputs, so its result is
identical regardless of whether a clear match came from a cross-encoder or an
ambiguous match came from browser-local WebLLM.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from common import (
    CandidateScoreBreakdown,
    CategoryScore,
    EligibilityResult,
    Requirement,
    RequirementMatch,
    ScoringProfile,
)


SCORING_POLICY_VERSION = "2026-09-26.1"
MANDATORY_COVERAGE_THRESHOLD = 0.60

# These are category importance defaults, not per-candidate model output. They
# are normalized over categories actually present in the JD policy.
DEFAULT_CATEGORY_WEIGHTS = {
    "mandatory_skills": 0.55,
    "relevant_experience": 0.15,
    "role_alignment": 0.10,
    "domain_alignment": 0.08,
    "preferred_skills": 0.05,
    "education": 0.04,
    "certifications": 0.03,
}
TIER_ORDER = {"excellent_match": 3, "strong_match": 2, "needs_review": 1, "ineligible": 0}


def _requirement(value: Requirement | dict) -> Requirement:
    if isinstance(value, Requirement):
        return value
    if not isinstance(value, dict):
        raise ValueError("Requirements must be Requirement objects or dictionaries.")
    return Requirement(
        id=value.get("id"), text=value.get("text"), skills=tuple(value.get("skills") or ()),
        category=value.get("category"), non_negotiable=bool(value.get("non_negotiable", False)),
        priority=value.get("priority", 1.0), source_section=value.get("source_section", ""),
        source_id=value.get("source_id", ""), alternative_group=value.get("alternative_group", ""),
    )


def _match(value: RequirementMatch | dict, candidate_id: str) -> RequirementMatch:
    if isinstance(value, RequirementMatch):
        match = value
    elif isinstance(value, dict):
        evidence_id = value.get("evidence_id")
        evidence_ids = tuple(value.get("evidence_ids") or (() if evidence_id is None else (evidence_id,)))
        match = RequirementMatch(
            requirement_id=value.get("requirement_id"), candidate_id=value.get("candidate_id", candidate_id),
            decision=value.get("decision"), confidence=value.get("confidence"), evidence_ids=evidence_ids,
            method=value.get("method", "cross_encoder"), reason=value.get("reason", ""),
        )
    else:
        raise ValueError("Matches must be RequirementMatch objects or dictionaries.")
    if match.candidate_id != candidate_id:
        raise ValueError("Match candidate isolation violation.")
    return match


def build_scoring_profile(role_id: str, requirements: Sequence[Requirement | dict]) -> ScoringProfile:
    """Build one deterministic profile for every candidate of a JD.

    Categories absent from a JD stay in the profile and receive full credit.
    This keeps the recruiter-facing scorecard stable across roles while making
    absence of a criterion neutral rather than a candidate penalty.
    """
    parsed = [_requirement(item) for item in requirements]
    if not parsed:
        raise ValueError("Cannot build a scoring profile without requirements.")
    weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    return ScoringProfile(
        profile_id=f"profile_{role_id}_{SCORING_POLICY_VERSION}", role_id=role_id, weights=weights,
        rationales={category: "Deterministic role-policy category weight." for category in weights},
        policy_version=SCORING_POLICY_VERSION,
    )


def merge_match_outcomes(
    candidate_id: str, final_matches: Iterable[dict], webllm_judgments: Iterable[dict] = (),
) -> list[RequirementMatch]:
    """Turn cross-encoder and validated WebLLM outputs into one strict contract.

    A requirement can be finalized once only. This prevents a browser judgment
    from overwriting a clear cross-encoder result or being counted twice.
    """
    merged: dict[str, RequirementMatch] = {}
    for item in final_matches:
        match = _match({**item, "candidate_id": candidate_id, "method": item.get("method", "cross_encoder")}, candidate_id)
        if match.requirement_id in merged:
            raise ValueError("Duplicate final match for a requirement.")
        merged[match.requirement_id] = match
    for item in webllm_judgments:
        evidence_id = item.get("evidence_id")
        match = _match({
            **item, "candidate_id": candidate_id, "method": "webllm",
            "evidence_ids": (() if item.get("decision") == "missing" else (evidence_id,)),
        }, candidate_id)
        if match.requirement_id in merged:
            raise ValueError("WebLLM cannot overwrite a finalized requirement.")
        merged[match.requirement_id] = match
    return list(merged.values())


def _match_value(match: RequirementMatch | None) -> float:
    if match is None or match.decision == "missing":
        return 0.0
    if match.decision == "weak":
        # Related evidence is intentionally yellow rather than a direct
        # green match, but approved deterministic groups and semantic review
        # receive the same 80% partial credit. Neither can clear a
        # direct-evidence gate.
        return 0.80
    return match.confidence


def _collapse_alternatives(requirements: Sequence[Requirement], matches: dict[str, RequirementMatch]) -> list[tuple[list[Requirement], RequirementMatch | None]]:
    """Alternative labels such as ETL/ELT get one best-of score, not two penalties."""
    groups: dict[str, list[Requirement]] = defaultdict(list)
    for requirement in requirements:
        groups[requirement.alternative_group or f"requirement:{requirement.id}"].append(requirement)
    collapsed = []
    for members in groups.values():
        best = max((matches.get(member.id) for member in members), key=_match_value, default=None)
        collapsed.append((members, best))
    return collapsed


def _coverage(requirements: Sequence[Requirement], matches: dict[str, RequirementMatch]) -> float:
    collapsed = _collapse_alternatives(requirements, matches)
    if not collapsed:
        return 1.0
    total = sum(max(member.priority for member in members) for members, _ in collapsed)
    supported = sum(
        max(member.priority for member in members)
        for members, match in collapsed
        if match is not None and match.decision == "matched"
    )
    return supported / total if total else 1.0


def _named_skill_confidence(requirements: Sequence[Requirement], matches: dict[str, RequirementMatch]) -> float:
    """Score each named skill exactly once for recruiter-visible skill coverage.

    Capability groups remain useful for finding related evidence and for
    eligibility checks, but they must not make the percentage disagree with
    the individual green/yellow/red chips shown to the recruiter.
    """
    if not requirements:
        return 1.0
    return sum(_match_value(matches.get(requirement.id)) for requirement in requirements) / len(requirements)


def _eligibility(requirements: Sequence[Requirement], matches: dict[str, RequirementMatch], final_score: float) -> EligibilityResult:
    mandatory = [item for item in requirements if item.category == "mandatory_skills"]
    non_negotiable = [item for item in requirements if item.non_negotiable]
    mandatory_coverage = _coverage(mandatory, matches)
    non_negotiable_coverage = _coverage(non_negotiable, matches)
    missing_non_negotiables = non_negotiable and non_negotiable_coverage < 1.0
    if missing_non_negotiables:
        return EligibilityResult(mandatory_coverage, non_negotiable_coverage, "ineligible", False, "One or more non-negotiable requirements are not directly supported.")
    if mandatory_coverage < MANDATORY_COVERAGE_THRESHOLD:
        return EligibilityResult(mandatory_coverage, non_negotiable_coverage, "needs_review", True, "Mandatory coverage is below the automatic-match threshold.")
    if final_score >= 85:
        return EligibilityResult(mandatory_coverage, non_negotiable_coverage, "excellent_match", True, "High score with required coverage.")
    return EligibilityResult(mandatory_coverage, non_negotiable_coverage, "strong_match", True, "Required coverage meets the automatic-match threshold.")


def score_candidate_matches(
    candidate_id: str, requirements: Sequence[Requirement | dict], matches: Sequence[RequirementMatch | dict],
    profile: ScoringProfile,
) -> CandidateScoreBreakdown:
    """Calculate a traceable score without calling an LLM or any external service."""
    parsed_requirements = [_requirement(item) for item in requirements]
    if profile.role_id == "":
        raise ValueError("Scoring profile needs a role id.")
    by_id = {item.id: item for item in parsed_requirements}
    if len(by_id) != len(parsed_requirements):
        raise ValueError("Requirement IDs must be unique.")
    parsed_matches = [_match(item, candidate_id) for item in matches]
    matches_by_id = {item.requirement_id: item for item in parsed_matches}
    if len(matches_by_id) != len(parsed_matches):
        raise ValueError("Only one final match is allowed per requirement.")
    unknown = sorted(set(matches_by_id) - set(by_id))
    if unknown:
        raise ValueError(f"Matches reference unknown requirements: {', '.join(unknown)}.")

    categories = []
    for category, weight in profile.weights.items():
        members = [item for item in parsed_requirements if item.category == category]
        if not members:
            categories.append(CategoryScore(
                category=category, weight=weight, confidence=1.0, contribution=100 * weight,
                matched_requirement_ids=(), weak_requirement_ids=(), missing_requirement_ids=(),
            ))
            continue
        if category in {"mandatory_skills", "preferred_skills"}:
            # Match the recruiter-visible formula exactly: green = 1,
            # yellow = .85, red = 0, divided by all named JD skills.
            confidence = _named_skill_confidence(members, matches_by_id)
        else:
            collapsed = _collapse_alternatives(members, matches_by_id)
            priority_total = sum(max(member.priority for member in group) for group, _ in collapsed)
            confidence = (
                sum(max(member.priority for member in group) * _match_value(match) for group, match in collapsed) / priority_total
                if priority_total else 0.0
            )
        category_matches = [matches_by_id.get(item.id) for item in members]
        categories.append(CategoryScore(
            category=category, weight=weight, confidence=confidence, contribution=100 * weight * confidence,
            matched_requirement_ids=tuple(item.requirement_id for item in category_matches if item and item.decision == "matched"),
            weak_requirement_ids=tuple(item.requirement_id for item in category_matches if item and item.decision == "weak"),
            missing_requirement_ids=tuple(item.id for item in members if not matches_by_id.get(item.id) or matches_by_id[item.id].decision == "missing"),
        ))
    final_score = round(sum(item.contribution for item in categories), 2)
    return CandidateScoreBreakdown(
        candidate_id=candidate_id, profile_id=profile.profile_id, final_score=final_score,
        categories=tuple(categories), eligibility=_eligibility(parsed_requirements, matches_by_id, final_score),
    )


def rank_candidate_breakdowns(breakdowns: Sequence[CandidateScoreBreakdown]) -> list[CandidateScoreBreakdown]:
    """Rank eligibility tier first, then score, then candidate ID for stable ties."""
    return sorted(
        breakdowns,
        key=lambda item: (-TIER_ORDER[item.eligibility.tier], -item.final_score, item.candidate_id),
    )
