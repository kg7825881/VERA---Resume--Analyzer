"""Server-owned semantic-session orchestration for the browser WebLLM handoff."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Callable

from common import ScoringProfile
from common import now_iso
from cross_encoder import CrossEncoderMatcher, route_candidate_matches
from decision_scoring import build_scoring_profile, merge_match_outcomes, score_candidate_matches
from job_title_matcher import score_job_titles
from retrieval import SemanticEvidenceIndex
from capability_groups import (
    explicit_capability_evidence,
    partial_capability_evidence,
    requires_named_tool_evidence,
)
from skill_aliases import canonical_skill, explicit_skill_in_values
from education_matching import education_keys
from webllm_contract import validate_ambiguity_judgments
from scorer import calculate_job_fit


SEMANTIC_SESSION_VERSION = "1.0"
LEGACY_DISPLAY_MAX = {
    "mandatory_skills": 30, "relevant_experience": 20, "education": 25,
    "role_alignment": 10, "job_title_match": 10, "domain_alignment": 10,
    "soft_skills": 10, "preferred_skills": 5, "certifications": 5,
}
LEGACY_CATEGORY_KEY = {
    "role_alignment": "job_title_match",
    "domain_alignment": "soft_skills",
}


def _candidate_skills(candidate: dict) -> set[str]:
    # ``skills_all_sources`` is the parsed resume skill list, while
    # ``skill_evidence`` contains verified skills recovered from experience,
    # project and summary sections.  Both are explicit resume evidence; using
    # both prevents a formatting quirk in a Skills heading from hiding skills
    # that were successfully extracted elsewhere in the same document.
    values = [*(candidate.get("skills") or []), *(candidate.get("skills_all_sources") or [])]
    values.extend(
        item.get("skill", "") for item in (candidate.get("skill_evidence") or [])
        if isinstance(item, dict)
    )
    # A named technology can appear solely in a project or experience bullet.
    # The source markdown is the verified resume text retained by ingestion,
    # so include it for word-bounded explicit-term matching.  This does not
    # turn broad semantic similarity into an exact match.
    if isinstance(candidate.get("source_markdown"), str):
        values.append(candidate["source_markdown"])
    return {canonical_skill(value).casefold() for value in values if isinstance(value, str) and value.strip()}


def _education_keys(candidate: dict) -> set[str]:
    """Return degree alternatives backed by extracted or source resume text.

    Education is a deterministic fact.  A layout/parser issue must not turn a
    verified ``B.Tech`` line in the retained resume source into a zero score.
    The source fallback only detects degree labels; it does not infer a degree
    from unrelated technical terms.
    """
    return education_keys(candidate)


def _deterministic_match(requirement: dict, candidate: dict) -> dict | None:
    """Finalize measurable facts before semantic evidence retrieval.

    Returning ``None`` leaves a skill for the cross-encoder/WebLLM cascade;
    a returned missing decision means a fact was deterministically absent.
    """
    candidate_id = str(candidate["candidate_id"])
    category, text = requirement.get("category"), requirement.get("text", "")
    derived_id = f"derived:{candidate_id}:{requirement['id']}"
    if category == "relevant_experience":
        import re
        required = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years", text.casefold())
        years = candidate.get("total_years_experience")
        try:
            actual = float(years)
        except (TypeError, ValueError):
            actual = None
        if not required or actual is None:
            return {"requirement_id": requirement["id"], "decision": "missing", "confidence": 1.0,
                    "evidence_ids": [], "method": "exact", "reason": "Resume has no deterministically extracted total experience."}
        minimum = float(required.group(1))
        return {"requirement_id": requirement["id"], "decision": "matched" if actual >= minimum else "missing",
                "confidence": 1.0, "evidence_ids": [derived_id] if actual >= minimum else [], "method": "exact",
                "reason": f"Deterministic experience check: {actual:g} years versus {minimum:g} required."}
    if category == "role_alignment":
        title = text.removeprefix("Approved job-title family for ").strip()
        result = score_job_titles(candidate, title)
        confidence = float(result["contribution"])
        return {"requirement_id": requirement["id"], "decision": "matched" if confidence else "missing",
                "confidence": confidence or 1.0, "evidence_ids": [derived_id] if confidence else [], "method": "exact",
                "reason": result.get("judge_reason", "Deterministic title-family check.")}
    if category == "education":
        required_key = "mca" if "mca" in text.casefold() else "undergraduate-engineering" if any(token in text.casefold() for token in ("b.tech", "b.e", "bachelor")) else ""
        if required_key:
            present = required_key in _education_keys(candidate)
            return {"requirement_id": requirement["id"], "decision": "matched" if present else "missing", "confidence": 1.0,
                    "evidence_ids": [derived_id] if present else [], "method": "exact",
                    "reason": "Deterministic education alternative check."}
    if category in {"mandatory_skills", "preferred_skills"} and requirement.get("skills"):
        required_skill = canonical_skill(requirement["skills"][0]).casefold()
        candidate_skills = _candidate_skills(candidate)
        explicit_skill = explicit_skill_in_values(requirement["skills"][0], candidate_skills)
        if required_skill in candidate_skills or explicit_skill:
            return {"requirement_id": requirement["id"], "decision": "matched", "confidence": 1.0,
                    "evidence_ids": [derived_id], "method": "exact",
                    "reason": f"Approved explicit skill match: {canonical_skill(requirement['skills'][0])}."}
        reason = explicit_capability_evidence(required_skill, candidate_skills)
        if reason:
            return {"requirement_id": requirement["id"], "decision": "weak", "confidence": 0.80,
                    "evidence_ids": [derived_id], "method": "lexical", "reason": reason}
        reason = partial_capability_evidence(required_skill, candidate_skills)
        if reason:
            return {"requirement_id": requirement["id"], "decision": "weak", "confidence": 0.80,
                    "evidence_ids": [derived_id], "method": "lexical",
                    "reason": reason}
        if requires_named_tool_evidence(required_skill):
            return {
                "requirement_id": requirement["id"], "decision": "missing", "confidence": 1.0,
                "evidence_ids": [], "method": "exact",
                "reason": (
                    "Named tooling requires explicit product evidence; generic capability "
                    "evidence is not sufficient."
                ),
            }
    return None


def _enforce_deterministic_skill_levels(requirements: list[dict], matches: list) -> list:
    """Reserve green named-skill matches for deterministic resume evidence.

    The cross-encoder and WebLLM are useful for related phrasing, but their
    semantic similarity must not claim that a JD's exact skill was directly
    evidenced. Deterministic exact aliases and approved capability rules run
    before this point and remain green; semantic-only positives become fixed
    yellow partial credit.
    """
    requirements_by_id = {item["id"]: item for item in requirements}

    def is_semantic_skill_match(requirement_id: str, decision: str, method: str) -> bool:
        requirement = requirements_by_id.get(requirement_id, {})
        return (
            decision == "matched"
            and method in {"cross_encoder", "webllm"}
            and requirement.get("category") in {"mandatory_skills", "preferred_skills"}
        )

    normalized = []
    for match in matches:
        if isinstance(match, dict):
            if is_semantic_skill_match(match.get("requirement_id", ""), match.get("decision", ""), match.get("method", "")):
                normalized.append({
                    **match, "decision": "weak", "confidence": 0.80,
                    "reason": f"{match.get('reason', '').strip()} Semantic evidence is related; direct named-skill evidence was not found.".strip(),
                })
            else:
                normalized.append(match)
        elif is_semantic_skill_match(match.requirement_id, match.decision, match.method):
            normalized.append(replace(
                match, decision="weak", confidence=0.80,
                reason=f"{match.reason.strip()} Semantic evidence is related; direct named-skill evidence was not found.".strip(),
            ))
        else:
            normalized.append(match)
    return normalized


def create_semantic_session(
    session_id: str, role: dict, candidate: dict, *, matcher: CrossEncoderMatcher | None = None,
    evidence_index_factory: Callable[[dict], SemanticEvidenceIndex] = SemanticEvidenceIndex,
) -> dict:
    """Create one candidate-isolated session, stopping only for WebLLM ambiguity."""
    metadata = role.get("requirement_metadata") or {}
    requirements = metadata.get("requirements") if isinstance(metadata, dict) else None
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("JD requirement metadata is missing; re-upload the JD before semantic analysis.")
    candidate_id = str(candidate.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("Candidate ID is required for semantic analysis.")
    index = evidence_index_factory(candidate)
    deterministic = [result for item in requirements if (result := _deterministic_match(item, candidate)) is not None]
    remaining = [item for item in requirements if item["id"] not in {result["requirement_id"] for result in deterministic}]
    evidence_by_requirement = {item["id"]: index.retrieve(item["text"], top_k=3) for item in remaining}
    routed = route_candidate_matches(candidate_id, remaining, evidence_by_requirement, matcher or CrossEncoderMatcher())
    profile = build_scoring_profile(role["role_id"], requirements)
    routed_matches = _enforce_deterministic_skill_levels(requirements, routed["final_matches"])
    session = {
        "schema_version": SEMANTIC_SESSION_VERSION,
        "session_id": session_id,
        "role_id": role["role_id"],
        "candidate_id": candidate_id,
        # Persist only the structured facts required to explain a stored score;
        # raw resume text remains in the resume table and never enters this
        # session payload.
        "candidate_context": {
            "current_role_title": candidate.get("current_role_title_from_summary", ""),
            "experience": candidate.get("experience") or [],
            "total_years_experience": candidate.get("total_years_experience"),
            "education": candidate.get("education") or [],
        },
        "requirements": requirements,
        "profile": profile.to_dict(),
        "final_matches": [*deterministic, *routed_matches],
        "ambiguity_batch": routed["ambiguity_batch"],
        "matching": {key: routed[key] for key in ("model", "model_version", "thresholds")},
        "status": "pending_webllm" if routed["ambiguity_batch"]["items"] else "completed",
        "result": None,
    }
    if session["status"] == "completed":
        session["result"] = _score_session(session, [])
    return session


def finalize_semantic_session(session: dict, response: dict) -> dict:
    """Validate browser output against the server-issued batch, then score once."""
    if session.get("status") != "pending_webllm":
        raise ValueError("This semantic session is not waiting for WebLLM judgments.")
    judgments = validate_ambiguity_judgments(session.get("ambiguity_batch", {}), response)
    session = dict(session)
    session["webllm_judgments"] = judgments
    session["result"] = _score_session(session, judgments)
    session["status"] = "completed"
    return session


def _score_session(session: dict, judgments: list[dict]) -> dict:
    matches = merge_match_outcomes(session["candidate_id"], session["final_matches"], judgments)
    matches = _enforce_deterministic_skill_levels(session["requirements"], matches)
    profile = ScoringProfile(**session["profile"])
    return score_candidate_matches(session["candidate_id"], session["requirements"], matches, profile).to_dict()


def legacy_result_for_session(session: dict, candidate: dict | None = None, role: dict | None = None) -> dict:
    """Return the established Local-Ollama score contract for a browser run.

    Browser WebLLM is an evidence classifier, not a second scoring engine.
    When the original candidate and JD are available, replay the validated
    browser decisions through ``calculate_job_fit`` so browser and Local
    Ollama runs share category weights, gate rules, percentage calculation,
    title treatment, and recruiter-facing result shape.  Deterministic
    explicit/alias matching still happens first inside that scorer; a WebLLM
    positive is deliberately capped as related evidence rather than being
    allowed to manufacture a green direct skill match.

    The fallback below keeps already-persisted sessions readable while old
    session rows are gradually replaced by fresh screenings.
    """
    if candidate is not None and role is not None:
        requirements_by_text: dict[str, list] = {}
        resolved = _enforce_deterministic_skill_levels(
            session.get("requirements", []),
            merge_match_outcomes(
                session["candidate_id"], session.get("final_matches", []), session.get("webllm_judgments", []),
            ),
        )
        requirement_by_id = {item["id"]: item for item in session.get("requirements", [])}
        for match in resolved:
            requirement = requirement_by_id.get(match.requirement_id)
            if requirement and requirement.get("skills"):
                requirements_by_text.setdefault(requirement["text"].casefold(), []).append(match)

        def browser_judge(requirement: str, _evidence_chunks: list[dict]) -> dict:
            """Translate one prevalidated browser decision into judge format."""
            matches = requirements_by_text.get((requirement or "").casefold(), [])
            match = matches.pop(0) if matches else None
            if not match or match.decision == "missing":
                return {
                    "match": "none", "confidence": 1.0,
                    "reason": "Browser semantic review found no grounded evidence.",
                }
            # Exact/direct evidence is recalculated by matcher.py before this
            # callback. Any evidence that reaches WebLLM remains yellow.
            return {
                "match": "related", "confidence": 0.80,
                "reason": match.reason or "Browser semantic review found related grounded evidence.",
            }

        return calculate_job_fit(candidate, role, judge_fn=browser_judge)

    """Fallback view adapter for historic sessions lacking source records."""
    result = session.get("result") or {}
    requirements = {item["id"]: item for item in session.get("requirements", [])}
    # Keep the provenance used by the deterministic scorer in the legacy
    # display contract. Yellow evidence is worth 80%, regardless of whether
    # it came from a deterministic capability group or semantic review.
    resolved_matches = _enforce_deterministic_skill_levels(
        session.get("requirements", []),
        merge_match_outcomes(
            session["candidate_id"],
            session.get("final_matches", []),
            session.get("webllm_judgments", []),
        ),
    )
    matches_by_requirement = {match.requirement_id: match for match in resolved_matches}

    def weak_match_detail(requirement_id: str) -> str:
        match = matches_by_requirement.get(requirement_id)
        if match and match.method == "lexical":
            return "Deterministic related-reference evidence; 80% scoring credit. Verify the named skill in interview."
        return "Semantic-only related evidence; 80% scoring credit. Verify the named skill in interview."

    category_scores, evidence = {}, {}
    context = session.get("candidate_context") or {}
    for category in result.get("categories", []):
        matched_ids = category.get("matched_requirement_ids", [])
        weak_ids = category.get("weak_requirement_ids", [])
        missing_ids = category.get("missing_requirement_ids", [])
        # Alternatives are one JD condition: "B.Tech / B.E / MCA" means any
        # one of the three, not three separate education failures.  Collapse
        # them for the legacy display model as the scoring engine already does.
        grouped_ids: dict[str, list[str]] = {}
        for requirement_id in [*matched_ids, *weak_ids, *missing_ids]:
            requirement = requirements.get(requirement_id, {})
            group = requirement.get("alternative_group") or requirement_id
            grouped_ids.setdefault(group, []).append(requirement_id)

        def group_text(ids: list[str]) -> str:
            return " / ".join(requirements[item]["text"] for item in ids if item in requirements)

        display_rows = []
        for ids in grouped_ids.values():
            status = "matched" if any(item in matched_ids for item in ids) else "weak_match" if any(item in weak_ids for item in ids) else "missing"
            display_rows.append({"skill": group_text(ids), "status": status, "match_type": "semantic", "evidence": []})
        source_category = category["category"]
        display_category = LEGACY_CATEGORY_KEY.get(source_category, source_category)
        not_required = not grouped_ids
        category_scores[display_category] = {
            # The legacy screens display fixed category maxima. Preserve the
            # semantic confidence there as coverage rather than showing a
            # normalized-policy contribution against the old, unrelated max.
            "score": round(category["confidence"] * LEGACY_DISPLAY_MAX.get(display_category, 0), 2),
            "required_count": len(grouped_ids),
            "matched_count": sum(row["status"] == "matched" for row in display_rows),
            "match_percentage": round(category["confidence"] * 100),
            "not_required": not_required,
            "matched": [row["skill"] for row in display_rows if row["status"] == "matched"],
            "missing": [row["skill"] for row in display_rows if row["status"] == "missing"],
            "gate_missing": [row["skill"] for row in display_rows if row["status"] == "missing"],
        }
        # Capability groups determine the score, but the recruiter-facing
        # evidence must retain each individual JD skill. Keep non-skill OR
        # conditions (for example B.Tech / B.E / MCA) grouped for clarity.
        if source_category in {"mandatory_skills", "preferred_skills", "domain_alignment"}:
            def individual_rows(ids: list[str], status: str) -> list[dict]:
                return [
                    {
                        "skill": requirements[item]["text"],
                        "status": status,
                        "match_type": "semantic",
                        "detail": weak_match_detail(item) if status == "weak_match" else None,
                        "evidence": [],
                    }
                    for item in ids if item in requirements
                ]
            # A capability group is a score-calibration tool, not proof that
            # every sibling tool was used.  Display each JD skill's own
            # direct, inferred, or no-explicit-evidence status to recruiters.
            evidence[display_category] = [
                *individual_rows(matched_ids, "matched"),
                *individual_rows(weak_ids, "weak_match"),
                *individual_rows(missing_ids, "missing"),
            ]
            # Keep both views in the result contract: capability coverage is
            # the calibrated score input, whereas named-skill coverage lets a
            # recruiter audit all explicit JD skills without mistaking groups
            # for a smaller JD.
            category_scores[display_category].update({
                "named_required_count": len(matched_ids) + len(weak_ids) + len(missing_ids),
                "named_matched_count": sum(row["status"] == "matched" for row in evidence[display_category]),
                "named_weak_count": sum(row["status"] == "weak_match" for row in evidence[display_category]),
                # Summaries and table evidence must not inherit a full match
                # from a sibling in the same score group.
                "matched": [requirements[item]["text"] for item in matched_ids if item in requirements],
                "weak": [requirements[item]["text"] for item in weak_ids if item in requirements],
                "missing": [requirements[item]["text"] for item in missing_ids if item in requirements],
            })
        else:
            evidence[display_category] = display_rows
        if source_category == "role_alignment":
            best_title = context.get("current_role_title") or next(
                (item.get("title") for item in context.get("experience", []) if isinstance(item, dict) and item.get("title")),
                "No current title extracted",
            )
            title_match = next((requirements[item] for item in matched_ids if item in requirements), None)
            title_match = title_match or next((requirements[item] for item in weak_ids if item in requirements), None)
            title_match = title_match or next((requirements[item] for item in missing_ids if item in requirements), None)
            confidence = category.get("confidence", 0)
            evidence["job_title"] = {
                "status": "matched" if confidence >= 0.99 else "related" if confidence else "missing",
                "match_level": "direct" if confidence >= 0.99 else "related" if confidence else "none",
                "best_match": {"title": best_title},
                "judge_reason": (
                    "Approved job-title family match."
                    if confidence >= 0.99 else "Related approved job-title family match."
                    if confidence else "No approved current job-title family match."
                ),
                "target_role": title_match["text"] if title_match else "",
            }
        if source_category == "relevant_experience":
            required = next((requirements[item]["text"] for item in matched_ids + weak_ids + missing_ids if item in requirements), "")
            evidence["experience"] = {
                "years": {
                    "status": "matched" if category.get("confidence", 0) else "missing",
                    "total_years_experience": context.get("total_years_experience"),
                    "min_years_required": required,
                },
                "roles": [],
            }
        if source_category == "education":
            evidence["education"] = [
                {
                    "required_degree_level": row["skill"],
                    "required_field": "",
                    "status": row["status"],
                }
                for row in display_rows
            ]
    eligibility = result.get("eligibility", {})
    return {
        "category_scores": category_scores, "evidence": evidence,
        "final_score": result["final_score"],
        "hard_gate_failed": eligibility.get("tier") == "ineligible",
        "hard_gate_reason": eligibility.get("reason", ""),
        "scored_at": now_iso(),
    }
