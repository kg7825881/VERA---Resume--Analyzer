import pytest

from common import (
    CandidateScoreBreakdown,
    CategoryScore,
    EligibilityResult,
    EvidenceChunk,
    Requirement,
    RequirementMatch,
    ScoringProfile,
    stable_requirement_id,
)


def test_requirement_id_is_stable_and_changes_when_the_policy_input_changes():
    first = stable_requirement_id("data-engineer", "mandatory_skills", "Python")
    second = stable_requirement_id("data-engineer", "mandatory_skills", " python ")
    changed = stable_requirement_id("data-engineer", "preferred_skills", "Python")

    assert first == second
    assert first.startswith("req_mandatory-skills_")
    assert first != changed


def test_contracts_round_trip_to_json_safe_dicts():
    requirement = Requirement(
        id="mandatory_001", text="Strong Python experience", skills=("Python",),
        category="mandatory_skills", non_negotiable=True, source_section="Required qualifications",
    )
    evidence = EvidenceChunk(
        evidence_id="candidate_1_exp_1", candidate_id="candidate-1", section="experience",
        text="Built Python APIs and data pipelines.", page=2, source_id="src_resume_1",
    )
    match = RequirementMatch(
        requirement_id=requirement.id, candidate_id=evidence.candidate_id, decision="matched",
        confidence=0.91, evidence_ids=(evidence.evidence_id,), method="cross_encoder",
    )
    profile = ScoringProfile(
        profile_id="profile_1", role_id="data-engineer",
        weights={"mandatory_skills": 0.7, "relevant_experience": 0.3},
        rationales={"mandatory_skills": "Required by JD."}, policy_version="2026.1",
    )
    eligibility = EligibilityResult(
        mandatory_coverage=0.8, non_negotiable_coverage=1.0,
        tier="excellent_match", eligible=True,
    )
    breakdown = CandidateScoreBreakdown(
        candidate_id="candidate-1", profile_id=profile.profile_id, final_score=84.5,
        categories=(CategoryScore(
            category="mandatory_skills", weight=0.7, confidence=0.9, contribution=63,
            matched_requirement_ids=(requirement.id,),
        ),),
        eligibility=eligibility,
    )

    assert requirement.to_dict()["skills"] == ["Python"]
    assert evidence.to_dict()["content_hash"]
    assert match.to_dict()["evidence_ids"] == [evidence.evidence_id]
    assert profile.to_dict()["weights"]["mandatory_skills"] == 0.7
    assert breakdown.to_dict()["eligibility"]["tier"] == "excellent_match"
    assert breakdown.to_dict()["categories"][0]["matched_requirement_ids"] == [requirement.id]


def test_contracts_reject_untraceable_matches_and_invalid_policy_values():
    with pytest.raises(ValueError, match="evidence_id"):
        RequirementMatch(
            requirement_id="req_1", candidate_id="candidate-1", decision="matched",
            confidence=0.9,
        )

    with pytest.raises(ValueError, match="sum to 1.0"):
        ScoringProfile(
            profile_id="profile_1", role_id="data-engineer",
            weights={"mandatory_skills": 0.6, "relevant_experience": 0.6},
        )

    with pytest.raises(ValueError, match="agree with the eligibility tier"):
        EligibilityResult(
            mandatory_coverage=1, non_negotiable_coverage=1,
            tier="ineligible", eligible=True,
        )
