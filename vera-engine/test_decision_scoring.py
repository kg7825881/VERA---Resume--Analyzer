from common import Requirement
from decision_scoring import build_scoring_profile, merge_match_outcomes, rank_candidate_breakdowns, score_candidate_matches


def _requirements():
    return [
        Requirement("req-python", "Python", ("Python",), "mandatory_skills", non_negotiable=True),
        Requirement("req-sql", "SQL", ("SQL",), "mandatory_skills"),
        Requirement("req-airflow", "Airflow", ("Airflow",), "preferred_skills"),
    ]


def test_score_is_deterministic_and_non_negotiables_make_candidates_ineligible():
    requirements = _requirements()
    profile = build_scoring_profile("data-engineer", requirements)
    matches = merge_match_outcomes("candidate-1", [
        {"requirement_id": "req-python", "decision": "matched", "confidence": 0.9, "evidence_id": "ev-python"},
        {"requirement_id": "req-sql", "decision": "matched", "confidence": 0.8, "evidence_id": "ev-sql"},
    ], [{"requirement_id": "req-airflow", "evidence_id": "ev-airflow", "decision": "weak", "confidence": 0.7, "reason": "Related orchestration."}])
    result = score_candidate_matches("candidate-1", requirements, matches, profile)
    assert result.eligibility.tier == "excellent_match"
    assert result.final_score > 80

    missing_non_negotiable = score_candidate_matches("candidate-2", requirements, [
        {"requirement_id": "req-python", "decision": "missing", "confidence": 0, "method": "cross_encoder"},
        {"requirement_id": "req-sql", "decision": "matched", "confidence": 1, "evidence_id": "ev-sql", "method": "cross_encoder"},
    ], profile)
    assert missing_non_negotiable.eligibility.tier == "ineligible"


def test_deterministic_related_matches_receive_fixed_partial_credit():
    requirements = [Requirement("req", "Containerization", ("Containerization",), "mandatory_skills")]
    profile = build_scoring_profile("platform-engineer", requirements)
    result = score_candidate_matches("candidate-1", requirements, [
        {"requirement_id": "req", "decision": "weak", "confidence": 0.31, "evidence_id": "ev", "method": "lexical"},
    ], profile)
    assert result.categories[0].confidence == 0.80


def test_mandatory_percentage_weights_exact_and_related_named_skills():
    requirements = [
        Requirement("exact", "Python", ("Python",), "mandatory_skills"),
        Requirement("related", "Grafana", ("Grafana",), "mandatory_skills"),
        Requirement("missing", "Kubernetes", ("Kubernetes",), "mandatory_skills"),
    ]
    profile = build_scoring_profile("platform-engineer", requirements)
    result = score_candidate_matches("candidate-1", requirements, [
        {"requirement_id": "exact", "decision": "matched", "confidence": 1, "evidence_id": "e1", "method": "exact"},
        {"requirement_id": "related", "decision": "weak", "confidence": 0.2, "evidence_id": "e2", "method": "lexical"},
        {"requirement_id": "missing", "decision": "missing", "confidence": 0, "method": "exact"},
    ], profile)

    mandatory = next(category for category in result.categories if category.category == "mandatory_skills")
    # (1 exact + 0.80 deterministic-related + 0 missing) / 3 requirements.
    assert round(mandatory.confidence * 100, 2) == 60.0


def test_skill_percentages_use_named_counts_not_priority_or_capability_groups():
    requirements = [
        Requirement("exact", "Python", ("Python",), "mandatory_skills", priority=10),
        Requirement("related", "Grafana", ("Grafana",), "mandatory_skills", priority=1),
        Requirement("missing", "Kubernetes", ("Kubernetes",), "mandatory_skills", priority=1),
        Requirement("preferred-exact", "Docker", ("Docker",), "preferred_skills", priority=10),
        Requirement("preferred-related", "Monitoring", ("Monitoring",), "preferred_skills", priority=1),
        Requirement("preferred-missing", "OCR", ("OCR",), "preferred_skills", priority=1),
    ]
    profile = build_scoring_profile("platform-engineer", requirements)
    result = score_candidate_matches("candidate-1", requirements, [
        {"requirement_id": "exact", "decision": "matched", "confidence": 1, "evidence_id": "ev-exact", "method": "exact"},
        {"requirement_id": "related", "decision": "weak", "confidence": .1, "method": "lexical"},
        {"requirement_id": "missing", "decision": "missing", "confidence": 0, "method": "exact"},
        {"requirement_id": "preferred-exact", "decision": "matched", "confidence": 1, "evidence_id": "ev-preferred-exact", "method": "exact"},
        {"requirement_id": "preferred-related", "decision": "weak", "confidence": .1, "method": "webllm"},
        {"requirement_id": "preferred-missing", "decision": "missing", "confidence": 0, "method": "exact"},
    ], profile)
    by_category = {category.category: category for category in result.categories}
    assert round(by_category["mandatory_skills"].confidence * 100, 2) == 60.0
    assert round(by_category["preferred_skills"].confidence * 100, 2) == 60.0


def test_semantic_related_matches_receive_eighty_five_percent_credit():
    requirements = [Requirement("req", "Python", ("Python",), "mandatory_skills")]
    profile = build_scoring_profile("engineer", requirements)
    result = score_candidate_matches("candidate-1", requirements, [
        {"requirement_id": "req", "decision": "weak", "confidence": 0.99, "evidence_id": "ev", "method": "cross_encoder"},
    ], profile)
    assert result.categories[0].confidence == 0.80


def test_ranking_is_tier_first_then_score():
    requirements = _requirements()
    profile = build_scoring_profile("data-engineer", requirements)
    excellent = score_candidate_matches("excellent", requirements, [
        {"requirement_id": "req-python", "decision": "matched", "confidence": 1, "evidence_id": "e1", "method": "cross_encoder"},
        {"requirement_id": "req-sql", "decision": "matched", "confidence": 1, "evidence_id": "e2", "method": "cross_encoder"},
        {"requirement_id": "req-airflow", "decision": "matched", "confidence": 1, "evidence_id": "e3", "method": "cross_encoder"},
    ], profile)
    ineligible = score_candidate_matches("ineligible", requirements, [
        {"requirement_id": "req-python", "decision": "missing", "confidence": 0, "method": "cross_encoder"},
        {"requirement_id": "req-sql", "decision": "matched", "confidence": 1, "evidence_id": "e2", "method": "cross_encoder"},
        {"requirement_id": "req-airflow", "decision": "matched", "confidence": 1, "evidence_id": "e3", "method": "cross_encoder"},
    ], profile)
    assert [item.candidate_id for item in rank_candidate_breakdowns([ineligible, excellent])] == ["excellent", "ineligible"]


def test_absent_jd_categories_are_visible_full_credit_not_candidate_penalties():
    requirements = [Requirement("req-python", "Python", ("Python",), "mandatory_skills")]
    profile = build_scoring_profile("data-engineer", requirements)
    result = score_candidate_matches("candidate-1", requirements, [
        {"requirement_id": "req-python", "decision": "matched", "confidence": 1, "evidence_id": "e1", "method": "exact"},
    ], profile)
    by_category = {item.category: item for item in result.categories}
    assert by_category["preferred_skills"].confidence == 1
    assert by_category["preferred_skills"].contribution == 5
    assert by_category["education"].confidence == 1
