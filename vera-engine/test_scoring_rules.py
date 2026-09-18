import pytest

from job_title_matcher import score_job_titles
from matcher import _exact_match, score_single_skill
from retrieval import CandidateEvidenceIndex
from scorer import calculate_job_fit, validate_jd_for_scoring
from scorer import WEIGHTS


def _no_match_judge(requirement, evidence):
    return {"match": "none", "confidence": 1.0, "reason": "test judge"}


def test_job_title_matches_any_accepted_jd_target_title():
    candidate = {
        "current_role_title_from_summary": "Product Analyst",
        "experience": [{"title": "Product Analyst", "company": "Example Co"}],
    }

    result = score_job_titles(
        candidate,
        ["Senior Business Analyst", "Business Analyst", "Product Analyst", "Systems Analyst"],
        judge_fn=_no_match_judge,
    )

    assert result["contribution"] == 1.0
    assert result["matched_target_title"] == "Product Analyst"
    assert result["best_match"]["title"] == "Product Analyst"
    assert result["best_match"]["company"] == "Example Co"


@pytest.mark.parametrize(
    ("jd_title", "candidate_title"),
    [
        ("Data Engineer — AI Data Platform", "Senior Data Engineer"),
        ("Data Engineer — AI Data Platform", "AI Architect"),
        ("Senior Business Analyst", "Business Systems Analyst"),
        ("DevOps Engineer", "Site Reliability Engineer"),
        ("Sr Java Lead Engineer", "Java Solutions Architect"),
    ],
)
def test_approved_title_reference_variants_earn_full_credit(jd_title, candidate_title):
    candidate = {
        "current_role_title_from_summary": candidate_title,
        "experience": [{"title": candidate_title, "company": "Example Co"}],
    }

    result = score_job_titles(candidate, jd_title, judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["matched_target_title"] == candidate_title


def test_approved_title_with_resume_qualifiers_keeps_full_credit():
    candidate = {
        "current_role_title_from_summary": "Senior Data Engineer (Data Platform)",
        "experience": [{
            "title": "Senior Data Engineer (Data Platform)",
            "company": "Example Co",
            "end_date_raw": "Present",
        }],
    }

    result = score_job_titles(candidate, "Data Engineer — AI Data Platform", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["matched_target_title"] == "Senior Data Engineer"


def test_base_role_explicitly_added_to_reference_earns_full_credit():
    candidate = {
        "current_role_title_from_summary": "Data Engineer",
        "experience": [{"title": "Data Engineer", "end_date_raw": "Present"}],
    }

    result = score_job_titles(candidate, "Data Engineer — AI Data Platform", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["match_level"] == "direct"


def test_data_scientist_explicitly_added_to_reference_earns_full_credit():
    candidate = {
        "current_role_title_from_summary": "Data Scientist",
        "experience": [{"title": "Data Scientist", "company": "Example Co"}],
    }

    result = score_job_titles(candidate, "Data Engineer — AI Data Platform", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["match_level"] == "direct"


def test_unrelated_title_gets_no_title_credit():
    candidate = {
        "current_role_title_from_summary": "Product Manager",
        "experience": [{"title": "Product Manager", "company": "Example Co"}],
    }

    result = score_job_titles(candidate, "DevOps Engineer", judge_fn=_no_match_judge)

    assert result["contribution"] == 0.0
    assert result["match_level"] == "none"


def test_latest_employment_title_overrides_legacy_summary_field_for_scoring():
    candidate = {
        # Old stored rows can still contain a title that was once read from a summary.
        "current_role_title_from_summary": "Data Scientist",
        "experience": [
            {"title": "Data Scientist", "company": "Former Co", "end_date_raw": "Dec 2023"},
            {"title": "Senior Data Engineer", "company": "Current Co", "end_date_raw": "Present"},
        ],
    }

    result = score_job_titles(candidate, "Data Engineer — AI Data Platform", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["best_match"]["title"] == "Senior Data Engineer"
    assert result["best_match"]["company"] == "Current Co"


@pytest.mark.parametrize(
    ("jd_skill", "resume_skill"),
    [
        ("Lakehouse", "Data Lakehouse"),
        ("Data Lakehouse", "Lakehouse"),
        ("Spark", "Apache Spark"),
        ("Spark", "PySpark"),
        ("PySpark", "Spark"),
        ("Airflow", "Apache Airflow"),
    ],
)
def test_named_skill_variants_match_bidirectionally(jd_skill, resume_skill):
    assert _exact_match(jd_skill, [resume_skill]) == resume_skill


def test_generic_word_overlap_does_not_create_a_skill_match():
    assert _exact_match("Data Lakehouse", ["Data Governance"]) is None


@pytest.mark.parametrize(
    ("requirement", "resume_skill"),
    [
        ("OCR", "PDF parsing"),
        ("Retrieval datasets", "semantic search"),
        ("GenAI", "Prompt engineering"),
        ("Dagster", "data orchestration"),
        ("dbt", "SQL modeling"),
        ("Alerting", "anomaly detection"),
        ("Monitoring", "distributed tracing"),
        ("Schema design", "dimensional modeling"),
        ("Embedding pipelines", "vector embeddings"),
    ],
)
def test_reference_keywords_supply_explainable_partial_skill_evidence(requirement, resume_skill):
    result = score_single_skill(
        requirement,
        [resume_skill],
        CandidateEvidenceIndex({"skills_all_sources": [resume_skill]}),
        judge_fn=_no_match_judge,
    )

    assert result["contribution"] == 0.7
    assert result["match_type"] == "reference"
    assert result["matched_against"] == resume_skill
    assert not result["gate_satisfied"]


def test_role_specific_requirements_do_not_become_technical_hard_gate_items():
    candidate = {
        "skills_all_sources": ["Python"],
        "current_role_title_from_summary": "Product Analyst",
        "experience": [{"title": "Product Analyst", "company": "Example Co"}],
        "education": [],
        "total_years_experience": 5,
    }
    jd = {
        "role_title": "Senior Business Analyst",
        "target_job_titles": ["Senior Business Analyst", "Product Analyst"],
        "mandatory_skills": ["Python", "  "],
        "mandatory_domain_requirements": [],
        "mandatory_role_specific_requirements": ["Executive workshops", ""],
        "preferred_technical_skills": [],
        "industry_keywords": [],
        "soft_preferred_skills": [],
        "education_requirements": [],
        "min_years_experience": None,
    }

    result = calculate_job_fit(candidate, jd, judge_fn=_no_match_judge)

    mandatory = result["category_scores"]["mandatory_skills"]
    assert mandatory["required_count"] == 1
    assert mandatory["matched"] == ["Python"]
    assert not result["hard_gate_failed"]
    row = result["evidence"]["mandatory_role_specific_requirements"][0]
    assert row["skill"] == "Executive workshops"
    assert row["status"] == "weak_match"


def test_sparse_jd_is_blocked_before_scoring():
    jd = {
        "role_title": "Senior Java Lead Engineer",
        "target_job_titles": [],
        "mandatory_skills": [],
        "mandatory_domain_requirements": [],
        "mandatory_role_specific_requirements": [],
        "preferred_technical_skills": [],
        "industry_keywords": [],
        "soft_preferred_skills": [],
        "education_requirements": [],
        "min_years_experience": None,
    }

    assert validate_jd_for_scoring(jd)
    with pytest.raises(ValueError, match="no extracted skills"):
        calculate_job_fit({}, jd, judge_fn=_no_match_judge)


def test_scoring_weights_sum_to_one_and_preferred_skills_cannot_exceed_five_points():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert WEIGHTS["preferred_skills"] * 100 == 5
