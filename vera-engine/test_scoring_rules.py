import pytest

from job_title_matcher import score_job_titles
from matcher import _exact_match
from scorer import calculate_job_fit, validate_jd_for_scoring


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


def test_only_exact_reference_titles_earn_full_credit_and_related_evidence_is_capped():
    candidate = {
        "current_role_title_from_summary": "Data Scientist",
        "experience": [{"title": "Data Scientist", "company": "Example Co"}],
    }

    result = score_job_titles(candidate, "Data Engineer — AI Data Platform", judge_fn=_no_match_judge)

    assert result["contribution"] == 0.8
    assert result["match_level"] == "related"


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
        "mandatory_skills": ["Python"],
        "mandatory_domain_requirements": [],
        "mandatory_role_specific_requirements": ["Executive workshops"],
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
    assert result["evidence"]["mandatory_role_specific_requirements"][0]["status"] == "not_technical_gate"


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
