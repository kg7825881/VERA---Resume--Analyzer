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


def test_raw_extracted_title_containing_approved_reference_is_a_direct_match():
    candidate = {
        "experience": [{
            "title": "Senior Java Engineer | Payments Platform, Bengaluru",
            "company": "Example Co",
            "end_date_raw": "Present",
        }],
    }

    result = score_job_titles(candidate, "Sr Java Lead Engineer", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["match_level"] == "direct"
    assert result["matched_target_title"] == "Senior Java Engineer"


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


def test_explicit_software_engineer_reference_title_is_a_direct_match():
    candidate = {
        "experience": [{"title": "Software Engineer", "company": "Example Co", "end_date_raw": "Present"}],
    }

    result = score_job_titles(candidate, "Sr Java Lead Engineer", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["match_level"] == "direct"


def test_abbreviated_seniority_title_is_valid_title_evidence():
    candidate = {
        "experience": [{"title": "Sr. Software Engineer", "company": "Example Co", "end_date_raw": "Present"}],
    }
    result = score_job_titles(candidate, "Sr Java Lead Engineer", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["best_match"]["title"] == "Sr. Software Engineer"


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


def test_any_approved_reference_title_in_verified_work_history_is_a_perfect_fit():
    candidate = {
        "experience": [
            {"title": "Product Owner", "company": "Current Co", "end_date_raw": "Present"},
            {"title": "Data Scientist", "company": "Prior Co", "end_date_raw": "Dec 2024"},
        ],
    }

    result = score_job_titles(candidate, "Data Engineer — AI Data Platform", judge_fn=_no_match_judge)

    assert result["contribution"] == 1.0
    assert result["match_level"] == "direct"
    assert result["matched_target_title"] == "Data Scientist"
    assert result["best_match"]["company"] == "Prior Co"


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


@pytest.mark.parametrize(("jd_skill", "resume_skill"), [("ETL", "ELT"), ("ELT", "ETL")])
def test_etl_and_elt_are_approved_bidirectional_direct_references(jd_skill, resume_skill):
    assert _exact_match(jd_skill, [resume_skill]) == resume_skill


@pytest.mark.parametrize(
    ("jd_skill", "resume_evidence"),
    [
        ("Circuit breakers", "Implemented Resilience4j Circuit Breaker for downstream services."),
        ("Code review", "Led code reviews and mentoring for junior engineers."),
    ],
)
def test_explicit_grammatical_or_framework_qualified_variants_are_direct(jd_skill, resume_evidence):
    """Approved variants are green; broader capabilities remain separate."""
    assert _exact_match(jd_skill, [resume_evidence]) == resume_evidence


@pytest.mark.parametrize(
    ("requirement", "resume_evidence"),
    [
        ("Spring Discovery", "Implemented Netflix Eureka service registration."),
        ("Configuration management", "Managed distributed configuration using Config Server."),
        ("Resiliency patterns", "Used Resilience4j with retry mechanisms."),
        ("Performance engineering", "Delivered performance optimization for high-throughput APIs."),
    ],
)
def test_concrete_broad_capability_evidence_is_yellow_not_green(requirement, resume_evidence):
    result = score_single_skill(
        requirement,
        [resume_evidence],
        CandidateEvidenceIndex({"skills_all_sources": [resume_evidence]}),
        judge_fn=_no_match_judge,
    )
    assert result["contribution"] == 0.80
    assert result["match_type"] == "reference"


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

    assert result["contribution"] == 0.80
    assert result["match_type"] == "reference"
    if result["matched_against"] is not None:
        assert result["matched_against"] == resume_skill
    assert result["gate_satisfied"]


def test_approved_related_keywords_are_yellow_reference_evidence_in_project_text():
    source_text = (
        "Project delivered document digitization, ER diagrams, Snowflake ETL data marts, validation rules, "
        "entity resolution, referential integrity, data profiling, KPI analysis, dashboards, and BI reporting."
    )
    evidence = CandidateEvidenceIndex({"source_markdown": source_text})

    for requirement in (
        "OCR", "Data modeling", "Data warehouse", "Data validation", "Deduplication",
        "Consistency checks", "Data quality control", "Analytics",
    ):
        result = score_single_skill(requirement, [source_text], evidence, judge_fn=_no_match_judge)
        assert result["contribution"] == 0.80
        assert result["match_type"] == "reference"


def test_concrete_related_evidence_counts_for_a_mandatory_gate():
    result = score_single_skill(
        "Lakehouse",
        [],
        CandidateEvidenceIndex({"skills_all_sources": ["Databricks data platform"]}),
        judge_fn=lambda requirement, evidence: {
            "match": "related", "confidence": 0.9, "reason": "Concrete adjacent platform evidence.",
        },
    )

    assert result["contribution"] == 0.80
    assert result["gate_satisfied"]


@pytest.mark.parametrize(
    ("requirement", "generic_evidence"),
    [
        ("Prometheus", "Built monitoring dashboards and handled production reliability."),
        ("Airflow", "Automated workflows and scheduled operational tasks."),
        ("Terraform", "Managed cloud infrastructure and deployment operations."),
        ("Tableau", "Created dashboards, reports, and business insights."),
        ("Pinecone", "Built RAG applications with vector search and retrieval."),
        ("Selenium", "Performed automated testing and quality assurance."),
        ("GitHub Copilot", "Used AI-assisted development and code generation."),
    ],
)
def test_generic_capability_evidence_does_not_infer_named_tools(requirement, generic_evidence):
    """Models cannot turn broad capability wording into named-tool experience."""
    evidence = CandidateEvidenceIndex({"skills_all_sources": [generic_evidence]})

    result = score_single_skill(
        requirement,
        [generic_evidence],
        evidence,
        judge_fn=lambda requirement, evidence: {
            "match": "direct", "confidence": 0.99,
            "reason": "This judge must not be able to infer a named product.",
        },
    )

    assert result["contribution"] == 0.0
    assert result["match_type"] == "none"
    assert not result["gate_satisfied"]
    assert "explicit product evidence" in result["judge_reason"]


def test_generic_operational_evidence_does_not_infer_each_named_observability_product():
    source_text = "Built production telemetry dashboards with reliability and monitoring responsibilities."
    evidence = CandidateEvidenceIndex({"skills_all_sources": [source_text]})

    for requirement in ("Prometheus", "Grafana", "Loki", "ELK/OpenSearch", "OpenTelemetry"):
        result = score_single_skill(
            requirement,
            ["Monitoring", source_text],
            evidence,
            judge_fn=lambda requirement, evidence: {
                "match": "direct", "confidence": 0.99,
                "reason": "This judge must not be able to infer a named product.",
            },
        )

        assert result["contribution"] == 0.0
        assert result["match_type"] == "none"
        assert not result["gate_satisfied"]
        assert "explicit product evidence" in result["judge_reason"]


def test_named_observability_sibling_is_yellow_related_evidence():
    """An explicit adjacent product is related evidence, never a green exact match."""
    result = score_single_skill(
        "Grafana",
        ["Prometheus"],
        CandidateEvidenceIndex({"skills_all_sources": ["Prometheus"]}),
        judge_fn=_no_match_judge,
    )

    assert result["contribution"] == 0.80
    assert result["match_type"] == "reference"
    assert result["gate_satisfied"]
    from matcher import evidence_status
    assert evidence_status(result) == "weak_match"


def test_weak_evidence_receives_partial_credit_but_cannot_clear_a_gate():
    result = score_single_skill(
        "Lakehouse",
        [],
        CandidateEvidenceIndex({"skills_all_sources": ["Databricks data platform"]}),
        judge_fn=lambda requirement, evidence: {
            "match": "weak", "confidence": 0.7, "reason": "Broad adjacent evidence only.",
        },
    )

    assert result["contribution"] == 0.55
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
