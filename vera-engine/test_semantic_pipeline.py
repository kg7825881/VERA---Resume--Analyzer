from cross_encoder import CrossEncoderMatcher
from semantic_pipeline import create_semantic_session, finalize_semantic_session, legacy_result_for_session


class FakeIndex:
    def __init__(self, candidate):
        self.candidate_id = candidate["candidate_id"]

    def retrieve(self, requirement, top_k=3):
        return [{"candidate_id": self.candidate_id, "evidence_id": f"ev-{requirement}", "text": f"Evidence for {requirement}"}]


def test_session_waits_only_for_ambiguous_items_then_scores_validated_judgment():
    role = {"role_id": "data-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-python", "text": "Python", "skills": ["Python"], "category": "mandatory_skills", "non_negotiable": True, "priority": 1},
        {"id": "req-rag", "text": "RAG", "skills": ["RAG"], "category": "preferred_skills", "priority": 1},
    ]}}
    matcher = CrossEncoderMatcher(predict_fn=lambda pairs: [0.91, 0.55])
    session = create_semantic_session("session-1", role, {"candidate_id": "candidate-1"}, matcher=matcher, evidence_index_factory=FakeIndex)
    assert session["status"] == "pending_webllm"
    completed = finalize_semantic_session(session, {"judgments": [{
        "requirement_id": "req-rag", "evidence_id": "ev-RAG", "decision": "weak", "confidence": 0.6, "reason": "Related retrieval evidence."
    }]})
    assert completed["status"] == "completed"
    assert completed["result"]["candidate_id"] == "candidate-1"


def test_semantic_only_skill_positive_is_related_not_direct_evidence():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-python", "text": "Python", "skills": ["Python"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-semantic-skill", role, {"candidate_id": "candidate-1"},
        matcher=CrossEncoderMatcher(predict_fn=lambda pairs: [0.95]), evidence_index_factory=FakeIndex,
    )

    match = session["final_matches"][0]
    assert match["decision"] == "weak"
    assert match["confidence"] == 0.80
    assert session["result"]["categories"][0]["confidence"] == 0.80


def test_deterministic_experience_alias_education_and_title_family_bypass_model():
    role = {"role_id": "sr-java-lead-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-java", "text": "Java", "skills": ["Java"], "category": "mandatory_skills", "priority": 1},
        {"id": "req-exp", "text": "4+ years relevant experience", "skills": [], "category": "relevant_experience", "priority": 1},
        {"id": "req-degree", "text": "B.Tech", "skills": [], "category": "education", "alternative_group": "degree", "priority": 1},
        {"id": "req-mca", "text": "MCA", "skills": [], "category": "education", "alternative_group": "degree", "priority": 1},
        {"id": "req-title", "text": "Approved job-title family for Sr Java Lead Engineer", "skills": [], "category": "role_alignment", "priority": 1},
    ]}}
    candidate = {"candidate_id": "candidate-1", "skills": ["JAVA"], "total_years_experience": 5,
                 "education": [{"degree_level": "B.Tech"}],
                 "experience": [{"title": "Senior Software Engineer", "end_date_raw": "Present"}]}
    session = create_semantic_session("session-facts", role, candidate, evidence_index_factory=FakeIndex)
    assert session["status"] == "completed"
    assert {item["requirement_id"] for item in session["final_matches"]} == {"req-java", "req-exp", "req-degree", "req-mca", "req-title"}
    assert session["result"]["final_score"] > 70
    legacy = legacy_result_for_session(session)
    assert legacy["category_scores"]["job_title_match"]["score"] == 10
    assert legacy["evidence"]["job_title"]["status"] == "matched"


def test_bachelor_in_technology_is_a_deterministic_btech_equivalent():
    role = {"role_id": "software-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-btech", "text": "B.Tech", "skills": [], "category": "education", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "education": [{"degree_level": "Bachelor", "field": "Technology"}],
    }

    session = create_semantic_session("session-bachelor-tech", role, candidate, evidence_index_factory=FakeIndex)

    assert session["status"] == "completed"
    assert session["final_matches"] == [{
        "requirement_id": "req-btech", "decision": "matched", "confidence": 1.0,
        "evidence_ids": ["derived:candidate-1:req-btech"], "method": "exact",
        "reason": "Deterministic education alternative check.",
    }]


def test_verified_source_text_recovers_btech_when_pdf_layout_hid_education_section():
    """Both scoring paths must credit an explicitly written degree."""
    role = {"role_id": "software-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-btech", "text": "B.Tech", "skills": [], "category": "education", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "education": [],
        "source_markdown": "Education: Bachelor of Technology in Computer Science and Engineering",
    }

    session = create_semantic_session("session-source-btech", role, candidate, evidence_index_factory=FakeIndex)

    assert session["status"] == "completed"
    assert session["final_matches"][0]["decision"] == "matched"
    assert session["final_matches"][0]["confidence"] == 1.0


def test_common_bachelors_degree_variants_are_recovered_from_resume_text():
    role = {"role_id": "software-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-btech", "text": "B.Tech", "skills": [], "category": "education", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "education": [],
        "source_markdown": "EDUCATION: Bachelors Degree in Computer Science and Engineering, 2020",
    }
    session = create_semantic_session("session-source-bachelors", role, candidate, evidence_index_factory=FakeIndex)
    assert session["status"] == "completed"
    assert session["final_matches"][0]["decision"] == "matched"


def test_verified_resume_evidence_and_qualified_skill_labels_match_compact_jd_labels():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "llms", "text": "LLMs", "skills": ["LLMs"], "category": "mandatory_skills", "priority": 1},
        {"id": "apis", "text": "APIs", "skills": ["APIs"], "category": "mandatory_skills", "priority": 1},
        {"id": "chunking", "text": "Chunking", "skills": ["Chunking"], "category": "mandatory_skills", "priority": 1},
        {"id": "evaluation", "text": "Model evaluation", "skills": ["Model evaluation"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "skills": ["Large Language Models (LLM)", "REST APIs"],
        "skill_evidence": [
            {"skill": "Document Chunking"},
            {"skill": "RAGAS"},
        ],
    }

    session = create_semantic_session("session-qualified-skills", role, candidate, evidence_index_factory=FakeIndex)

    assert session["status"] == "completed"
    assert {item["requirement_id"] for item in session["final_matches"]} == {"llms", "apis", "chunking", "evaluation"}
    matches = {item["requirement_id"]: item for item in session["final_matches"]}
    assert {item["decision"] for item in matches.values()} == {"matched", "weak"}
    assert matches["evaluation"]["decision"] == "weak"
    assert matches["evaluation"]["confidence"] == 0.80


def test_project_text_is_explicit_evidence_and_rag_testing_is_related_evidence():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "chunking", "text": "Chunking", "skills": ["Chunking"], "category": "mandatory_skills", "priority": 1},
        {"id": "llm-testing", "text": "LLM testing", "skills": ["LLM testing"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": "Project: built document chunking and RAG testing workflows for answer quality.",
    }

    session = create_semantic_session("session-project-evidence", role, candidate, evidence_index_factory=FakeIndex)
    matches = {item["requirement_id"]: item for item in session["final_matches"]}

    assert matches["chunking"]["decision"] == "matched"
    assert matches["chunking"]["confidence"] == 1.0
    assert matches["llm-testing"]["decision"] == "weak"
    assert matches["llm-testing"]["confidence"] == 0.80


def test_project_framework_evidence_establishes_a_broad_capability():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "backend", "text": "Backend development", "skills": ["Backend development"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {"candidate_id": "candidate-1", "source_markdown": "Experience: built REST services with FastAPI."}

    session = create_semantic_session("session-project-framework", role, candidate, evidence_index_factory=FakeIndex)

    match = session["final_matches"][0]
    assert match["decision"] == "weak"
    assert match["confidence"] == 0.80


def test_document_and_operational_vocabulary_matches_across_resume_sections():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "parsing", "text": "Document parsing", "skills": ["Document parsing"], "category": "mandatory_skills", "priority": 1},
        {"id": "monitoring", "text": "Monitoring", "skills": ["Monitoring"], "category": "mandatory_skills", "priority": 1},
        {"id": "logging", "text": "Logging", "skills": ["Logging"], "category": "mandatory_skills", "priority": 1},
        {"id": "fullstack", "text": "Full-stack development", "skills": ["Full-stack development"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": "Built a PDF text-extraction pipeline using PyMuPDF with Prometheus metrics and structured logging. Worked on React UI.",
    }

    session = create_semantic_session("session-vocabulary", role, candidate, evidence_index_factory=FakeIndex)
    matches = {item["requirement_id"]: item for item in session["final_matches"]}

    assert matches["parsing"]["decision"] == "weak"
    assert matches["monitoring"]["decision"] == "weak"
    assert matches["logging"]["decision"] == "matched"
    assert matches["fullstack"]["decision"] == "weak"


def test_generalized_ai_document_and_workflow_related_evidence_is_deterministic():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "llm-apps", "text": "LLM applications", "skills": ["LLM applications"], "category": "mandatory_skills", "priority": 1},
        {"id": "llm-testing", "text": "LLM testing", "skills": ["LLM testing"], "category": "mandatory_skills", "priority": 1},
        {"id": "parsing", "text": "Document parsing", "skills": ["Document parsing"], "category": "mandatory_skills", "priority": 1},
        {"id": "workflow", "text": "Workflow automation", "skills": ["Workflow automation"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": "Built Generative AI applications with prompt engineering; benchmarked hallucination detection, used information extraction, and automated process orchestration pipelines.",
    }

    session = create_semantic_session("session-generalized-related-evidence", role, candidate, evidence_index_factory=FakeIndex)

    for match in session["final_matches"]:
        assert match["decision"] == "weak"
        assert match["confidence"] == 0.80
        assert match["method"] == "lexical"


def test_data_business_and_devops_umbrella_capabilities_use_direct_evidence():
    role = {"role_id": "mixed-platform-role", "requirement_metadata": {"requirements": [
        {"id": "pipelines", "text": "Data pipelines", "skills": ["Data pipelines"], "category": "mandatory_skills", "priority": 1},
        {"id": "quality", "text": "Data quality", "skills": ["Data quality"], "category": "mandatory_skills", "priority": 1},
        {"id": "requirements", "text": "Requirements elicitation", "skills": ["Requirements elicitation"], "category": "mandatory_skills", "priority": 1},
        {"id": "agile", "text": "Agile delivery", "skills": ["Agile delivery"], "category": "mandatory_skills", "priority": 1},
        {"id": "deployments", "text": "Deployment automation", "skills": ["Deployment automation"], "category": "mandatory_skills", "priority": 1},
        {"id": "iac", "text": "Infrastructure as code", "skills": ["Infrastructure as code"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": "Built ETL with Airflow, data validation and deduplication checks; led discovery workshops and Scrum sprint planning; automated GitOps deployments with Argo CD and provisioned Terraform infrastructure.",
    }

    session = create_semantic_session("session-cross-role-umbrella-evidence", role, candidate, evidence_index_factory=FakeIndex)

    assert {match["requirement_id"] for match in session["final_matches"] if match["decision"] == "weak"} == {
        "pipelines", "quality", "requirements", "agile", "deployments", "iac",
    }
    assert all(match["confidence"] == 0.80 for match in session["final_matches"])


def test_insurance_platforms_remain_related_not_interchangeable_exact_matches():
    role = {"role_id": "business-analyst", "requirement_metadata": {"requirements": [
        {"id": "guidewire", "text": "Guidewire ClaimCenter", "skills": ["Guidewire ClaimCenter"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-insurance-platform-related", role,
        {"candidate_id": "candidate-1", "skills": ["Duck Creek", "Claims operations"]},
        evidence_index_factory=FakeIndex,
    )
    match = session["final_matches"][0]
    assert match["decision"] == "weak"
    assert match["confidence"] == 0.80


def test_spring_ecosystem_is_a_related_capability_group_match():
    role = {"role_id": "sr-java-lead-engineer", "requirement_metadata": {"requirements": [
        {"id": "req-spring-cloud", "text": "Spring Cloud Gateway", "skills": ["Spring Cloud Gateway"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-spring", role, {"candidate_id": "candidate-1", "skills": ["Spring Boot"]}, evidence_index_factory=FakeIndex,
    )
    match = session["final_matches"][0]
    assert match["decision"] == "weak"
    assert match["confidence"] == 0.80


def test_capability_group_members_are_related_not_exact_matches():
    role = {"role_id": "platform", "requirement_metadata": {"requirements": [
        {"id": "grafana", "text": "Grafana", "skills": ["Grafana"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-capability-group", role, {"candidate_id": "candidate-1", "skills": ["Prometheus"]},
        evidence_index_factory=FakeIndex,
    )

    match = session["final_matches"][0]
    assert match["decision"] == "weak"
    assert match["confidence"] == 0.80
    assert "capability-group" in match["reason"]


def test_capability_group_evidence_is_found_in_project_and_experience_text():
    """A raw resume section must be treated like the parsed Skills list."""
    role = {"role_id": "data-engineer", "requirement_metadata": {"requirements": [
        {"id": "elt", "text": "ELT", "skills": ["ELT"], "category": "mandatory_skills", "priority": 1},
        {"id": "transform", "text": "Data transformation logic", "skills": ["Data transformation logic"], "category": "mandatory_skills", "priority": 1},
        {"id": "alerting", "text": "Alerting", "skills": ["Alerting"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": "Experience: set up ETL pipelines and monitoring for a centralized data platform.",
    }

    session = create_semantic_session("session-raw-group-evidence", role, candidate, evidence_index_factory=FakeIndex)

    matches = {match["requirement_id"]: match for match in session["final_matches"]}
    # ETL and ELT are approved, bidirectional direct references. Other
    # transformation/operational siblings remain capability-group evidence.
    assert matches["elt"]["decision"] == "matched"
    assert matches["elt"]["confidence"] == 1.0
    assert matches["transform"]["decision"] == "weak"
    assert matches["alerting"]["decision"] == "weak"
    assert matches["transform"]["confidence"] == matches["alerting"]["confidence"] == 0.80
    assert "capability-group" in matches["transform"]["reason"]


def test_genai_frameworks_are_related_reference_evidence_in_project_text():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "genai", "text": "GenAI", "skills": ["GenAI"], "category": "preferred_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": "Built a retrieval assistant with LangChain, Azure OpenAI and GPT-4o.",
    }

    session = create_semantic_session("session-genai-framework", role, candidate, evidence_index_factory=FakeIndex)

    assert session["final_matches"][0]["decision"] == "weak"
    assert session["final_matches"][0]["confidence"] == 0.80


def test_browser_session_uses_the_local_scorer_policy_for_persisted_results():
    """WebLLM provides evidence decisions; it must not introduce new weights."""
    role = {
        "role_id": "shared-policy", "role_title": "Shared Policy Engineer",
        "mandatory_skills": ["Python"], "mandatory_domain_requirements": [],
        "mandatory_role_specific_requirements": [], "preferred_technical_skills": [],
        "industry_keywords": [], "soft_preferred_skills": [],
        "education_requirements": [], "min_years_experience": None,
        "requirement_metadata": {"requirements": [
            {"id": "python", "text": "Python", "skills": ["Python"], "category": "mandatory_skills", "priority": 1},
        ]},
    }
    candidate = {"candidate_id": "candidate-1", "skills": ["Python"]}
    session = create_semantic_session("session-shared-policy", role, candidate, evidence_index_factory=FakeIndex)

    persisted = legacy_result_for_session(session, candidate, role)

    # Local calculate_job_fit assigns the mandatory category a 30-point
    # maximum. The old browser-specific profile assigned it 55 points.
    assert persisted["category_scores"]["mandatory_skills"]["score"] == 30


def test_approved_related_keywords_in_resume_text_are_yellow():
    role = {"role_id": "data-engineer", "requirement_metadata": {"requirements": [
        {"id": "ocr", "text": "OCR", "skills": ["OCR"], "category": "mandatory_skills", "priority": 1},
        {"id": "modeling", "text": "Data modeling", "skills": ["Data modeling"], "category": "mandatory_skills", "priority": 1},
        {"id": "warehouse", "text": "Data warehouse", "skills": ["Data warehouse"], "category": "mandatory_skills", "priority": 1},
        {"id": "validation", "text": "Data validation", "skills": ["Data validation"], "category": "mandatory_skills", "priority": 1},
        {"id": "dedupe", "text": "Deduplication", "skills": ["Deduplication"], "category": "mandatory_skills", "priority": 1},
        {"id": "consistency", "text": "Consistency checks", "skills": ["Consistency checks"], "category": "mandatory_skills", "priority": 1},
        {"id": "quality", "text": "Data quality control", "skills": ["Data quality"], "category": "mandatory_skills", "priority": 1},
        {"id": "analytics", "text": "Analytics", "skills": ["Analytics"], "category": "mandatory_skills", "priority": 1},
    ]}}
    candidate = {
        "candidate_id": "candidate-1",
        "source_markdown": (
            "Built document digitization with document intelligence; designed ER diagrams and data architecture; "
            "delivered Snowflake ETL data marts; implemented validation rules, entity resolution, referential integrity, "
            "data profiling, anomaly detection, KPI analysis, dashboards, and data visualization."
        ),
    }

    session = create_semantic_session("session-direct-related-keywords", role, candidate, evidence_index_factory=FakeIndex)

    assert len(session["final_matches"]) == 8
    assert {match["decision"] for match in session["final_matches"]} == {"weak"}
    assert {match["confidence"] for match in session["final_matches"]} == {0.80}


def test_singular_vector_database_is_an_explicit_plural_jd_match():
    role = {"role_id": "ai-engineer", "requirement_metadata": {"requirements": [
        {"id": "vector-db", "text": "Vector databases", "skills": ["Vector databases"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-vector-db", role, {"candidate_id": "candidate-1", "source_markdown": "Skills: Vector Database, Vector Search."},
        evidence_index_factory=FakeIndex,
    )

    match = session["final_matches"][0]
    assert match["decision"] == "matched"
    assert match["confidence"] == 1.0


def test_skill_evidence_stays_individual_while_capability_score_is_grouped():
    role = {"role_id": "role", "requirement_metadata": {"requirements": [
        {"id": "spring-boot", "text": "Spring Boot", "skills": ["Spring Boot"], "category": "mandatory_skills", "alternative_group": "spring-ecosystem", "priority": 1},
        {"id": "spring-cloud", "text": "Spring Cloud Gateway", "skills": ["Spring Cloud Gateway"], "category": "mandatory_skills", "alternative_group": "spring-ecosystem", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-display", role, {"candidate_id": "candidate-1", "skills": ["Spring Boot"]}, evidence_index_factory=FakeIndex,
    )
    legacy = legacy_result_for_session(session)

    assert legacy["category_scores"]["mandatory_skills"]["required_count"] == 1
    assert legacy["category_scores"]["mandatory_skills"]["named_required_count"] == 2
    assert legacy["category_scores"]["mandatory_skills"]["named_matched_count"] == 1
    assert legacy["category_scores"]["mandatory_skills"]["named_weak_count"] == 1
    assert [row["skill"] for row in legacy["evidence"]["mandatory_skills"]] == ["Spring Boot", "Spring Cloud Gateway"]
    assert [row["status"] for row in legacy["evidence"]["mandatory_skills"]] == ["matched", "weak_match"]
    assert "80% scoring credit" in legacy["evidence"]["mandatory_skills"][1]["detail"]


def test_docker_and_kubernetes_are_partial_evidence_for_containerization():
    role = {"role_id": "platform-engineer", "requirement_metadata": {"requirements": [
        {"id": "containerization", "text": "Containerization", "skills": ["Containerization"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-container", role,
        {"candidate_id": "candidate-1", "skills": ["Docker", "Kubernetes"]},
        evidence_index_factory=FakeIndex,
    )
    match = session["final_matches"][0]
    assert match["decision"] == "weak"
    assert match["confidence"] == 0.80


def test_named_nosql_stores_explicitly_establish_nosql_capability():
    role = {"role_id": "data-platform", "requirement_metadata": {"requirements": [
        {"id": "nosql", "text": "NoSQL", "skills": ["NoSQL"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-nosql", role,
        {"candidate_id": "candidate-1", "skills": ["MongoDB (NoSQL)", "Aerospike"]},
        evidence_index_factory=FakeIndex,
    )
    # The resume explicitly names NoSQL in ``MongoDB (NoSQL)``.
    assert session["final_matches"][0]["decision"] == "matched"
    assert session["final_matches"][0]["confidence"] == 1.0


def test_named_relational_databases_explicitly_establish_sql_capability():
    role = {"role_id": "data-platform", "requirement_metadata": {"requirements": [
        {"id": "sql", "text": "SQL", "skills": ["SQL"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-sql", role,
        {"candidate_id": "candidate-1", "skills": ["Oracle DB", "MySQL"]},
        evidence_index_factory=FakeIndex,
    )
    assert session["final_matches"][0]["decision"] == "weak"
    assert session["final_matches"][0]["confidence"] == 0.80


def test_platform_and_cloud_service_relationships_are_related_not_exact():
    role = {"role_id": "platform-engineer", "requirement_metadata": {"requirements": [
        {"id": "github", "text": "GitHub", "skills": ["GitHub"], "category": "mandatory_skills", "priority": 1},
        {"id": "sqs", "text": "AWS SQS", "skills": ["AWS SQS"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-platform", role,
        {"candidate_id": "candidate-1", "skills": ["GitLab", "AWS"]},
        evidence_index_factory=FakeIndex,
    )
    assert {match["requirement_id"]: match["decision"] for match in session["final_matches"]} == {
        "github": "weak", "sqs": "weak",
    }


def test_github_capitalization_variant_is_an_exact_platform_match():
    role = {"role_id": "platform-engineer", "requirement_metadata": {"requirements": [
        {"id": "github", "text": "GitHub", "skills": ["GitHub"], "category": "mandatory_skills", "priority": 1},
    ]}}
    session = create_semantic_session(
        "session-github-spelling", role,
        {"candidate_id": "candidate-1", "skills": ["Github"]},
        evidence_index_factory=FakeIndex,
    )
    assert session["final_matches"][0]["decision"] == "matched"
