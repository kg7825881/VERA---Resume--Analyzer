import pytest

import db
from jd_policy import (
    POLICY_SCHEMA_VERSION,
    apply_recruiter_overrides_to_metadata,
    build_requirement_metadata,
    refresh_stale_requirement_metadata,
)


def _jd():
    return {
        "role_id": "data-engineer-ai-data-platform",
        "role_title": "Data Engineer — AI Data Platform",
        "source_id": "src_jd_1",
        "mandatory_skills": ["Python", "SQL", "ETL", "ELT", "Data modeling", "Airflow"],
        "mandatory_domain_requirements": ["Fintech"],
        "mandatory_role_specific_requirements": [],
        "preferred_technical_skills": ["Spark"],
        "soft_preferred_skills": [],
        "industry_keywords": [],
        "min_years_experience": 3,
        "education_requirements": [{"degree_level": "Bachelor's", "field": "Computer Science"}],
        "relevant_certifications": ["AWS Data Engineer"],
    }


def test_policy_creates_stable_requirements_and_only_marks_jd_present_role_defaults():
    metadata = build_requirement_metadata(_jd())
    requirements = metadata["requirements"]
    by_text = {item["text"]: item for item in requirements}

    assert metadata["schema_version"] == POLICY_SCHEMA_VERSION
    assert by_text["Python"]["non_negotiable"] is True
    assert by_text["SQL"]["non_negotiable"] is True
    assert by_text["ETL"]["alternative_group"] == "data-etl-elt"
    assert by_text["Spark"]["non_negotiable"] is False
    assert "Kubernetes" not in by_text
    assert by_text["3+ years relevant experience"]["category"] == "relevant_experience"
    assert by_text["Bachelor's in Computer Science"]["category"] == "education"
    assert by_text["AWS Data Engineer"]["category"] == "certifications"
    assert all(item["source_id"] == "src_jd_1" for item in requirements)


def test_java_policy_preserves_jd_required_skills_and_limits_only_the_hard_gate():
    jd = _jd() | {
        "role_id": "sr-java-lead-engineer", "role_title": "Sr Java Lead Engineer",
        "mandatory_skills": ["JAVA", "Spring Boot", "Apache Kafka", "RESTful APIs", "Monitoring"],
        "education_requirements": [{"degree_level": "B.Tech", "field": ""}, {"degree_level": "MCA", "field": ""}],
    }
    by_text = {item["text"]: item for item in build_requirement_metadata(jd)["requirements"]}
    assert by_text["JAVA"]["category"] == "mandatory_skills"
    assert by_text["Apache Kafka"]["category"] == "mandatory_skills"
    assert by_text["RESTful APIs"]["skills"] == ["API design"]
    assert by_text["Monitoring"]["category"] == "mandatory_skills"
    assert by_text["Monitoring"]["non_negotiable"] is False
    assert by_text["Spring Boot"]["alternative_group"] == "spring-ecosystem"
    assert by_text["Apache Kafka"]["alternative_group"] == "event-streaming-messaging"
    assert by_text["B.Tech"]["alternative_group"] == "education-degree-options"
    assert by_text["MCA"]["alternative_group"] == "education-degree-options"
    assert any(item["category"] == "role_alignment" for item in by_text.values())


def test_capability_groups_are_reusable_across_role_families():
    jd = _jd() | {
        "mandatory_skills": ["Prometheus", "Grafana", "Kubernetes", "Docker", "ETL", "ELT"],
    }
    by_text = {item["text"]: item for item in build_requirement_metadata(jd)["requirements"]}
    assert by_text["Prometheus"]["alternative_group"] == by_text["Grafana"]["alternative_group"] == "observability"
    assert by_text["Kubernetes"]["alternative_group"] == by_text["Docker"]["alternative_group"] == "cloud-native-platform"
    assert by_text["ETL"]["alternative_group"] == by_text["ELT"]["alternative_group"] == "data-etl-elt"


def test_cloud_and_genai_capability_groups_are_generic_and_specific():
    jd = _jd() | {
        "mandatory_skills": ["AWS", "Lambda", "S3", "LLMs", "RAG", "Pinecone", "LangChain"],
    }
    by_text = {item["text"]: item for item in build_requirement_metadata(jd)["requirements"]}
    assert by_text["AWS"]["alternative_group"] == by_text["Lambda"]["alternative_group"] == by_text["S3"]["alternative_group"] == "aws-cloud-platform"
    assert by_text["LLMs"]["alternative_group"] == "llm-genai-development"
    assert by_text["RAG"]["alternative_group"] == by_text["Pinecone"]["alternative_group"] == by_text["LangChain"]["alternative_group"] == "rag-retrieval-systems"


def test_stale_policy_rebuild_does_not_create_absent_preferred_categories():
    jd = _jd() | {
        "preferred_technical_skills": [], "soft_preferred_skills": [], "industry_keywords": [],
        "requirement_metadata": {
            "schema_version": "2.1",
            "requirements": [{"id": "old", "category": "preferred_skills"}],
            "recruiter_overrides": {},
        },
    }
    refreshed = refresh_stale_requirement_metadata(jd)
    assert refreshed is not None
    assert refreshed["schema_version"] == POLICY_SCHEMA_VERSION
    assert "preferred_skills" not in {item["category"] for item in refreshed["requirements"]}


def test_recruiter_override_wins_and_unknown_ids_are_rejected():
    metadata = build_requirement_metadata(_jd())
    spark = next(item for item in metadata["requirements"] if item["text"] == "Spark")
    updated = apply_recruiter_overrides_to_metadata(
        metadata, {spark["id"]: {"non_negotiable": True, "priority": 1.5}}, reject_unknown=True,
    )

    updated_spark = next(item for item in updated["requirements"] if item["id"] == spark["id"])
    assert updated_spark["non_negotiable"] is True
    assert updated_spark["priority"] == 1.5
    assert updated["recruiter_overrides"][spark["id"]]["priority"] == 1.5

    with pytest.raises(ValueError, match="Unknown requirement override ids"):
        apply_recruiter_overrides_to_metadata(metadata, {"req_missing": {"priority": 1.2}}, reject_unknown=True)


def test_db_persists_requirement_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    record = _jd()
    record.update({"document_id": "doc_1", "file_name": "jd.docx", "requirement_metadata": build_requirement_metadata(record)})
    db.insert_jd(record)
    python_requirement = next(item for item in record["requirement_metadata"]["requirements"] if item["text"] == "Python")

    updated = db.update_jd_requirement_overrides(
        record["role_id"], {python_requirement["id"]: {"non_negotiable": False}},
    )

    stored = next(item for item in updated["requirement_metadata"]["requirements"] if item["id"] == python_requirement["id"])
    assert stored["non_negotiable"] is False


def test_db_reupload_upgrades_legacy_empty_requirement_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    record = _jd()
    record.update({"document_id": "doc_legacy", "file_name": "jd.docx", "requirement_metadata": []})
    db.insert_jd(record)

    upgraded = _jd()
    upgraded.update({
        "document_id": "doc_current",
        "file_name": "jd.docx",
        "requirement_metadata": build_requirement_metadata(upgraded),
    })
    assert db.insert_jd(upgraded) == "refreshed"
    assert db.get_jd_by_role_id(upgraded["role_id"])["requirement_metadata"]["requirements"]
