import db

from chunking import build_evidence_chunks, build_parent_child_chunks


def _candidate(candidate_id="candidate-1"):
    return {
        "candidate_id": candidate_id,
        "document_id": "doc-1",
        "source_id": "src_resume_1",
        "skills_all_sources": ["Python", "SQL", "Airflow"],
        "experience": [{
            "title": "Data Engineer", "company": "Acme", "page": 2,
            "description": "Built production ETL pipelines.",
            "technologies_used": ["Python", "Airflow"],
        }],
        "projects": [{"title": "Warehouse migration", "description": "Migrated warehouse jobs.", "page": 3}],
        "education": [{"degree_level": "B.Tech", "field": "Computer Science", "institution": "Example University"}],
        "certifications": ["AWS Certified Data Engineer"],
    }


def test_evidence_chunks_are_stable_section_aware_and_candidate_isolated():
    candidate = _candidate()
    first = build_evidence_chunks(candidate)
    second = build_evidence_chunks(candidate)
    other_candidate = build_evidence_chunks(_candidate("candidate-2"))

    assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]
    assert {chunk.section for chunk in first} == {"experience", "project", "skills", "education", "certifications"}
    experience = next(chunk for chunk in first if chunk.section == "experience")
    assert experience.page == 2
    assert experience.source_id == "src_resume_1"
    assert experience.source_label == "Data Engineer at Acme"
    assert experience.content_hash
    assert experience.evidence_id != next(chunk for chunk in other_candidate if chunk.section == "experience").evidence_id
    assert {chunk.candidate_id for chunk in first} == {"candidate-1"}


def test_parent_child_retrieval_uses_persisted_evidence_without_rebuilding_it():
    candidate = _candidate()
    persisted = [chunk.to_dict() for chunk in build_evidence_chunks(candidate)]
    parents, children = build_parent_child_chunks({"candidate_id": "wrong-candidate", "evidence_chunks": persisted})

    assert len(parents) == len(persisted)
    assert len(children) >= len(parents)
    assert {parent.candidate_id for parent in parents} == {"candidate-1"}
    assert parents[0].to_dict()["evidence_id"] == persisted[0]["evidence_id"]


def test_db_persists_evidence_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    candidate = _candidate()
    candidate["evidence_chunks"] = [chunk.to_dict() for chunk in build_evidence_chunks(candidate)]
    db.insert_resume(candidate)

    stored = db.get_resume_by_candidate_id("candidate-1")
    assert stored["evidence_chunks"] == candidate["evidence_chunks"]
