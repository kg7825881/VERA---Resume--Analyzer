from workflow import cache_key, validate_analysis_request


def test_cache_key_changes_when_scoring_evidence_changes():
    jd = {"role_title": "Data Engineer", "mandatory_skills": ["Python"]}
    candidate = {"candidate_id": "a", "skills": ["Python"], "experience": []}
    original = cache_key(jd, candidate)
    candidate["skills"] = ["Python", "SQL"]
    assert cache_key(jd, candidate) != original


def test_audit_mode_uses_a_separate_cache_key():
    jd = {"role_title": "Data Engineer", "mandatory_skills": ["Python"]}
    candidate = {"candidate_id": "a", "skills": ["Python"], "experience": []}
    assert cache_key(jd, candidate) != cache_key(jd, candidate, audit_mode=True)


def test_analysis_request_validation_rejects_unknown_and_duplicate_ids():
    assert validate_analysis_request(["a", "a"], {"a"}) == ["candidate_ids contains duplicates: a."]
    assert validate_analysis_request(["missing"], {"a"}) == ["Unknown candidate_ids: missing."]


def test_run_task_transitions_are_counted_once(tmp_path, monkeypatch):
    import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    db.create_screening_run("run-1", "role-1", ["a", "b"], "2026-01-01T00:00:00Z", audit_mode=True)
    db.start_screening_task("run-1", "a", "2026-01-01T00:00:01Z")
    db.complete_screening_task("run-1", "a", True, "2026-01-01T00:00:02Z")
    db.complete_screening_task("run-1", "a", True, "2026-01-01T00:00:03Z")
    db.fail_screening_task("run-1", "b", "model unavailable", "2026-01-01T00:00:04Z")

    run = db.get_screening_run("run-1")
    assert run["completed_count"] == 1
    assert run["audit_mode"] is True
    assert run["failed_count"] == 1
    assert run["progress"] == {"total": 2, "completed": 1, "failed": 1, "pending": 0}
    assert [task["status"] for task in run["tasks"]] == ["completed", "failed"]
