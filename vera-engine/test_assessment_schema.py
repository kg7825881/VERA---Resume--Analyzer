from assessment_schema import build_assessment


def test_build_assessment_is_compact_and_preserves_decision_fields():
    result = build_assessment(
        {
            "role_id": "data-engineer",
            "role_title": "Data Engineer",
            "target_job_titles": ["Data Engineer", "data engineer"],
            "mandatory_skills": ["Python", "SQL"],
            "min_years_experience": 3,
            "education_requirements": [{"degree_level": "Bachelor's", "field": "Computer Science"}],
        },
        {
            "candidate_id": "candidate-1",
            "candidate_name": "Ada Lovelace",
            "total_years_experience": 4,
            "skills": ["Python", "SQL", "Python"],
            "experience": [{"title": "Data Engineer", "company": "Analytical Engines"}],
            "raw_resume_text": "must never appear in the response",
        },
        {
            "final_score": 88.5,
            "hard_gate_failed": False,
            "hard_gate_reason": "",
            "scored_at": "2026-09-18T00:00:00Z",
            "category_scores": {
                "mandatory_skills": {"score": 30, "matched": ["Python"], "missing": ["SQL"]},
            },
            "evidence": {"very": "large"},
        },
    )

    assert result["schema_version"] == "1.0"
    assert result["jd"]["requirements"] == [
        {"name": "Python", "kind": "mandatory"},
        {"name": "SQL", "kind": "mandatory"},
        {"name": "3+ years experience", "kind": "experience"},
        {"name": "Bachelor's in Computer Science", "kind": "education"},
    ]
    assert result["candidate"]["headline"] == "Data Engineer"
    assert result["candidate"]["skills"] == ["Python", "SQL"]
    assert result["assessment"]["eligible"] is True
    assert "evidence" not in result["assessment"]
    assert "raw_resume_text" not in str(result)


def test_build_assessment_tolerates_persisted_string_numbers_and_bad_optional_values():
    result = build_assessment(
        {"min_years_experience": "2.5", "education_requirements": [None]},
        {"candidate_id": "candidate-1", "total_years_experience": "not available"},
        {"final_score": "87.5", "category_scores": {"skills": {"score": "12.5"}}},
    )

    assert result["jd"]["requirements"] == [{"name": "2.5+ years experience", "kind": "experience"}]
    assert result["candidate"]["years_experience"] == 0
    assert result["assessment"]["score"] == 87.5
    assert result["assessment"]["dimensions"] == [
        {"key": "skills", "score": 12.5, "matched": [], "missing": []}
    ]
