from extractor import extract_structured_evidence
from job_title_matcher import score_job_titles


def _no_match_judge(requirement, evidence):
    return {"match": "none", "confidence": 1.0, "reason": "test judge"}


def test_flattened_company_prefix_is_not_persisted_as_job_title():
    resume = """
    ROSHAN SINGH
    Work Experience
    Bosch Global Software Technologies (BGSW) GenAI Engineer
    Jan 2023 - Present
    Built RAG services using Python.
    """

    result = extract_structured_evidence(resume)

    assert result["experience"][0]["title"] == "GenAI Engineer"
    assert result["experience"][0]["company"] == "Bosch Global Software Technologies (BGSW)"
    assert result["current_role_title_from_summary"] == "GenAI Engineer"


def test_ai_genai_title_family_expands_through_document_qualifiers():
    jd_title = "AI Engineer — GenAI Product Engineering [Production focus Role]"

    for candidate_title in ("GenAI Engineer", "Software Engineer", "AI/ML Engineer"):
        result = score_job_titles(
            {"experience": [{"title": candidate_title, "company": "Example", "end_date_raw": "Present"}]},
            jd_title,
            judge_fn=_no_match_judge,
        )
        assert result["contribution"] == 1.0
        assert result["best_match"]["title"] == candidate_title
