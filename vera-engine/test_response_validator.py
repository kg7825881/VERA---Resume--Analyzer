import pytest

from assessment_schema import build_assessment
from response_validator import ResponseValidationError, validate_assessment_response


def _valid_response():
    return build_assessment(
        {"role_id": "role-1", "role_title": "Engineer", "mandatory_skills": ["Python"]},
        {"candidate_id": "candidate-1", "candidate_name": "Ada", "skills": ["Python"]},
        {"final_score": 75, "hard_gate_failed": False, "category_scores": {}},
    )


def test_response_validator_accepts_the_published_assessment_shape():
    response = _valid_response()
    assert validate_assessment_response(response) is response


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda response: response["assessment"].update({"evidence": {}}), "exactly"),
        (lambda response: response["candidate"].update({"source_markdown": "secret"}), "exactly"),
        (lambda response: response["assessment"].update({"score": 101}), "at most"),
        (lambda response: response.update({"schema_version": "0.9"}), "schema_version"),
    ],
)
def test_response_validator_rejects_contract_drift_and_internal_data(mutate, message):
    response = _valid_response()
    mutate(response)
    with pytest.raises(ResponseValidationError, match=message):
        validate_assessment_response(response)
