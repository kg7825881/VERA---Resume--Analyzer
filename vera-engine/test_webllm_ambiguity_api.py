from fastapi.testclient import TestClient

from api import app


def test_webllm_ambiguity_validation_endpoint_returns_only_validated_judgments():
    client = TestClient(app)
    response = client.post("/webllm/ambiguity/validate", json={
        "batch": {"candidate_id": "candidate-1", "items": [{"requirement_id": "req-1", "evidence_id": "ev-1"}]},
        "response": {"judgments": [{
            "requirement_id": "req-1", "evidence_id": "ev-1", "decision": "weak", "confidence": 0.58,
            "reason": "The evidence is related but indirect."
        }]},
    })
    assert response.status_code == 200
    assert response.json()["validated"] is True


def test_webllm_ambiguity_validation_endpoint_rejects_unissued_evidence():
    client = TestClient(app)
    response = client.post("/webllm/ambiguity/validate", json={
        "batch": {"candidate_id": "candidate-1", "items": [{"requirement_id": "req-1", "evidence_id": "ev-1"}]},
        "response": {"judgments": [{
            "requirement_id": "req-1", "evidence_id": "ev-other", "decision": "matched", "confidence": 0.9,
            "reason": "This should be rejected."
        }]},
    })
    assert response.status_code == 422
