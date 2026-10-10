import pytest

from cross_encoder import CrossEncoderMatcher, route_candidate_matches
from webllm_contract import validate_ambiguity_judgments


def test_cross_encoder_routes_a_candidate_in_one_batch():
    calls = []

    def predict(pairs):
        calls.append(pairs)
        return [0.91 if requirement == "Python" else 0.56 for requirement, _ in pairs]

    matcher = CrossEncoderMatcher(predict_fn=predict, model="test-cross-encoder", model_version="v1")
    result = route_candidate_matches(
        "candidate-1",
        [{"id": "req-python", "text": "Python"}, {"id": "req-rag", "text": "RAG"}, {"id": "req-sql", "text": "SQL"}],
        {
            "req-python": [{"candidate_id": "candidate-1", "evidence_id": "ev-python", "text": "Built Python services."}],
            "req-rag": [{"candidate_id": "candidate-1", "evidence_id": "ev-rag", "text": "Built retrieval pipelines."}],
        },
        matcher,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert {match["requirement_id"] for match in result["final_matches"]} == {"req-python", "req-sql"}
    assert result["ambiguity_batch"]["candidate_id"] == "candidate-1"
    assert result["ambiguity_batch"]["items"][0]["requirement_id"] == "req-rag"


def test_cross_encoder_rejects_evidence_from_another_candidate():
    matcher = CrossEncoderMatcher(predict_fn=lambda pairs: [0.9] * len(pairs))
    with pytest.raises(ValueError, match="isolation"):
        route_candidate_matches(
            "candidate-1", [{"id": "req-python", "text": "Python"}],
            {"req-python": [{"candidate_id": "candidate-2", "evidence_id": "ev", "text": "Python"}]}, matcher,
        )


def test_webllm_contract_only_accepts_the_issued_ambiguity_pair():
    batch = {"candidate_id": "candidate-1", "items": [{"requirement_id": "req-rag", "evidence_id": "ev-rag"}]}
    accepted = validate_ambiguity_judgments(batch, {"judgments": [{
        "requirement_id": "req-rag", "evidence_id": "ev-rag", "decision": "matched", "confidence": 0.82,
        "reason": "The retrieval pipeline explicitly describes RAG-style retrieval."
    }]})
    assert accepted[0]["confidence"] == 0.82

    with pytest.raises(ValueError, match="unknown"):
        validate_ambiguity_judgments(batch, {"judgments": [{
            "requirement_id": "req-rag", "evidence_id": "wrong", "decision": "matched", "confidence": 0.82,
            "reason": "Wrong evidence id."
        }]})
