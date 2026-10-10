"""Batched cross-encoder matching and deterministic ambiguity routing."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Sequence


CROSS_ENCODER_MODEL = os.environ.get("TALENTLENS_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_MODEL_VERSION = os.environ.get("TALENTLENS_CROSS_ENCODER_MODEL_VERSION", CROSS_ENCODER_MODEL)
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.40


def calibrate_confidence(raw_score: float) -> float:
    """Convert a probability or cross-encoder logit into a bounded confidence."""
    value = float(raw_score)
    if not math.isfinite(value):
        raise ValueError("Cross-encoder returned a non-finite score.")
    if 0 <= value <= 1:
        return value
    # MS MARCO cross-encoders commonly emit logits, not probabilities.
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


class CrossEncoderMatcher:
    """Lazy model adapter with an injectable batch predictor for tests/hosting."""

    def __init__(self, *, model: str = CROSS_ENCODER_MODEL, model_version: str = CROSS_ENCODER_MODEL_VERSION,
                 predict_fn: Callable[[list[tuple[str, str]]], Sequence[float]] | None = None):
        self.model = model
        self.model_version = model_version
        self._predict_fn = predict_fn
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-encoder support requires sentence-transformers. Install the backend requirements first."
                ) from exc
            self._model = CrossEncoder(self.model)
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        raw = self._predict_fn(pairs) if self._predict_fn else self._load().predict(pairs, batch_size=32, show_progress_bar=False)
        if len(raw) != len(pairs):
            raise ValueError("Cross-encoder returned the wrong number of scores.")
        return [calibrate_confidence(score) for score in raw]


@dataclass(frozen=True)
class RoutedMatch:
    requirement_id: str
    candidate_id: str
    decision: str  # matched | missing | ambiguous
    confidence: float
    evidence_id: str | None
    method: str
    evidence: dict | None = None

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id, "candidate_id": self.candidate_id,
            "decision": self.decision, "confidence": round(self.confidence, 4),
            "evidence_id": self.evidence_id, "method": self.method, "evidence": self.evidence,
        }


def route_candidate_matches(
    candidate_id: str, requirements: Sequence[dict], evidence_by_requirement: dict[str, Sequence[dict]],
    matcher: CrossEncoderMatcher,
    *, high_threshold: float = HIGH_CONFIDENCE_THRESHOLD, low_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> dict:
    """Score every retrieved pair in one batch and route only uncertain items.

    Requirements with no retrieved evidence are deterministically missing. The
    return object is ready for exactly one browser WebLLM call per candidate.
    """
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= low < high <= 1.")
    pairs, bindings, routed = [], [], []
    for requirement in requirements:
        requirement_id = str(requirement.get("id") or "")
        text = str(requirement.get("text") or "")
        if not requirement_id or not text:
            raise ValueError("Each requirement needs id and text.")
        evidence_items = list(evidence_by_requirement.get(requirement_id) or [])
        if not evidence_items:
            routed.append(RoutedMatch(requirement_id, candidate_id, "missing", 0.0, None, "no_evidence"))
            continue
        for evidence in evidence_items:
            if evidence.get("candidate_id") != candidate_id:
                raise ValueError("Evidence candidate isolation violation.")
            evidence_id, evidence_text = evidence.get("evidence_id"), evidence.get("text")
            if not isinstance(evidence_id, str) or not isinstance(evidence_text, str) or not evidence_text.strip():
                raise ValueError("Evidence needs evidence_id and text.")
            pairs.append((text, evidence_text))
            bindings.append((requirement_id, evidence))
    scores = matcher.score_pairs(pairs)
    best: dict[str, tuple[float, dict]] = {}
    for (requirement_id, evidence), score in zip(bindings, scores):
        if requirement_id not in best or score > best[requirement_id][0]:
            best[requirement_id] = (score, evidence)
    for requirement_id, (confidence, evidence) in best.items():
        if confidence >= high_threshold:
            routed.append(RoutedMatch(requirement_id, candidate_id, "matched", confidence, evidence["evidence_id"], "cross_encoder", evidence))
        elif confidence < low_threshold:
            routed.append(RoutedMatch(requirement_id, candidate_id, "missing", confidence, evidence["evidence_id"], "cross_encoder", evidence))
        else:
            routed.append(RoutedMatch(requirement_id, candidate_id, "ambiguous", confidence, evidence["evidence_id"], "cross_encoder", evidence))
    routed.sort(key=lambda item: item.requirement_id)
    requirement_texts = {str(requirement["id"]): str(requirement["text"]) for requirement in requirements}
    final = [item.to_dict() for item in routed if item.decision != "ambiguous"]
    ambiguous = []
    for item in routed:
        if item.decision != "ambiguous":
            continue
        payload = item.to_dict()
        # The browser receives only the exact requirement/evidence pair it may
        # judge; it never needs the candidate's entire resume.
        payload["requirement_text"] = requirement_texts[item.requirement_id]
        payload["evidence_text"] = str((item.evidence or {}).get("text") or "")
        ambiguous.append(payload)
    return {
        "candidate_id": candidate_id, "model": matcher.model, "model_version": matcher.model_version,
        "thresholds": {"high": high_threshold, "low": low_threshold},
        "final_matches": final, "ambiguity_batch": {"candidate_id": candidate_id, "items": ambiguous},
    }
