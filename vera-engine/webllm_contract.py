"""Validation boundary for browser-provided WebLLM ambiguity judgments."""
from __future__ import annotations

import math


def validate_ambiguity_judgments(batch: dict, response: dict) -> list[dict]:
    """Accept only decisions for the exact ambiguous requirement/evidence pairs sent."""
    items = batch.get("items") if isinstance(batch, dict) else None
    judgments = response.get("judgments") if isinstance(response, dict) else None
    if not isinstance(items, list) or not isinstance(judgments, list):
        raise ValueError("Ambiguity batch and response must contain arrays.")
    allowed = {
        (item.get("requirement_id"), item.get("evidence_id"))
        for item in items if isinstance(item, dict)
    }
    if len(judgments) != len(allowed):
        raise ValueError("WebLLM must return exactly one judgment per ambiguous item.")
    validated, seen = [], set()
    for judgment in judgments:
        if not isinstance(judgment, dict) or set(judgment) != {"requirement_id", "evidence_id", "decision", "confidence", "reason"}:
            raise ValueError("WebLLM judgment has an invalid shape.")
        key = (judgment["requirement_id"], judgment["evidence_id"])
        if key not in allowed or key in seen:
            raise ValueError("WebLLM judgment references an unknown or duplicate evidence pair.")
        if judgment["decision"] not in {"matched", "weak", "missing"}:
            raise ValueError("WebLLM judgment has an invalid decision.")
        confidence = judgment["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("WebLLM confidence must be a finite number from 0 through 1.")
        if not isinstance(judgment["reason"], str) or not judgment["reason"].strip():
            raise ValueError("WebLLM judgment requires a non-empty reason.")
        seen.add(key)
        validated.append({**judgment, "confidence": float(confidence)})
    return validated
