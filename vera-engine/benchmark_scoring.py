"""Recruiter-approved benchmark evaluation for score-policy releases.

The benchmark deliberately stores expected ranking and score bands, not model
prompts.  It is suitable for a checked-in, reviewed set of recruiter labels.
"""
from __future__ import annotations

from typing import Iterable


def evaluate_recruiter_benchmark(expected: Iterable[dict], actual_scores: Iterable[dict]) -> dict:
    """Compare scored candidates against approved rank and score bands.

    Each expected item requires ``candidate_id`` and may provide
    ``min_score``, ``max_score``, and ``expected_rank``.  The return value is
    machine-readable so CI can block a scoring-policy change that breaks an
    approved recruiter decision.
    """
    by_id = {str(item.get("candidate_id")): item for item in actual_scores}
    failures = []
    checked = 0
    for target in expected:
        candidate_id = str(target.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("Benchmark cases need candidate_id.")
        actual = by_id.get(candidate_id)
        if actual is None:
            failures.append({"candidate_id": candidate_id, "reason": "candidate was not scored"})
            continue
        checked += 1
        score = float(actual.get("final_score", 0))
        if "min_score" in target and score < float(target["min_score"]):
            failures.append({"candidate_id": candidate_id, "reason": f"score {score} is below approved minimum {target['min_score']}"})
        if "max_score" in target and score > float(target["max_score"]):
            failures.append({"candidate_id": candidate_id, "reason": f"score {score} is above approved maximum {target['max_score']}"})
        if "expected_rank" in target and int(actual.get("rank", 0)) != int(target["expected_rank"]):
            failures.append({"candidate_id": candidate_id, "reason": f"rank {actual.get('rank')} differs from approved rank {target['expected_rank']}"})
    return {"checked": checked, "passed": not failures, "failures": failures}
