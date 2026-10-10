from benchmark_scoring import evaluate_recruiter_benchmark


def test_recruiter_benchmark_checks_score_bands_and_rank():
    result = evaluate_recruiter_benchmark(
        [{"candidate_id": "a", "expected_rank": 1, "min_score": 80, "max_score": 95}],
        [{"candidate_id": "a", "rank": 1, "final_score": 88}],
    )
    assert result == {"checked": 1, "passed": True, "failures": []}
