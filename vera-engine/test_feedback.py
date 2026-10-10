import pytest

from feedback import validate_feedback


def test_feedback_contract_accepts_a_disposition_and_score_snapshot():
    assert validate_feedback("shortlist", "Strong evidence.", 88.5) == ("shortlist", "Strong evidence.", 88.5)


@pytest.mark.parametrize("disposition", ["maybe", "", None])
def test_feedback_contract_rejects_unknown_dispositions(disposition):
    with pytest.raises(ValueError, match="Disposition"):
        validate_feedback(disposition)
