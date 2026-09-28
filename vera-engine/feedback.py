"""Small, explicit human-feedback contract for later offline evaluation and LTR."""
from __future__ import annotations

import math


RECRUITER_DISPOSITIONS = frozenset({"shortlist", "hold", "reject"})
MAX_NOTE_LENGTH = 2_000


def validate_feedback(disposition: object, note: object = "", score_snapshot: object = None) -> tuple[str, str, float | None]:
    if not isinstance(disposition, str) or disposition not in RECRUITER_DISPOSITIONS:
        raise ValueError("Disposition must be shortlist, hold, or reject.")
    if not isinstance(note, str):
        raise ValueError("Feedback note must be text.")
    cleaned_note = note.strip()
    if len(cleaned_note) > MAX_NOTE_LENGTH:
        raise ValueError(f"Feedback note must be at most {MAX_NOTE_LENGTH} characters.")
    if score_snapshot is None:
        return disposition, cleaned_note, None
    if isinstance(score_snapshot, bool) or not isinstance(score_snapshot, (int, float)) or not math.isfinite(score_snapshot) or not 0 <= score_snapshot <= 100:
        raise ValueError("Score snapshot must be a number from 0 through 100.")
    return disposition, cleaned_note, float(score_snapshot)
