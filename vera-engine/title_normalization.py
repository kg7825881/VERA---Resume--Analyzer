"""Small, evidence-preserving helpers for employment-title parsing.

Resumes often flatten a company and role into one line (for example,
``Acme Technologies (AT) GenAI Engineer``).  A company name must never become
part of the title supplied to title matching.
"""
from __future__ import annotations

import re


_TITLE_TAIL_RE = re.compile(
    r"\b"
    r"(?:(?:senior|sr\.?|junior|jr\.?|lead|principal|staff|associate|assistant)\s+)?"
    r"(?:(?:ai/ml|ai|ml|genai|generative ai|machine learning|llm|nlp|software|data|"
    r"backend|front[- ]end|full[- ]stack|devops|platform|cloud|business|product|"
    r"systems|solutions|java|python|qa|test)\s+)?"
    r"(?:engineer|developer|scientist|analyst|manager|consultant|architect|specialist|"
    r"designer|administrator|coordinator|owner|lead)"
    r"(?:\s*(?:[-—|/]\s*)?(?:java|python|ai|ml|data platform))?"
    r"\b",
    re.IGNORECASE,
)


def split_flattened_company_title(value: str) -> tuple[str, str]:
    """Return ``(title, company_prefix)`` for a flattened company/role line.

    The role phrase is recognised only when it occurs after a non-empty prefix;
    clean role-only strings are returned untouched.  This keeps the original
    resume wording and avoids guessing from generic company names.
    """
    text = re.sub(r"\s+", " ", (value or "")).strip(" -–—,|")
    if not text:
        return "", ""

    candidates = list(_TITLE_TAIL_RE.finditer(text))
    for match in reversed(candidates):
        prefix = text[:match.start()].strip(" -–—,|")
        title = match.group(0).strip(" -–—,|")
        # One arbitrary leading word is usually part of a longer title; a
        # multiword or parenthesized prefix is strong evidence of an employer.
        if (len(title.split()) >= 2 and prefix
                and (len(prefix.split()) >= 2 or "(" in prefix or ")" in prefix)):
            return title, prefix
    return text, ""


def normalize_employment_title(value: str) -> str:
    """Remove only a demonstrable flattened employer prefix from a title."""
    return split_flattened_company_title(value)[0]
