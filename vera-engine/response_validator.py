"""Runtime guard for the versioned public assessment response.

The persistence model intentionally contains raw Markdown, resume extraction
detail, and judge evidence.  This validator is the last boundary before the
compact assessment is returned from the API: it detects contract drift and
prevents those internal fields from accidentally becoming public.
"""
from __future__ import annotations

import math
from typing import Any

from assessment_schema import SCHEMA_VERSION


class ResponseValidationError(ValueError):
    """Raised when an API response does not satisfy the published contract."""


_REQUIREMENT_KINDS = {"mandatory", "preferred", "role_specific", "experience", "education"}
_FORBIDDEN_FIELDS = {"raw_resume_text", "source_markdown", "evidence", "raw_output"}


def _fail(path: str, message: str) -> None:
    raise ResponseValidationError(f"Invalid assessment response at {path}: {message}")


def _object(value: Any, path: str, keys: set[str]) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    actual = set(value)
    if actual != keys:
        _fail(path, f"must contain exactly {sorted(keys)}; got {sorted(actual)}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    return value


def _number(value: Any, path: str, minimum: float | None = None, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(path, "must be a finite number")
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be at most {maximum}")


def _strings(value: Any, path: str) -> None:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    for index, item in enumerate(value):
        _string(item, f"{path}[{index}]")


def validate_assessment_response(response: Any) -> dict:
    """Validate and return an assessment response without coercing data.

    Deliberately avoiding coercion makes contract failures observable instead
    of silently changing the meaning of an assessment at the HTTP boundary.
    """
    response = _object(response, "$", {"schema_version", "jd", "candidate", "assessment"})
    if response["schema_version"] != SCHEMA_VERSION:
        _fail("$.schema_version", f"must equal {SCHEMA_VERSION!r}")

    jd = _object(response["jd"], "$.jd", {"id", "title", "target_titles", "requirements"})
    _string(jd["id"], "$.jd.id")
    _string(jd["title"], "$.jd.title")
    _strings(jd["target_titles"], "$.jd.target_titles")
    if not isinstance(jd["requirements"], list):
        _fail("$.jd.requirements", "must be an array")
    for index, requirement in enumerate(jd["requirements"]):
        requirement = _object(requirement, f"$.jd.requirements[{index}]", {"name", "kind"})
        _string(requirement["name"], f"$.jd.requirements[{index}].name")
        if requirement["kind"] not in _REQUIREMENT_KINDS:
            _fail(f"$.jd.requirements[{index}].kind", "is not a supported requirement kind")

    candidate = _object(response["candidate"], "$.candidate", {"id", "name", "headline", "years_experience", "skills"})
    for key in ("id", "name", "headline"):
        _string(candidate[key], f"$.candidate.{key}")
    _number(candidate["years_experience"], "$.candidate.years_experience", minimum=0)
    _strings(candidate["skills"], "$.candidate.skills")

    assessment = _object(response["assessment"], "$.assessment", {"score", "eligible", "decision_reason", "dimensions", "scored_at"})
    _number(assessment["score"], "$.assessment.score", minimum=0, maximum=100)
    if not isinstance(assessment["eligible"], bool):
        _fail("$.assessment.eligible", "must be a boolean")
    _string(assessment["decision_reason"], "$.assessment.decision_reason")
    _string(assessment["scored_at"], "$.assessment.scored_at")
    if not isinstance(assessment["dimensions"], list):
        _fail("$.assessment.dimensions", "must be an array")
    for index, dimension in enumerate(assessment["dimensions"]):
        dimension = _object(dimension, f"$.assessment.dimensions[{index}]", {"key", "score", "matched", "missing"})
        _string(dimension["key"], f"$.assessment.dimensions[{index}].key")
        _number(dimension["score"], f"$.assessment.dimensions[{index}].score", minimum=0, maximum=100)
        _strings(dimension["matched"], f"$.assessment.dimensions[{index}].matched")
        _strings(dimension["missing"], f"$.assessment.dimensions[{index}].missing")

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            leaked = _FORBIDDEN_FIELDS.intersection(value)
            if leaked:
                _fail("$", f"contains forbidden internal field(s): {sorted(leaked)}")
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(response)
    return response
