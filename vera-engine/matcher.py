"""
matcher.py — skill/requirement matching for TalentLens.

Two-stage matching per requirement:
  1. Exact / word-boundary match against the candidate's own skill list
     (cheap, deterministic, no model calls at all).
  2. If no exact match: BM25 retrieves the candidate's most relevant resume
     evidence for the requirement (retrieval.py), then a small local LLM
     classifies how well that evidence actually supports the requirement
     (judge.py). The LLM is never asked for a score — only a classification
     ("direct" / "related" / "weak" / "none") — and that classification is
     converted into a numeric contribution deterministically, right here.

exact_only is a generic strictness flag, not tied to any one category. As of
this revision, scorer.py calls score_skill_list with exact_only=False for
BOTH mandatory and preferred skills — an exact match still earns free full
credit with no model call either way, but a non-exact match is now scored via
BM25 + judge evidence for both, rather than mandatory skills being forced to
zero out. The flag remains available for a caller that wants literal-match-
only behavior for some other requirement category in the future.

This replaces the previous embedding-cosine-similarity semantic stage.
Retrieval + judgment is more inspectable than a bare cosine number: every
non-exact match now carries the actual evidence quote it was judged against
and a one-sentence reason, which is what actually gets shown to HR. It also
removes an entire class of problems that came from embedding-model-specific
similarity thresholds (SEM_LOW/SEM_HIGH) needing separate recalibration
every time the embedding model changed — see calibrate_embeddings.py, which
existed only because of that.

Embeddings aren't gone from the codebase, just no longer part of the default
scoring path — see embeddings.py.
"""

import re

from capability_groups import (
    explicit_capability_evidence,
    partial_capability_evidence,
    requires_named_tool_evidence,
)
from skill_aliases import explicit_skill_in_values
from judge import judge_evidence
from retrieval import CandidateEvidenceIndex

# Minimum contribution a match must reach to satisfy a hard gate. A direct
# match and a concrete related match both clear this bar. This deliberately
# avoids requiring exact resume wording for a capability, while weak evidence
# remains partial credit only and cannot satisfy a mandatory-skill gate.
GATE_MIN_CONTRIBUTION = 0.8

# How many BM25-retrieved evidence chunks to hand the judge per requirement.
# Kept small: the judge only needs the strongest evidence, not the whole
# resume, and every extra chunk is more tokens per call across what can be
# dozens of requirements x candidates in one /analyze run.
DEFAULT_TOP_K = 3

# Deterministic mapping from the judge's 4-level classification to a numeric
# contribution. Tuned here, not in judge.py — the judge only classifies, it
# never sees or produces these numbers (same "LLM never does arithmetic"
# principle as experience.py / extractor.py). Shared by mandatory and
# preferred skills alike; split this into two mappings if mandatory and
# preferred should ever weight evidence confidence differently.
MATCH_LEVEL_CONTRIBUTION = {
    "direct": 1.0,
    # A deterministic reference phrase, capability-group signal, or model
    # judgment is useful but is not the JD's exact named skill. Keep it
    # visibly yellow and consistently below an approved alias/exact match.
    "related": 0.80,
    "weak": 0.55,
    "none": 0.0,
}

# Interchangeable skill names that satisfy exact matching bidirectionally.
#
# These are deliberately complete, named technologies/capabilities rather than
# arbitrary individual words.  For example, "Spark" can satisfy "PySpark",
# but "Data" must never satisfy "Data Lakehouse".  The resume extractor keeps
# the candidate's original wording; this table is used only at match time.
_EQUIVALENT_SKILL_GROUPS = (
    ("etl", "elt"),
    ("lakehouse", "data lakehouse"),
    ("spark", "apache spark", "pyspark"),
    ("airflow", "apache airflow", "airflow dags", "apache airflow dags"),
    ("rag", "retrieval augmented generation"),
    ("ocr", "optical character recognition"),
    ("genai", "generative ai"),
    ("llm", "llms", "large language model", "large language models"),
    ("postgres", "postgresql"),
    ("aws", "amazon web services"),
    ("gcp", "google cloud platform"),
    ("azure", "microsoft azure"),
)
EQUIVALENT_SKILLS = {
    item: set(group)
    for group in _EQUIVALENT_SKILL_GROUPS
    for item in group
}

# Evidence phrases that can support a JD capability when the capability name
# itself is absent. These are concrete related matches: they receive related
# credit and can satisfy a mandatory gate, just as a related LLM judgment can.
REFERENCE_SKILL_KEYWORDS = {
    "ocr": {"document digitization", "text extraction", "image preprocessing", "layout analysis", "handwriting recognition", "pdf parsing", "document classification", "table extraction", "quality validation", "multilingual processing", "PyMuPDF4LLM",},
    "retrieval datasets": {"knowledge corpus", "vector database", "document chunking", "metadata tagging", "relevance ranking", "semantic search", "hybrid retrieval", "data indexing", "source attribution", "retrieval evaluation"},
    "genai": {"large language models", "prompt engineering", "retrieval augmented generation", "fine tuning", "inference optimization", "content generation", "ai agents", "model evaluation", "guardrails", "responsible ai"},
    "dagster": {"data orchestration", "software defined assets", "pipeline scheduling", "asset lineage", "job execution", "data quality checks", "partition management", "resource configuration", "observability", "workflow automation"},
    "dbt": {"data transformation", "sql modeling", "analytics engineering", "data testing", "model documentation", "incremental models", "semantic layer", "data lineage", "warehouse optimization", "version control"},
    "alerting": {"incident notification", "threshold alerts", "anomaly detection", "escalation policies", "alert routing", "severity classification", "on call management", "root cause analysis", "service level objectives", "alert fatigue reduction"},
    "monitoring": {"system health", "performance metrics", "log analysis", "infrastructure visibility", "application telemetry", "uptime tracking", "capacity planning", "error tracking", "distributed tracing", "operational dashboards"},
    "schema design": {"database modeling", "normalization", "entity relationships", "primary keys", "foreign keys", "data types", "index strategy", "constraints", "dimensional modeling", "schema evolution"},
    "data modeling": {"database modeling", "dimensional modeling", "sql modeling", "data transformation", "entity relationships", "schema evolution"},
    "data warehouse": {"warehouse optimization", "dimensional modeling", "etl pipeline", "etl pipelines", "data integration", "data marts"},
    "lakehouse": {"databricks", "delta lake", "data lake", "lake storage", "medallion architecture"},
    "data validation": {"data quality checks", "data testing", "quality validation", "data quality control", "data integrity checks"},
    "data quality checks": {"data validation", "data testing", "quality validation", "data quality control", "data integrity checks"},
    "data quality control": {"data validation", "data testing", "quality validation", "data quality checks", "data integrity checks"},
    "embedding pipelines": {"vector embeddings", "text chunking", "embedding models", "semantic indexing", "batch processing", "similarity search", "vector storage", "metadata enrichment", "embedding refresh", "retrieval optimization"},
}


def _normalize(text: str) -> str:
    return text.strip().lower()


def _word_boundary_contains(needle: str, haystack: str) -> bool:
    """
    True if `needle` appears in `haystack` as a whole token/phrase, not just
    as a raw substring. Prevents false positives like "java" matching inside
    "javascript", while still allowing compound-phrase matches like "aws"
    matching "aws lambda" or "react" matching "react native".

    Boundary is defined as "not immediately adjacent to another letter/digit",
    so punctuation (spaces, +, #, /, etc.) counts as a valid edge. Note this
    means single/short tokens glued to symbols (e.g. required "c" against
    candidate "c++") can still match - that ambiguity is inherent to
    substring-based compound matching and isn't fully resolved by word
    boundaries alone.
    """
    if not needle:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _exact_match(required_skill: str, candidate_skills: list[str]) -> str | None:
    """Returns the candidate skill string that satisfies an exact/word-boundary
    match, or None if there isn't one."""
    required_norm = _normalize(required_skill)

    # Keep exact matching aligned with the approved alias registry used by
    # the Browser Semantic pipeline.  The registry contains only spelling,
    # grammatical, and recruiter-approved interchangeable variants, so this
    # can award green only for genuinely explicit evidence (for example,
    # "code reviews" for "Code review"), never for broad capability overlap.
    approved_alias = explicit_skill_in_values(required_skill, candidate_skills)
    if approved_alias is not None:
        return approved_alias

    # 1. Direct equality & interchangeable term checks (e.g., ETL <-> ELT)
    for cand in candidate_skills:
        cand_norm = _normalize(cand)
        if cand_norm == required_norm:
            return cand
        if required_norm in EQUIVALENT_SKILLS and cand_norm in EQUIVALENT_SKILLS[required_norm]:
            return cand

    # 2. Word-boundary containment checks
    req_variants = EQUIVALENT_SKILLS.get(required_norm, {required_norm})

    for cand in candidate_skills:
        cand_norm = _normalize(cand)

        # Direction 1: Required term (or its equivalent variant) appears inside candidate skill
        for req_var in req_variants:
            if _word_boundary_contains(req_var, cand_norm):
                return cand

        # Direction 2: Multi-token candidate skill appears inside required phrase
        if len(cand_norm.split()) > 1 and _word_boundary_contains(cand_norm, required_norm):
            return cand

    return None


def _reference_keyword_match(required_skill: str, candidate_skills: list[str]) -> tuple[str, str] | None:
    """Return candidate evidence and reference phrase for a partial capability match."""
    for candidate in candidate_skills:
        candidate_norm = _normalize(candidate)
        for phrase in REFERENCE_SKILL_KEYWORDS.get(_normalize(required_skill), set()):
            if _word_boundary_contains(phrase, candidate_norm):
                return candidate, phrase
    return None


def score_single_skill(
    required_skill: str,
    candidate_skills: list[str],
    evidence_index: CandidateEvidenceIndex,
    judge_fn=judge_evidence,
    exact_only: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Scores one required skill/requirement against a candidate.

    evidence_index: a CandidateEvidenceIndex built ONCE per candidate (see
    retrieval.py / scorer.py) and reused across every requirement scored
    against that candidate — the BM25 index doesn't need rebuilding per-skill.

    If exact_only is True, evidence-based matches are still retrieved and
    judged — so the UI can show *why* something is a near-miss — but
    contribute 0.0 either way: only a literal exact/word-boundary match is
    accepted. As of this revision neither mandatory nor preferred skills call
    this with exact_only=True by default; it's kept available for any future
    requirement category that should behave that strictly.
    """
    matched_against = _exact_match(required_skill, candidate_skills)
    if matched_against is not None:
        return {
            "skill": required_skill,
            "contribution": 1.0,
            "match_type": "exact",
            "gate_satisfied": True,
            "matched_against": matched_against,
            "evidence": [],
            "judge_reason": "",
        }

    # Approved related phrases are deterministic resume evidence, but not the
    # JD's exact named skill. Keep them as yellow reference evidence; only an
    # exact term or approved interchangeable alias above is green.
    direct_reason = explicit_capability_evidence(required_skill, set(candidate_skills))
    if direct_reason:
        return {
            "skill": required_skill,
            "contribution": MATCH_LEVEL_CONTRIBUTION["related"],
            "match_type": "reference",
            "gate_satisfied": MATCH_LEVEL_CONTRIBUTION["related"] >= GATE_MIN_CONTRIBUTION,
            "matched_against": None,
            "evidence": [],
            "judge_reason": direct_reason,
        }

    reference_match = _reference_keyword_match(required_skill, candidate_skills)
    if reference_match is not None:
        matched_against, reference_phrase = reference_match
        return {
            "skill": required_skill,
            "contribution": MATCH_LEVEL_CONTRIBUTION["related"],
            "match_type": "reference",
            "gate_satisfied": MATCH_LEVEL_CONTRIBUTION["related"] >= GATE_MIN_CONTRIBUTION,
            "matched_against": matched_against,
            "evidence": [],
            "judge_reason": f"Related capability evidence: '{reference_phrase}' supports '{required_skill}'.",
        }

    # Capability-group evidence is intentionally separate from the approved
    # direct vocabulary above: it remains yellow partial credit. It runs after
    # the legacy reference lookup so the UI can continue to show the concrete
    # phrase that triggered an existing yellow rule.
    group_reason = partial_capability_evidence(required_skill, set(candidate_skills))
    if group_reason:
        return {
            "skill": required_skill,
            "contribution": MATCH_LEVEL_CONTRIBUTION["related"],
            "match_type": "reference",
            "gate_satisfied": MATCH_LEVEL_CONTRIBUTION["related"] >= GATE_MIN_CONTRIBUTION,
            "matched_against": None,
            "evidence": [],
            "judge_reason": group_reason,
        }

    # A resume's generic capability language does not establish use of a
    # specific product. Keep named tools red unless exact/approved evidence
    # above supports them. The shared boundary is used by both Local Ollama
    # and Browser Semantic replay.
    if requires_named_tool_evidence(required_skill):
        return {
            "skill": required_skill,
            "contribution": 0.0,
            "match_type": "none",
            "gate_satisfied": False,
            "matched_against": None,
            "evidence": [],
            "judge_reason": (
                "Named tooling requires explicit product evidence; generic capability "
                "evidence is not sufficient."
            ),
        }

    evidence_chunks = evidence_index.retrieve(required_skill, top_k=top_k)
    judgment = judge_fn(required_skill, evidence_chunks)
    match_level = judgment["match"]
    # The model can classify evidence as highly convincing, but it is not
    # permitted to manufacture a green named-skill match.  Exact terms and
    # approved interchangeable aliases have already returned above.  Every
    # model-mediated positive is therefore related evidence: yellow in the
    # UI and 80% credit in both Local Ollama and Browser Semantic runs.
    #
    # This normalization belongs at the shared scorer boundary rather than
    # inside either model prompt, so changing models cannot loosen the policy.
    model_positive = match_level in {"direct", "related"}
    normalized_match_level = "related" if model_positive else match_level
    base_contribution = MATCH_LEVEL_CONTRIBUTION.get(normalized_match_level, 0.0)
    found_evidence = base_contribution > 0.0
    evidence_label = evidence_chunks[0]["source_label"] if (found_evidence and evidence_chunks) else None
    judge_reason = judgment.get("reason", "")
    if match_level == "direct":
        judge_reason = (
            f"{judge_reason} Model evidence is related; direct named-skill evidence was not found."
        ).strip()

    if exact_only:
        # Evidence was found and judged (useful for the UI's "near miss" view),
        # but exact_only means only a literal skill-list match is accepted —
        # zero contribution ensures it fails the gate and renders red.
        return {
            "skill": required_skill,
            "contribution": 0.0,
            "match_type": "evidence" if found_evidence else "none",
            "gate_satisfied": False,
            "matched_against": evidence_label,
            "evidence": evidence_chunks,
            "judge_reason": judge_reason,
        }

    return {
        "skill": required_skill,
        "contribution": base_contribution,
        "match_type": "evidence" if found_evidence else "none",
        "gate_satisfied": base_contribution >= GATE_MIN_CONTRIBUTION,
        "matched_against": evidence_label,
        "evidence": evidence_chunks,
        "judge_reason": judge_reason,
    }


def score_skill_list(
    required_skills: list[str],
    candidate_skills: list[str],
    evidence_index: CandidateEvidenceIndex,
    judge_fn=judge_evidence,
    exact_only: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Scores an entire list of required skills/requirements against a candidate's skill list.
    """
    required_skills = [skill.strip() for skill in (required_skills or []) if isinstance(skill, str) and skill.strip()]
    if not required_skills:
        return {"results": [], "matched": [], "missing": [], "gate_missing": [], "average_contribution": 1.0}

    results = [
        score_single_skill(skill, candidate_skills, evidence_index, judge_fn, exact_only, top_k)
        for skill in required_skills
    ]
    matched = [r["skill"] for r in results if r["contribution"] > 0.0]
    missing = [r["skill"] for r in results if r["contribution"] == 0.0]
    gate_missing = [r["skill"] for r in results if not r["gate_satisfied"]]
    avg = sum(r["contribution"] for r in results) / len(results)

    return {"results": results, "matched": matched, "missing": missing, "gate_missing": gate_missing, "average_contribution": avg}


def evidence_status(result: dict) -> str:
    """
    Maps one score_single_skill result to a 3-state UI status for the evidence view.

    Green is reserved for an exact term or an approved interchangeable alias.
    Every positive non-exact result (capability group, reference phrase, or
    model evidence) remains yellow, even when its 80% credit satisfies the
    mandatory coverage gate.  This keeps the recruiter-visible color aligned
    with the cross-engine scoring policy.
    """
    if result["contribution"] == 0.0:
        return "missing"
    if result.get("match_type") != "exact":
        return "weak_match"
    return "matched"
