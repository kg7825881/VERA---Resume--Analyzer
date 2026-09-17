"""
job_title_matcher.py — job-title relevance scoring (the "Job Titles" scoring
category — one of the ATS-style comparison points alongside Industry Keywords
and Soft Skills).

Previous revision used pure token-overlap (Jaccard) — deliberately deterministic
and cheap, but with a known, explicitly-documented limitation: it misses true
synonyms/related roles that share no words at all (e.g. "AI Architect" vs
required "Data Engineer" scores 0 despite plausibly being a related role,
depending on context). That limitation is exactly what this revision fixes.

The matcher uses the extracted current/latest employment title only. Exact
matches to the JD title or its approved title-family references receive full
credit. A supported adjacent title can receive partial credit; an unsupported
title receives none. The judge is used only for that non-exact decision.

Jaccard token overlap is retained only as inspectable display metadata. It
does not decide the score.
"""

import re
from judge import judge_evidence
from matcher import MATCH_LEVEL_CONTRIBUTION

STOPWORDS = {"the", "a", "an", "of", "and", "for"}

# Approved title families supplied by the hiring workflow.  These are exact
# acceptable alternatives for a JD's primary title, not broad synonym guesses:
# a candidate earns full title credit only when their resolved latest title is
# one of these stated titles (or the JD title itself).  New role families can
# be added as data without changing matching logic.
ROLE_TITLE_REFERENCE = {
    "Data Engineer — AI Data Platform": (
        "AI Data Engineer",
        "Data Platform Engineer",
        "Machine Learning Data Engineer",
        "Senior Data Engineer",
        "Big Data Engineer",
    ),
    "Senior Business Analyst": (
        "Lead Business Analyst",
        "Senior IT Business Analyst",
        "Business Systems Analyst",
        "Senior Product Analyst",
        "Functional Business Analyst",
    ),
    "DevOps Engineer": (
        "Site Reliability Engineer",
        "Cloud DevOps Engineer",
        "Platform Engineer",
        "Infrastructure Engineer",
        "CI/CD Engineer",
    ),
    "Sr Java Lead Engineer": (
        "Java Technical Lead",
        "Senior Java Developer",
        "Lead Software Engineer — Java",
        "Java Solutions Architect",
        "Backend Engineering Lead",
    ),
}


def _title_key(title: str) -> str:
    """Comparison key that ignores presentation-only punctuation and case."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").casefold())


def _bounded_related_title(title: str, target_role_title: str) -> str | None:
    """Recognise a small set of defensible, adjacent title families.

    This is deliberately a weak match, not a synonym system.  It captures
    roles such as Data Scientist and AI Technical Lead for an AI data-engineer
    opening without letting an unrelated Product Manager receive title credit
    merely because the resume mentions AI elsewhere.
    """
    title_text = (title or "").lower()
    target_text = (target_role_title or "").lower()
    if "data engineer" not in target_text:
        return None
    if "data scientist" in title_text or "data analyst" in title_text:
        return "Adjacent data role; relevant but not the same title."
    if "ai technical lead" in title_text or "technical lead" in title_text and "ai" in title_text:
        return "AI technical leadership is adjacent to the AI data-engineering role."
    # A resume can state the base role while the JD title adds a speciality,
    # e.g. "Data Engineer" versus "Data Engineer — AI Data Platform".  This
    # is relevant but not an approved full-credit title variant.  Match only
    # a contiguous phrase of at least two role words so generic words such as
    # "Engineer" never create title credit by themselves.
    candidate_words = _title_words(title)
    target_words = _title_words(target_role_title)
    role_words = {"engineer", "analyst", "developer", "manager", "architect", "consultant", "lead"}
    for width in range(min(4, len(candidate_words)), 1, -1):
        for index in range(len(candidate_words) - width + 1):
            phrase = candidate_words[index:index + width]
            if not (set(phrase) & role_words):
                continue
            if any(target_words[target_index:target_index + width] == phrase
                   for target_index in range(len(target_words) - width + 1)):
                return "Matches the JD's base role but does not state its full speciality."
    return None


def _is_plausible_title(value: str) -> bool:
    """Reject a resume-summary sentence accidentally extracted as a title."""
    text = (value or "").strip()
    words = re.findall(r"[A-Za-z0-9+#.&/-]+", text)
    return bool(text) and len(text) <= 100 and len(words) <= 12 and not re.search(r"[.;\n]", text)


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _employment_recency_key(entry: dict) -> tuple[int, int]:
    """Sort an employment record by its stated end date without guessing dates."""
    value = (entry.get("end_date_raw") or "").strip().casefold()
    if re.fullmatch(r"(?:present|current|ongoing|till date)", value):
        return (9999, 12)
    year = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep(?:t)?|oct|nov|dec)[a-z]*\.?", value)
    return (int(year.group(1)) if year else 0, _MONTHS.get(month.group(1)[:3], 12) if month else 12)


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS}


def _title_words(text: str) -> list[str]:
    """Words used for strict title-family comparison, preserving their order."""
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _is_accepted_title_variant(candidate_title: str, accepted_title: str) -> bool:
    """True when an approved title appears intact in a qualified resume title.

    Resumes frequently add non-role detail after the title, such as
    ``Senior Data Engineer (Data Platform)`` or ``Data Engineer (via BluePi)``.
    Those still state the approved title.  The reverse is intentionally not
    accepted: a plain ``Data Engineer`` is not promoted to ``Senior Data
    Engineer`` merely because the two share words.
    """
    candidate_words = _title_words(candidate_title)
    accepted_words = _title_words(accepted_title)
    if not candidate_words or not accepted_words:
        return False
    if candidate_words == accepted_words:
        return True
    width = len(accepted_words)
    return any(candidate_words[index:index + width] == accepted_words
               for index in range(len(candidate_words) - width + 1))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# --- Groundedness check: catch the judge citing the JD's own title as if it
# were evidence -----------------------------------------------------------
#
# Confirmed in production: for a candidate whose actual titles share zero
# words with the JD's target_role_title, the judge still returned "direct"
# with a reason like "the evidence explicitly states 'Manager' ... directly
# related to Data Engineer roles involving AI data platforms" - where
# "Data Engineer" / "AI data platforms" are the JD's OWN words, appearing
# nowhere in the candidate's actual titles, but cited as if they were found
# in the evidence. This happened for two different candidates with two
# different (unrelated) actual titles, both judged "direct" against the
# same JD title with the same tell: the reasoning quotes JD language back
# as though it were evidence.
#
# This is deliberately NOT a Jaccard floor - the whole point of this
# revision was to let a true zero-overlap synonym match (e.g. "AI
# Architect" for "Data Engineer") still earn credit via the judge, and a
# hard token-overlap gate would defeat that. Instead this checks something
# narrower and fully deterministic: does every quoted phrase in the judge's
# OWN reason actually appear in the evidence text it was given? A reason
# quoting something absent from the evidence is a demonstrably false
# citation regardless of whether the underlying match might otherwise be
# real - so this can't (falsely) reject a genuine synonym call as long as
# the judge's reasoning doesn't misattribute JD language to the evidence
# while making it.
_QUOTED_PHRASE_RE = re.compile(r"['\u2018\u2019]([^'\u2018\u2019]{2,80})['\u2018\u2019]|[\"\u201c\u201d]([^\"\u201c\u201d]{2,80})[\"\u201c\u201d]")


def _extract_quoted_phrases(text: str) -> list[str]:
    phrases = []
    for m in _QUOTED_PHRASE_RE.finditer(text or ""):
        phrase = m.group(1) or m.group(2)
        if phrase:
            phrases.append(phrase.strip())
    return phrases


def _normalize_for_containment(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/(),]", " ", (text or "").lower())).strip()


def _phrase_grounded_in(phrase: str, source_text: str) -> bool:
    phrase_norm = _normalize_for_containment(phrase)
    if not phrase_norm:
        return True  # nothing meaningful to check - don't penalize on empty/punctuation-only quotes
    return phrase_norm in _normalize_for_containment(source_text)


def _reason_falsely_cites_jd_title(judge_reason: str, evidence_titles_text: str, target_role_title: str) -> str | None:
    """
    Returns a description of the false citation if the judge's reason
    contains language drawn from the JD's target_role_title that does not
    appear anywhere in the candidate's actual evidence titles - i.e. the
    judge attributed JD language to the candidate's evidence. Checks two
    shapes of this, since it showed up both ways in production:

    1. Quoted citation - the reason explicitly quotes a phrase (e.g. "the
       evidence explicitly states 'Data Engineer'") that isn't in the
       evidence but IS in the JD title.
    2. Unquoted leakage - the reason just uses JD-title language directly in
       its own prose (e.g. "...directly related to Data Engineer roles
       involving AI data platforms") without quoting it at all. This one
       needs the JD title split on its own delimiters (em-dash, slash, pipe)
       into segments, since a compound title like "Data Engineer — AI Data
       Platform" leaking as either half is just as ungrounded as the whole
       phrase leaking.

    Returns None if no such phrase is found (reasoning is grounded, or at
    least not provably mis-citing the JD title specifically).
    """
    quoted_phrases = _extract_quoted_phrases(judge_reason)

    title_segments = [seg.strip() for seg in re.split(r"[—\-/|]", target_role_title) if len(seg.strip().split()) >= 2]
    if len(target_role_title.split()) >= 2:
        title_segments.append(target_role_title.strip())

    for phrase in quoted_phrases + title_segments:
        if not phrase:
            continue
        if _phrase_grounded_in(phrase, evidence_titles_text):
            continue  # actually present in the candidate's evidence - fine
        if _phrase_grounded_in(phrase, judge_reason) and _phrase_grounded_in(phrase, target_role_title):
            return (
                f"judge's reasoning uses '{phrase}', which appears only in the JD's target role "
                f"title ('{target_role_title}'), not in any of the candidate's actual titles — "
                f"treating as an unsupported match rather than trusting it."
            )
    return None


def _collect_candidate_titles(candidate_data: dict) -> list[dict]:
    """Return only the extracted current/latest employment title for scoring.

    The persistence key retains its old name for database compatibility, but
    extractor.py now stores the latest work-history title in it.  Matching a
    single role prevents an older, unrelated job or a summary sentence from
    becoming the displayed job-title evidence.
    """
    roles = [
        entry for entry in (candidate_data.get("experience", []) or [])
        if _is_plausible_title((entry.get("title") or "").strip())
    ]
    if roles:
        latest = max(roles, key=_employment_recency_key)
        return [{
            "title": latest["title"].strip(),
            "company": (latest.get("company") or "").strip(),
        }]

    current_title = (candidate_data.get("current_role_title_from_summary") or "").strip()
    if not _is_plausible_title(current_title):
        return []

    # This is only reachable for a valid headline fallback where the resume
    # had no structured employment title.  It is deliberately not described
    # as summary evidence in any response or UI surface.
    return [{"title": current_title, "company": ""}]


def _target_titles(target_role_titles: str | list[str]) -> list[str]:
    """Normalise one JD title or a JD-provided list of acceptable titles."""
    raw_titles = [target_role_titles] if isinstance(target_role_titles, str) else (target_role_titles or [])
    titles, seen = [], set()
    # First retain the JD's own title variants, then expand a matching primary
    # title with its explicitly approved alternatives from ROLE_TITLE_REFERENCE.
    expanded_titles = list(raw_titles)
    reference_by_key = {_title_key(primary): alternatives for primary, alternatives in ROLE_TITLE_REFERENCE.items()}
    for value in raw_titles:
        expanded_titles.extend(reference_by_key.get(_title_key(value), ()))
    for value in expanded_titles:
        title = (value or "").strip()
        key = _title_key(title)
        if title and key not in seen:
            seen.add(key)
            titles.append(title)
    return titles


def score_job_titles(candidate_data: dict, target_role_titles: str | list[str], judge_fn=judge_evidence) -> dict:
    """
    Scores the extracted current/latest employment title against the accepted
    JD role titles. Exact accepted titles receive 100%, supported adjacent
    titles receive 80%, and unsupported titles receive 0%.
    """
    titles = _collect_candidate_titles(candidate_data)
    accepted_titles = _target_titles(target_role_titles)

    if not titles or not accepted_titles:
        return {
            "contribution": 0.0,
            "best_match": None,
            "status": "missing",
            "all_titles": [],
            "match_level": "none",
            "judge_reason": "" if accepted_titles else "No JD target job titles to compare against.",
            "target_role_titles": accepted_titles,
        }

    target_token_sets = [(title, _tokenize(title)) for title in accepted_titles]
    all_titles = [
        {"title": t["title"], "company": t["company"], "jaccard": round(max(
            _jaccard(tokens, _tokenize(t["title"])) for _, tokens in target_token_sets
        ), 3)}
        for t in titles
    ]
    # Always score the current/latest title.  Older titles remain available for
    # context but must not be presented as the candidate's current match.
    latest_title_dict = titles[0]
    latest_tokens = _tokenize(latest_title_dict["title"])
    best_target_title, target_tokens = max(
        target_token_sets, key=lambda pair: _jaccard(pair[1], latest_tokens)
    )
    overlap = _jaccard(target_tokens, latest_tokens)

    exact_target_title = next(
        (title for title, _ in target_token_sets
         if _is_accepted_title_variant(latest_title_dict["title"], title)),
        None,
    )
    if exact_target_title:
        # Bypass the judge and return deterministic score
        contribution = 1.0
        match_level = "direct"
        status = "matched"
        judge_reason = f"Exact current-title match to accepted JD title: '{exact_target_title}'."
        
        best_match = dict(latest_title_dict)
        best_match["jaccard"] = round(_jaccard(target_tokens, _tokenize(latest_title_dict["title"])), 3)
        best_match["match_level"] = match_level
        best_match["judge_reason"] = judge_reason
        
        return {
            "contribution": contribution,
            "best_match": best_match,
            "status": status,
            "all_titles": all_titles,
            "match_level": match_level,
            "judge_reason": judge_reason,
            "matched_target_title": exact_target_title,
            "target_role_titles": accepted_titles,
        }

    evidence_chunks = [{
        "text": latest_title_dict["title"],
        "source_type": "job_title",
        "source_label": latest_title_dict["title"],
    }]
    requirement = "Accepted JD job titles (match any one): " + "; ".join(accepted_titles)
    judgment = judge_fn(requirement, evidence_chunks)
    match_level = judgment.get("match", "none")
    judge_reason = judgment.get("reason", "")
    contribution = MATCH_LEVEL_CONTRIBUTION.get(match_level, 0.0)

    # Groundedness check (only runs when the judge claimed a real match)
    groundedness_warning = None
    if contribution > 0.0:
        evidence_titles_text = " | ".join(t["title"] for t in titles)
        groundedness_warning = next((
            _reason_falsely_cites_jd_title(judge_reason, evidence_titles_text, target)
            for target in accepted_titles
            if _reason_falsely_cites_jd_title(judge_reason, evidence_titles_text, target)
        ), None)
        
        if groundedness_warning:
            match_level = "none"
            contribution = 0.0

    # Keep the judge grounded, but do not turn sensible, bounded adjacent
    # titles into an automatic zero merely because they do not share the
    # exact JD wording.
    related_title_reason = next((
        _bounded_related_title(latest_title_dict["title"], target) for target in accepted_titles
        if _bounded_related_title(latest_title_dict["title"], target)
    ), None)
    if contribution == 0.0 and related_title_reason:
        match_level = "related"
        contribution = 0.8
        judge_reason = related_title_reason
        groundedness_warning = None
    elif contribution == 0.0:
        judge_reason = "No supported current-title match."

    # Exact wording earns full credit above.  Any grounded semantic/adjacent
    # title is intentionally capped at the business-approved 80% level.
    if contribution > 0.0:
        match_level = "related"
        contribution = 0.8

    if contribution >= 1.0:
        status = "matched"
    elif contribution > 0.0:
        status = "related"
    else:
        status = "missing"

    best_match = dict(latest_title_dict)
    best_match["jaccard"] = round(overlap, 3)
    best_match["match_level"] = match_level
    best_match["judge_reason"] = judge_reason

    result = {
        "contribution": round(contribution, 3),
        "best_match": best_match,
        "status": status,
        "all_titles": all_titles,
        "match_level": match_level,
        "judge_reason": judge_reason,
        "matched_target_title": best_target_title,
        "target_role_titles": accepted_titles,
    }
    if groundedness_warning:
        result["groundedness_warning"] = groundedness_warning
    return result
