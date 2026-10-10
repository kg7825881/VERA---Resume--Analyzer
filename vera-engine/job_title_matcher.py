"""
job_title_matcher.py — job-title relevance scoring (the "Job Titles" scoring
category — one of the ATS-style comparison points alongside Industry Keywords
and Soft Skills).

Previous revision used pure token-overlap (Jaccard) — deliberately deterministic
and cheap, but with a known, explicitly-documented limitation: it misses true
synonyms/related roles that share no words at all (e.g. "AI Architect" vs
required "Data Engineer" scores 0 despite plausibly being a related role,
depending on context). That limitation is exactly what this revision fixes.

Now reuses the SAME judge (judge.py) already used for skill/requirement
evidence classification, rather than introducing a second similarity
mechanism (embeddings) or a synonym table to maintain. All of a candidate's
past titles — plus, if extracted, their current-role title pulled from a
resume Summary/Profile section (see extractor.py's
current_role_title_from_summary field) — are handed to the judge as evidence
chunks in ONE call, same shape as a BM25-retrieved evidence list for a skill.
This deliberately costs exactly one extra judge call per candidate (not one
per past title), keeping this in line with the rest of the pipeline's
"one model, minimal calls" design.

Jaccard token overlap is NOT gone — it's kept purely as a deterministic,
inspectable way to pick which past title to surface as "best_match" for
display (e.g. in the Evidence panel), sorted alongside every other title in
all_titles. It no longer drives the actual score; the judge's
direct/related/weak/none classification does, via the same
MATCH_LEVEL_CONTRIBUTION mapping matcher.py already uses for skills — so a
job-title match and a skill match mean the same thing on the same 0-1 scale.
"""

import re
from judge import judge_evidence
from matcher import MATCH_LEVEL_CONTRIBUTION
from title_normalization import normalize_employment_title

STOPWORDS = {"the", "a", "an", "of", "and", "for"}

# A title need not repeat the JD verbatim to be useful evidence. These groups
# make that rule reusable across roles without giving unrelated functions
# title credit merely because they share a seniority word such as "lead".
_SENIORITY_WORDS = {"junior", "jr", "senior", "sr", "lead", "principal", "staff", "chief", "manager"}
_ENGINEERING_ROLE_WORDS = {"engineer", "developer", "architect"}
_GENERIC_ENGINEERING_WORDS = {
    "software", "application", "applications", "backend", "frontend",
    "fullstack", "full", "stack", "platform", "systems",
}


def _generic_title_family_match(candidate_title: str, target_title: str) -> str | None:
    """Return a bounded related-title reason for compatible role families.

    A ``Software Engineer`` can therefore support a specialised engineering
    JD such as ``Sr Java Lead Engineer``. It is related evidence, not an exact
    title match. Specific adjacent roles still belong in ROLE_TITLE_REFERENCE.
    """
    candidate_tokens = _tokenize(candidate_title)
    target_tokens = _tokenize(target_title)
    if not (candidate_tokens & _ENGINEERING_ROLE_WORDS and target_tokens & _ENGINEERING_ROLE_WORDS):
        return None

    role_words = _ENGINEERING_ROLE_WORDS | _SENIORITY_WORDS
    substantive_shared = (candidate_tokens & target_tokens) - role_words
    if substantive_shared:
        return "Related engineering title family with shared role specialisation."
    if candidate_tokens & _GENERIC_ENGINEERING_WORDS or target_tokens & _GENERIC_ENGINEERING_WORDS:
        return "Related engineering title family; the resume uses a broader technical title than the JD."
    return None

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
        "Data Engineer",
        "AI Architect",
        "Data Scientist",
        "Analytics Engineer",
    ),
    "Senior Business Analyst": (
        "Lead Business Analyst",
        "Senior IT Business Analyst",
        "Business Systems Analyst",
        "Senior Product Analyst",
        "Functional Business Analyst",
        "Product Manager",
        "Business Analyst",
        "Product Analyst",
        "Systems Analyst",
        "Insurance Technology Consultant",
    ),
    "DevOps Engineer": (
        "Site Reliability Engineer",
        "Cloud DevOps Engineer",
        "Platform Engineer",
        "Infrastructure Engineer",
        "CI/CD Engineer",
        "Cloud Engineer", 
        "DevSecOps Engineer", 
        "Build and Release Engineer", 
        "Systems Engineer", 
        "Cloud Infrastructure Engineer", 
        "Automation Engineer", 
        "Kubernetes Engineer", 
        "Reliability Engineer", 
        "Platform Operations Engineer", 
        "Infrastructure Automation Engineer",
    ),
    "Sr Java Lead Engineer": (
        "Java Technical Lead",
        "Senior Java Developer",
        "Lead Software Engineer",
        "Java Solutions Architect",
        "Backend Engineering Lead",
        "Senior Java Engineer",  
        "Lead Java Developer",  
        "Java Lead Engineer", 
        "Java Engineering Lead", 
        "Lead Backend Engineer", 
        "Senior Backend Engineer", 
        "Java Software Architect",  
        "Principal Java Engineer", 
        "Java Application Lead", 
        "Senior Software Engineer",
        "Lead Software Engineer",
        "Java Microservices Lead", 
        "Spring Boot Lead Developer", 
        "Full Stack Java Lead",
        "Software Engineer",
        "Java Developer",
        "Java Engineer",
        "Backend Engineer",
    ),
    "AI Engineer — GenAI Product Engineering": (
        "AI Engineer",
        "AI/ML Engineer",
        "ML Engineer",
        "Machine Learning Engineer",
        "GenAI Engineer",
        "Generative AI Engineer",
        "LLM Engineer",
        "NLP Engineer",
        "AI Software Engineer",
        "AI Product Engineer",
        "Software Engineer",
        "Backend Engineer",
        "AI Architect",
    ),
}


def _title_key(title: str) -> str:
    """Comparison key that ignores presentation-only punctuation and case."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").casefold())


def _reference_key(title: str) -> str:
    """Compare JD titles without publishing-only bracketed qualifiers.

    A role can be headed "AI Engineer — GenAI Product Engineering
    [Production focus Role]" while its approved title family is maintained
    under the stable role name without that document annotation.
    """
    unqualified = re.sub(r"\s*[\[(][^\])]{0,120}[\])]\s*", " ", title or "")
    return _title_key(unqualified)


def _candidate_title_key(title: str) -> str:
    """Ignore a clarifying parenthetical, not meaningful title words.

    For example, a resume may state ``Senior Data Engineer (Data Platform)``
    while the approved family lists ``Senior Data Engineer``.
    """
    return _title_key(re.sub(r"\s*\([^)]{0,80}\)", "", title or ""))


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
    return None


def _is_plausible_title(value: str) -> bool:
    """Reject prose accidentally extracted as a title without rejecting abbreviations.

    ``Sr. Software Engineer`` and ``S/W Engineer`` are normal, explicit
    resume titles.  A full stop must therefore not invalidate title evidence;
    sentence-like entries are controlled by bounded length, word count and a
    required role noun instead.
    """
    text = (value or "").strip()
    words = re.findall(r"[A-Za-z0-9+#.&/-]+", text)
    return (
        bool(text)
        and len(text) <= 100
        and 1 <= len(words) <= 12
        and not re.search(r"[;\n]", text)
        and bool(re.search(
            r"\b(?:engineer|developer|scientist|analyst|manager|consultant|architect|"
            r"specialist|designer|administrator|coordinator|owner|lead)\b",
            text,
            re.I,
        ))
    )


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS}


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
    """
    Every distinct title worth judging against the JD's role title: the
    candidate's past experience-entry titles, plus — if extractor.py found
    one — a title stated in the resume's Summary/Profile section
    (current_role_title_from_summary). The summary title is included even
    when it duplicates the first experience entry's title (harmless — dedup
    below handles it) and matters most when a resume's most recent role
    ISN'T clearly the first dated experience entry (unusual ordering,
    a title that only appears in prose, etc.) — exactly the gap the summary
    field exists to cover.

    Deduplicates case-insensitively.  A dated current/latest employment role
    takes precedence over legacy summary fields; this makes matching stable
    even for existing records created before the role-resolution upgrade.
    """
    titles: list[dict] = []
    seen = set()

    experience = candidate_data.get("experience", []) or []

    def recency(entry: dict) -> tuple[int, int]:
        value = (entry.get("end_date_raw") or "").casefold()
        if re.search(r"\b(?:present|current|ongoing)\b", value):
            return (9999, 12)
        year = re.search(r"\b(19\d{2}|20\d{2})\b", value)
        month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep(?:t)?|oct|nov|dec)\b", value)
        month_number = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        return (int(year.group(1)) if year else 0, month_number.get((month.group(1) if month else "")[:3], 0))

    for entry in sorted(experience, key=recency, reverse=True):
        raw_title = (entry.get("title") or "").strip()
        # Preserve the extracted title alongside its normalized form. A
        # normalizer can fail to strip a company/location suffix, but that
        # raw string can still contain an approved title verbatim.
        title = normalize_employment_title(raw_title) or raw_title
        if not _is_plausible_title(title):
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append({"title": title, "raw_title": raw_title, "company": entry.get("company") or "unknown company"})

    raw_summary_title = (candidate_data.get("current_role_title_from_summary") or "").strip()
    summary_title = normalize_employment_title(raw_summary_title) or raw_summary_title
    if _is_plausible_title(summary_title) and summary_title.lower() not in seen:
        titles.append({"title": summary_title, "raw_title": raw_summary_title, "company": "(resume headline)"})
        seen.add(summary_title.lower())

    return titles


def _target_titles(target_role_titles: str | list[str]) -> list[str]:
    """Normalise one JD title or a JD-provided list of acceptable titles."""
    raw_titles = [target_role_titles] if isinstance(target_role_titles, str) else (target_role_titles or [])
    titles, seen = [], set()
    # First retain the JD's own title variants, then expand a matching primary
    # title with its explicitly approved alternatives from ROLE_TITLE_REFERENCE.
    expanded_titles = list(raw_titles)
    reference_by_key = {_reference_key(primary): alternatives for primary, alternatives in ROLE_TITLE_REFERENCE.items()}
    for value in raw_titles:
        expanded_titles.extend(reference_by_key.get(_reference_key(value), ()))
    for value in expanded_titles:
        title = (value or "").strip()
        key = _title_key(title)
        if title and key not in seen:
            seen.add(key)
            titles.append(title)
    return titles


def _contains_title_phrase(extracted_title: str, approved_title: str) -> bool:
    """Whether an extracted title contains an approved title as whole words.

    This handles values such as ``Senior Java Engineer | Payments Platform``
    without allowing a broad one-word label such as ``Engineer`` to turn any
    engineering role into an exact match.
    """
    extracted = re.findall(r"[a-z0-9]+", (extracted_title or "").casefold())
    approved = re.findall(r"[a-z0-9]+", (approved_title or "").casefold())
    if len(approved) < 2 or len(approved) > len(extracted):
        return False
    width = len(approved)
    return any(extracted[index:index + width] == approved for index in range(len(extracted) - width + 1))


def score_job_titles(candidate_data: dict, target_role_titles: str | list[str], judge_fn=judge_evidence) -> dict:
    """
    Scores how closely a candidate's job titles (past roles + summary-stated
    current title) relate to the JD's role_title.
    
    Applies a deterministic check on the latest role first: >= 50% similarity 
    yields an 85% match. Falls back to a single judge call if not met.
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
    # An approved reference title is recruiter-authorized evidence of a
    # perfect title fit, wherever it appears in the candidate's verified work
    # history.  This is intentionally different from a semantic related-title
    # result: it is an explicit, maintained equivalence list for the role.
    # Keep the matching title in the evidence so the recruiter can see why
    # the score is 100%, even if the person has since taken a broader title.
    for candidate_title in titles:
        exact_reference = next(
            (title for title in accepted_titles if _candidate_title_key(title) == _candidate_title_key(candidate_title["title"])),
            None,
        )
        contained_reference = exact_reference or next(
            (
                title for title in accepted_titles
                if any(_contains_title_phrase(value, title) for value in (
                    candidate_title.get("raw_title", ""), candidate_title["title"],
                ))
            ),
            None,
        )
        if contained_reference:
            best_match = dict(candidate_title)
            best_match.update({
                "jaccard": round(max(_jaccard(tokens, _tokenize(candidate_title["title"])) for _, tokens in target_token_sets), 3),
                "match_level": "direct",
                "judge_reason": f"Approved reference title match: '{contained_reference}'.",
            })
            return {
                "contribution": 1.0, "best_match": best_match, "status": "matched", "all_titles": all_titles,
                "match_level": "direct", "judge_reason": best_match["judge_reason"],
                "matched_target_title": contained_reference, "target_role_titles": accepted_titles,
            }

    # No approved reference title was found. Evaluate the current/latest role
    # for bounded related-title evidence and, only then, semantic fallback.
    latest_title_dict = titles[0]
    latest_tokens = _tokenize(latest_title_dict["title"])
    best_target_title, target_tokens = max(
        target_token_sets, key=lambda pair: _jaccard(pair[1], latest_tokens)
    )
    overlap = _jaccard(target_tokens, latest_tokens)

    # A resume commonly omits a technology suffix (for example, "Senior
    # Software Engineer" instead of the approved family title "Senior
    # Software Engineer – Java").  This is a bounded family match: the
    # candidate title must be the complete leading title phrase, never merely
    # share a token such as "Engineer".
    latest_title_tokens = re.findall(r"[a-z0-9]+", latest_title_dict["title"].casefold())
    family_target = next((
        title for title in accepted_titles
        if len(latest_title_tokens) >= 2
        and re.findall(r"[a-z0-9]+", title.casefold())[:len(latest_title_tokens)] == latest_title_tokens
    ), None)
    if family_target:
        best_match = dict(latest_title_dict)
        best_match.update({"jaccard": round(overlap, 3), "match_level": "related",
                           "judge_reason": f"Current title is an approved title-family prefix of '{family_target}'."})
        return {
            "contribution": 0.8, "best_match": best_match, "status": "related", "all_titles": all_titles,
            "match_level": "related", "judge_reason": best_match["judge_reason"],
            "matched_target_title": family_target, "target_role_titles": accepted_titles,
        }

    # Apply the deterministic family policy before the semantic judge. This
    # makes common title variance stable for every JD, rather than only for
    # explicitly maintained role-reference entries.
    generic_family_target = next(
        (title for title in accepted_titles if _generic_title_family_match(latest_title_dict["title"], title)),
        None,
    )
    if generic_family_target:
        reason = _generic_title_family_match(latest_title_dict["title"], generic_family_target)
        best_match = dict(latest_title_dict)
        best_match.update({"jaccard": round(overlap, 3), "match_level": "related", "judge_reason": reason})
        return {
            "contribution": 0.8, "best_match": best_match, "status": "related", "all_titles": all_titles,
            "match_level": "related", "judge_reason": reason,
            "matched_target_title": generic_family_target, "target_role_titles": accepted_titles,
        }

    evidence_chunks = [{
        "text": latest_title_dict["title"],
        "source_type": "job_title",
        "source_label": f"{latest_title_dict['title']} at {latest_title_dict['company']}",
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
