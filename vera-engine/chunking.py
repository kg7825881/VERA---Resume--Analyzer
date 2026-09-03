"""
chunking.py — parent/child evidence chunking for TalentLens's retrieval stage.

This does NOT touch Stage 2 (extractor.py's LLM structuring). Field
extraction — title, company, start_date_raw, end_date_raw, degree, atomic
skills — stays an LLM job; that's information extraction, not chunking, and
no text splitter produces those fields. This module only reshapes what
retrieval.py hands to BM25 and, downstream, to judge.py:

  - PARENT chunk: one whole experience/project/skills entry, exactly what
    retrieval.py's build_evidence_chunks already produced. This is what the
    judge sees — full role/project context (title, company, domain, tools)
    intact, never an isolated sentence stripped of its source.
  - CHILD chunk(s): the searchable unit(s) BM25 actually scores against. A
    short parent (<= MAX_PARENT_TOKENS) has exactly one child, identical to
    the parent — nothing changes for the common case. A long parent (a dense
    multi-line role description, a long project writeup) is recursively
    split into overlapping children, so a requirement can be found via a
    narrow match inside a long block without diluting that block's BM25
    score with unrelated sentences from the rest of it.

retrieval.py deduplicates child hits back to their parent before returning
results — the judge is never shown a bare child fragment, only the parent it
came from. matcher.py / judge.py need no changes: retrieve() still returns
the same {"text", "source_type", "source_label", "bm25_score"} shape.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain splitters work on characters, not tokens. This is a conservative
# chars-per-token approximation for English text, used only to decide when a
# parent is "long enough" to split — not for any scoring math.
CHARS_PER_TOKEN = 4
MAX_PARENT_TOKENS = 500          # parents at/under this stay a single child
CHILD_CHUNK_TOKENS = 150         # target size of each searchable child
CHILD_CHUNK_OVERLAP_TOKENS = 30  # overlap so a sentence isn't cut mid-thought

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_TOKENS * CHARS_PER_TOKEN,
    chunk_overlap=CHILD_CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN,
    # Preserve larger units first (paragraphs, then lines, then sentences),
    # only splitting further if a fragment still doesn't fit — same
    # rationale LangChain's own docs give for this separator ordering.
    separators=["\n\n", "\n", ". ", " ", ""],
)


class ParentChunk:
    """One whole experience/project/skills entry — never split further than
    this. parent_id ties every child back to exactly one of these."""

    __slots__ = ("text", "source_type", "source_label", "parent_id")

    def __init__(self, text: str, source_type: str, source_label: str, parent_id: int):
        self.text = text
        self.source_type = source_type      # "experience" | "project" | "skills"
        self.source_label = source_label    # e.g. "Data Engineer at Foo Corp"
        self.parent_id = parent_id

    def to_dict(self) -> dict:
        return {"text": self.text, "source_type": self.source_type, "source_label": self.source_label}


class ChildChunk:
    """One BM25-searchable fragment. Carries parent_id so a child hit can be
    mapped back to its full parent before anything reaches the judge."""

    __slots__ = ("text", "parent_id")

    def __init__(self, text: str, parent_id: int):
        self.text = text
        self.parent_id = parent_id


def _build_parents(candidate_data: dict) -> list[ParentChunk]:
    """Reads the exact same structured fields retrieval.py's original
    build_evidence_chunks did (experience/projects/skills) — only what
    happens to each assembled entry afterward is different."""
    parents: list[ParentChunk] = []
    parent_id = 0

    for entry in candidate_data.get("experience", []) or []:
        title = entry.get("title") or "unknown title"
        company = entry.get("company") or "unknown company"
        label = f"{title} at {company}"
        parts = []
        if entry.get("description"):
            parts.append(entry["description"])
        if entry.get("domain"):
            parts.append(f"Industry/domain: {entry['domain']}")
        techs = entry.get("technologies_used") or []
        if techs:
            parts.append("Tools/technologies used: " + ", ".join(techs))
        text = " ".join(parts).strip()
        if text:
            parents.append(ParentChunk(text, "experience", label, parent_id))
            parent_id += 1

    for proj in candidate_data.get("projects", []) or []:
        label = proj.get("title") or "unknown project"
        parts = []
        if proj.get("description"):
            parts.append(proj["description"])
        techs = proj.get("technologies_used") or []
        if techs:
            parts.append("Tools/technologies used: " + ", ".join(techs))
        text = " ".join(parts).strip()
        if text:
            parents.append(ParentChunk(text, "project", label, parent_id))
            parent_id += 1

    # Skills list stays one compact parent, same as before — it's already
    # atomic per-item, so there's no long-prose block here worth splitting.
    skills = candidate_data.get("skills_all_sources") or candidate_data.get("skills") or []
    if skills:
        parents.append(ParentChunk(", ".join(skills), "skills", "Skills section", parent_id))
        parent_id += 1

    return parents


def build_parent_child_chunks(candidate_data: dict) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Returns (parents, children).

    Every parent gets at least one child:
      - short parent (<= MAX_PARENT_TOKENS): one child, text identical to the
        parent — this is the common case and behaves exactly like the old
        flat one-chunk-per-entry corpus.
      - long parent: recursively split into overlapping children for finer
        BM25 matching, while the parent itself is kept whole for the judge.
    """
    parents = _build_parents(candidate_data)
    children: list[ChildChunk] = []

    for parent in parents:
        approx_tokens = len(parent.text) / CHARS_PER_TOKEN
        if approx_tokens <= MAX_PARENT_TOKENS:
            children.append(ChildChunk(parent.text, parent.parent_id))
        else:
            for fragment in _splitter.split_text(parent.text):
                children.append(ChildChunk(fragment, parent.parent_id))

    return parents, children
