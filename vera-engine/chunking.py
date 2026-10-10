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

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised only in minimal local installs
    class RecursiveCharacterTextSplitter:
        """Small dependency-free fallback for the optional LangChain splitter."""

        def __init__(self, *, chunk_size: int, chunk_overlap: int, separators: list[str]):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators

        def split_text(self, text: str) -> list[str]:
            if len(text) <= self.chunk_size:
                return [text]
            chunks, start = [], 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                if end < len(text):
                    boundary = max((text.rfind(separator, start, end) for separator in self.separators if separator), default=-1)
                    if boundary > start:
                        end = boundary + 1
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(text):
                    break
                start = max(end - self.chunk_overlap, start + 1)
            return chunks

from common import EvidenceChunk

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

    __slots__ = ("text", "source_type", "source_label", "source_id", "parent_id", "evidence_id", "candidate_id", "page", "content_hash")

    def __init__(self, evidence: EvidenceChunk, parent_id: int):
        self.text = evidence.text
        self.source_type = evidence.section  # "experience" | "project" | "skills"
        self.source_label = evidence.source_label or evidence.section.title()
        self.source_id = evidence.source_id
        self.parent_id = parent_id
        self.evidence_id = evidence.evidence_id
        self.candidate_id = evidence.candidate_id
        self.page = evidence.page
        self.content_hash = evidence.content_hash

    def to_dict(self) -> dict:
        return {"text": self.text, "source_type": self.source_type,
                "source_label": self.source_label, "source_id": self.source_id,
                "evidence_id": self.evidence_id, "candidate_id": self.candidate_id,
                "page": self.page, "content_hash": self.content_hash}


class ChildChunk:
    """One BM25-searchable fragment. Carries parent_id so a child hit can be
    mapped back to its full parent before anything reaches the judge."""

    __slots__ = ("text", "parent_id")

    def __init__(self, text: str, parent_id: int):
        self.text = text
        self.parent_id = parent_id


def _chunk_id(candidate_id: str, section: str, label: str, text: str, page: int | None) -> str:
    # EvidenceChunk calculates a full content hash; reuse that deterministic
    # digest in an identifier that is stable for the same candidate/source text.
    seed = "\x1f".join((candidate_id, section, label, text, str(page or "")))
    import hashlib
    return f"ev_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _new_evidence(candidate_id: str, section: str, label: str, text: str, source_id: str, page: int | None = None) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=_chunk_id(candidate_id, section, label, text, page), candidate_id=candidate_id,
        section=section, text=text, page=page, source_id=source_id, source_label=label,
    )


def build_evidence_chunks(candidate_data: dict) -> list[EvidenceChunk]:
    """Create stable, candidate-isolated evidence records from extracted fields.

    Page data is optional because current DOCX/OCR extraction does not reliably
    preserve it. A future page-aware extractor can populate ``page`` without
    changing the evidence ID or retrieval contract shape.
    """
    chunks: list[EvidenceChunk] = []
    candidate_id = str(candidate_data.get("candidate_id") or "candidate-unknown")
    source_id = candidate_data.get("source_id") or candidate_data.get("document_id") or ""

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
            chunks.append(_new_evidence(candidate_id, "experience", label, text, source_id, entry.get("page")))

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
            chunks.append(_new_evidence(candidate_id, "project", label, text, source_id, proj.get("page")))

    # Skills list stays one compact parent, same as before — it's already
    # atomic per-item, so there's no long-prose block here worth splitting.
    skills = candidate_data.get("skills_all_sources") or candidate_data.get("skills") or []
    if skills:
        chunks.append(_new_evidence(candidate_id, "skills", "Skills section", ", ".join(skills), source_id))

    for education in candidate_data.get("education", []) or []:
        if not isinstance(education, dict):
            continue
        text = " in ".join(part.strip() for part in (
            str(education.get("degree_level") or ""), str(education.get("field") or ""),
        ) if part.strip())
        institution = str(education.get("institution") or "").strip()
        if institution:
            text = f"{text}; {institution}" if text else institution
        if text:
            chunks.append(_new_evidence(candidate_id, "education", "Education", text, source_id, education.get("page")))

    certifications = [value.strip() for value in candidate_data.get("certifications", []) or [] if isinstance(value, str) and value.strip()]
    if certifications:
        chunks.append(_new_evidence(candidate_id, "certifications", "Certifications", ", ".join(certifications), source_id))

    return chunks


def _evidence_from_persisted(candidate_data: dict) -> list[EvidenceChunk]:
    chunks = []
    for item in candidate_data.get("evidence_chunks", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            chunks.append(EvidenceChunk(
                evidence_id=item["evidence_id"], candidate_id=item["candidate_id"], section=item["section"],
                text=item["text"], page=item.get("page"), source_id=item.get("source_id", ""),
                content_hash=item.get("content_hash", ""), source_label=item.get("source_label", ""),
            ))
        except (KeyError, ValueError, TypeError):
            # A malformed historic chunk must never contaminate retrieval. The
            # deterministic fallback below rebuilds a safe corpus from fields.
            return []
    return chunks


def _build_parents(candidate_data: dict) -> list[ParentChunk]:
    """Load persisted chunks when available, otherwise build compatible ones."""
    evidence_chunks = _evidence_from_persisted(candidate_data) or build_evidence_chunks(candidate_data)
    parents = [ParentChunk(evidence, parent_id) for parent_id, evidence in enumerate(evidence_chunks)]

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
