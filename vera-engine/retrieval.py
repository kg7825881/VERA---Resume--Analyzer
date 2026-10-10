"""
retrieval.py — BM25-based evidence retrieval for candidate resumes.

Given one JD requirement (e.g. "Production ETL/ELT pipelines"), retrieves the
most relevant chunks of a candidate's resume — experience descriptions,
project descriptions, and their technologies_used, plus the raw skills list —
using BM25 lexical ranking (rank_bm25's BM25Okapi).

This module is retrieval ONLY. It does not decide whether the evidence it
finds actually supports the requirement — that judgment belongs to the LLM
judge (see judge.py). BM25's job is narrower and cheaper: cut a candidate's
whole resume down to the handful of chunks worth asking the judge about, and
do it in a way that's transparent — every retrieval has an inspectable score
and a source quote, unlike a bare cosine-similarity number.

Why BM25 and not embeddings for this stage: it's a first-stage retrieval
mechanism precisely because it's cheap, fast, and needs no calibration
(compare calibrate_embeddings.py, which exists ONLY because embedding cosine
similarity needs per-model threshold tuning). BM25 is lexical, not semantic —
it will miss e.g. "data pipeline development" (JD) vs "Built ETL workflows
using Apache Airflow" (resume) if the vocabulary doesn't overlap. That's
expected and fine here: a retrieval miss just means an empty evidence list,
which judge.py already treats as "no evidence" — a safe, inspectable default,
not a wrong answer smuggled in from an unrelated match.

Chunk construction itself (parent/child splitting for long entries) lives in
chunking.py, built on LangChain's RecursiveCharacterTextSplitter — see that
module's docstring for why chunking and Stage 2's LLM structuring are
different problems and why this only touches the former.
"""

import re
from collections.abc import Callable

try:
    from rank_bm25 import BM25Plus
except ImportError:  # pragma: no cover - only for minimal local installs
    class BM25Plus:
        """Dependency-free lexical fallback; production uses rank-bm25."""

        def __init__(self, corpus):
            self.corpus = [set(document) for document in corpus]

        def get_scores(self, query):
            query_tokens = set(query)
            return [float(len(query_tokens & document)) for document in self.corpus]

from chunking import build_parent_child_chunks
from embeddings import EMBED_MODEL, EMBED_MODEL_VERSION, cosine_similarity, embed_texts

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _lexical_overlap(query: str, text: str) -> float:
    """A bounded lexical signal used only to break semantic-retrieval ties."""
    query_tokens, text_tokens = set(_tokenize(query)), set(_tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0.0


class SemanticEvidenceIndex:
    """Candidate-isolated embedding retrieval with a small lexical blend.

    It intentionally returns evidence only; cross-encoder/WebLLM decision
    routing belongs to later slices. Vectors are batched and persisted by
    ``embed_texts`` so every unchanged resume chunk is embedded only once per
    model version.
    """

    def __init__(
        self, candidate_data: dict, *, embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
        model: str = EMBED_MODEL, model_version: str = EMBED_MODEL_VERSION,
    ):
        self.parents, _ = build_parent_child_chunks(candidate_data)
        self.candidate_id = str(candidate_data.get("candidate_id") or "candidate-unknown")
        self.model = model
        self.model_version = model_version
        self._embedding_fn = embedding_fn
        self._vectors = embed_texts(
            [parent.text for parent in self.parents], model=model, model_version=model_version,
            embedding_fn=embedding_fn,
        ) if self.parents else []

    def retrieve(self, requirement: str, top_k: int = 3) -> list[dict]:
        if top_k < 1 or not self.parents:
            return []
        query_vector = embed_texts(
            [requirement], model=self.model, model_version=self.model_version,
            embedding_fn=self._embedding_fn,
        )[0]
        ranked = []
        for parent, vector in zip(self.parents, self._vectors):
            semantic = cosine_similarity(query_vector, vector)
            lexical = _lexical_overlap(requirement, parent.text)
            # Semantic relevance is primary; lexical overlap provides a stable,
            # explainable preference where similarities are close.
            score = 0.85 * semantic + 0.15 * lexical
            ranked.append((score, semantic, lexical, parent))
        ranked.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, semantic, lexical, parent in ranked[:top_k]:
            item = parent.to_dict()
            if item["candidate_id"] != self.candidate_id:
                raise ValueError("Evidence candidate isolation violation.")
            item.update({
                "semantic_similarity": round(float(semantic), 4),
                "lexical_similarity": round(float(lexical), 4),
                "retrieval_score": round(float(score), 4),
                "retrieval_model": self.model,
                "retrieval_model_version": self.model_version,
            })
            results.append(item)
        return results


class CandidateEvidenceIndex:
    """
    One BM25 index built ONCE per candidate and reused across every
    requirement scored against that candidate in a single scoring run — a JD
    can easily have 15-20 mandatory + preferred requirements, so rebuilding
    the index per-requirement would be wasted repeated work over the same
    fixed corpus.

    BM25 scores CHILD chunks (chunking.py's parent/child split — short
    entries have one child identical to the parent, long entries are
    recursively split into several). retrieve() always returns PARENT
    chunks, though: the judge should see the whole role/project an evidence
    snippet came from, never a bare fragment missing its title/company/
    domain context.
    """

    def __init__(self, candidate_data: dict):
        self.parents, self.children = build_parent_child_chunks(candidate_data)
        self._parent_by_id = {p.parent_id: p for p in self.parents}
        self._corpus_tokens = [_tokenize(c.text) for c in self.children]
        # BM25Plus, not the classic BM25Okapi. This corpus is small by construction —
        # a handful of chunks per candidate, not thousands of documents — and Okapi's
        # classic IDF, log((N - n + 0.5) / (n + 0.5)), collapses to exactly 0 whenever a
        # term appears in precisely half the chunks (e.g. N=2, n=1: log(1.5/1.5) = 0),
        # silently zeroing out the score for a term that DOES appear in the evidence.
        # Confirmed on this exact corpus shape during testing. BM25Plus's IDF,
        # log((N + 1) / n), stays positive for any n <= N, which is the property this
        # small-corpus retrieval actually needs.
        self._bm25 = BM25Plus(self._corpus_tokens) if self._corpus_tokens else None

    def retrieve(self, requirement: str, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
        """
        Returns up to top_k evidence PARENT chunks for `requirement`, each as
        a dict (see ParentChunk.to_dict in chunking.py) plus a "bm25_score"
        field, sorted by score descending. Returns [] if the candidate has no
        evidence chunks at all, the requirement tokenizes to nothing, or
        nothing scores above min_score (i.e. zero lexical overlap with
        anything on the resume).

        Matching happens at the child level (so a narrow match inside one
        long role description isn't diluted by the rest of that block), but
        results are deduplicated back to their parent before being returned —
        if two children of the same parent both score well (e.g. two
        sentences in one role both mention Kubernetes), the judge sees that
        parent once, with the best of those child scores attached, not the
        same role twice.
        """
        if not self._bm25 or not self.children:
            return []

        query_tokens = _tokenize(requirement)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.children, scores), key=lambda pair: pair[1], reverse=True)

        best_score_per_parent: dict[int, float] = {}
        for child, score in ranked:
            if score <= min_score:
                continue
            if child.parent_id not in best_score_per_parent or score > best_score_per_parent[child.parent_id]:
                best_score_per_parent[child.parent_id] = score

        ranked_parents = sorted(best_score_per_parent.items(), key=lambda kv: kv[1], reverse=True)

        results = []
        for parent_id, score in ranked_parents[:top_k]:
            d = self._parent_by_id[parent_id].to_dict()
            d["bm25_score"] = round(float(score), 3)
            results.append(d)
        return results
