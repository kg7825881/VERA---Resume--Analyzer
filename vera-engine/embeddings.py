"""Versioned, persistent embedding cache and vector helpers."""
from __future__ import annotations

import hashlib
import math
import os
import threading
from collections.abc import Callable, Sequence

try:
    import ollama
except ImportError:  # Allows deterministic/injected retrieval tests without Ollama installed.
    ollama = None


EMBED_MODEL = os.environ.get("TALENTLENS_EMBED_MODEL", "nomic-embed-text")
EMBED_MODEL_VERSION = os.environ.get("TALENTLENS_EMBED_MODEL_VERSION", EMBED_MODEL)

_embedding_cache: dict[str, list[float]] = {}
_cache_lock = threading.Lock()


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def embedding_cache_key(text: str, model: str = EMBED_MODEL, model_version: str = EMBED_MODEL_VERSION) -> str:
    return hashlib.sha256(f"{content_hash(text)}\x1f{model}\x1f{model_version}".encode("utf-8")).hexdigest()


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if len(vec1) != len(vec2) or not vec1:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0


def _validated_vector(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding provider returned an invalid vector.")
    return values


def _ollama_batch_embed(texts: list[str], model: str) -> list[list[float]]:
    """Use modern Ollama batch embedding when available, with legacy fallback."""
    if ollama is None:
        raise RuntimeError("Ollama is required for embeddings. Install dependencies or supply an embedding provider.")
    if hasattr(ollama, "embed"):
        response = ollama.embed(model=model, input=texts)
        vectors = response.get("embeddings") if isinstance(response, dict) else getattr(response, "embeddings", None)
        if isinstance(vectors, list) and len(vectors) == len(texts):
            return [_validated_vector(vector) for vector in vectors]
    return [_validated_vector(ollama.embeddings(model=model, prompt=text)["embedding"]) for text in texts]


def embed_texts(
    texts: Sequence[str], *, model: str = EMBED_MODEL, model_version: str = EMBED_MODEL_VERSION,
    embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[list[float]]:
    """Embed a batch, reusing memory and SQLite cache entries by content/model version."""
    clean = [str(text or "") for text in texts]
    if not clean:
        return []
    keys = [embedding_cache_key(text, model, model_version) for text in clean]
    results: list[list[float] | None] = [None] * len(clean)
    missing: dict[str, tuple[str, list[int]]] = {}
    for index, (text, key) in enumerate(zip(clean, keys)):
        with _cache_lock:
            cached = _embedding_cache.get(key)
        if cached is not None:
            results[index] = cached
            continue
        try:
            import db
            cached = db.get_cached_embedding(key)
        except Exception:
            cached = None
        if cached is not None:
            vector = _validated_vector(cached)
            with _cache_lock:
                _embedding_cache[key] = vector
            results[index] = vector
        else:
            existing = missing.get(key)
            if existing:
                existing[1].append(index)
            else:
                missing[key] = (text, [index])
    if missing:
        missing_keys = list(missing)
        missing_texts = [missing[key][0] for key in missing_keys]
        vectors = embedding_fn(missing_texts) if embedding_fn else _ollama_batch_embed(missing_texts, model)
        if len(vectors) != len(missing_texts):
            raise ValueError("Embedding provider returned the wrong number of vectors.")
        for key, text, vector in zip(missing_keys, missing_texts, vectors):
            normalized = _validated_vector(vector)
            with _cache_lock:
                _embedding_cache[key] = normalized
            try:
                import db
                from common import now_iso
                db.put_cached_embedding(key, content_hash(text), model, model_version, normalized, now_iso())
            except Exception:
                pass  # Memory cache keeps scoring usable if persistence is unavailable.
            for index in missing[key][1]:
                results[index] = normalized
    return [vector for vector in results if vector is not None]


def get_embedding(text: str) -> list[float]:
    return embed_texts([text])[0]


def cache_stats() -> dict:
    with _cache_lock:
        return {"cached_vectors": len(_embedding_cache), "model": EMBED_MODEL, "model_version": EMBED_MODEL_VERSION}
