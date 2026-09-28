import db
import embeddings

from embeddings import embed_texts, embedding_cache_key
from retrieval import SemanticEvidenceIndex


def _embed(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([
            1.0 if any(word in lowered for word in ("python", "etl", "pipeline")) else 0.0,
            1.0 if any(word in lowered for word in ("warehouse", "snowflake")) else 0.0,
        ])
    return vectors


def test_embeddings_batch_deduplicates_and_reuses_sqlite_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    embeddings._embedding_cache.clear()
    calls = []

    def provider(texts):
        calls.append(texts)
        return _embed(texts)

    first = embed_texts(["Python pipeline", "Warehouse", "Python pipeline"], model="test", model_version="v1", embedding_fn=provider)
    assert calls == [["Python pipeline", "Warehouse"]]
    assert first[0] == first[2]
    assert embedding_cache_key("Python pipeline", "test", "v1") != embedding_cache_key("Python pipeline", "test", "v2")

    embeddings._embedding_cache.clear()  # Prove the second call uses SQLite, not RAM.
    second = embed_texts(["Warehouse", "Python pipeline"], model="test", model_version="v1", embedding_fn=lambda _: (_ for _ in ()).throw(AssertionError("provider should not run")))
    assert second == [first[1], first[0]]


def test_semantic_retrieval_returns_top_three_grounded_candidate_isolated_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    embeddings._embedding_cache.clear()
    candidate = {
        "candidate_id": "candidate-1", "source_id": "src-1",
        "experience": [{"title": "Data Engineer", "company": "Acme", "description": "Built ETL pipelines in Python."}],
        "projects": [{"title": "Warehouse migration", "description": "Migrated Snowflake warehouse jobs."}],
        "skills": ["Python", "SQL"],
    }

    index = SemanticEvidenceIndex(candidate, embedding_fn=_embed, model="test", model_version="v1")
    results = index.retrieve("Python pipeline engineering", top_k=3)

    assert 1 <= len(results) <= 3
    assert results[0]["source_label"] == "Data Engineer at Acme"
    assert results[0]["candidate_id"] == "candidate-1"
    assert results[0]["evidence_id"].startswith("ev_")
    assert results[0]["semantic_similarity"] >= results[-1]["semantic_similarity"]
    assert all(item["retrieval_model_version"] == "v1" for item in results)
