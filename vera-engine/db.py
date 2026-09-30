"""
db.py — lightweight persistence layer using SQLite (stdlib, zero setup).
Stores JDs, resumes, and score records per the Phase 1 schema.

Swappable later: if you outgrow SQLite (concurrent HR users, larger scale),
migrate to Postgres by replacing this module with a SQLAlchemy version —
the function signatures below (insert_jd, insert_resume, etc.) can stay
the same so api.py doesn't need to change.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get("TALENTLENS_DB_PATH", "VERA.db")


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jds (
                role_id TEXT PRIMARY KEY,
                document_id TEXT,
                source_hash TEXT,
                source_id TEXT,
                source_markdown TEXT,
                source_bundle_version TEXT,
                role_title TEXT,
                department TEXT,
                target_job_titles TEXT,
                mandatory_skills TEXT,
                mandatory_domain_requirements TEXT,
                mandatory_role_specific_requirements TEXT,
                preferred_technical_skills TEXT,
                soft_preferred_skills TEXT,
                industry_keywords TEXT,
                requirement_metadata TEXT,
                min_years_experience REAL,
                education_requirements TEXT,
                relevant_certifications TEXT,
                responsibilities TEXT,
                file_name TEXT,
                extraction_method TEXT,
                extraction_warnings TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                candidate_id TEXT PRIMARY KEY,
                document_id TEXT,
                source_id TEXT,
                source_hash TEXT,
                source_markdown TEXT,
                source_bundle_version TEXT,
                candidate_name TEXT,
                skills TEXT,
                skills_all_sources TEXT,
                current_role_title_from_summary TEXT,
                total_years_experience REAL,
                experience TEXT,
                latest_role_period TEXT,
                education TEXT,
                certifications TEXT,
                projects TEXT,
                skill_evidence TEXT,
                evidence_chunks TEXT,
                file_name TEXT,
                extraction_method TEXT,
                extraction_warnings TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT,
                role_id TEXT,
                run_id TEXT,
                category_scores TEXT,
                evidence TEXT,
                final_score REAL,
                hard_gate_failed INTEGER,
                hard_gate_reason TEXT,
                scored_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screening_runs (
                run_id TEXT PRIMARY KEY, role_id TEXT NOT NULL, status TEXT NOT NULL,
                audit_mode INTEGER NOT NULL DEFAULT 0,
                candidate_count INTEGER NOT NULL, completed_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0, validation_errors TEXT,
                started_at TEXT NOT NULL, completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screening_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                status TEXT NOT NULL, cache_hit INTEGER NOT NULL DEFAULT 0, error TEXT,
                started_at TEXT NOT NULL, completed_at TEXT,
                UNIQUE(run_id, candidate_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assessment_cache (
                cache_key TEXT PRIMARY KEY, result TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY, content_hash TEXT NOT NULL, model TEXT NOT NULL,
                model_version TEXT NOT NULL, vector TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_sessions (
                session_id TEXT PRIMARY KEY, role_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recruiter_feedback (
                role_id TEXT NOT NULL, candidate_id TEXT NOT NULL, disposition TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '', score_snapshot REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (role_id, candidate_id)
            )
        """)

        # Migration: a talentlens.db created before this change has a "scores" table
        # WITHOUT the evidence column — CREATE TABLE IF NOT EXISTS only applies to a
        # brand-new table, it does nothing to an existing one with a different shape.
        # This adds the column in place so existing databases don't need a manual reset.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(scores)").fetchall()}
        if "evidence" not in existing_cols:
            conn.execute("ALTER TABLE scores ADD COLUMN evidence TEXT")
        # Runs were introduced after the original score table.  Keep old local
        # databases usable: historic scores simply have a NULL run_id and are
        # not mistaken for a current tracked run.
        if "run_id" not in existing_cols:
            conn.execute("ALTER TABLE scores ADD COLUMN run_id TEXT")

        run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(screening_runs)").fetchall()}
        if "audit_mode" not in run_cols:
            conn.execute("ALTER TABLE screening_runs ADD COLUMN audit_mode INTEGER NOT NULL DEFAULT 0")

        # These fields are used by matching after ingestion.  Keep migrations here so
        # an existing local database gains the same data model as a fresh install.
        jd_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jds)").fetchall()}
        if "industry_keywords" not in jd_cols:
            conn.execute("ALTER TABLE jds ADD COLUMN industry_keywords TEXT")
        if "mandatory_domain_requirements" not in jd_cols:
            conn.execute("ALTER TABLE jds ADD COLUMN mandatory_domain_requirements TEXT")
        if "mandatory_role_specific_requirements" not in jd_cols:
            conn.execute("ALTER TABLE jds ADD COLUMN mandatory_role_specific_requirements TEXT")
        if "requirement_metadata" not in jd_cols:
            conn.execute("ALTER TABLE jds ADD COLUMN requirement_metadata TEXT")
        if "source_hash" not in jd_cols:
            conn.execute("ALTER TABLE jds ADD COLUMN source_hash TEXT")
        for column in ("source_id", "source_markdown", "source_bundle_version"):
            if column not in jd_cols:
                conn.execute(f"ALTER TABLE jds ADD COLUMN {column} TEXT")
        if "target_job_titles" not in jd_cols:
            conn.execute("ALTER TABLE jds ADD COLUMN target_job_titles TEXT")

        resume_cols = {row["name"] for row in conn.execute("PRAGMA table_info(resumes)").fetchall()}
        if "skills_all_sources" not in resume_cols:
            conn.execute("ALTER TABLE resumes ADD COLUMN skills_all_sources TEXT")
        if "current_role_title_from_summary" not in resume_cols:
            conn.execute("ALTER TABLE resumes ADD COLUMN current_role_title_from_summary TEXT")
        if "latest_role_period" not in resume_cols:
            conn.execute("ALTER TABLE resumes ADD COLUMN latest_role_period TEXT")
        if "skill_evidence" not in resume_cols:
            conn.execute("ALTER TABLE resumes ADD COLUMN skill_evidence TEXT")
        if "evidence_chunks" not in resume_cols:
            conn.execute("ALTER TABLE resumes ADD COLUMN evidence_chunks TEXT")
        for column in ("source_id", "source_hash", "source_markdown", "source_bundle_version"):
            if column not in resume_cols:
                conn.execute(f"ALTER TABLE resumes ADD COLUMN {column} TEXT")


@contextmanager
def _connect():
    # timeout: with parallel resume ingestion/scoring, multiple threads may write around
    # the same time — without a timeout, SQLite raises "database is locked" immediately
    # instead of waiting briefly for the other writer to finish.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _dumps(value) -> str:
    return json.dumps(value if value is not None else [])


def insert_jd(record: dict) -> str:
    with _connect() as conn:
        existing = conn.execute("SELECT document_id, source_hash, requirement_metadata FROM jds WHERE role_id = ?", (record["role_id"],)).fetchone()
        if existing and existing["source_hash"] == record.get("source_hash"):
            record["document_id"] = existing["document_id"]
        # A recruiter decision must survive a re-upload when the corresponding
        # stable requirement still exists. Removed/rewritten requirements simply
        # lose their stale override instead of being silently recreated.
        if existing and existing["requirement_metadata"] and record.get("requirement_metadata"):
            from jd_policy import apply_recruiter_overrides_to_metadata
            try:
                prior = json.loads(existing["requirement_metadata"])
                # Older JD records stored an empty JSON list ([]) before the
                # semantic-policy schema was introduced. Treat that legacy
                # value as "no saved overrides" rather than failing the
                # re-upload which upgrades it to the new metadata object.
                if isinstance(prior, dict):
                    record["requirement_metadata"] = apply_recruiter_overrides_to_metadata(
                        record["requirement_metadata"], prior.get("recruiter_overrides"), reject_unknown=False,
                    )
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        conn.execute("""
            INSERT OR REPLACE INTO jds
            (role_id, document_id, source_hash, source_id, source_markdown, source_bundle_version, role_title, department, target_job_titles, mandatory_skills, mandatory_domain_requirements, mandatory_role_specific_requirements, preferred_technical_skills,
             soft_preferred_skills, industry_keywords, requirement_metadata, min_years_experience, education_requirements,
             relevant_certifications, responsibilities, file_name, extraction_method, extraction_warnings, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record["role_id"], record["document_id"], record.get("source_hash", ""), record.get("source_id", ""), record.get("source_markdown", ""), record.get("source_bundle_version", ""), record.get("role_title", ""),
            record.get("department", ""), _dumps(record.get("target_job_titles")), _dumps(record.get("mandatory_skills")),
            _dumps(record.get("mandatory_domain_requirements")),
            _dumps(record.get("mandatory_role_specific_requirements")),
            _dumps(record.get("preferred_technical_skills")), _dumps(record.get("soft_preferred_skills")),
            _dumps(record.get("industry_keywords")), _dumps(record.get("requirement_metadata")), record.get("min_years_experience", 0), _dumps(record.get("education_requirements")),
            _dumps(record.get("relevant_certifications")), _dumps(record.get("responsibilities")),
            record.get("file_name", ""), record.get("extraction_method", ""),
            _dumps(record.get("extraction_warnings")), record.get("uploaded_at", ""),
        ))
        return "refreshed" if existing else "created"


def update_jd_requirement_overrides(role_id: str, recruiter_overrides: dict) -> dict | None:
    """Persist recruiter policy choices and return the updated JD, or None if absent."""
    with _connect() as conn:
        row = conn.execute("SELECT requirement_metadata FROM jds WHERE role_id = ?", (role_id,)).fetchone()
        if not row:
            return None
        if not row["requirement_metadata"]:
            raise ValueError("This JD has no requirement metadata; upload it again to create the policy.")
        from jd_policy import apply_recruiter_overrides_to_metadata
        metadata = apply_recruiter_overrides_to_metadata(
            json.loads(row["requirement_metadata"]), recruiter_overrides, reject_unknown=True,
        )
        conn.execute("UPDATE jds SET requirement_metadata = ? WHERE role_id = ?", (json.dumps(metadata), role_id))
    return get_jd_by_role_id(role_id)


def replace_jd_requirement_metadata(role_id: str, metadata: dict) -> dict | None:
    """Persist a safe JD-policy schema refresh without touching source fields."""
    with _connect() as conn:
        updated = conn.execute(
            "UPDATE jds SET requirement_metadata = ? WHERE role_id = ?",
            (json.dumps(metadata), role_id),
        ).rowcount
    return get_jd_by_role_id(role_id) if updated else None


def insert_resume(record: dict):
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO resumes
            (candidate_id, document_id, source_id, source_hash, source_markdown, source_bundle_version, candidate_name, skills, skills_all_sources, current_role_title_from_summary,
             total_years_experience, experience, latest_role_period, education, certifications, projects, skill_evidence, evidence_chunks, file_name, extraction_method,
             extraction_warnings, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record["candidate_id"], record["document_id"], record.get("source_id", ""), record.get("source_hash", ""), record.get("source_markdown", ""), record.get("source_bundle_version", ""), record.get("candidate_name", ""),
            _dumps(record.get("skills")), _dumps(record.get("skills_all_sources")),
            record.get("current_role_title_from_summary", ""), record.get("total_years_experience", 0),
            _dumps(record.get("experience")), _dumps(record.get("latest_role_period")), _dumps(record.get("education")),
            _dumps(record.get("certifications")), _dumps(record.get("projects")), _dumps(record.get("skill_evidence")),
            _dumps(record.get("evidence_chunks")),
            record.get("file_name", ""), record.get("extraction_method", ""),
            _dumps(record.get("extraction_warnings")), record.get("uploaded_at", ""),
        ))


def clear_screening_data() -> dict:
    """Clear the previous candidate batch and its scores, preserving the JD library."""
    with _connect() as conn:
        score_count = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        resume_count = conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
        # Scores reference candidates, so remove them first.  JDs remain available
        # for the next screening through the existing role library.
        conn.execute("DELETE FROM scores")
        conn.execute("DELETE FROM resumes")
        conn.execute("DELETE FROM semantic_sessions")
    return {"removed_resumes": resume_count, "removed_scores": score_count}


def create_semantic_session(session: dict, created_at: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO semantic_sessions (session_id, role_id, candidate_id, status, payload, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (session["session_id"], session["role_id"], session["candidate_id"], session["status"], json.dumps(session), created_at, created_at),
        )


def get_semantic_session(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT payload FROM semantic_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return json.loads(row["payload"]) if row else None


def update_semantic_session(session: dict, updated_at: str):
    with _connect() as conn:
        changed = conn.execute(
            "UPDATE semantic_sessions SET status = ?, payload = ?, updated_at = ? WHERE session_id = ?",
            (session["status"], json.dumps(session), updated_at, session["session_id"]),
        ).rowcount
        if not changed:
            raise ValueError("Semantic session not found.")


def get_recruiter_feedback(role_id: str, candidate_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT role_id, candidate_id, disposition, note, score_snapshot, created_at, updated_at FROM recruiter_feedback WHERE role_id = ? AND candidate_id = ?",
            (role_id, candidate_id),
        ).fetchone()
        return dict(row) if row else None


def save_recruiter_feedback(role_id: str, candidate_id: str, disposition: str, note: str, score_snapshot: float | None, timestamp: str) -> dict:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO recruiter_feedback (role_id, candidate_id, disposition, note, score_snapshot, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(role_id, candidate_id) DO UPDATE SET disposition=excluded.disposition, note=excluded.note,
               score_snapshot=excluded.score_snapshot, updated_at=excluded.updated_at""",
            (role_id, candidate_id, disposition, note, score_snapshot, timestamp, timestamp),
        )
    return get_recruiter_feedback(role_id, candidate_id)


def insert_score(candidate_id: str, role_id: str, run_id: str, result: dict):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO scores (candidate_id, role_id, run_id, category_scores, evidence, final_score,
                                 hard_gate_failed, hard_gate_reason, scored_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            candidate_id, role_id, run_id, json.dumps(result["category_scores"]),
            json.dumps(result.get("evidence", {})), result["final_score"],
            int(result["hard_gate_failed"]), result["hard_gate_reason"], result["scored_at"],
        ))


def create_screening_run(run_id: str, role_id: str, candidate_ids: list[str], started_at: str, *, audit_mode: bool = False):
    with _connect() as conn:
        conn.execute("INSERT INTO screening_runs (run_id, role_id, status, audit_mode, candidate_count, started_at) VALUES (?,?,?,?,?,?)",
                     (run_id, role_id, "running", int(audit_mode), len(candidate_ids), started_at))
        conn.executemany("INSERT INTO screening_tasks (run_id, candidate_id, status, started_at) VALUES (?,?,?,?)",
                         [(run_id, candidate_id, "queued", started_at) for candidate_id in candidate_ids])


def start_screening_task(run_id: str, candidate_id: str, started_at: str):
    """Mark one queued task as actively executing; idempotent for retries."""
    with _connect() as conn:
        conn.execute(
            "UPDATE screening_tasks SET status = ?, started_at = ? "
            "WHERE run_id = ? AND candidate_id = ? AND status = ?",
            ("running", started_at, run_id, candidate_id, "queued"),
        )


def complete_screening_task(run_id: str, candidate_id: str, cache_hit: bool, completed_at: str):
    with _connect() as conn:
        changed = conn.execute(
            "UPDATE screening_tasks SET status = ?, cache_hit = ?, completed_at = ? "
            "WHERE run_id = ? AND candidate_id = ? AND status IN (?, ?)",
            ("completed", int(cache_hit), completed_at, run_id, candidate_id, "queued", "running"),
        ).rowcount
        if changed:
            conn.execute("UPDATE screening_runs SET completed_count = completed_count + 1 WHERE run_id = ?", (run_id,))


def fail_screening_task(run_id: str, candidate_id: str, error: str, completed_at: str):
    with _connect() as conn:
        changed = conn.execute(
            "UPDATE screening_tasks SET status = ?, error = ?, completed_at = ? "
            "WHERE run_id = ? AND candidate_id = ? AND status IN (?, ?)",
            ("failed", error[:1000], completed_at, run_id, candidate_id, "queued", "running"),
        ).rowcount
        if changed:
            conn.execute("UPDATE screening_runs SET failed_count = failed_count + 1 WHERE run_id = ?", (run_id,))


def finish_screening_run(run_id: str, status: str, completed_at: str):
    with _connect() as conn:
        conn.execute("UPDATE screening_runs SET status = ?, completed_at = ? WHERE run_id = ?", (status, completed_at, run_id))


def get_screening_run(run_id: str) -> dict | None:
    with _connect() as conn:
        run = conn.execute("SELECT * FROM screening_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not run:
            return None
        result = dict(run)
        result["audit_mode"] = bool(result["audit_mode"])
        result["validation_errors"] = json.loads(result["validation_errors"]) if result["validation_errors"] else []
        tasks = conn.execute("SELECT candidate_id, status, cache_hit, error, started_at, completed_at FROM screening_tasks WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        result["tasks"] = [{**dict(task), "cache_hit": bool(task["cache_hit"]),
                            "duration_ms": _duration_ms(task["started_at"], task["completed_at"])} for task in tasks]
        result["progress"] = {
            "total": result["candidate_count"],
            "completed": result["completed_count"],
            "failed": result["failed_count"],
            "pending": max(0, result["candidate_count"] - result["completed_count"] - result["failed_count"]),
        }
        result["performance"] = {
            "elapsed_ms": _duration_ms(result["started_at"], result["completed_at"]),
            "cache_hits": sum(1 for task in result["tasks"] if task["cache_hit"]),
        }
        return result


def _duration_ms(started_at: str | None, completed_at: str | None) -> int | None:
    """Best-effort elapsed time for UI/benchmarking; bad historic timestamps stay null."""
    if not started_at or not completed_at:
        return None
    try:
        return max(0, round((datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds() * 1000))
    except (TypeError, ValueError):
        return None


def list_screening_runs(role_id: str | None = None, limit: int = 25) -> list[dict]:
    """Return recent run summaries; task detail stays on GET /runs/{run_id}."""
    safe_limit = max(1, min(int(limit), 100))
    with _connect() as conn:
        if role_id:
            rows = conn.execute(
                "SELECT * FROM screening_runs WHERE role_id = ? ORDER BY started_at DESC LIMIT ?",
                (role_id, safe_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM screening_runs ORDER BY started_at DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [{
            **dict(row), "audit_mode": bool(row["audit_mode"]),
            "validation_errors": json.loads(row["validation_errors"]) if row["validation_errors"] else [],
            "progress": {
                "total": row["candidate_count"], "completed": row["completed_count"],
                "failed": row["failed_count"],
                "pending": max(0, row["candidate_count"] - row["completed_count"] - row["failed_count"]),
            },
        } for row in rows]


def get_cached_assessment(cache_key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT result FROM assessment_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        return json.loads(row["result"]) if row else None


def put_cached_assessment(cache_key: str, result: dict, created_at: str):
    with _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO assessment_cache (cache_key, result, created_at) VALUES (?,?,?)",
                     (cache_key, json.dumps(result), created_at))


def get_cached_embedding(cache_key: str) -> list[float] | None:
    with _connect() as conn:
        row = conn.execute("SELECT vector FROM embedding_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        return json.loads(row["vector"]) if row else None


def put_cached_embedding(cache_key: str, content_hash: str, model: str, model_version: str, vector: list[float], created_at: str):
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (cache_key, content_hash, model, model_version, vector, created_at) VALUES (?,?,?,?,?,?)",
            (cache_key, content_hash, model, model_version, json.dumps(vector), created_at),
        )


def get_all_roles() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT document_id, role_id, role_title, department, uploaded_at FROM jds").fetchall()
        return [dict(r) for r in rows]


def get_jd_by_role_id(role_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jds WHERE role_id = ? ORDER BY uploaded_at DESC LIMIT 1", (role_id,)).fetchone()
        if not row:
            return None
        return _row_to_jd_dict(row)


def _row_to_jd_dict(row) -> dict:
    d = dict(row)
    for field in ["target_job_titles", "mandatory_skills", "mandatory_domain_requirements", "mandatory_role_specific_requirements", "preferred_technical_skills", "soft_preferred_skills", "industry_keywords", "requirement_metadata",
                  "education_requirements", "relevant_certifications", "responsibilities", "extraction_warnings"]:
        d[field] = json.loads(d[field]) if d[field] else []
    return d


def get_resumes(candidate_ids: list[str] = None) -> list[dict]:
    with _connect() as conn:
        if candidate_ids:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = conn.execute(f"SELECT * FROM resumes WHERE candidate_id IN ({placeholders})", candidate_ids).fetchall()
        else:
            rows = conn.execute("SELECT * FROM resumes").fetchall()
        return [_row_to_resume_dict(r) for r in rows]


def get_resume_by_candidate_id(candidate_id: str) -> dict | None:
    """Return the canonical extracted resume record for one candidate."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM resumes WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return _row_to_resume_dict(row) if row else None


def update_candidate_name(candidate_id: str, candidate_name: str) -> dict | None:
    """Persist a recruiter-corrected candidate name without changing resume evidence."""
    with _connect() as conn:
        updated = conn.execute(
            "UPDATE resumes SET candidate_name = ? WHERE candidate_id = ?",
            (candidate_name, candidate_id),
        ).rowcount
    return get_resume_by_candidate_id(candidate_id) if updated else None


def update_resume_role_facts(candidate_id: str, experience: list[dict], current_role_title: str) -> dict | None:
    """Persist deterministic role recovery from an already stored resume source."""
    from experience import latest_role_period

    with _connect() as conn:
        updated = conn.execute(
            "UPDATE resumes SET experience = ?, latest_role_period = ?, current_role_title_from_summary = ? WHERE candidate_id = ?",
            (_dumps(experience), _dumps(latest_role_period(experience)), current_role_title, candidate_id),
        ).rowcount
    return get_resume_by_candidate_id(candidate_id) if updated else None


def _row_to_resume_dict(row) -> dict:
    d = dict(row)
    for field in ["skills", "skills_all_sources", "experience", "latest_role_period", "education", "certifications", "projects", "skill_evidence", "evidence_chunks", "extraction_warnings"]:
        d[field] = json.loads(d[field]) if d[field] else []
    return d


def get_scores_by_role(role_id: str) -> list[dict]:
    with _connect() as conn:
        # Only the most recent /analyze run for this role — otherwise re-running analysis
        # (even with an overlapping or different candidate set) would pile every historical
        # score row on top of the previous ones, showing candidates from old runs too.
        latest_run = conn.execute(
            "SELECT run_id FROM scores WHERE role_id = ? ORDER BY scored_at DESC LIMIT 1", (role_id,)
        ).fetchone()
        if not latest_run:
            return []
        run_id = latest_run["run_id"]

        rows = conn.execute("""
            SELECT s.*, sr.audit_mode, r.candidate_name, r.file_name, r.source_id, r.skills, r.total_years_experience,
                   r.current_role_title_from_summary,
                   r.experience, r.latest_role_period, r.education, r.certifications, r.projects
            FROM scores s JOIN resumes r ON s.candidate_id = r.candidate_id
            LEFT JOIN screening_runs sr ON s.run_id = sr.run_id
            WHERE s.role_id = ? AND s.run_id = ?
            ORDER BY s.hard_gate_failed ASC, s.final_score DESC
        """, (role_id, run_id)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["category_scores"] = json.loads(d["category_scores"])
            # Rows scored before the evidence column existed will have NULL here —
            # fall back to an empty dict instead of crashing on json.loads(None).
            d["evidence"] = json.loads(d["evidence"]) if d.get("evidence") else {}
            d["hard_gate_failed"] = bool(d["hard_gate_failed"])
            d["audit_mode"] = bool(d["audit_mode"])
            # Raw extracted resume fields — needed for the candidate comparison table,
            # not just the computed scores.
            for field in ["skills", "experience", "latest_role_period", "education", "certifications", "projects"]:
                d[field] = json.loads(d[field]) if d[field] else []
            results.append(d)
        return results


def get_score_by_role_and_candidate(role_id: str, candidate_id: str) -> dict | None:
    """Return one candidate from the latest completed run for a role."""
    return next(
        (score for score in get_scores_by_role(role_id) if score["candidate_id"] == candidate_id),
        None,
    )