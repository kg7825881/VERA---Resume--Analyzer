"""
api.py — FastAPI layer for TalentLens.

Endpoints:
  POST /jds/upload         — upload a JD (PDF/DOCX), ingest, store
  GET  /jds                — list all JD roles in the library
  POST /resumes/upload      — upload one or more resumes, ingest, store (partial-failure tolerant)
  POST /analyze              — resolve a role query, score resumes against it, store results
  GET  /results/{role_id}   — fetch the ranked list of scores for a role

No auth — built for a single internal HR user against the Next.js frontend.

Run locally with: uvicorn api:app --reload
"""

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
from common import new_id, now_iso
from extractor import ingest_resume, extract_structured_evidence, _is_plausible_candidate_name
from jd_extractor import ingest_jd
from scorer import calculate_job_fit, validate_jd_for_scoring
from role_resolver import resolve_role
from assessment_schema import build_assessment
from response_validator import ResponseValidationError, validate_assessment_response
from webllm_contract import validate_ambiguity_judgments
from semantic_pipeline import create_semantic_session, finalize_semantic_session, legacy_result_for_session
from cross_encoder import CrossEncoderMatcher
from jd_policy import build_requirement_metadata, refresh_stale_requirement_metadata
from feedback import validate_feedback
from analysis_workflow import cache_key, validate_analysis_request

# Ollama serves requests over HTTP, so ingestion/scoring calls are I/O-bound — a thread pool
# lets several run concurrently even though each individual call is a normal blocking function.
# This does NOT by itself guarantee the Ollama *server* processes them in parallel — check
# OLLAMA_NUM_PARALLEL (and that you have RAM headroom for it) if wall-clock time doesn't improve.
MAX_WORKERS = int(os.environ.get("TALENTLENS_MAX_WORKERS", "4"))
logger = logging.getLogger("talentlens.api")

app = FastAPI(title="VERA Resume Analyzer")

# Allows the Next.js dev server (and same-origin prod builds you configure later) to call this API.
FRONTEND_ORIGINS = os.environ.get("TALENTLENS_FRONTEND_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_JDS = "storage/jds"
STORAGE_RESUMES = "storage/resumes"
os.makedirs(STORAGE_JDS, exist_ok=True)
os.makedirs(STORAGE_RESUMES, exist_ok=True)


@app.on_event("startup")
def on_startup():
    db.init_db()
    repaired_names = 0
    repaired_roles = 0
    # Name corrections are display metadata, not scoring evidence. Repair only
    # objectively invalid legacy values (for example a technology list or
    # "PROJECT EXPERIENCE") and only when the original stored resume text
    # yields a plausible replacement under the current extractor rules.
    for resume in db.get_resumes():
        current_name = resume.get("candidate_name", "")
        if not _is_plausible_candidate_name(current_name):
            extracted_name = extract_structured_evidence(resume.get("source_markdown", "")).get("candidate_name", "")
            if extracted_name and _is_plausible_candidate_name(extracted_name):
                db.update_candidate_name(resume["candidate_id"], extracted_name)
                repaired_names += 1

        # Older records can have a valid source bundle but an empty role because
        # a multi-column PDF hid the Experience heading during the original
        # extraction. Re-run only the deterministic extractor and fill missing
        # role facts; do not overwrite an existing verified work history.
        existing_roles = [
            item for item in (resume.get("experience") or [])
            if isinstance(item, dict) and item.get("title")
        ]
        if resume.get("current_role_title_from_summary") and existing_roles:
            continue
        repaired = extract_structured_evidence(resume.get("source_markdown", ""))
        recovered_roles = repaired.get("experience") or []
        recovered_current_role = repaired.get("current_role_title_from_summary", "")
        if recovered_current_role and recovered_roles:
            db.update_resume_role_facts(
                resume["candidate_id"],
                existing_roles or recovered_roles,
                resume.get("current_role_title_from_summary") or recovered_current_role,
            )
            repaired_roles += 1
    if repaired_names:
        logger.info("Repaired %s invalid candidate name(s) from verified resume text.", repaired_names)
    if repaired_roles:
        logger.info("Recovered current/latest role evidence for %s stored resume(s).", repaired_roles)


def _save_upload(upload_file: UploadFile, target_dir: str) -> str:
    dest_path = os.path.join(target_dir, upload_file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return dest_path


# --- JDs ---

@app.post("/jds/upload")
def upload_jd(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(400, f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")

    path = _save_upload(file, STORAGE_JDS)
    try:
        record = ingest_jd(path)
        # Semantic mode evaluates stable JD requirement objects, not the legacy
        # display-only skill arrays. Build and persist that policy at ingestion
        # time so every newly uploaded JD can start a semantic session.
        record["requirement_metadata"] = build_requirement_metadata(record)
    except Exception as e:
        raise HTTPException(500, f"Failed to process JD: {str(e)}")

    storage_status = db.insert_jd(record)
    # extraction_warnings is intentionally NOT included here — those are extraction-quality
    # signals for whoever runs the pipeline (already logged server-side by jd_extractor.py),
    # not candidate/role-facing data. The full record (including extraction_warnings) is
    # still in the DB via db.insert_jd() above if an admin/debug view ever wants it —
    # GET /jds/{role_id} exposes that full record already.
    return {
        "role_id": record["role_id"],
        "role_title": record.get("role_title", ""),
        "document_id": record["document_id"],
        "storage_status": storage_status,
        "extraction_method": record["extraction_method"],
    }


@app.get("/jds")
def list_jds():
    return db.get_all_roles()


@app.get("/jds/{role_id}")
def get_jd(role_id: str):
    """Full extracted JD record (mandatory_skills, responsibilities, etc.) — used by the frontend
    to display the raw structured JD data, not just the summary from GET /jds."""
    jd = db.get_jd_by_role_id(role_id)
    if not jd:
        raise HTTPException(404, f"No JD found for role_id '{role_id}'.")
    return jd


class RequirementOverridesRequest(BaseModel):
    recruiter_overrides: Dict[str, Dict[str, Any]]


@app.put("/jds/{role_id}/requirement-overrides")
def update_requirement_overrides(role_id: str, request: RequirementOverridesRequest):
    """Save recruiter non-negotiable/priority overrides for an existing JD policy."""
    try:
        jd = db.update_jd_requirement_overrides(role_id, request.recruiter_overrides)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if not jd:
        raise HTTPException(404, f"No JD found for role_id '{role_id}'.")
    return {"role_id": role_id, "requirement_metadata": jd["requirement_metadata"]}


class WebLLMAmbiguityValidationRequest(BaseModel):
    """Client-to-server handoff for one browser-local ambiguity batch.

    Slice 6 deliberately validates but does not score or persist these results;
    the scoring slice will bind this handoff to a server-created analysis run.
    """

    batch: Dict[str, Any]
    response: Dict[str, Any]


@app.post("/webllm/ambiguity/validate")
def validate_webllm_ambiguity(request: WebLLMAmbiguityValidationRequest):
    try:
        judgments = validate_ambiguity_judgments(request.batch, request.response)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {
        "candidate_id": request.batch.get("candidate_id"),
        "judgments": judgments,
        "validated": True,
    }


class SemanticSessionRequest(BaseModel):
    role_id: str
    candidate_ids: List[str]


class SemanticJudgmentsRequest(BaseModel):
    response: Dict[str, Any]


@app.post("/semantic-sessions")
def start_semantic_sessions(request: SemanticSessionRequest):
    """Issue server-owned candidate batches for the cross-encoder/WebLLM cascade."""
    role = db.get_jd_by_role_id(request.role_id)
    if not role:
        raise HTTPException(404, f"No JD found for role_id '{request.role_id}'.")
    # Upgrade policies created before Required and Preferred sections were
    # faithfully preserved.  This only touches explicitly versioned old
    # policies; it does not invent absent JD categories.
    refreshed_metadata = refresh_stale_requirement_metadata(role)
    if refreshed_metadata is not None:
        role = db.replace_jd_requirement_metadata(request.role_id, refreshed_metadata) or role
    errors = validate_analysis_request(request.candidate_ids, {item["candidate_id"] for item in db.get_resumes()})
    if errors:
        raise HTTPException(422, " ".join(errors))
    sessions = []
    semantic_run_id = new_id()
    # Reuse the lazy model adapter across the whole batch. Previously each
    # candidate created and downloaded/loaded an independent cross-encoder.
    matcher = CrossEncoderMatcher()
    for candidate_id in request.candidate_ids:
        candidate = db.get_resume_by_candidate_id(candidate_id)
        session_id = new_id()
        try:
            session = create_semantic_session(session_id, role, candidate, matcher=matcher)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(503, f"Semantic matching could not start: {exc}")
        session["run_id"] = semantic_run_id
        db.create_semantic_session(session, now_iso())
        if session["status"] == "completed":
            db.insert_score(candidate_id, request.role_id, semantic_run_id, legacy_result_for_session(session, candidate, role))
        sessions.append(_semantic_session_response(session))
    return {"role_id": request.role_id, "run_id": semantic_run_id, "sessions": sessions}


@app.get("/semantic-sessions/{session_id}")
def get_semantic_session(session_id: str):
    session = db.get_semantic_session(session_id)
    if not session:
        raise HTTPException(404, "Semantic session not found.")
    return _semantic_session_response(session)


@app.post("/semantic-sessions/{session_id}/webllm-judgments")
def submit_semantic_judgments(session_id: str, request: SemanticJudgmentsRequest):
    session = db.get_semantic_session(session_id)
    if not session:
        raise HTTPException(404, "Semantic session not found.")
    try:
        completed = finalize_semantic_session(session, request.response)
        db.update_semantic_session(completed, now_iso())
        candidate = db.get_resume_by_candidate_id(completed["candidate_id"])
        role = db.get_jd_by_role_id(completed["role_id"])
        if not candidate or not role:
            raise ValueError("The candidate or JD was removed before browser semantic scoring completed.")
        db.insert_score(
            completed["candidate_id"], completed["role_id"], completed["run_id"],
            legacy_result_for_session(completed, candidate, role),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return _semantic_session_response(completed)


def _semantic_session_response(session: dict) -> dict:
    """Do not leak the server's full working state beyond the necessary browser batch."""
    response = {
        "session_id": session["session_id"], "role_id": session["role_id"],
        "candidate_id": session["candidate_id"], "status": session["status"],
        "matching": session.get("matching", {}), "result": session.get("result"),
    }
    if session["status"] == "pending_webllm":
        response["ambiguity_batch"] = session.get("ambiguity_batch")
    return response


class RecruiterFeedbackRequest(BaseModel):
    disposition: str
    note: str = ""
    score_snapshot: float | None = None


@app.get("/feedback/{role_id}/{candidate_id}")
def get_recruiter_feedback(role_id: str, candidate_id: str):
    feedback = db.get_recruiter_feedback(role_id, candidate_id)
    return feedback or {"role_id": role_id, "candidate_id": candidate_id, "disposition": None, "note": "", "score_snapshot": None}


@app.put("/feedback/{role_id}/{candidate_id}")
def save_recruiter_feedback(role_id: str, candidate_id: str, request: RecruiterFeedbackRequest):
    if not db.get_jd_by_role_id(role_id):
        raise HTTPException(404, f"No JD found for role_id '{role_id}'.")
    if not db.get_resume_by_candidate_id(candidate_id):
        raise HTTPException(404, f"No resume found for candidate_id '{candidate_id}'.")
    try:
        disposition, note, score_snapshot = validate_feedback(request.disposition, request.note, request.score_snapshot)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return db.save_recruiter_feedback(role_id, candidate_id, disposition, note, score_snapshot, now_iso())


# --- Resumes ---

@app.post("/screenings/new")
def start_new_screening():
    """Start a clean candidate batch while keeping the saved JD library intact."""
    return db.clear_screening_data()

@app.post("/resumes/upload")
def upload_resumes(files: List[UploadFile] = File(...)):
    """Batch upload — individual file failures don't stop the rest (per Phase 2 design).
    Streams results back as newline-delimited JSON (NDJSON) so the frontend can add each
    candidate to the list the moment ITS ingestion finishes, instead of waiting for the
    entire batch. Files are still ingested concurrently — this changes *when the client
    finds out*, not the concurrency itself (that was already fixed).

    Response shape: one JSON object per line —
      {"type": "meta", "total": N}                                   — sent first
      {"type": "result", "file_name": ..., "status": "ok"|"failed", ...}  — one per file,
                                                                            in COMPLETION order,
                                                                            not upload order.
    """

    # Validate + save synchronously first (fast, and keeps UploadFile handling on the main
    # thread — file.read() isn't safe to call concurrently from a worker thread).
    to_process = []  # (file_name, saved_path)
    immediate_failures = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".pdf", ".docx"):
            immediate_failures.append({
                "type": "result", "file_name": file.filename,
                "status": "failed", "error": f"Unsupported file type: {ext}",
            })
            continue
        path = _save_upload(file, STORAGE_RESUMES)
        to_process.append((file.filename, path))

    total = len(files)

    def _process(item):
        file_name, path = item
        try:
            record = ingest_resume(path)
            db.insert_resume(record)
            # extraction_warnings intentionally NOT included in the frontend-facing result —
            # already logged server-side by extractor.py, and still in the full DB record.
            return {
                "type": "result",
                "file_name": file_name,
                "status": "ok",
                "candidate_id": record["candidate_id"],
                "candidate_name": record.get("candidate_name", ""),
            }
        except Exception as e:
            return {"type": "result", "file_name": file_name, "status": "failed", "error": str(e)}

    def stream():
        yield json.dumps({"type": "meta", "total": total}) + "\n"

        for failure in immediate_failures:
            yield json.dumps(failure) + "\n"

        if to_process:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(_process, item) for item in to_process]
                # as_completed yields whichever finishes first — that's the whole point:
                # a fast resume shouldn't wait behind a slow one before reaching the client.
                for future in as_completed(futures):
                    yield json.dumps(future.result()) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class CandidateNameUpdateRequest(BaseModel):
    candidate_name: str


@app.put("/candidates/{candidate_id}/name")
def update_candidate_name(candidate_id: str, request: CandidateNameUpdateRequest):
    """Save an HR correction to a candidate name used across rankings and details."""
    candidate_name = " ".join((request.candidate_name or "").split())
    if not 2 <= len(candidate_name) <= 120:
        raise HTTPException(422, "Candidate name must contain between 2 and 120 characters.")
    if any(ord(char) < 32 for char in candidate_name):
        raise HTTPException(422, "Candidate name contains unsupported characters.")
    record = db.update_candidate_name(candidate_id, candidate_name)
    if not record:
        raise HTTPException(404, f"No resume found for candidate_id '{candidate_id}'.")
    return {"candidate_id": candidate_id, "candidate_name": record["candidate_name"]}


# --- Analyze ---

class AnalyzeRequest(BaseModel):
    role_id: Optional[str] = None
    role_query: Optional[str] = None
    candidate_ids: Optional[List[str]] = None  # if omitted, scores ALL stored resumes
    async_mode: bool = False  # return a run_id immediately so clients can poll /runs/{run_id}
    audit_mode: bool = False  # bypass LLM judgment; score from deterministic rules only


@app.post("/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    # The UI already has a selected role_id.  Use it directly rather than trying
    # to infer the same record again from a non-unique display title.
    if req.role_id:
        role = db.get_jd_by_role_id(req.role_id)
        if not role:
            raise HTTPException(404, f"No JD found for role_id '{req.role_id}'.")
    elif req.role_query:
        roles = db.get_all_roles()
        resolution = resolve_role(req.role_query, roles)

        if resolution["status"] == "no_match":
            raise HTTPException(404, f"No JD found matching '{req.role_query}'. Upload a JD for this role first.")

        if resolution["status"] == "ambiguous":
            return {
                "status": "ambiguous",
                "message": f"'{req.role_query}' matches multiple roles — please specify which one.",
                "candidates": [{"role_id": c["role_id"], "role_title": c["role_title"]} for c in resolution["candidates"]],
            }

        role = resolution["role"]
    else:
        raise HTTPException(422, "Provide role_id or role_query.")

    jd_data = db.get_jd_by_role_id(role["role_id"])
    validation_errors = validate_jd_for_scoring(jd_data)
    if validation_errors:
        raise HTTPException(422, " ".join(validation_errors))
    # Load the complete local batch first so explicitly requested IDs can be
    # validated instead of silently dropping a typo from the analysis.
    available_resumes = db.get_resumes()
    validation_errors = validate_analysis_request(
        req.candidate_ids, {resume["candidate_id"] for resume in available_resumes}
    )
    if validation_errors:
        raise HTTPException(422, " ".join(validation_errors))
    selected_ids = set(req.candidate_ids) if req.candidate_ids is not None else None
    resumes = [resume for resume in available_resumes if selected_ids is None or resume["candidate_id"] in selected_ids]

    if not resumes:
        raise HTTPException(404, "No resumes found to analyze (upload resumes first, or check candidate_ids).")

    run_id = new_id()  # ties every score from this /analyze call together as one run
    db.create_screening_run(
        run_id, role["role_id"], [resume["candidate_id"] for resume in resumes], now_iso(),
        audit_mode=req.audit_mode,
    )

    def audit_judge(requirement, evidence_chunks):
        """Fail closed without model inference for reproducible audit runs."""
        return {
            "match": "none", "confidence": 1.0,
            "reason": "Audit mode: semantic model judgment is disabled.",
        }

    def _score(resume):
        candidate_id = resume["candidate_id"]
        try:
            db.start_screening_task(run_id, candidate_id, now_iso())
            key = cache_key(jd_data, resume, audit_mode=req.audit_mode)
            result = db.get_cached_assessment(key)
            cache_hit = result is not None
            if result is None:
                result = (
                    calculate_job_fit(resume, jd_data, judge_fn=audit_judge)
                    if req.audit_mode else calculate_job_fit(resume, jd_data)
                )
                db.put_cached_assessment(key, result, now_iso())
            # Scores are recorded for every run, including a cache hit, so
            # result history and the existing results endpoint stay complete.
            db.insert_score(candidate_id, role["role_id"], run_id, result)
            db.complete_screening_task(run_id, candidate_id, cache_hit, now_iso())
            return {
                "candidate_id": candidate_id,
                "candidate_name": resume.get("candidate_name", ""),
                "final_score": result["final_score"],
                "hard_gate_failed": result["hard_gate_failed"],
                "hard_gate_reason": result["hard_gate_reason"],
                "cache_hit": cache_hit,
            }
        except Exception as exc:
            db.fail_screening_task(run_id, candidate_id, str(exc), now_iso())
            return {"candidate_id": candidate_id, "candidate_name": resume.get("candidate_name", ""), "error": str(exc)}

    def execute_run() -> dict:
        """Run all candidate work and always leave a terminal run state."""
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                scored_results = [future.result() for future in as_completed([pool.submit(_score, resume) for resume in resumes])]
        except Exception as exc:
            # This is a run-level failure (rather than a single candidate
            # failure handled in _score). Mark any unfinished tasks so a
            # polling client never sees a permanently-running run.
            logger.exception("Analysis run %s failed before candidate completion", run_id)
            for resume in resumes:
                db.fail_screening_task(run_id, resume["candidate_id"], str(exc), now_iso())
            db.finish_screening_run(run_id, "failed", now_iso())
            return {"status": "failed", "run_id": run_id, "role_id": role["role_id"], "role_title": role["role_title"], "failed_tasks": [{"error": str(exc)}]}

        failures = [result for result in scored_results if "error" in result]
        scored_results = [result for result in scored_results if "error" not in result]
        db.finish_screening_run(run_id, "completed_with_errors" if failures else "completed", now_iso())
        scored_results.sort(key=lambda r: r["final_score"], reverse=True)
        ranked = [r for r in scored_results if not r["hard_gate_failed"]]
        excluded = [r for r in scored_results if r["hard_gate_failed"]]
        return {
            "status": "scored", "run_id": run_id, "role_id": role["role_id"],
            "role_title": role["role_title"], "ranked": ranked,
            "excluded_hard_gate_failed": excluded, "failed_tasks": failures,
        }

    if req.async_mode:
        background_tasks.add_task(execute_run)
        return {
            "status": "accepted", "run_id": run_id, "role_id": role["role_id"],
            "role_title": role["role_title"], "candidate_count": len(resumes),
        }
    return execute_run()


# --- Results ---

@app.get("/runs")
def list_runs(role_id: Optional[str] = None, limit: int = Query(default=25, ge=1, le=100)):
    """List recent analysis runs; use GET /runs/{run_id} for task detail."""
    return db.list_screening_runs(role_id=role_id, limit=limit)

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Return a durable run record with the state of every candidate task."""
    run = db.get_screening_run(run_id)
    if not run:
        raise HTTPException(404, f"No screening run found for run_id '{run_id}'.")
    return run

@app.get("/assessments/{role_id}/{candidate_id}")
def get_assessment(role_id: str, candidate_id: str):
    """Compact, versioned decision view for one candidate/JD pairing.

    Unlike ``/results``, this response deliberately omits the raw extraction
    payload and full judge evidence. It is suitable for lightweight clients and
    decision exports that only need the JD criteria, candidate summary, and
    outcome of the latest scoring run.
    """
    jd = db.get_jd_by_role_id(role_id)
    if not jd:
        raise HTTPException(404, f"No JD found for role_id '{role_id}'.")
    score = db.get_score_by_role_and_candidate(role_id, candidate_id)
    if not score:
        raise HTTPException(404, "No score found for this candidate and role in the latest analysis run.")
    # Use the canonical resume record rather than the joined score row.  The
    # score query intentionally selects only fields needed by /results, while
    # this contract owns a compact candidate summary (including the enriched
    # all-sources skill list).
    candidate = db.get_resume_by_candidate_id(candidate_id)
    if not candidate:
        # This should be unreachable because get_score_by_role_and_candidate
        # joins resumes, but make the contract's failure mode explicit if old
        # or manually modified database rows ever violate that assumption.
        raise HTTPException(404, f"No resume found for candidate_id '{candidate_id}'.")
    response = build_assessment(jd, candidate, score)
    try:
        return validate_assessment_response(response)
    except ResponseValidationError:
        # Do not return a partial response if an internal shape changes. The
        # logged exception gives operators the contract mismatch while callers
        # get a stable failure rather than a potentially unsafe payload.
        logger.exception("Assessment response failed contract validation")
        raise HTTPException(500, "Assessment response failed contract validation.")

@app.get("/results/{role_id}")
def get_results(role_id: str):
    scores = db.get_scores_by_role(role_id)
    if not scores:
        raise HTTPException(404, f"No scores found for role_id '{role_id}'. Run /analyze first.")

    ranked = [s for s in scores if not s["hard_gate_failed"]]
    excluded = [s for s in scores if s["hard_gate_failed"]]
    return {"role_id": role_id, "ranked": ranked, "excluded_hard_gate_failed": excluded}
