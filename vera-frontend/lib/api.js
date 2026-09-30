// lib/api.js
// Thin client for the TalentLens FastAPI backend (api.py — no-auth version).
// All functions throw an Error with a readable message on non-2xx responses,
// so callers can just try/catch and show err.message.

import { adaptResultsResponse } from "./resultsAdapter";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  let res;
  try {
    // CHANGE 1: Added headers merge to include ngrok bypass
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers: {
        ...options.headers,
        "ngrok-skip-browser-warning": "1",
      },
    });
  } catch (err) {
    throw new Error(
      `Couldn't reach the TalentLens API at ${BASE_URL}. Is the backend running (uvicorn api:app --reload)?`
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  if (res.status === 204) return null;
  return res.json();
}

/** Upload a single JD file (.pdf/.docx). Returns { role_id, role_title, document_id, extraction_method, extraction_warnings }. */
export function uploadJD(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/jds/upload", { method: "POST", body: form });
}

/** List every JD role currently in the library. */
export function listJDs() {
  return request("/jds");
}

/** Fetch the full extracted JD record (mandatory_skills, responsibilities, etc.) for one role. */
export function getJD(roleId) {
  return request(`/jds/${encodeURIComponent(roleId)}`);
}

/** Remove the previous candidate batch and its results, while retaining saved JDs. */
export function startNewScreening() {
  return request("/screenings/new", { method: "POST" });
}

/**
 * Upload a batch of resumes (.pdf/.docx) and stream results back as each one finishes,
 * instead of waiting for the whole batch. `onEvent` is called once per line the backend
 * sends:
 *   { type: "meta", total }                                              — sent first
 *   { type: "result", file_name, status: "ok"|"failed", candidate_id?, ... } — one per file,
 *                                                                            in COMPLETION
 *                                                                            order, not
 *                                                                            upload order.
 * Throws if the request itself fails (network error, non-2xx before streaming starts).
 * A malformed/partial line from a dropped connection is skipped rather than thrown, so one
 * bad chunk doesn't lose results that already arrived.
 */
export async function uploadResumesStreaming(files, onEvent) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  let res;
  try {
    // CHANGE 2: Added headers object to include ngrok bypass
    res = await fetch(`${BASE_URL}/resumes/upload`, { 
      method: "POST", 
      body: form,
      headers: {
        "ngrok-skip-browser-warning": "1",
      },
    });
  } catch (err) {
    throw new Error(
      `Couldn't reach the TalentLens API at ${BASE_URL}. Is the backend running (uvicorn api:app --reload)?`
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // not JSON — keep statusText
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  // Fallback for any environment where streaming reads aren't available — parse the
  // whole NDJSON body at once rather than failing outright.
  if (!res.body || !res.body.getReader) {
    const text = await res.text();
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed));
      } catch {
        // skip a malformed line rather than aborting everything already parsed
      }
    }
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex;
    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;
      try {
        onEvent(JSON.parse(line));
      } catch {
        // skip a malformed line rather than aborting everything already parsed
      }
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    try {
      onEvent(JSON.parse(trailing));
    } catch {
      // ignore a truncated trailing chunk
    }
  }
}

/**
 * Upload a batch of resumes (.pdf/.docx) and wait for the whole batch. Individual failures
 * don't stop the rest. Kept for callers that want a single Promise instead of progressive
 * events — prefer uploadResumesStreaming for anything showing upload progress in the UI.
 * Returns { results: [{ file_name, status: 'ok'|'failed', candidate_id?, candidate_name?, extraction_warnings?, error? }] }.
 */
export async function uploadResumes(files) {
  const results = [];
  await uploadResumesStreaming(files, (msg) => {
    if (msg.type === "result") results.push(msg);
  });
  return { results };
}

/**
 * Resolve a role query and score resumes against it.
 * candidateIds: pass the ids from the current upload batch, or omit to score every stored resume.
 */
export function analyze(roleId, candidateIds, { asyncMode = false, auditMode = false } = {}) {
  return request("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      role_id: roleId,
      candidate_ids: candidateIds && candidateIds.length ? candidateIds : null,
      async_mode: asyncMode,
      audit_mode: auditMode,
    }),
  });
}

/** Fetch the lifecycle and per-candidate task state for one analysis run. */
export function getRun(runId) {
  return request(`/runs/${encodeURIComponent(runId)}`);
}

/** List recent analysis-run summaries, optionally limited to one role. */
export function listRuns({ roleId, limit } = {}) {
  const params = new URLSearchParams();
  if (roleId) params.set("role_id", roleId);
  if (limit) params.set("limit", String(limit));
  const query = params.toString();
  return request(`/runs${query ? `?${query}` : ""}`);
}

/** Fetch the full ranked list of scores (with category breakdowns) for a role. */
export function getResults(roleId) {
  return request(`/results/${encodeURIComponent(roleId)}`).then(adaptResultsResponse);
}

/** Fetch the compact, versioned assessment for one scored JD/candidate pair. */
export function getAssessment(roleId, candidateId) {
  return request(
    `/assessments/${encodeURIComponent(roleId)}/${encodeURIComponent(candidateId)}`
  );
}

/** Submit one browser-local WebLLM ambiguity pass for server-side contract validation. */
export function validateWebLLMAmbiguity(batch, response) {
  return request("/webllm/ambiguity/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ batch, response: { judgments: response } }),
  });
}

/** Start server-owned semantic sessions; each may request one local WebLLM batch. */
export function startSemanticSessions(roleId, candidateIds) {
  return request("/semantic-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role_id: roleId, candidate_ids: candidateIds }),
  });
}

export function getSemanticSession(sessionId) {
  return request(`/semantic-sessions/${encodeURIComponent(sessionId)}`);
}

/** Submit browser-local judgments to the specific server-issued session. */
export function submitSemanticJudgments(sessionId, judgments) {
  return request(`/semantic-sessions/${encodeURIComponent(sessionId)}/webllm-judgments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ response: { judgments } }),
  });
}

export function getRecruiterFeedback(roleId, candidateId) {
  return request(`/feedback/${encodeURIComponent(roleId)}/${encodeURIComponent(candidateId)}`);
}

export function saveRecruiterFeedback(roleId, candidateId, feedback) {
  return request(`/feedback/${encodeURIComponent(roleId)}/${encodeURIComponent(candidateId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(feedback),
  });
}

/** Save a recruiter-corrected candidate name in the canonical resume record. */
export function updateCandidateName(candidateId, candidateName) {
  return request(`/candidates/${encodeURIComponent(candidateId)}/name`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_name: candidateName }),
  });
}

export { BASE_URL };