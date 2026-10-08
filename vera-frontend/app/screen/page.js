"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { uploadJD, listJDs, uploadResumesStreaming, analyze, getResults, getJD, getRun, startSemanticSessions, updateCandidateName } from "../../lib/api";
import { resolveSemanticSession } from "../../lib/semanticRun";
import { useAppState, useToast } from "../providers";
import JDSummaryCard from "../../components/JDSummaryCard";

const STEPS = [
  { key: "jd", title: "Parse Job Description", detail: " (Extract mandatory and preferred requirements)" },
  { key: "resumes", title: "Understand resumes", detail: " (Extract skills, experience, projects and education)" },
  { key: "matching", title: "Semantic matching", detail: " (Compare evidence against JD clauses)" },
  { key: "scoring", title: "Calculate Job Fit Scores", detail: " (Apply deterministic weighted scoring)" },
  { key: "ranking", title: "Rank candidates", detail: " (Generate explainable shortlist)" },
];

export default function ScreenPage() {
  const router = useRouter();
  const toast = useToast();
  const { state, setCurrentRole, setBatch, setResultsForRole } = useAppState();

  const [existingRoles, setExistingRoles] = useState([]);
  const [selectedExistingRoleId, setSelectedExistingRoleId] = useState("");

  const [jdFile, setJdFile] = useState(null);
  const [jdRecord, setJdRecord] = useState(null); // { role_id, role_title, extraction_method, extraction_warnings }
  const [jdUploading, setJdUploading] = useState(false);
  const [jdFull, setJdFull] = useState(null); // full extracted JD record (mandatory_skills, etc.) for the summary card
  const [jdFullLoading, setJdFullLoading] = useState(false);

  const [resumeFiles, setResumeFiles] = useState([]);
  const [resumeResults, setResumeResults] = useState([]); // per-file upload status, appended as each one finishes
  const [resumesUploading, setResumesUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ done: 0, total: 0 });

  const [analyzing, setAnalyzing] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(-1); // -1 = idle
  const [doneSteps, setDoneSteps] = useState(new Set());
  const [statusText, setStatusText] = useState("Ready to analyze");
  const [activeRun, setActiveRun] = useState(null);
  const [auditMode, setAuditMode] = useState(false);
  const [semanticMode, setSemanticMode] = useState(false);
  const [semanticError, setSemanticError] = useState("");
  const [editingResume, setEditingResume] = useState(null);
  const [candidateNameDraft, setCandidateNameDraft] = useState("");
 

  const resumeInputRef = useRef(null);
  const stepTimerRef = useRef(null);
  const pollingStoppedRef = useRef(false);

  useEffect(() => {
    listJDs()
      .then(setExistingRoles)
      .catch(() => {
        // library listing is a nice-to-have; ignore failure here, upload flow still works
      });
    return () => {
      clearInterval(stepTimerRef.current);
      pollingStoppedRef.current = true;
    };
  }, []);

  const successfulCandidates = resumeResults.filter((r) => r.status === "ok");
  const activeRole = jdRecord || (selectedExistingRoleId
    ? existingRoles.find((r) => r.role_id === selectedExistingRoleId)
    : null);

  async function handleJdUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setJdFile(file);
    setJdUploading(true);
    setJdRecord(null);
    setJdFull(null);
    setSelectedExistingRoleId("");
    try {
      const record = await uploadJD(file);
      setJdRecord(record);
      toast(`Parsed JD: ${record.role_title || "role title not detected"}`);
      setJdFullLoading(true);
      try {
        const full = await getJD(record.role_id);
        setJdFull(full);
      } catch {
        // extracted-data card is a nice-to-have; the upload itself already succeeded
      } finally {
        setJdFullLoading(false);
      }
    } catch (err) {
      toast(err.message, "error");
      setJdFile(null);
    } finally {
      setJdUploading(false);
    }
  }

  async function handlePickExistingRole(roleId) {
    setSelectedExistingRoleId(roleId);
    setJdRecord(null);
    setJdFile(null);
    setJdFull(null);
    if (!roleId) return;
    setJdFullLoading(true);
    try {
      const full = await getJD(roleId);
      setJdFull(full);
    } catch (err) {
      toast(`Couldn't load JD details: ${err.message}`, "error");
    } finally {
      setJdFullLoading(false);
    }
  }

  async function handleResumeUpload(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setResumeFiles((prev) => [...prev, ...files]);
    setResumesUploading(true);
    setUploadProgress({ done: 0, total: files.length });

    let okCount = 0;
    let failCount = 0;

    try {
      await uploadResumesStreaming(files, (msg) => {
        if (msg.type === "meta") {
          setUploadProgress({ done: 0, total: msg.total });
          return;
        }
        // msg.type === "result" — arrives the moment THIS file finishes, not the whole batch.
        // Results can arrive out of upload order (fastest resume first), by design.
        setResumeResults((prev) => [...prev, msg]);
        setUploadProgress((p) => ({ ...p, done: p.done + 1 }));
        if (msg.status === "ok") okCount += 1;
        else failCount += 1;
      });

      toast(
        failCount
          ? `${okCount} resume(s) ready, ${failCount} failed to process`
          : `${okCount} resume(s) ready`,
        failCount && !okCount ? "error" : "info"
      );
    } catch (err) {
      toast(err.message, "error");
    } finally {
      setResumesUploading(false);
      if (resumeInputRef.current) resumeInputRef.current.value = "";
    }
  }

  function removeResume(index) {
    setResumeResults((prev) => prev.filter((_, i) => i !== index));
  }

  function editQueuedCandidateName(index) {
    const candidate = resumeResults[index];
    if (!candidate?.candidate_id) return;
 
    setEditingResume({
      index,
      candidateId: candidate.candidate_id,
    });
    setCandidateNameDraft(candidate.candidate_name || "");
  }
 
  async function saveQueuedCandidateName(event) {
    event.preventDefault();
 
    const candidateName = candidateNameDraft.trim().replace(/\s+/g, " ");
    if (!candidateName) {
      toast("Enter a candidate name before saving.", "error");
      return;
    }
 
    try {
      const saved = await updateCandidateName(
        editingResume.candidateId,
        candidateName
      );
 
      setResumeResults((previous) =>
        previous.map((item, itemIndex) =>
          itemIndex === editingResume.index
            ? { ...item, candidate_name: saved.candidate_name }
            : item
        )
      );
 
      setEditingResume(null);
      toast("Candidate name updated.", "success");
    } catch (err) {
      toast(`Couldn't update the candidate name: ${err.message}`, "error");
    }
  }

  function clearResumeBatch() {
    setResumeResults([]);
    setResumeFiles([]);
    setUploadProgress({ done: 0, total: 0 });
    if (resumeInputRef.current) resumeInputRef.current.value = "";
  }

  function showRunProgress(run) {
    const progress = run.progress || { total: run.candidate_count || 0, completed: 0, failed: 0, pending: 0 };
    const finished = progress.completed + progress.failed;
    const terminal = ["completed", "completed_with_errors", "failed"].includes(run.status);
    if (terminal) {
      setDoneSteps(new Set(STEPS.map((step) => step.key)));
      setActiveStepIndex(STEPS.length - 1);
      setStatusText(
        progress.failed
          ? `Analysis finished · ${progress.completed} scored · ${progress.failed} failed`
          : `Analysis complete · ${progress.completed} candidate(s) ranked`
      );
    } else if (finished) {
      setDoneSteps(new Set(["jd", "resumes", "matching"]));
      setActiveStepIndex(3);
      setStatusText(`Scoring ${finished}/${progress.total} candidate(s)…`);
    } else {
      setDoneSteps(new Set(["jd", "resumes"]));
      setActiveStepIndex(2);
      setStatusText(`Matching evidence for ${progress.total} candidate(s)…`);
    }
  }

  async function handleRunAnalysis() {
    if (!activeRole) {
      toast("Upload or select a job description first", "error");
      return;
    }
    if (!successfulCandidates.length) {
      toast("Upload at least one resume first", "error");
      return;
    }

    setAnalyzing(true);
    setActiveRun(null);
    setSemanticError("");
    pollingStoppedRef.current = false;
    setDoneSteps(new Set(["jd", "resumes"]));
    setActiveStepIndex(2);
    setStatusText(`Starting analysis for ${successfulCandidates.length} resume(s)…`);

    const roleId = activeRole.role_id;
    const candidateIds = successfulCandidates.map((r) => r.candidate_id);

    try {
      if (semanticMode) {
        setStatusText("Preparing semantic evidence for each candidate…");
        const kickoff = await startSemanticSessions(roleId, candidateIds);
        const completed = [];
        for (let index = 0; index < kickoff.sessions.length; index += 1) {
          const session = kickoff.sessions[index];
          setActiveStepIndex(2);
          setStatusText(`Semantic matching ${index + 1}/${kickoff.sessions.length} candidate(s)…`);
          const resolved = await resolveSemanticSession(session, {
            onProgress: (progress) => {
              const detail = typeof progress?.text === "string" ? progress.text : "Loading local browser model…";
              setStatusText(`Browser semantic model · ${detail}`);
            },
          });
          if (resolved.status !== "completed") {
            throw new Error(resolved.local_resolution?.reason || "Browser-local semantic matching was not completed.");
          }
          completed.push(resolved);
          setDoneSteps(new Set(["jd", "resumes", "matching", "scoring"]));
        }
        setDoneSteps(new Set(STEPS.map((step) => step.key)));
        setActiveStepIndex(STEPS.length - 1);
        setStatusText(`Semantic analysis complete · ${completed.length} candidate(s) ranked`);
        const full = await getResults(roleId);
        setCurrentRole({ role_id: roleId, role_title: activeRole.role_title });
        setBatch({ roleId, candidateIds, candidates: successfulCandidates });
        setResultsForRole(roleId, { ...full, role_title: activeRole.role_title });
        router.push(`/results/${roleId}`);
        return;
      }
      const result = await analyze(roleId, candidateIds, { asyncMode: true, auditMode });

      if (result.status === "ambiguous") {
        setAnalyzing(false);
        setActiveStepIndex(-1);
        setStatusText("Ready to analyze");
        toast(result.message, "error");
        return;
      }

      if (result.status !== "accepted") {
        throw new Error("The analysis run was not accepted.");
      }

      let run = {
        ...result,
        progress: { total: result.candidate_count, completed: 0, failed: 0, pending: result.candidate_count },
        tasks: [],
        audit_mode: auditMode,
      };
      setActiveRun(run);
      showRunProgress(run);

      while (!pollingStoppedRef.current) {
        await new Promise((resolve) => setTimeout(resolve, 700));
        run = await getRun(result.run_id);
        if (pollingStoppedRef.current) return;
        setActiveRun(run);
        showRunProgress(run);
        if (["completed", "completed_with_errors", "failed"].includes(run.status)) break;
      }

      if (pollingStoppedRef.current) return;
      setAnalyzing(false);
      if (run.progress.completed === 0) {
        toast("Analysis finished without any scored candidates. Review the failed tasks below.", "error");
        return;
      }

      setCurrentRole({ role_id: result.role_id, role_title: result.role_title });
      setBatch({
        roleId: result.role_id,
        candidateIds,
        candidates: successfulCandidates,
      });

      // Pull the full record set (with category_scores) so the results/detail pages have everything.
      const full = await getResults(result.role_id);
      setResultsForRole(result.role_id, { ...full, role_title: result.role_title });

      router.push(`/results/${result.role_id}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err || "Analysis failed.");
      setAnalyzing(false);
      setActiveStepIndex(-1);
      setStatusText("Semantic analysis failed");
      if (semanticMode) setSemanticError(message);
      toast(message, "error");
    }
  }

  return (
    <>
      <section className="view active">
        <div className="hero">
          <div>
            <div className="ey">AI-powered recruitment workflow</div>
            <h1>Screen candidates in minutes.</h1>
            <p>
              Upload one JD and a batch of resumes. GigaForce Resume Analyser extracts requirements, understands
              candidate experience, and produces an explainable ranking.
            </p>
          </div>
          <button className="btn primary" onClick={handleRunAnalysis} disabled={analyzing}>
            {analyzing ? "Analyzing…" : "▶ Run live analysis"}
          </button>
        </div>

        <div className="grid">
          {/* --- JD panel --- */}
          <div className="panel">
            <div className="head">
              <h3>1. Job Description</h3>
              <span className="tag">{jdRecord ? jdRecord.extraction_method : "AI parsed"}</span>
            </div>
            <div className="body">
              <div className="drop">
                {jdUploading ? (
                  <>
                    <strong><span className="spinner" />Parsing…</strong>
                    <small>{jdFile?.name}</small>
                  </>
                ) : jdRecord ? (
                  <>
                    <strong>{jdRecord.role_title || "Role title not detected"}</strong>
                    <small>{jdFile?.name} · Parsed successfully</small>
                  </>
                ) : (
                  <>
                    <strong>Drop a job description here</strong>
                    <small>PDF or DOCX</small>
                  </>
                )}
                <span className="fake">{jdRecord ? "Replace JD" : "Choose JD file"}</span>
                <input type="file" accept=".pdf,.docx" onChange={handleJdUpload} />
              </div>

              {existingRoles.length > 0 && (
                <div className="field" style={{ marginTop: 14 }}>
                  <label>Or use an existing role from the library</label>
                  <select
                    value={selectedExistingRoleId}
                    onChange={(e) => handlePickExistingRole(e.target.value)}
                  >
                    <option value="">— Select a role —</option>
                    {existingRoles.map((r) => (
                      <option key={r.role_id} value={r.role_id}>
                        {r.role_title || r.role_id}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {jdRecord && (
                <div className="reqs">
                  <div className="req">
                    <span>{(jdRecord.role_title || "Role")} · document parsed</span>
                    <span className="tag">{jdRecord.document_id?.slice(0, 8)}</span>
                  </div>
                  {jdRecord.extraction_warnings?.length > 0 &&
                    jdRecord.extraction_warnings.map((w, i) => (
                      <div className="req" key={i}>
                        <span style={{ color: "var(--w)" }}>{w}</span>
                        <span className="tag pref">Warning</span>
                      </div>
                    ))}
                </div>
              )}

              {jdFullLoading && (
                <div className="center-pad" style={{ padding: "14px 0" }}>
                  <span className="spinner" /> Loading extracted requirements…
                </div>
              )}
              {jdFull && <JDSummaryCard jd={jdFull} />}
            </div>
          </div>

          {/* --- Resume panel --- */}
          <div className="panel">
            <div className="head">
              <h3>2. Resume Batch</h3>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="tag">
                  {resumesUploading
                    ? `${uploadProgress.done}/${uploadProgress.total} processed`
                    : `${resumeResults.length} resumes`}
                </span>
                {resumeResults.length > 0 && !resumesUploading && (
                  <button onClick={clearResumeBatch} style={{ fontSize: 11 }}>
                    Clear all
                  </button>
                )}
              </div>
            </div>
            <div className="body">
              <div className="drop">
                <strong>
                  {resumesUploading
                    ? `Processing ${uploadProgress.done}/${uploadProgress.total}…`
                    : "Drop resumes here"}
                </strong>
                <small>PDF or DOCX · multiple files supported · results appear as each finishes</small>
                <span className="fake">Choose resumes</span>
                <input
                  ref={resumeInputRef}
                  type="file"
                  accept=".pdf,.docx"
                  multiple
                  disabled={resumesUploading}
                  onChange={handleResumeUpload}
                />
              </div>

              {resumesUploading && uploadProgress.total > 0 && (
                <div className="bar" style={{ marginTop: 14 }}>
                  <i style={{ width: `${(uploadProgress.done / uploadProgress.total) * 100}%` }} />
                </div>
              )}

              <div className="files">
                {resumeResults.map((r, i) => (
                  <div className="file" key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ minWidth: 0 }}>
                      ▧ {r.candidate_name || r.file_name}
                      {r.candidate_name && r.file_name && (
                        <small style={{ display: "block", marginLeft: 18, color: "var(--muted)" }}>
                          {r.file_name}
                        </small>
                      )}
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ color: r.status === "ok" ? "var(--a)" : "var(--r)", fontSize: 10 }}>
                        {r.status === "ok" ? "Ready" : r.error || "Failed"}
                      </span>
                      {r.status === "ok" && r.candidate_id && (
                        <button
                          onClick={() => editQueuedCandidateName(i)}
                          title="Edit candidate name"
                          aria-label={`Edit candidate name for ${r.candidate_name || r.file_name}`}
                          style={{ fontSize: 11, padding: "2px 8px" }}
                        >
                          Edit
                        </button>
                      )}
                      <button
                        onClick={() => removeResume(i)}
                        title="Remove from this batch"
                        style={{ fontSize: 11, padding: "2px 8px", color: "var(--r)" }}
                      >
                        ✕
                      </button>
                    </span>
                  </div>
                ))}
                {resumeResults.length === 0 && !resumesUploading && (
                  <div className="muted" style={{ padding: "8px 0" }}>
                    No resumes queued yet — only what you add here gets analyzed.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="panel" style={{ marginTop: 18 }}>
          <div className="head">
            <h3>Live analysis pipeline</h3>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <label className="audit-toggle" title="Uses deterministic rules only and does not call the semantic judge">
                <input
                  type="checkbox"
                  checked={auditMode}
                  disabled={analyzing}
                  onChange={(event) => {
                    setAuditMode(event.target.checked);
                    if (event.target.checked) setSemanticMode(false);
                  }}
                />
                Deterministic audit mode
              </label>
              <label className="audit-toggle" title="Uses embeddings and a cross-encoder first, then asks WebLLM only about uncertain evidence pairs.">
                <input
                  type="checkbox"
                  checked={semanticMode}
                  disabled={analyzing || auditMode}
                  onChange={(event) => setSemanticMode(event.target.checked)}
                />
                Browser semantic cascade
              </label>
              <span style={{ color: "var(--m)", fontSize: 11 }}>{statusText}</span>
            </div>
          </div>
          <div className="body">
            <div className="process">
              {STEPS.map((step, i) => (
                <div
                  key={step.key}
                  className={`step ${activeStepIndex === i ? "active" : ""} ${
                    doneSteps.has(step.key) ? "done" : ""
                  }`}
                >
                  <div className="dot">{doneSteps.has(step.key) ? "✓" : i + 1}</div>
                  <div>
                    <b>{step.title}</b>
                    <small>{step.detail}</small>
                  </div>
                </div>
              ))}
            </div>
            <div className="bar">
              <i
                style={{
                  width:
                    activeStepIndex >= 0
                      ? `${(Math.min(STEPS.length, doneSteps.size + 1) / STEPS.length) * 100}%`
                      : "0%",
                }}
              />
            </div>
            {semanticError && (
              <div role="alert" style={{ marginTop: 14, border: "1px solid var(--r)", borderRadius: 8, padding: "10px 12px", color: "var(--r)", fontSize: 12 }}>
                <b>Browser semantic matching failed.</b> {semanticError}
              </div>
            )}
            {activeRun && (
              <div className="run-progress" aria-live="polite">
                <div className="run-progress-head">
                  <span>Run {activeRun.run_id?.slice(0, 8)}</span>
                  <span>
                    {activeRun.audit_mode ? "Audit · " : ""}
                    {activeRun.progress?.completed || 0}/{activeRun.progress?.total || 0} scored
                    {activeRun.progress?.failed ? ` · ${activeRun.progress.failed} failed` : ""}
                    {activeRun.performance?.elapsed_ms != null ? ` · ${(activeRun.performance.elapsed_ms / 1000).toFixed(1)}s` : ""}
                  </span>
                </div>
                <div className="run-task-list">
                  {activeRun.tasks?.map((task) => {
                    const candidate = successfulCandidates.find((item) => item.candidate_id === task.candidate_id);
                    return (
                      <div className={`run-task ${task.status}`} key={task.candidate_id}>
                        <span>{candidate?.candidate_name || candidate?.file_name || task.candidate_id}</span>
                        <span>
                          {task.status === "completed" && task.cache_hit ? "Completed · cached" : task.status}
                          {task.duration_ms != null ? ` · ${(task.duration_ms / 1000).toFixed(1)}s` : ""}
                          {task.error ? ` · ${task.error}` : ""}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {editingResume && (
              <div className="name-dialog-backdrop">
                <form className="name-dialog" onSubmit={saveQueuedCandidateName}>
                  <h3>Edit candidate name</h3>
 
                  <label htmlFor="candidate-name">Candidate name</label>
                  <input
                      id="candidate-name"
                      value={candidateNameDraft}
                      onChange={(event) => setCandidateNameDraft(event.target.value)}
                      autoFocus
                  />
 
                  <div className="name-dialog-actions">
                    <button
                      type="button"
                      className="btn ghost"
                      onClick={() => setEditingResume(null)}
                   >
                      Cancel
                    </button>
                    <button type="submit" className="btn primary">
                      Save name
                    </button>
                  </div>
                </form>
              </div>
            )}
          </div>
        </div>
      </section>
    </>
  );
}
