"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { getResults, getRecruiterFeedback, saveRecruiterFeedback } from "../../../lib/api";
import { useAppState } from "../../providers";
import Pill from "../../../components/Pill";
import {
  statusFor,
  rankAll,
  evidenceSections,
  detailedCandidateSummary,
  CATEGORY_MAX,
  CATEGORY_LABELS,
} from "../../../lib/scoring";
import { reviewSourcesForResult } from "../../../lib/resultsAdapter";
import { generateAssessmentReviewWithFallback } from "../../../lib/webllm";

export default function CandidateDetailPage({ params }) {
  return (
    <Suspense
      fallback={
        <section className="view active">
          <div className="center-pad">
            <span className="spinner" /> Loading candidate…
          </div>
        </section>
      }
    >
      <CandidateDetail params={params} />
    </Suspense>
  );
}

const STATUS_CHIP_STYLE = {
  matched: { background: "#12362b", color: "#67e19b" },
  weak_match: { background: "#3a2e18", color: "#f6c66c" },
  missing: { background: "#3a1f27", color: "#ff8192" },
};

const STATUS_ICON = {
  matched: "✓",
  weak_match: "~",
  missing: "×",
};

function EvidenceChip({ status, label, detail }) {
  const style = STATUS_CHIP_STYLE[status] || { background: "#14332f", color: "var(--a)" };
  const icon = STATUS_ICON[status] || "";
  return (
    <span className="tag" style={{ ...style, fontWeight: 600 }} title={detail || undefined}>
      {icon} {label}{status === "weak_match" ? " · inferred" : ""}
    </span>
  );
}

function EvidenceCitations({ citations }) {
  if (!citations?.length) return null;
  return (
    <div className="evidence-citations">
      {citations.map((citation, index) => (
        <div className="evidence-quote" key={`${citation.sourceId}-${index}`}>
          <span>{citation.sourceLabel}{citation.sourceId ? ` · ${citation.sourceId.slice(0, 12)}` : ""}</span>
          <q>{citation.quote}</q>
        </div>
      ))}
    </div>
  );
}

function EvidenceSection({ title, emptyMessage, children }) {
  return (
    <div className="evidence-section">
      <h4
        style={{
          color: "var(--m)",
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: 0.5,
          margin: "16px 0 8px",
        }}
      >
        {title}
      </h4>
      {emptyMessage ? (
        <div className="muted">{emptyMessage}</div>
      ) : (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>{children}</div>
      )}
    </div>
  );
}

async function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  // Fallback for browsers that do not expose the asynchronous Clipboard API.
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function CandidateDetail({ params }) {
  const { candidateId } = params;
  const searchParams = useSearchParams();
  const router = useRouter();
  const { state, setResultsForRole } = useAppState();

  const roleId = searchParams.get("role") || state.currentRole?.role_id;
  const [loading, setLoading] = useState(!state.resultsCache[roleId]);
  const [error, setError] = useState(null);
  const [summaryCopied, setSummaryCopied] = useState(false);
  const [localReview, setLocalReview] = useState(null);
  const [reviewing, setReviewing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [feedbackNote, setFeedbackNote] = useState("");
  const [savingFeedback, setSavingFeedback] = useState(false);

  useEffect(() => {
    if (!roleId) return;
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const full = await getResults(roleId);
        if (cancelled) return;
        const roleTitle = state.resultsCache[roleId]?.role_title || state.currentRole?.role_title || "";
        setResultsForRole(roleId, { ...full, role_title: roleTitle });
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roleId]);

  useEffect(() => {
    if (!roleId || !candidateId) return;
    let cancelled = false;
    getRecruiterFeedback(roleId, candidateId)
      .then((value) => {
        if (cancelled) return;
        setFeedback(value);
        setFeedbackNote(value.note || "");
      })
      .catch(() => {
        // Feedback is optional; a scoring result remains usable when this
        // non-critical read is unavailable.
      });
    return () => { cancelled = true; };
  }, [roleId, candidateId]);

  const data = roleId ? state.resultsCache[roleId] : null;
  const all = useMemo(() => (data ? rankAll(data.ranked, data.excluded_hard_gate_failed) : []), [data]);
  const record = all.find((r) => r.candidate_id === candidateId);
  const roleTitle = data?.role_title || state.currentRole?.role_title || "";

  if (!roleId) {
    return (
      <section className="view active">
        <div className="center-pad">
          No role in context. <Link href="/results">Pick a role</Link> to view candidates from.
        </div>
      </section>
    );
  }

  if (loading && !data) {
    return (
      <section className="view active">
        <div className="center-pad">
          <span className="spinner" /> Loading candidate…
        </div>
      </section>
    );
  }

  if ((error && !data) || (data && !record)) {
    return (
      <section className="view active">
        <div className="hero">
          <div>
            <div className="ey">Candidate analysis</div>
            <h1>Candidate not found</h1>
            <p>{error || "This candidate hasn't been scored for the current role."}</p>
          </div>
          <button className="btn primary" onClick={() => router.push(`/results/${roleId}`)}>
            ← Back to ranking
          </button>
        </div>
      </section>
    );
  }

  const status = statusFor(record.final_score);
  const sections = evidenceSections(record);
  const candidateSummary = detailedCandidateSummary(record, roleTitle);
  const ringGradient = `conic-gradient(var(--a) 0 ${record.final_score}%, #173047 ${record.final_score}%)`;
  const hasExperienceRequirement = Boolean(sections.experience.years);
  // Education is visible only when the JD actually included an education
  // condition. Candidate education alone is context, not a scoring criterion.
  const hasEducationRequirement = sections.education.length > 0;
  const candidateRole = sections.jobTitle?.best_match?.title || record.current_role_title_from_summary || "";
  const copyCandidateSummary = async () => {
    const text = [`Candidate summary — ${record.candidate_name || "Unnamed candidate"}`, "", ...candidateSummary.map((item) => `• ${item}`)].join("\n");
    try {
      await copyToClipboard(text);
      setSummaryCopied(true);
      window.setTimeout(() => setSummaryCopied(false), 2000);
    } catch {
      setSummaryCopied(false);
    }
  };
  const runLocalReview = async () => {
    setReviewing(true);
    try {
      const response = await generateAssessmentReviewWithFallback({
        assessment: {
          jd: { id: roleId, title: roleTitle },
          candidate: { id: record.candidate_id, name: record.candidate_name || "" },
          assessment: {
            score: record.final_score,
            eligible: !record.hard_gate_failed,
            decision_reason: record.hard_gate_reason || "",
          },
        },
        sources: reviewSourcesForResult(record),
      });
      setLocalReview(response);
    } finally {
      setReviewing(false);
    }
  };
  const saveFeedback = async (disposition) => {
    setSavingFeedback(true);
    try {
      const saved = await saveRecruiterFeedback(roleId, candidateId, {
        disposition,
        note: feedbackNote,
        score_snapshot: record.final_score,
      });
      setFeedback(saved);
    } finally {
      setSavingFeedback(false);
    }
  };

  return (
    <section className="view active">
      <div className="hero">
        <div>
          <div className="ey">Candidate analysis</div>
          <h1>{record.candidate_name || "Unnamed candidate"}</h1>
          <p>
            {candidateRole ? `${candidateRole}  ` : ""}
          </p>
        </div>
        <button className="btn primary" onClick={() => router.push(`/results/${roleId}`)}>
          ← Back to ranking
        </button>
      </div>

      <div className="detail">
        <div className="panel scorecard">
          <div style={{ color: "var(--m)", fontSize: 11 }}>JOB FIT SCORE</div>
          <div className="ring" style={{ background: ringGradient }}>
            <b>{record.final_score}</b>
          </div>
          <Pill kind={status.key}>{status.label}</Pill>
        </div>

        <div className="panel">
          <div className="head">
            <h3>Score breakdown</h3>
            <span className="tag">Explainable</span>
          </div>
          <div className="body metrics">
            {Object.entries(CATEGORY_MAX).filter(([key]) => record.category_scores?.[key] && !record.category_scores[key]?.not_applicable).map(([key, max]) => {
              const category = record.category_scores?.[key] || {};
              const score = category.score ?? 0;
              const hasSkillCoverage = Number.isFinite(category.required_count);
              const pct = hasSkillCoverage
                ? category.match_percentage
                : Math.round((score / max) * 100);
              const summary = hasSkillCoverage
                ? (category.not_required
                  ? "100% (Not required)"
                  : `${pct}% `)
                : `${pct}%`;
              const evidenceRows = Array.isArray(record.evidence?.[key]) ? record.evidence[key] : [];
              // Older stored results predate named_* fields. Derive the same
              // audit count from their individual evidence chips so a user
              // does not need to re-run a completed screening just to see it.
              const namedRequired = Number.isFinite(category.named_required_count)
                ? category.named_required_count
                : evidenceRows.length;
              const namedMatched = Number.isFinite(category.named_matched_count)
                ? category.named_matched_count
                : evidenceRows.filter((row) => row.status === "matched").length;
              const namedWeak = Number.isFinite(category.named_weak_count)
                ? category.named_weak_count
                : evidenceRows.filter((row) => row.status === "weak_match").length;
              const namedSummary = namedRequired
                ? `${namedMatched}/${namedRequired} named skills matched${namedWeak ? ` · ${namedWeak} related` : ""}`
                : "";
              return (
                <div className="metric" key={key}>
                  <small>{CATEGORY_LABELS[key] || key}</small>
                  <div className="bar">
                    <i style={{ width: `${Math.min(pct, 100)}%` }} />
                  </div>
                  <strong>
                    {summary}
                  </strong>
                  {namedSummary && <small>{namedSummary}</small>}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="cols">
        <div className="panel">
          <div className="head">
            <h3>Evidence</h3>
            <button className="btn ghost local-review-btn" disabled={reviewing} onClick={runLocalReview}>
              {reviewing ? "Preparing local review…" : "Review locally"}
            </button>
          </div>
          <div className="body">
            {localReview && (
              <div className={`local-review ${localReview.mode === "deterministic_fallback" ? "fallback" : ""}`}>
                <div>
                  <b>{localReview.mode === "webllm" ? "Local AI review" : "Deterministic fallback"}</b>
                  <span>{localReview.review.recommendation}</span>
                </div>
                <p>{localReview.review.summary}</p>
                {localReview.mode === "webllm" && (
                  <small>
                    Local model · {localReview.performance.initialization_ms} ms init · {localReview.performance.generation_ms} ms generation
                  </small>
                )}
                {localReview.mode === "deterministic_fallback" && <small>{localReview.fallback_reason}</small>}
              </div>
            )}
            {/* Only JD requirement categories that contain criteria are shown. */}
            {sections.skillSections.map((section) => (
              <EvidenceSection
                title={section.label}
                key={section.key}
              >
                {section.items.map((item, i) => (
                  <div key={i} className="evidence-item">
                    <EvidenceChip
                      status={item.status}
                      label={item.label}
                      detail={
                        item.detail ||
                        (item.status === "weak_match"
                          ? "Related evidence; 85% scoring credit. Verify the named skill in interview."
                          : item.status === "missing"
                          ? "No explicit evidence found in the resume."
                          : "Explicit resume evidence.")
                      }
                    />
                    <EvidenceCitations citations={item.citations} />
                  </div>
                ))}
              </EvidenceSection>
            ))}

            {/* Job Title Match Evidence */}
            {sections.jobTitle?.best_match && (
              <EvidenceSection title="Job Title Match">
                <EvidenceChip
                  status={
                    sections.jobTitle.status === "matched"
                      ? "matched"
                      : sections.jobTitle.status === "related"
                      ? "weak_match"
                      : "missing"
                  }
                  label={sections.jobTitle.best_match.title}
                  detail={sections.jobTitle.judge_reason || `Match Level: ${sections.jobTitle.match_level}`}
                />
              </EvidenceSection>
            )}

            {/* Experience is only a scored section when the JD specifies a minimum. */}
            {hasExperienceRequirement && <EvidenceSection title="Experience">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {sections.experience.years && (
                  <EvidenceChip
                    status={sections.experience.years.status}
                    label={`${sections.experience.years.total_years_experience} yrs total`}
                    detail={`${sections.experience.years.min_years_required} yrs required`}
                  />
                )}
                {sections.experience.roles.map((r, i) => (
                  <EvidenceChip
                    key={i}
                    status={r.status}
                    label={r.label}
                    detail={[r.detail, `relevance ${r.similarity}`].filter(Boolean).join(" · ")}
                  />
                ))}
              </div>
              {sections.experience.roles.length === 0 && !sections.experience.years && (
                <div className="muted" style={{ marginTop: 6 }}>No experience entries recorded.</div>
              )}
            </EvidenceSection>}

            {/* Education Evidence */}
            {hasEducationRequirement && <EvidenceSection
              title="Education"
              emptyMessage={sections.education.length === 0 ? "No education extracted from resume." : null}
            >
              {sections.education.map((e, i) => (
                <EvidenceChip 
                  key={i} 
                  status={e.status} 
                  label={e.label} 
                  detail={e.detail} 
                />
              ))}
            </EvidenceSection>}

            {/* Additional Skills */}
            {sections.additionalSkills.length > 0 && (
              <EvidenceSection title={sections.additionalSkillsTotal > sections.additionalSkills.length
                ? `Additional candidate skills (showing ${sections.additionalSkills.length} of ${sections.additionalSkillsTotal})`
                : "Additional candidate skills"}>
                {sections.additionalSkills.map((skill, i) => (
                  <span key={i} className="tag pref">
                    {skill}
                  </span>
                ))}
              </EvidenceSection>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="head">
            <h3>Candidate summary</h3>
            <button
              type="button"
              className="btn"
              onClick={copyCandidateSummary}
              aria-label="Copy candidate summary"
              style={{ padding: "7px 11px", fontSize: 12 }}
            >
              {summaryCopied ? "Copied" : "Copy"}
            </button>
          </div>
          <div className="body explain">
            <ul style={{ paddingLeft: "16px", margin: 0, textAlign: "left" }}>
              {candidateSummary.map((item, idx) => (
                <li key={idx} style={{ marginBottom: "8px" }}>{item}</li>
              ))}
            </ul>
            <div style={{ borderTop: "1px solid var(--line)", marginTop: 18, paddingTop: 16 }}>
              <h4 style={{ margin: "0 0 8px" }}>Recruiter decision</h4>
              <p className="muted" style={{ margin: "0 0 10px" }}>
                Capture your decision for future evaluation. It does not change this score.
              </p>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[
                  ["shortlist", "Shortlist"],
                  ["hold", "Hold"],
                  ["reject", "Reject"],
                ].map(([value, label]) => (
                  <button
                    className={`btn ${feedback?.disposition === value ? "primary" : "ghost"}`}
                    disabled={savingFeedback}
                    key={value}
                    onClick={() => saveFeedback(value)}
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
              <textarea
                aria-label="Recruiter decision note"
                value={feedbackNote}
                onChange={(event) => setFeedbackNote(event.target.value)}
                maxLength={2000}
                placeholder="Optional note for this decision"
                style={{ width: "100%", minHeight: 72, marginTop: 10, resize: "vertical" }}
              />
              {feedback?.disposition && (
                <small className="muted">Saved as {feedback.disposition} · score snapshot {feedback.score_snapshot ?? record.final_score}</small>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
