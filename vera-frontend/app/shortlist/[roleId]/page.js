"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getResults, listJDs } from "../../../lib/api";
import { detailedCandidateSummary, passesMandatory, rankAll } from "../../../lib/scoring";
import { downloadShortlistDocx, downloadShortlistPdf, shortlistText } from "../../../lib/shortlistExport";
import { useAppState, useToast } from "../../providers";

const SCORE_THRESHOLD = 65;

export default function QualifiedCandidatesPage({ params }) {
  const { roleId } = params;
  const router = useRouter();
  const toast = useToast();
  const { state, setCurrentRole, setResultsForRole } = useAppState();
  const cached = state.resultsCache[roleId];
  const [loading, setLoading] = useState(!cached);
  const [roleTitle, setRoleTitle] = useState(cached?.role_title || state.currentRole?.role_title || "");
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const [full, roles] = await Promise.all([getResults(roleId), listJDs().catch(() => [])]);
        if (cancelled) return;
        const title = roles.find((role) => role.role_id === roleId)?.role_title || roleTitle || roleId;
        setRoleTitle(title);
        setCurrentRole({ role_id: roleId, role_title: title });
        setResultsForRole(roleId, { ...full, role_title: title });
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roleId]);

  const records = useMemo(() => {
    if (!cached) return [];
    return rankAll(cached.ranked, cached.excluded_hard_gate_failed)
      .filter((record) => record.final_score >= SCORE_THRESHOLD || passesMandatory(record));
  }, [cached]);
  const candidates = useMemo(() => records.map((record) => ({
    ...record,
    summary: detailedCandidateSummary(record, roleTitle),
  })), [records, roleTitle]);

  async function copyAll() {
    try {
      await navigator.clipboard.writeText(shortlistText({ roleTitle, candidates }));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      toast("Couldn't copy the qualified-candidate list.", "error");
    }
  }

  function exportDocx() {
    try { downloadShortlistDocx({ roleTitle, candidates }); } catch (err) { toast(`Couldn't create DOCX: ${err.message}`, "error"); }
  }
  function exportPdf() {
    try { downloadShortlistPdf({ roleTitle, candidates }); } catch (err) { toast(`Couldn't open PDF export: ${err.message}`, "error"); }
  }

  if (loading && !cached) return <section className="view active"><div className="center-pad"><span className="spinner" /> Loading qualified candidates…</div></section>;
  if (error && !cached) return <section className="view active"><div className="center-pad">Couldn&apos;t load qualified candidates: {error}</div></section>;

  return (
    <section className="view active">
      <div className="hero">
        <div>
          <div className="ey">Recruiter shortlist</div>
          <h1>Qualified candidates</h1>
          <p>{roleTitle || "Selected role"} · {candidates.length} candidate{candidates.length === 1 ? "" : "s"} included.</p>
          <p className="muted">A candidate appears here when their Job Fit is at least {SCORE_THRESHOLD}% or they pass the mandatory-skill threshold.</p>
        </div>
        <div className="export-actions">
          <button className="btn ghost" onClick={copyAll}>{copied ? "✓ Copied" : "Copy all"}</button>
          <button className="btn ghost" onClick={exportDocx} disabled={!candidates.length}>Download DOCX</button>
          <button className="btn primary" onClick={exportPdf} disabled={!candidates.length}>Download PDF</button>
        </div>
      </div>
      <div className="shortlist-list">
        {candidates.map((candidate) => (
          <article className="panel shortlist-card" key={candidate.candidate_id}>
            <div className="head">
              <div>
                <h3>{candidate.candidate_name || "Unnamed candidate"}</h3>
                <small className="muted">Rank #{candidate.rank} · {candidate.final_score}% Job Fit</small>
              </div>
              <button className="btn ghost shortlist-view" onClick={() => router.push(`/candidate/${candidate.candidate_id}?role=${roleId}`)}>View evidence</button>
            </div>
            <ul className="shortlist-summary">
              {candidate.summary.map((line) => <li key={line}>{line}</li>)}
            </ul>
          </article>
        ))}
        {!candidates.length && <div className="panel center-pad">No candidates meet either qualification condition for this role.</div>}
      </div>
    </section>
  );
}
