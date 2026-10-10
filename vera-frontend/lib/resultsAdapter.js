// Converts persistence-shaped /results data into the browser's stable view
// model. Keep this boundary small: result pages should not need to know how
// SQLite rows, scorer evidence, or source bundles are represented internally.

const MAX_EVIDENCE_QUOTE_CHARS = 1_200;
const INTERNAL_FIELDS = new Set(["source_markdown", "raw_resume_text", "raw_output"]);

function strings(value) {
  return Array.isArray(value) ? value.filter((item) => typeof item === "string" && item.trim()) : [];
}

function adaptChunk(chunk, fallbackSourceId) {
  if (!chunk || typeof chunk !== "object" || typeof chunk.text !== "string") return null;
  const sourceId = typeof chunk.source_id === "string" && chunk.source_id
    ? chunk.source_id
    : fallbackSourceId || null;
  return {
    sourceId,
    quote: chunk.text.slice(0, MAX_EVIDENCE_QUOTE_CHARS),
    sourceLabel: typeof chunk.source_label === "string" ? chunk.source_label : "Resume evidence",
    sourceType: typeof chunk.source_type === "string" ? chunk.source_type : "resume",
    score: Number.isFinite(chunk.bm25_score) ? chunk.bm25_score : null,
  };
}

function adaptEvidenceRow(row, fallbackSourceId) {
  if (!row || typeof row !== "object") return null;
  return {
    ...row,
    citations: (Array.isArray(row.evidence) ? row.evidence : [])
      .map((chunk) => adaptChunk(chunk, fallbackSourceId))
      .filter(Boolean),
    // The original evidence chunk list is retained for the existing detail
    // view, but it has already been clipped and normalized into citations.
    evidence: undefined,
  };
}

function adaptEvidence(evidence, fallbackSourceId) {
  if (!evidence || typeof evidence !== "object") return {};
  const output = {};
  for (const [key, value] of Object.entries(evidence)) {
    if (Array.isArray(value)) {
      output[key] = value.map((row) => adaptEvidenceRow(row, fallbackSourceId) || row);
    } else {
      output[key] = value;
    }
  }
  return output;
}

function adaptCategoryScores(categoryScores) {
  const scores = categoryScores && typeof categoryScores === "object" ? { ...categoryScores } : {};
  // Results persisted before the display-category rename used
  // `role_alignment`. Keep those candidate pages useful without requiring a
  // database reset or a second upload.
  if (!scores.job_title_match && scores.role_alignment) {
    scores.job_title_match = scores.role_alignment;
  }
  if (!scores.soft_skills && scores.domain_alignment) {
    scores.soft_skills = scores.domain_alignment;
  }
  return scores;
}

export function adaptResultRecord(record) {
  if (!record || typeof record !== "object") return null;
  const clean = Object.fromEntries(
    Object.entries(record).filter(([key]) => !INTERNAL_FIELDS.has(key))
  );
  const categoryScores = adaptCategoryScores(clean.category_scores);
  const evidence = adaptEvidence(clean.evidence, clean.source_id);
  if (!evidence.job_title && categoryScores.job_title_match) {
    const score = Number(categoryScores.job_title_match.score || 0);
    const title = clean.current_role_title_from_summary
      || clean.experience?.find((item) => item?.title)?.title
      || "No current title extracted";
    evidence.job_title = {
      status: score >= 9.9 ? "matched" : score > 0 ? "related" : "missing",
      match_level: score >= 9.9 ? "direct" : score > 0 ? "related" : "none",
      best_match: { title },
      judge_reason: categoryScores.job_title_match.notes || "Job-title evidence reviewed.",
    };
  }
  return {
    ...clean,
    source_id: typeof clean.source_id === "string" ? clean.source_id : null,
    skills: strings(clean.skills),
    category_scores: categoryScores,
    evidence,
  };
}

export function adaptResultsResponse(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Results response must be an object.");
  }
  return {
    role_id: typeof payload.role_id === "string" ? payload.role_id : "",
    ranked: (Array.isArray(payload.ranked) ? payload.ranked : []).map(adaptResultRecord).filter(Boolean),
    excluded_hard_gate_failed: (Array.isArray(payload.excluded_hard_gate_failed) ? payload.excluded_hard_gate_failed : [])
      .map(adaptResultRecord)
      .filter(Boolean),
  };
}

/**
 * Produce the only source payload the browser-local reviewer needs: retrieved
 * evidence quotes keyed by immutable source ID. Full source Markdown never
 * enters the results cache or local-review prompt.
 */
export function reviewSourcesForResult(record) {
  const byId = new Map();
  for (const rows of Object.values(record?.evidence || {})) {
    if (!Array.isArray(rows)) continue;
    for (const row of rows) {
      for (const citation of row?.citations || []) {
        if (!citation.sourceId || !citation.quote) continue;
        const current = byId.get(citation.sourceId) || [];
        if (!current.includes(citation.quote)) current.push(citation.quote);
        byId.set(citation.sourceId, current);
      }
    }
  }
  return [...byId].map(([source_id, quotes]) => ({ source_id, markdown: quotes.join("\n\n") }));
}
