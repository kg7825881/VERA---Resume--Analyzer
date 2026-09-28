const MAX_SOURCE_CHARS = 12_000;
const MAX_TOTAL_SOURCE_CHARS = 30_000;

export const REVIEW_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    summary: { type: "string" },
    recommendation: { type: "string", enum: ["advance", "review", "decline"] },
    citations: {
      type: "array",
      items: {
        type: "object",
        properties: {
          source_id: { type: "string" },
          quote: { type: "string" },
        },
        required: ["source_id", "quote"],
      },
    },
  },
  required: ["summary", "recommendation", "citations"],
};

function boundedSources(sources) {
  if (!Array.isArray(sources)) return [];
  let remaining = MAX_TOTAL_SOURCE_CHARS;
  return sources.flatMap((source) => {
    if (!source || typeof source.source_id !== "string" || typeof source.markdown !== "string") {
      return [];
    }
    const markdown = source.markdown.slice(0, Math.min(MAX_SOURCE_CHARS, remaining));
    remaining -= markdown.length;
    return markdown ? [{ source_id: source.source_id, markdown }] : [];
  });
}

/**
 * Conservative no-model fallback. It intentionally makes no claim about the
 * candidate beyond the already-computed score and always asks for human
 * review, so it never needs to invent a source citation.
 */
export function buildFallbackAssessmentReview(assessment) {
  const decision = assessment?.assessment || assessment || {};
  const score = Number.isFinite(decision.score) ? `${decision.score}/100` : "unavailable";
  const eligibility = decision.eligible === false ? " The mandatory gate was not met." : "";
  return {
    summary: `Local model review is unavailable. Deterministic Job Fit score: ${score}.${eligibility} Please review the scored evidence directly.`,
    recommendation: "review",
    citations: [],
  };
}

/**
 * Build an intentionally narrow, source-grounded prompt for browser-local
 * review. The scoring result is context, not evidence: any factual statement
 * about the candidate must cite an exact quote from a supplied source.
 */
export function buildAssessmentReviewPrompt({ assessment, sources }) {
  const selectedSources = boundedSources(sources);
  return [
    "You are VERA's local hiring-review assistant.",
    "Use only the supplied assessment and source excerpts. Do not infer unstated skills, experience, education, or identity details.",
    "Return JSON only, with exactly: summary, recommendation, citations.",
    "recommendation must be one of advance, review, decline.",
    "Each citation must contain a supplied source_id and an exact, short quote copied from that source. Cite every candidate-specific claim. If no source supports a claim, omit it and choose review.",
    "Never reproduce an entire resume, contact details, or more than 240 characters from any one source.",
    "\nASSESSMENT:\n" + JSON.stringify(assessment),
    "\nSOURCES:\n" + selectedSources.map((source) => (
      `[source_id: ${source.source_id}]\n${source.markdown}`
    )).join("\n\n"),
  ].join("\n");
}

/** Reject model output that is malformed or cites text not in the source bundle. */
export function validateAssessmentReview(response, sources) {
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    throw new Error("Local review must be a JSON object.");
  }
  const keys = Object.keys(response).sort().join(",");
  if (keys !== "citations,recommendation,summary") {
    throw new Error("Local review has an unexpected response shape.");
  }
  if (typeof response.summary !== "string" || response.summary.trim() === "") {
    throw new Error("Local review requires a non-empty summary.");
  }
  if (!["advance", "review", "decline"].includes(response.recommendation)) {
    throw new Error("Local review has an invalid recommendation.");
  }
  if (!Array.isArray(response.citations)) {
    throw new Error("Local review citations must be an array.");
  }

  const sourceText = new Map(boundedSources(sources).map((source) => [source.source_id, source.markdown]));
  for (const citation of response.citations) {
    if (!citation || typeof citation !== "object" || Object.keys(citation).sort().join(",") !== "quote,source_id") {
      throw new Error("Local review has an invalid citation.");
    }
    if (typeof citation.source_id !== "string" || typeof citation.quote !== "string" || !citation.quote.trim()) {
      throw new Error("Local review citation values must be non-empty strings.");
    }
    if (citation.quote.length > 240 || !sourceText.get(citation.source_id)?.includes(citation.quote)) {
      throw new Error("Local review citation is not grounded in its source.");
    }
  }
  return response;
}
