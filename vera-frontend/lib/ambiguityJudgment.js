/**
 * Browser-side contract for one WebLLM ambiguity pass per candidate.
 * The backend remains the authority for final scoring; this module only
 * resolves the narrow band of cross-encoder matches that require judgment.
 */

const MAX_EVIDENCE_CHARACTERS = 3_000;

export const AMBIGUITY_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    judgments: {
      type: "array",
      items: {
        type: "object",
        properties: {
          requirement_id: { type: "string" },
          evidence_id: { type: "string" },
          decision: { type: "string", enum: ["matched", "weak", "missing"] },
          confidence: { type: "number", minimum: 0, maximum: 1 },
          reason: { type: "string" },
        },
        required: ["requirement_id", "evidence_id", "decision", "confidence", "reason"],
        additionalProperties: false,
      },
    },
  },
  required: ["judgments"],
  additionalProperties: false,
};

function assertBatch(batch) {
  if (!batch || typeof batch !== "object" || typeof batch.candidate_id !== "string" || !Array.isArray(batch.items)) {
    throw new Error("Invalid WebLLM ambiguity batch.");
  }
  for (const item of batch.items) {
    if (!item || typeof item.requirement_id !== "string" || typeof item.evidence_id !== "string") {
      throw new Error("Each ambiguity item needs a requirement and evidence ID.");
    }
  }
}

export function buildAmbiguityJudgmentPrompt(batch) {
  assertBatch(batch);
  const items = batch.items.map(({ requirement_id, evidence_id, requirement_text, evidence_text, confidence }) => ({
    requirement_id,
    evidence_id,
    requirement: String(requirement_text || "").trim(),
    evidence: String(evidence_text || "").trim().slice(0, MAX_EVIDENCE_CHARACTERS),
    cross_encoder_confidence: confidence,
  }));
  return [
    "You are resolving ambiguous resume-to-job-requirement matches for one candidate.",
    "Use only the evidence supplied for each item. A close tool, workflow, or outcome may be a weak match; do not invent experience.",
    "Return exactly one judgment for every item. Preserve requirement_id and evidence_id exactly.",
    "Choose matched for clearly supported, weak for plausible but indirect support, or missing for unsupported.",
    "Return JSON only: { judgments: [{ requirement_id, evidence_id, decision, confidence, reason }] }.",
    `Candidate ID: ${batch.candidate_id}`,
    `Items: ${JSON.stringify(items)}`,
  ].join("\n\n");
}

export function validateAmbiguityJudgments(batch, response) {
  assertBatch(batch);
  if (!response || typeof response !== "object" || !Array.isArray(response.judgments)) {
    throw new Error("WebLLM returned an invalid ambiguity response.");
  }
  const allowed = new Set(batch.items.map((item) => `${item.requirement_id}::${item.evidence_id}`));
  if (response.judgments.length !== allowed.size) {
    throw new Error("WebLLM must return one judgment for every ambiguous item.");
  }
  const seen = new Set();
  return response.judgments.map((judgment) => {
    const expectedKeys = ["requirement_id", "evidence_id", "decision", "confidence", "reason"];
    if (!judgment || typeof judgment !== "object" || Object.keys(judgment).length !== expectedKeys.length || !expectedKeys.every((key) => key in judgment)) {
      throw new Error("WebLLM returned a judgment with an invalid shape.");
    }
    const key = `${judgment.requirement_id}::${judgment.evidence_id}`;
    if (!allowed.has(key) || seen.has(key)) throw new Error("WebLLM referenced unknown or duplicate evidence.");
    if (!['matched', 'weak', 'missing'].includes(judgment.decision)) throw new Error("WebLLM returned an invalid decision.");
    if (!Number.isFinite(judgment.confidence) || judgment.confidence < 0 || judgment.confidence > 1) {
      throw new Error("WebLLM confidence must be between 0 and 1.");
    }
    if (typeof judgment.reason !== "string" || !judgment.reason.trim()) throw new Error("WebLLM judgment needs a reason.");
    seen.add(key);
    return { ...judgment, confidence: Number(judgment.confidence) };
  });
}
