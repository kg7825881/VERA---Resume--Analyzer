import {
  buildAssessmentReviewPrompt,
  buildFallbackAssessmentReview,
  REVIEW_RESPONSE_SCHEMA,
  validateAssessmentReview,
} from "./assessmentReview";
import {
  AMBIGUITY_RESPONSE_SCHEMA,
  buildAmbiguityJudgmentPrompt,
  validateAmbiguityJudgments,
} from "./ambiguityJudgment";

export const DEFAULT_WEBLLM_MODEL = "Llama-3.2-1B-Instruct-q4f32_1-MLC";

// WebLLM 0.2.x compiles JSON mode through its grammar matcher. Supplying only
// `{ type: "json_object" }` leaves its schema undefined and causes the wasm
// grammar binding to reject a non-string value. Serialize the schemas once so
// every local request uses the supported API shape.
const AMBIGUITY_JSON_RESPONSE_FORMAT = {
  type: "json_object",
  schema: JSON.stringify(AMBIGUITY_RESPONSE_SCHEMA),
};
const REVIEW_JSON_RESPONSE_FORMAT = {
  type: "json_object",
  schema: JSON.stringify(REVIEW_RESPONSE_SCHEMA),
};

let modulePromise;
let enginePromise;
let engineWorker;
let activeModel;

/** Convert worker-thrown values into a useful, safe diagnostic for the UI. */
export function describeWebLLMError(error) {
  if (error instanceof Error && error.message) return error.message;
  if (error && typeof error === "object" && typeof error.message === "string") return error.message;
  if (typeof error === "string" && error.trim()) return error;
  try {
    const serialized = JSON.stringify(error);
    if (serialized && serialized !== "{}") return serialized;
  } catch {
    // Fall through to the stable fallback below.
  }
  return "Local WebLLM ambiguity matching failed.";
}

export class LocalReviewUnavailableError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = "LocalReviewUnavailableError";
    this.cause = cause;
  }
}

function nowMs() {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

function ambiguityMaxTokens(batch) {
  // A judgment repeats opaque IDs and a reason. Leave enough room that a
  // longer batch is not cut off while the JSON array is still open.
  return Math.max(500, (batch?.items?.length || 0) * 180);
}

function parseAmbiguityResponse(batch, content) {
  if (typeof content !== "string") throw new Error("Local model returned no ambiguity judgment.");
  return validateAmbiguityJudgments(batch, JSON.parse(content));
}

async function requestAmbiguityJudgments(engine, batch) {
  const request = (prompt) => engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 0,
    max_tokens: ambiguityMaxTokens(batch),
    response_format: AMBIGUITY_JSON_RESPONSE_FORMAT,
  });

  const first = await request(buildAmbiguityJudgmentPrompt(batch));
  try {
    return parseAmbiguityResponse(batch, first.choices?.[0]?.message?.content);
  } catch (error) {
    // Do not try to repair or infer a hiring decision from malformed output.
    // Instead, request one clean, schema-constrained answer from the same
    // browser-local model.
    if (!(error instanceof SyntaxError)) throw error;
    console.warn("WebLLM returned malformed ambiguity JSON; retrying once.", error);
  }

  const retryPrompt = [
    "Your previous response was invalid JSON.",
    "Return only one valid JSON object: no markdown, explanation, or comments.",
    "Use exactly: {\"judgments\":[{\"requirement_id\":\"...\",\"evidence_id\":\"...\",\"decision\":\"matched|weak|missing\",\"confidence\":0.0,\"reason\":\"...\"}]}",
    buildAmbiguityJudgmentPrompt(batch),
  ].join("\n\n");
  const retry = await request(retryPrompt);
  return parseAmbiguityResponse(batch, retry.choices?.[0]?.message?.content);
}

function loadWebLLM() {
  // WebLLM is intentionally loaded only after a user asks for a local review;
  // importing it on the server or initial page render would break SSR and add
  // unnecessary download/initialization cost.
  modulePromise ||= import("@mlc-ai/web-llm");
  return modulePromise;
}

export function webGpuAvailable() {
  return typeof window !== "undefined" && "gpu" in navigator;
}

/** A lightweight capability check suitable for an explicit UI status message. */
export function getWebLLMAvailability() {
  if (typeof window === "undefined") return { available: false, reason: "WebLLM runs only in the browser." };
  if (!webGpuAvailable()) return { available: false, reason: "This browser does not expose WebGPU." };
  return { available: true, reason: null };
}

function disposeEngine() {
  engineWorker?.terminate();
  engineWorker = undefined;
  enginePromise = undefined;
  activeModel = undefined;
}

function createEngineWorker() {
  const worker = new Worker(
    new URL("../workers/webllm-worker.js", import.meta.url),
    { type: "module" }
  );
  // Errors from a module worker are often delivered as plain event objects,
  // not Error instances. Keep their details visible during local debugging.
  worker.addEventListener("error", (event) => {
    console.error("WebLLM worker error:", event.message || event.error || event);
  });
  worker.addEventListener("messageerror", (event) => {
    console.error("WebLLM worker message error:", event.data || event);
  });
  return worker;
}

function hasIncompleteModelCache(error) {
  const detail = describeWebLLMError(error);
  return detail.includes("Tensor-cache record range") && detail.includes("shard size 0");
}

export async function getWebLLMEngine({ model = DEFAULT_WEBLLM_MODEL, onProgress } = {}) {
  if (!webGpuAvailable()) {
    throw new LocalReviewUnavailableError("WebGPU is required for local browser review.");
  }
  // One worker/model is kept warm. A requested model change must create a new
  // engine; otherwise WebLLM would silently keep using the previous model.
  if (enginePromise && activeModel !== model) disposeEngine();
  activeModel ||= model;
  enginePromise ||= (async () => {
    const { CreateWebWorkerMLCEngine, deleteModelAllInfoInCache } = await loadWebLLM();
    const initialize = async () => {
      engineWorker = createEngineWorker();
      return CreateWebWorkerMLCEngine(engineWorker, model, { initProgressCallback: onProgress });
    };
    try {
      return await initialize();
    } catch (error) {
      // Interrupted model downloads can leave a zero-byte shard in Cache
      // Storage. WebLLM then cannot recover by itself; delete just this
      // model's local artifacts and retry once with a fresh worker.
      if (hasIncompleteModelCache(error)) {
        console.warn(`WebLLM cache for ${model} is incomplete; clearing it and retrying once.`, error);
        engineWorker?.terminate();
        engineWorker = undefined;
        onProgress?.({ text: "Repairing incomplete local model download…" });
        try {
          await deleteModelAllInfoInCache(model);
          onProgress?.({ text: "Re-downloading local browser model…" });
          return await initialize();
        } catch (retryError) {
          disposeEngine();
          const retryDetail = describeWebLLMError(retryError);
          console.error(`WebLLM retry failed for ${model}:`, retryError);
          throw new LocalReviewUnavailableError(`The local model could not be initialized after repairing its cache: ${retryDetail}`, retryError);
        }
      }
      disposeEngine();
      const detail = describeWebLLMError(error);
      console.error(`WebLLM initialization failed for ${model}:`, error);
      throw new LocalReviewUnavailableError(`The local model could not be initialized: ${detail}`, error);
    }
  })();
  return enginePromise;
}

/** Generate and source-validate a browser-local hiring review. */
export async function generateLocalAssessmentReview({ assessment, sources, model, onProgress }) {
  const engine = await getWebLLMEngine({ model, onProgress });
  const completion = await engine.chat.completions.create({
    messages: [{ role: "user", content: buildAssessmentReviewPrompt({ assessment, sources }) }],
    temperature: 0,
    max_tokens: 700,
    // JSON mode makes parsing deterministic; validateAssessmentReview applies
    // source-grounding checks after generation.
    response_format: REVIEW_JSON_RESPONSE_FORMAT,
  });
  const content = completion.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    throw new Error("Local model returned no review content.");
  }
  let parsed;
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error("Local model returned invalid JSON.");
  }
  const review = validateAssessmentReview(parsed, sources);
  return review;
}

/** Resolve only the cross-encoder's uncertain matches for one candidate. */
export async function generateAmbiguityJudgments({ batch, model, onProgress }) {
  const engine = await getWebLLMEngine({ model, onProgress });
  try {
    return await requestAmbiguityJudgments(engine, batch);
  } catch (error) {
    if (error instanceof SyntaxError) throw new Error("Local model returned invalid ambiguity JSON.");
    throw error;
  }
}

/**
 * Execute exactly one candidate ambiguity batch locally. Unlike the existing
 * review helper, this never fabricates a deterministic substitute: unresolved
 * matches remain unresolved when WebGPU or the model is unavailable.
 */
export async function generateAmbiguityJudgmentsWithStatus({ batch, model, onProgress }) {
  const startedAt = nowMs();
  const availability = getWebLLMAvailability();
  if (!availability.available) {
    return { mode: "unavailable", judgments: [], reason: availability.reason, performance: { initialization_ms: 0, generation_ms: 0 } };
  }
  try {
    const engine = await getWebLLMEngine({ model, onProgress });
    const initializedAt = nowMs();
    return {
      mode: "webllm",
      judgments: await requestAmbiguityJudgments(engine, batch),
      performance: {
        initialization_ms: Math.round(initializedAt - startedAt),
        generation_ms: Math.round(nowMs() - initializedAt),
      },
    };
  } catch (error) {
    const reason = describeWebLLMError(error);
    console.error("WebLLM ambiguity matching failed:", error);
    return {
      mode: "unavailable",
      judgments: [],
      reason,
      performance: { initialization_ms: Math.round(nowMs() - startedAt), generation_ms: 0 },
    };
  }
}

export function unloadWebLLMModel() {
  disposeEngine();
}

/**
 * Preferred UI entry point: emits measured timing data and safely falls back
 * when WebGPU, model download, worker initialization, or generation fails.
 */
export async function generateAssessmentReviewWithFallback({ assessment, sources, model, onProgress }) {
  const startedAt = nowMs();
  try {
    const engine = await getWebLLMEngine({ model, onProgress });
    const initializedAt = nowMs();
    const completion = await engine.chat.completions.create({
      messages: [{ role: "user", content: buildAssessmentReviewPrompt({ assessment, sources }) }],
      temperature: 0,
      max_tokens: 700,
      response_format: REVIEW_JSON_RESPONSE_FORMAT,
    });
    const content = completion.choices?.[0]?.message?.content;
    if (typeof content !== "string") throw new Error("Local model returned no review content.");
    const review = validateAssessmentReview(JSON.parse(content), sources);
    return {
      review,
      mode: "webllm",
      performance: {
        initialization_ms: Math.round(initializedAt - startedAt),
        generation_ms: Math.round(nowMs() - initializedAt),
        source_characters: (sources || []).reduce((total, source) => total + (source?.markdown?.length || 0), 0),
      },
    };
  } catch (error) {
    const review = validateAssessmentReview(buildFallbackAssessmentReview(assessment), sources);
    return {
      review,
      mode: "deterministic_fallback",
      fallback_reason: error instanceof Error ? error.message : "Local review failed.",
      performance: {
        initialization_ms: Math.round(nowMs() - startedAt),
        generation_ms: 0,
        source_characters: (sources || []).reduce((total, source) => total + (source?.markdown?.length || 0), 0),
      },
    };
  }
}
