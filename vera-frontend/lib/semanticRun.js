import { generateAmbiguityJudgmentsWithStatus } from "./webllm";
import { submitSemanticJudgments } from "./api";

/**
 * Resolve an issued candidate session. It deliberately never sends a local
 * result for a session that did not ask for WebLLM, and never substitutes an
 * unavailable browser model with another decision source.
 */
export async function resolveSemanticSession(session, { model, onProgress } = {}) {
  if (session.status === "completed") return session;
  if (session.status !== "pending_webllm" || !session.ambiguity_batch) {
    throw new Error("Semantic session is not ready for browser-local resolution.");
  }
  const local = await generateAmbiguityJudgmentsWithStatus({
    batch: session.ambiguity_batch,
    model,
    onProgress,
  });
  if (local.mode !== "webllm") return { ...session, local_resolution: local };
  const completed = await submitSemanticJudgments(session.session_id, local.judgments);
  return { ...completed, local_resolution: local };
}
