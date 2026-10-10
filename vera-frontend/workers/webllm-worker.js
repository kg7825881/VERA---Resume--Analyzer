// Bundled as a dedicated browser worker by the client module. It never runs in
// Next's server runtime, so WebGPU inference cannot block page rendering.
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (event) => handler.onmessage(event);

// Surface worker-side failures in the browser console instead of leaving the
// page with only a generic failed-state message.
self.addEventListener("error", (event) => {
  console.error("WebLLM worker crashed:", event.message || event.error || event);
});
