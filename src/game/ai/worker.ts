import { searchTurn } from "./search";
import type { AIConfig, AIObservation } from "./types";

self.onmessage = (
  event: MessageEvent<{ view: AIObservation; config: AIConfig }>,
) => {
  try {
    self.postMessage({
      result: searchTurn(event.data.view, event.data.config),
    });
  } catch (error) {
    self.postMessage({
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
