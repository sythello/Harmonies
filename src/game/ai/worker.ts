import { searchTurn } from "./search";
import type { AIConfig, AIObservation } from "./types";

self.onmessage = (
  event: MessageEvent<{
    view: AIObservation;
    config: AIConfig;
    planId?: string;
  }>,
) => {
  try {
    self.postMessage({
      result: searchTurn(event.data.view, event.data.config, {
        planId: event.data.planId,
      }),
    });
  } catch (error) {
    self.postMessage({
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
