import { Network, type Weights } from "./network";
import { search, type SearchConfig } from "./mcts";
import type { AIObservation } from "../types";

let network: Network | undefined;
/** Initialize once with exported JSON; subsequent requests only send public views. */
self.onmessage = (
  event: MessageEvent<{
    id?: string;
    weights?: Weights;
    view?: AIObservation;
    config?: Partial<SearchConfig>;
  }>,
) => {
  const { id, weights, view, config } = event.data;
  try {
    if (weights) network = new Network(weights);
    if (!network)
      throw new Error("Initialize the RL worker with model weights first");
    self.postMessage({
      id,
      ...(view ? { result: search(view, network, config) } : { ready: true }),
    });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
