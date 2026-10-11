import { Network, type Weights } from "./network";
import { SearchTree, type SearchConfig } from "./mcts";
import type { AIObservation } from "../types";

let tree: SearchTree | undefined;
/** Initialize once with exported JSON; subsequent requests only send public views. */
self.onmessage = (
  event: MessageEvent<{
    id?: string;
    weights?: Weights;
    view?: AIObservation;
    config?: Partial<SearchConfig>;
    reset?: boolean;
  }>,
) => {
  const { id, weights, view, config, reset } = event.data;
  try {
    if (weights) {
      tree = undefined;
      tree = new SearchTree(new Network(weights));
    }
    if (!tree)
      throw new Error("Initialize the RL worker with model weights first");
    if (reset) tree.reset();
    self.postMessage({
      id,
      ...(view ? { result: tree.search(view, config) } : { ready: true }),
    });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
