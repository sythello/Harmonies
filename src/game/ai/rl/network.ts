import {
  ACTION_SIZE,
  CARD_INPUTS,
  FEATURE_SIZE,
  VERSION,
  encode,
  type Features,
} from "./encoding";
import type { AIObservation } from "../types";

export type Pair = [number, number];
/** Each pair is [active player, opponent]; score heads are in units of 100 points. */
export interface Values {
  rank: Pair;
  relative: Pair;
  absolute: Pair;
}
export interface Prediction {
  logits: number[];
  value: Values;
}
export interface Evaluator {
  predict(view: AIObservation): Prediction;
}
export interface Dense {
  weight: number[][];
  bias: number[];
}
export interface Weights {
  version: typeof VERSION;
  embedding: number[][];
  trunk: Dense[];
  heads: { policy: Dense; rank: Dense; relative: Dense; absolute: Dense };
}
const dense = (x: number[], layer: Dense) => {
  // Board one-hots and ReLU outputs are sparse. Keep the original summation
  // order while skipping zero products; the large input layer benefits most.
  const nonzero: number[] = [];
  for (let j = 0; j < x.length; j++) if (x[j] !== 0) nonzero.push(j);
  return layer.weight.map((row, i) => {
    let value = layer.bias[i];
    for (let k = 0; k < nonzero.length; k++) {
      const j = nonzero[k];
      value += row[j] * x[j];
    }
    return value;
  });
};
function validateMatrix(matrix: number[][], rows: number, columns: number) {
  if (
    !Array.isArray(matrix) ||
    matrix.length !== rows ||
    matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== columns ||
        row.some((x) => !Number.isFinite(x)),
    )
  )
    throw new Error("Invalid model matrix shape or nonfinite weights");
}
export function validateWeights(weights: Weights) {
  if (
    weights.version !== VERSION ||
    !weights.embedding?.[0]?.length ||
    !weights.trunk?.length
  )
    throw new Error(`Incompatible RL model; expected ${VERSION}`);
  const dim = weights.embedding[0].length;
  validateMatrix(weights.embedding, 33, dim);
  if (weights.embedding[0].some((x) => x !== 0))
    throw new Error("Padding embedding must be zero");
  let size = FEATURE_SIZE + CARD_INPUTS * dim;
  const check = (layer: Dense, input: number, output: number) => {
    if (
      !layer ||
      !Array.isArray(layer.bias) ||
      layer.bias.length !== output ||
      layer.bias.some((x) => !Number.isFinite(x))
    )
      throw new Error("Invalid model bias");
    validateMatrix(layer.weight, output, input);
  };
  for (const layer of weights.trunk) {
    if (!layer.bias?.length) throw new Error("Empty trunk layer");
    check(layer, size, layer.bias.length);
    size = layer.bias.length;
  }
  for (const head of ["policy", "rank", "relative", "absolute"] as const)
    check(weights.heads?.[head], size, head === "policy" ? ACTION_SIZE : 2);
}
export class Network implements Evaluator {
  constructor(readonly weights: Weights) {
    validateWeights(weights);
  }
  forward(features: Features): Prediction {
    let x = [
      ...features.numeric,
      ...features.cards.flatMap((id) => this.weights.embedding[id]),
    ];
    for (const layer of this.weights.trunk)
      x = dense(x, layer).map((v) => Math.max(0, v));
    return {
      logits: dense(x, this.weights.heads.policy),
      value: {
        rank: dense(x, this.weights.heads.rank).map(Math.tanh) as Pair,
        relative: dense(x, this.weights.heads.relative) as Pair,
        absolute: dense(x, this.weights.heads.absolute) as Pair,
      },
    };
  }
  predict(view: AIObservation) {
    return this.forward(encode(view));
  }
}

/** Explicit bootstrap baseline, never silently substituted for a missing model. */
export const uniformEvaluator: Evaluator = {
  predict: () => ({
    logits: Array(ACTION_SIZE).fill(0),
    value: { rank: [0, 0], relative: [0, 0], absolute: [0, 0] },
  }),
};
