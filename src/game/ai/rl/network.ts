import {
  ACTION_SIZE,
  CARD_INPUTS,
  FEATURE_SIZE,
  VERSION,
  encode,
  type Features,
} from "./encoding";
import type { AIObservation } from "../types";
import { spatialSchema, SPATIAL_SIZE } from "./spatial";

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
  /** Data schema stays v2; absent architecture denotes a legacy FFN. */
  architecture?: "ffn-v2" | "spatial-v3";
  embedding: number[][];
  trunk: Dense[];
  heads: { policy: Dense; rank: Dense; relative: Dense; absolute: Dense };
  spatial?: { version: string; filters: Record<string, Dense> };
  towers?: { policy: Dense; rank: Dense; relative: Dense; absolute: Dense };
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
/** Skip the zeros in cell one-hots without materializing 230 dense patches. */
export function spatialFeatures(
  numeric: number[],
  filters: Record<string, Dense>,
): number[] {
  const { cells, cellTypes, kernels, playerOffsets } = spatialSchema;
  const boards = playerOffsets.map((start) => {
    const board = Array.from({ length: cells }, (_, i) => {
      const active: [number, number][] = [];
      for (let c = 0; c < cellTypes; c++) {
        const value = numeric[start + i * cellTypes + c];
        if (value !== 0) active.push([c, value]);
      }
      return active;
    });
    board.push([[cellTypes, 1]]);
    return board;
  });
  const output: number[] = [];
  // Match PyTorch flatten order and dense summation: kernel, player, anchor, channel.
  for (const kernel of kernels)
    for (const board of boards)
      for (const patch of kernel.indices) {
        const indices: number[] = [],
          values: number[] = [];
        patch.forEach((cell, slot) => {
          for (const [category, value] of board[cell]) {
            indices.push(slot * (cellTypes + 1) + category);
            values.push(value);
          }
        });
        const layer = filters[kernel.name];
        for (let channel = 0; channel < layer.bias.length; channel++) {
          let value = layer.bias[channel];
          for (let i = 0; i < indices.length; i++)
            value += layer.weight[channel][indices[i]] * values[i];
          output.push(Math.max(0, value));
        }
      }
  return output;
}
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
  const spatial = weights.architecture === "spatial-v3";
  if (
    weights.architecture !== undefined &&
    !["ffn-v2", "spatial-v3"].includes(weights.architecture)
  )
    throw new Error("Unsupported network architecture");
  if (
    spatial !== Boolean(weights.spatial) ||
    spatial !== Boolean(weights.towers)
  )
    throw new Error("Spatial architecture requires filters and target towers");
  let size = FEATURE_SIZE + CARD_INPUTS * dim + (spatial ? SPATIAL_SIZE : 0);
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
  if (spatial) {
    if (
      weights.spatial!.version !== spatialSchema.version ||
      Object.keys(weights.spatial!.filters).join() !==
        spatialSchema.kernels.map((k) => k.name).join()
    )
      throw new Error("Incompatible spatial kernel schema");
    for (const k of spatialSchema.kernels)
      check(
        weights.spatial!.filters[k.name],
        k.offsets.length * (spatialSchema.cellTypes + 1),
        spatialSchema.channels,
      );
  }
  for (const layer of weights.trunk) {
    if (!layer.bias?.length) throw new Error("Empty trunk layer");
    check(layer, size, layer.bias.length);
    size = layer.bias.length;
  }
  for (const head of ["policy", "rank", "relative", "absolute"] as const) {
    if (spatial) check(weights.towers![head], size, 64);
    check(
      weights.heads?.[head],
      spatial ? 64 : size,
      head === "policy" ? ACTION_SIZE : 2,
    );
  }
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
    if (this.weights.spatial)
      x.push(
        ...spatialFeatures(features.numeric, this.weights.spatial.filters),
      );
    for (const layer of this.weights.trunk)
      x = dense(x, layer).map((v) => Math.max(0, v));
    const head = (name: keyof Weights["heads"]) =>
      dense(
        this.weights.towers
          ? dense(x, this.weights.towers[name]).map((v) => Math.max(0, v))
          : x,
        this.weights.heads[name],
      );
    return {
      logits: head("policy"),
      value: {
        rank: head("rank").map(Math.tanh) as Pair,
        relative: head("relative") as Pair,
        absolute: head("absolute") as Pair,
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
