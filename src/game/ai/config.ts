import type { AIConfig } from "./types";

export const EVALUATION_VERSION = "local-projects-v1";
export const MAX_PROJECTS = 6;
export const INITIAL_COUNTS = [23, 23, 21, 19, 19, 15];
export const DEFAULT_AI: AIConfig = {
  depth: 2,
  timeMs: 3500,
  beamWidth: 6,
  topK: 3,
  seed: 1729,
};
export function normalizeAI(config: Partial<AIConfig> = {}): AIConfig {
  const integer = (
    value: number | undefined,
    fallback: number,
    min: number,
    max: number,
  ) =>
    Number.isFinite(value)
      ? Math.max(min, Math.min(max, Math.floor(value!)))
      : fallback;
  return {
    depth: integer(config.depth, DEFAULT_AI.depth, 1, 3),
    timeMs: integer(config.timeMs, DEFAULT_AI.timeMs, 100, 30000),
    beamWidth: integer(config.beamWidth, DEFAULT_AI.beamWidth, 1, 32),
    topK: integer(config.topK, DEFAULT_AI.topK, 1, 10),
    seed: integer(config.seed, DEFAULT_AI.seed, 0, 2147483647),
  };
}
