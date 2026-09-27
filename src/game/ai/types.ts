import type { Action, Color, Game, Score } from "../engine";

export interface AIConfig {
  depth: number;
  timeMs: number;
  beamWidth: number;
  topK: number;
  seed: number;
}
/** Vectors use indices 0..5 for colors 1..6. */
export type Counts = number[];
export interface AIObservation extends Omit<Game, "bag" | "deck" | "log"> {
  bagCount: number;
  deckCount: number;
  bagCounts: Counts;
  unseenAnimals: number[];
  needsSpiritChoice: number[];
}
export interface Project {
  kind: "animal" | "terrain";
  cardId?: number;
  targetCell?: number;
  additions: { cell: number; colors: Color[] }[];
  key: number[];
}
export interface ProjectEstimate {
  project: Project;
  G: number;
  m: number;
  mByColor: Counts;
  d: Counts;
  p: Counts;
  H: Counts;
  A: Counts;
  B: number;
  F: number;
  q: number;
  W: number;
}
export interface Evaluation {
  actualScore: Score;
  value: number;
  potential: number;
  R: number;
  B: number;
  F: number;
  H: Counts;
  A: Counts;
  p: Counts;
  supplyGroup?: number;
  projects: ProjectEstimate[];
}
export interface CandidateLog {
  action: Action;
  label: string;
  value: number;
  gain: number;
  evaluation: Evaluation;
}
export interface AIDecisionLog {
  position: AIObservation;
  round: number;
  playerId: number;
  playerName: string;
  gameLogIndex: number;
  config: AIConfig;
  before: Evaluation;
  selected: CandidateLog;
  topByOperation: Partial<Record<Action["type"], CandidateLog[]>>;
  legalCounts: Partial<Record<Action["type"], number>>;
  evaluatedCounts: Partial<Record<Action["type"], number>>;
  targetDepth: number;
  completedDepth: number;
  searchValue: number;
  elapsedMs: number;
  nodes: number;
  timedOut: boolean;
  fallback: boolean;
}
export interface TurnPlan {
  actions: Action[];
  decisions: AIDecisionLog[];
  completedDepth: number;
  elapsedMs: number;
  nodes: number;
  timedOut: boolean;
}
