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
  terrainGain?: number;
  animalGain?: number;
  spiritGain?: number;
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
  L?: number;
  T?: number;
  canonicalRotation?: number;
}
export interface CandidateLog {
  scoreKind?: "order" | "full";
  action: Action;
  label: string;
  value: number;
  gain?: number;
  evaluation: Evaluation;
}
export interface AIDecisionLog {
  evaluationVersion?: string;
  planId?: string;
  decisionIndex?: number;
  planSummary?: PlanSummary;
  actualScoreAfter?: Score;
  scoreGain?: number;
  potentialGain?: number;
  usesSampledRefill?: boolean;
  sampleSeed?: number;
  position: AIObservation;
  round: number;
  playerId: number;
  playerName: string;
  gameLogIndex: number;
  config: AIConfig;
  before?: Evaluation;
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
export interface PlanCandidate {
  actions: Action[];
  labels?: string[];
  value: number;
  scenarioValues: number[];
  terminal: boolean;
  evaluation?: Evaluation;
}
export interface PlanSummary {
  rootValue?: number;
  value: number;
  gain?: number;
  targetDepth: number;
  completedDepth: number;
  elapsedMs: number;
  expandedActions: number;
  evaluations: number;
  stopReason: "completed" | "time" | "evaluations" | "stepLimit";
  topPlans: PlanCandidate[];
  searchStats?: SearchStats;
}
export interface SearchStats {
  generatedActions: number;
  uniqueStates: number;
  duplicateStates: number;
  orderEvaluations: number;
  fullEvaluations: number;
  completePlans: number;
  widthsCompleted: number[];
  widthAttempted: number;
  logEvaluations: number;
  searchMs: number;
  logMs: number;
}
export interface TurnPlan {
  actions: Action[];
  decisions: AIDecisionLog[];
  completedDepth: number;
  elapsedMs: number;
  nodes: number;
  timedOut: boolean;
  evaluations?: number;
}
