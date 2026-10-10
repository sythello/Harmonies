import { applyAction, type Action, type Game } from "../engine";
import { EVALUATION_VERSION, normalizeAI } from "./config";
import { createEvaluator } from "./evaluate";
import { observe, sampleGame } from "./observation";
import {
  actionLabel,
  compareActions,
  freeCardSlots,
  legalActions,
  positionKey,
} from "./actions";
import type {
  AIConfig,
  AIDecisionLog,
  AIObservation,
  CandidateLog,
  Evaluation,
  PlanSummary,
  SearchStats,
  TurnPlan,
} from "./types";

interface State {
  freeCardSlots: number;
  game: Game;
  key: string;
  order?: Evaluation;
  full?: Evaluation;
  edges?: Edge[];
}
interface Edge {
  action: Action;
  state: State;
}
interface Path {
  state: State;
  actions: Action[];
}
export interface SearchOptions {
  /** Total uncached order + full evaluations. Mandatory fallback is exempt. */
  maxNodes?: number;
  now?: () => number;
  planId?: string;
  log?: boolean;
  onEvaluation?: (
    kind: "order" | "full",
    game: Game,
    value: Evaluation,
  ) => void;
  onComplete?: (actions: readonly Action[], value: number) => void;
}
class BudgetExceeded extends Error {}
function compareSequences(a: Action[], b: Action[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = compareActions(a[i], b[i]);
    if (d) return d;
  }
  return a.length - b.length;
}
const comparePaths = (a: Path, b: Path, kind: "order" | "full") =>
  b.state[kind]!.value - a.state[kind]!.value ||
  b.state.freeCardSlots - a.state.freeCardSlots ||
  b.state[kind]!.actualScore.total - a.state[kind]!.actualScore.total ||
  compareSequences(a.actions, b.actions);

/** Current-turn beam search. All widening passes share states, scores and incumbents. */
export function searchTurn(
  view: AIObservation,
  partial: Partial<AIConfig> = {},
  options: SearchOptions = {},
): TurnPlan {
  // Older saved depths remain readable; this planner always searches one own turn.
  const config = { ...normalizeAI(partial), depth: 1 };
  const now = options.now ?? (() => performance.now()),
    start = now();
  const rootId = view.players[view.active].id;
  const sampleSeed = config.seed + view.round * 1009 + rootId;
  const sampled = sampleGame(view, sampleSeed);
  if (sampled.phase === "ended")
    return {
      actions: [],
      decisions: [],
      completedDepth: 0,
      elapsedMs: 0,
      nodes: 0,
      evaluations: 0,
      timedOut: false,
    };
  const evaluator = createEvaluator("v3.1");
  const states = new Map<string, State>(),
    complete = new Map<string, Path>();
  let generatedActions = 0,
    duplicateStates = 0,
    stopReason: PlanSummary["stopReason"] = "completed";
  const check = () => {
    if (now() >= start + config.timeMs) {
      stopReason = "time";
      throw new BudgetExceeded();
    }
    if (
      evaluator.computations + evaluator.orderComputations >=
      (options.maxNodes ?? Infinity)
    ) {
      stopReason = "evaluations";
      throw new BudgetExceeded();
    }
  };
  // A single root sample is shared by every path. No draws occur until endTurn,
  // so identical public states here also have identical hidden draw orders.
  const intern = (game: Game): State => {
    const key = positionKey(game),
      old = states.get(key);
    if (old) {
      duplicateStates++;
      return old;
    }
    const state = {
      key,
      game,
      freeCardSlots: freeCardSlots(game.players.find((p) => p.id === rootId)!),
    };
    states.set(key, state);
    return state;
  };
  const root = intern(sampled);
  const edges = (state: State, bounded: boolean): Edge[] => {
    if (state.edges) return state.edges;
    const result: Edge[] = [];
    for (const action of legalActions(state.game)) {
      if (bounded) check();
      const game = applyAction(state.game, action);
      game.log = [];
      generatedActions++;
      result.push({ action, state: intern(game) });
    }
    state.edges = result;
    return result;
  };
  const score = (state: State, kind: "order" | "full", bounded: boolean) => {
    if (state[kind]) return;
    const value =
      kind === "order"
        ? evaluator.order(
            observe(state.game, rootId),
            rootId,
            bounded ? check : undefined,
          )
        : evaluator.evaluate(
            observe(state.game, rootId),
            rootId,
            bounded ? check : undefined,
          );
    state[kind] = value;
    options.onEvaluation?.(kind, state.game, value);
  };
  let best: Path | undefined;
  const finish = (path: Path, bounded: boolean) => {
    score(path.state, "full", bounded);
    const old = complete.get(path.state.key);
    if (!old || compareSequences(path.actions, old.actions) < 0)
      complete.set(path.state.key, path);
    if (!best || comparePaths(path, best, "full") < 0) best = path;
    options.onComplete?.(path.actions, path.state.full!.value);
  };
  // v1's inexpensive greedy fallback, plus one terminal V. No extra root V.
  let fallback: Path = { state: root, actions: [] };
  for (let step = 0; step < 64; step++) {
    const candidates = edges(fallback.state, false);
    for (const edge of candidates) score(edge.state, "order", false);
    const chosen = candidates
      .map((edge) => ({
        state: edge.state,
        actions: [...fallback.actions, edge.action],
      }))
      .sort((a, b) => comparePaths(a, b, "order"))[0];
    if (!chosen) throw new Error("AI 未找到合法动作。");
    fallback = chosen;
    if (chosen.actions.at(-1)!.type === "endTurn") {
      finish(chosen, false);
      break;
    }
  }
  if (!best) throw new Error("AI 回合超过动作上限。");
  let width = config.beamWidth,
    completedDepth = 0;
  const widthsCompleted: number[] = [];
  try {
    for (;;) {
      let frontier: Path[] = [{ state: root, actions: [] }],
        pruned = false;
      for (let step = 0; frontier.length; step++) {
        check();
        if (step >= 64) {
          stopReason = "stepLimit";
          break;
        }
        const next = new Map<string, Path>(),
          terminals = new Map<string, Path>();
        // Merge the entire layer before scoring or pruning. No per-parent top-k.
        for (const parent of frontier) {
          for (const edge of edges(parent.state, true)) {
            const path = {
              state: edge.state,
              actions: [...parent.actions, edge.action],
            };
            const target = edge.action.type === "endTurn" ? terminals : next;
            const old = target.get(path.state.key);
            if (!old || compareSequences(path.actions, old.actions) < 0)
              target.set(path.state.key, path);
          }
        }
        // Complete plans become eligible immediately, even if a later sibling
        // or wider pass runs out of time. Every one is compared by full V.
        for (const path of terminals.values()) {
          check();
          finish(path, true);
          completedDepth = 1;
        }
        for (const path of next.values()) {
          check();
          score(path.state, "order", true);
        }
        const ordered = [...next.values()].sort((a, b) =>
          comparePaths(a, b, "order"),
        );
        pruned ||= ordered.length > width;
        frontier = ordered.slice(0, width);
      }
      if (stopReason === "stepLimit") break;
      widthsCompleted.push(width);
      if (!pruned) break;
      width *= 2;
    }
  } catch (error) {
    if (!(error instanceof BudgetExceeded)) throw error;
  }
  const searchEnd = now();
  const beforeLog = evaluator.computations + evaluator.orderComputations;
  const decisions: AIDecisionLog[] = [];
  let current = root;
  if (options.log !== false) {
    for (const [decisionIndex, action] of best.actions.entries()) {
      // Read only. Missing scores are left absent, never calculated for logging.
      const candidates = current.edges!;
      const selectedEdge = candidates.find(
        (c) => compareActions(c.action, action) === 0,
      )!;
      const kind = (edge: Edge) =>
        edge.action.type === "endTurn" ? "full" : "order";
      if (!selectedEdge?.state[kind(selectedEdge)])
        throw new Error("AI 计划缺少搜索评分。");
      const before = current[kind(selectedEdge)];
      const makeLog = (edge: Edge): CandidateLog => {
        const scoreKind = kind(edge),
          evaluation = edge.state[scoreKind]!,
          parent = current[scoreKind];
        return {
          freeCardSlots: edge.state.freeCardSlots,
          scoreKind,
          action: edge.action,
          label: actionLabel(current.game, edge.action),
          value: evaluation.value,
          gain: parent ? evaluation.value - parent.value : undefined,
          evaluation,
        };
      };
      const selected = makeLog(selectedEdge);
      const topByOperation: AIDecisionLog["topByOperation"] = {},
        legalCounts: AIDecisionLog["legalCounts"] = {},
        evaluatedCounts: AIDecisionLog["evaluatedCounts"] = {};
      for (const type of new Set(candidates.map((c) => c.action.type))) {
        const all = candidates.filter((c) => c.action.type === type);
        const scored = all
          .filter((c) => c.state[kind(c)])
          .sort(
            (a, b) =>
              b.state[kind(b)]!.value - a.state[kind(a)]!.value ||
              b.state.freeCardSlots - a.state.freeCardSlots ||
              b.state[kind(b)]!.actualScore.total -
                a.state[kind(a)]!.actualScore.total ||
              compareActions(a.action, b.action),
          );
        legalCounts[type] = all.length;
        evaluatedCounts[type] = scored.length;
        topByOperation[type] = scored.slice(0, config.topK).map(makeLog);
      }
      decisions.push({
        evaluationVersion: EVALUATION_VERSION,
        planId: options.planId ?? `offline:${view.round}:${rootId}`,
        decisionIndex,
        position: observe(current.game, rootId),
        round: current.game.round,
        playerId: rootId,
        playerName: view.players[view.active].name,
        gameLogIndex: 0,
        config,
        before,
        selected,
        topByOperation,
        legalCounts,
        evaluatedCounts,
        scoreGain: before
          ? selected.evaluation.actualScore.total - before.actualScore.total
          : undefined,
        potentialGain: before
          ? selected.evaluation.potential - before.potential
          : undefined,
        usesSampledRefill:
          action.type === "endTurn" &&
          selectedEdge.state.game.phase !== "ended",
        sampleSeed,
        targetDepth: 1,
        completedDepth,
        searchValue: best.state.full!.value,
        elapsedMs: 0,
        nodes: generatedActions,
        timedOut: false,
        fallback: completedDepth === 0,
      });
      current = selectedEdge.state;
    }
    const topPlans = [...complete.values()]
      .sort((a, b) => comparePaths(a, b, "full"))
      .slice(0, config.topK)
      .map((path) => {
        let state = root;
        const labels = path.actions.map((action) => {
          const label = actionLabel(state.game, action);
          state = state.edges!.find(
            (edge) => compareActions(edge.action, action) === 0,
          )!.state;
          return label;
        });
        return {
          freeCardSlots: path.state.freeCardSlots,
          actions: path.actions,
          labels,
          value: path.state.full!.value,
          scenarioValues: [path.state.full!.value],
          terminal: path.state.game.phase === "ended",
          evaluation: path.state.full,
        };
      });
    decisions[0].planSummary = {
      value: best.state.full!.value,
      targetDepth: 1,
      completedDepth,
      elapsedMs: 0,
      expandedActions: generatedActions,
      evaluations: evaluator.computations,
      stopReason,
      topPlans,
    };
  }
  const end = now(),
    elapsedMs = Math.round(end - start);
  const stats: SearchStats = {
    generatedActions,
    uniqueStates: states.size,
    duplicateStates,
    orderEvaluations: evaluator.orderComputations,
    fullEvaluations: evaluator.computations,
    completePlans: complete.size,
    widthsCompleted,
    widthAttempted: width,
    logEvaluations:
      evaluator.computations + evaluator.orderComputations - beforeLog,
    searchMs: searchEnd - start,
    logMs: end - searchEnd,
  };
  if (stats.logEvaluations !== 0) throw new Error("AI 日志发生了额外估值。");
  const timedOut =
    ["time", "evaluations"].includes(stopReason) || elapsedMs >= config.timeMs;
  for (const decision of decisions) {
    decision.elapsedMs = elapsedMs;
    decision.timedOut = timedOut;
  }
  if (decisions[0]?.planSummary) {
    decisions[0].planSummary.elapsedMs = elapsedMs;
    decisions[0].planSummary.searchStats = stats;
  }
  return {
    actions: best.actions,
    decisions,
    completedDepth,
    elapsedMs,
    nodes: generatedActions,
    evaluations: beforeLog,
    timedOut,
  };
}
