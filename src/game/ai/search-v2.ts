import { applyAction, type Action, type Game } from "../engine";
import { normalizeAI } from "./config";
import { createEvaluator, rounded } from "./evaluate";
import { observe, sampleGame } from "./observation";
import {
  actionLabel,
  compareActions,
  legalActions,
  positionKey,
} from "./actions";
import type {
  AIConfig,
  AIDecisionLog,
  AIObservation,
  CandidateLog,
  PlanSummary,
  TurnPlan,
} from "./types";

interface Node {
  game: Game;
  actions: Action[];
  value: number;
  actual: number;
}
export interface SearchOptions {
  /** Limit uncached complete evaluations, including the mandatory fallback's work. */
  maxNodes?: number;
  now?: () => number;
  planId?: string;
  /** Optional audit hooks; never used by the UI or for ranking. */
  onExpanded?: (
    game: Game,
    playerId: number,
    candidates: readonly { action: Action; value: number }[],
  ) => void;
  onScenario?: (depth: number, candidate: number, sample: number) => void;
}
class BudgetExceeded extends Error {}
function compareSequences(a: Action[], b: Action[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const diff = compareActions(a[i], b[i]);
    if (diff) return diff;
  }
  return a.length - b.length;
}
const compareNodes = (a: Node, b: Node) =>
  rounded(b.value) - rounded(a.value) ||
  b.actual - a.actual ||
  compareSequences(a.actions, b.actions);

export function searchTurn(
  view: AIObservation,
  partial: Partial<AIConfig> = {},
  options: SearchOptions = {},
): TurnPlan {
  const config = normalizeAI({ depth: 2, ...partial }),
    now = options.now ?? (() => performance.now()),
    start = now();
  const deadline = start + config.timeMs;
  const rootId = view.players[view.active].id,
    sampleSeed = config.seed + view.round * 1009 + rootId;
  const root = sampleGame(view, sampleSeed);
  if (root.phase === "ended")
    return {
      actions: [],
      decisions: [],
      completedDepth: 0,
      elapsedMs: 0,
      nodes: 0,
      evaluations: 0,
      timedOut: false,
    };
  const evaluator = createEvaluator(),
    transitions = new Map<string, Node[]>();
  let expandedActions = 0,
    completedDepth = 0,
    stopReason: PlanSummary["stopReason"] = "completed";
  const check = () => {
    if (now() >= deadline) {
      stopReason = "time";
      throw new BudgetExceeded();
    }
    if (evaluator.computations >= (options.maxNodes ?? Infinity)) {
      stopReason = "evaluations";
      throw new BudgetExceeded();
    }
  };
  const evaluate = (game: Game, id: number, bounded = false) =>
    evaluator.evaluate(observe(game, id), id, bounded ? check : undefined);
  const transitionKey = (game: Game, id: number) =>
    `${id}:${positionKey(game)}:${game.bag.join(",")}:${game.deck.join(",")}`;
  const expand = (game: Game, id: number, bounded: boolean): Node[] => {
    if (bounded) check();
    const key = transitionKey(game, id),
      cached = transitions.get(key);
    if (cached) return cached;
    // Cache the parent as well, so the executed decision's diagnostics need no new evaluation.
    evaluate(game, id, bounded);
    const result: Node[] = [];
    for (const action of legalActions(game)) {
      if (bounded) check();
      const next = applyAction(game, action);
      next.log = [];
      expandedActions++;
      const evaluation = evaluate(next, id, bounded);
      result.push({
        game: next,
        actions: [action],
        value: evaluation.value,
        actual: evaluation.actualScore.total,
      });
    }
    result.sort(compareNodes);
    options.onExpanded?.(
      game,
      id,
      result.map((node) => ({ action: node.actions[0], value: node.value })),
    );
    // An interrupted sibling set is never published or used to choose a move.
    transitions.set(key, result);
    return result;
  };
  const greedy = (initial: Game, bounded: boolean): Node => {
    let game = initial;
    const id = game.players[game.active].id,
      actions: Action[] = [];
    for (let step = 0; step < 64; step++) {
      const chosen = expand(game, id, bounded)[0];
      if (!chosen) throw new Error("AI 未找到合法动作。");
      const action = chosen.actions[0];
      actions.push(action);
      game = chosen.game;
      if (action.type === "endTurn") return { ...chosen, actions };
    }
    throw new Error("AI 回合超过动作上限。");
  };
  // The only time-budget exemption: guarantee one full, legal, full-V greedy turn.
  const fallback = greedy(root, false);
  let best = fallback,
    bestValue = fallback.value;
  type Ranked = {
    node: Node;
    value: number;
    scenarioValues: number[];
    terminal: boolean;
  };
  let ranked: Ranked[] = [
    {
      node: fallback,
      value: fallback.value,
      scenarioValues: [fallback.value],
      terminal: fallback.game.phase === "ended",
    },
  ];
  const rankedCompare = (a: Ranked, b: Ranked) =>
    rounded(b.value) - rounded(a.value) ||
    b.node.actual - a.node.actual ||
    compareSequences(a.node.actions, b.node.actions);

  const completeTurns = (
    initial: Game,
    width: number,
    baseline: Node,
    allowPartial: boolean,
  ) => {
    const id = initial.players[initial.active].id;
    let frontier: Node[] = [
        { game: initial, actions: [], value: 0, actual: 0 },
      ],
      inspected = false;
    const complete = new Map<string, Node>([
      [positionKey(baseline.game), baseline],
    ]);
    const addComplete = (node: Node) => {
      const key = positionKey(node.game),
        old = complete.get(key);
      if (!old || compareNodes(node, old) < 0) complete.set(key, node);
      const ordered = [...complete.entries()].sort((a, b) =>
        compareNodes(a[1], b[1]),
      );
      for (const [key] of ordered.slice(width)) complete.delete(key);
    };
    try {
      for (let step = 0; frontier.length && step < 64; step++) {
        check();
        const next = new Map<string, Node>();
        for (const node of frontier) {
          const children = expand(node.game, id, true);
          for (const child of children) {
            const combined = {
              ...child,
              actions: [...node.actions, ...child.actions],
            };
            if (child.actions[0].type === "endTurn") {
              inspected = true;
              addComplete(combined);
            } else {
              const key = positionKey(child.game),
                old = next.get(key);
              if (!old || compareNodes(combined, old) < 0)
                next.set(key, combined);
            }
          }
        }
        frontier = [...next.values()].sort(compareNodes).slice(0, width);
      }
      if (frontier.length) stopReason = "stepLimit";
    } catch (error) {
      if (!(error instanceof BudgetExceeded) || !allowPartial) throw error;
    }
    return { turns: [...complete.values()].sort(compareNodes), inspected };
  };
  const shallow = completeTurns(root, config.beamWidth, fallback, true);
  const roots = shallow.turns;
  if (shallow.inspected) completedDepth = 1;
  best = roots[0];
  bestValue = best.value;
  ranked = roots.map((node) => ({
    node,
    value: node.value,
    scenarioValues: [node.value],
    terminal: node.game.phase === "ended",
  }));
  if (completedDepth > 0 && stopReason === "completed") {
    try {
      for (let depth = 2; depth <= config.depth; depth++) {
        const layer: Ranked[] = [];
        for (const [candidateIndex, candidate] of roots.entries()) {
          const scenarioValues: number[] = [];
          let terminal = true;
          for (let sample = 0; sample < 2; sample++) {
            options.onScenario?.(depth, candidateIndex, sample);
            check();
            let simulated = sampleGame(view, sampleSeed + sample * 7919);
            for (const action of candidate.actions)
              simulated = applyAction(simulated, action);
            for (
              let future = 1;
              future < depth && simulated.phase !== "ended";
              future++
            ) {
              while (
                simulated.phase !== "ended" &&
                simulated.players[simulated.active].id !== rootId
              )
                simulated = greedy(simulated, true).game;
              if (simulated.phase !== "ended") {
                const baseline = greedy(simulated, true);
                simulated = completeTurns(
                  simulated,
                  Math.min(2, config.beamWidth),
                  baseline,
                  false,
                ).turns[0].game;
              }
            }
            scenarioValues.push(evaluate(simulated, rootId, true).value);
            terminal &&= simulated.phase === "ended";
          }
          layer.push({
            node: candidate,
            value: (scenarioValues[0] + scenarioValues[1]) / 2,
            scenarioValues,
            terminal,
          });
        }
        check();
        layer.sort(rankedCompare);
        ranked = layer;
        best = layer[0].node;
        bestValue = layer[0].value;
        completedDepth = depth;
      }
    } catch (error) {
      if (!(error instanceof BudgetExceeded)) throw error;
    }
  }
  const rootValue = evaluate(root, rootId).value;
  const evaluationsBeforeLog = evaluator.computations;
  const decisions: AIDecisionLog[] = [];
  let current = root;
  for (const [decisionIndex, action] of best.actions.entries()) {
    const before = evaluate(current, rootId);
    const candidates = transitions.get(transitionKey(current, rootId));
    if (!candidates) throw new Error("AI 决策缺少完整候选集。");
    const makeLog = (candidate: Node): CandidateLog => {
      const evaluation = evaluate(candidate.game, rootId);
      return {
        action: candidate.actions[0],
        label: actionLabel(current, candidate.actions[0]),
        value: evaluation.value,
        gain: evaluation.value - before.value,
        evaluation,
      };
    };
    const selectedNode = candidates.find(
      (n) => compareActions(n.actions[0], action) === 0,
    )!;
    if (!selectedNode) throw new Error("AI 计划与局面不一致。");
    const selected = makeLog(selectedNode);
    const legalCounts: AIDecisionLog["legalCounts"] = {},
      topByOperation: AIDecisionLog["topByOperation"] = {};
    for (const type of new Set(candidates.map((n) => n.actions[0].type))) {
      const group = candidates.filter((n) => n.actions[0].type === type);
      legalCounts[type] = group.length;
      topByOperation[type] = group.slice(0, config.topK).map(makeLog);
    }
    decisions.push({
      evaluationVersion: "local-projects-v2",
      planId: options.planId ?? `offline:${view.round}:${rootId}`,
      decisionIndex,
      position: observe(current, rootId),
      round: current.round,
      playerId: rootId,
      playerName: current.players[current.active].name,
      gameLogIndex: 0,
      config,
      before,
      selected,
      topByOperation,
      legalCounts,
      evaluatedCounts: { ...legalCounts },
      scoreGain:
        selected.evaluation.actualScore.total - before.actualScore.total,
      potentialGain: selected.evaluation.potential - before.potential,
      usesSampledRefill:
        action.type === "endTurn" && selectedNode.game.phase !== "ended",
      sampleSeed,
      targetDepth: config.depth,
      completedDepth,
      searchValue: bestValue,
      elapsedMs: 0,
      nodes: expandedActions,
      timedOut: false,
      fallback: completedDepth === 0,
    });
    current = selectedNode.game;
  }
  if (evaluator.computations !== evaluationsBeforeLog)
    throw new Error("AI 日志发生了额外估值。");
  const topPlans = ranked.slice(0, config.topK).map((r) => {
    let state = root;
    const labels = r.node.actions.map((action) => {
      const label = actionLabel(state, action);
      state = applyAction(state, action);
      return label;
    });
    return {
      actions: r.node.actions,
      labels,
      value: r.value,
      scenarioValues: r.scenarioValues,
      terminal: r.terminal,
    };
  });
  const elapsedMs = Math.round(now() - start);
  const timedOut =
    ["time", "evaluations"].includes(stopReason) || elapsedMs >= config.timeMs;
  decisions.forEach((d) => {
    d.elapsedMs = elapsedMs;
    d.timedOut = timedOut;
  });
  decisions[0].planSummary = {
    rootValue,
    value: bestValue,
    gain: bestValue - rootValue,
    targetDepth: config.depth,
    completedDepth,
    elapsedMs,
    expandedActions,
    evaluations: evaluator.computations,
    stopReason,
    topPlans,
  };
  return {
    actions: best.actions,
    decisions,
    completedDepth,
    elapsedMs,
    nodes: expandedActions,
    evaluations: evaluator.computations,
    timedOut,
  };
}
