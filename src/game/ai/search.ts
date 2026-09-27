import {
  applyAction,
  CARDS,
  scorePlayer,
  spiritScore,
  type Action,
  type Game,
} from "../engine";
import { normalizeAI } from "./config";
import { discount, evaluatePosition, horizon } from "./evaluate";
import { counts, observe, sampleGame } from "./observation";
import { projects } from "./projects";
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
  Evaluation,
  TurnPlan,
} from "./types";

interface Node {
  game: Game;
  actions: Action[];
  value: number;
  actual: number;
}
export interface SearchOptions {
  maxNodes?: number;
  now?: () => number;
}
class BudgetExceeded extends Error {}

/** Cheap ordering only; the returned decision values always use the six-project evaluator. */
function orderValue(game: Game, playerId: number, actual: number): number {
  const player = game.players.find((p) => p.id === playerId)!;
  if (game.phase === "ended") return actual;
  const view = observe(game, playerId),
    h = horizon(view, playerId);
  if (!h.turns) return actual;
  const H =
    game.players[game.active].id === playerId
      ? counts(game.turn.pending)
      : [0, 0, 0, 0, 0, 0];
  const supplyCounts = counts(game.supply.flat());
  const A = view.bagCounts.map((n, c) => n + supplyCounts[c]);
  const best = new Map<number, number>();
  for (const project of projects(player, false)) {
    const card = player.cards.find((c) => c.id === project.cardId)!;
    const data = CARDS[card.id];
    const G = data.spirit
      ? spiritScore(player.board, card.id)
      : data.points[card.placed] - (data.points[card.placed - 1] ?? 0);
    const terms = discount(
      counts(project.additions.flatMap((a) => a.colors)),
      H,
      A,
      h.B,
    );
    if (terms) best.set(card.id, Math.max(best.get(card.id) ?? 0, G * terms.q));
  }
  return actual + [...best.values()].reduce((a, b) => a + b, 0);
}
const compareNodes = (a: Node, b: Node) =>
  b.value - a.value ||
  b.actual - a.actual ||
  compareSequences(a.actions, b.actions);
function compareSequences(a: Action[], b: Action[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const diff = compareActions(a[i], b[i]);
    if (diff) return diff;
  }
  return a.length - b.length;
}

export function searchTurn(
  view: AIObservation,
  partial: Partial<AIConfig> = {},
  options: SearchOptions = {},
): TurnPlan {
  const config = normalizeAI(partial),
    now = options.now ?? (() => performance.now()),
    start = now();
  const searchDeadline = start + config.timeMs * 0.65,
    traceDeadline = start + config.timeMs;
  const rootId = view.players[view.active].id,
    root = sampleGame(view, config.seed + view.round * 1009 + rootId);
  if (root.phase === "ended")
    return {
      actions: [],
      decisions: [],
      completedDepth: 0,
      elapsedMs: 0,
      nodes: 0,
      timedOut: false,
    };
  let nodes = 0,
    timedOut = false;
  const cache = new Map<string, Evaluation>();
  const transitions = new Map<string, Node[]>();
  const evaluate = (game: Game, id = rootId): Evaluation => {
    const key = `${id}:${positionKey(game)}`;
    let value = cache.get(key);
    if (!value) {
      value = evaluatePosition(observe(game, id), id);
      cache.set(key, value);
    }
    return value;
  };
  const check = () => {
    if (now() >= searchDeadline || nodes >= (options.maxNodes ?? Infinity))
      throw new BudgetExceeded();
  };
  const expand = (game: Game, id: number, bounded: boolean): Node[] => {
    if (bounded) check();
    // Hypothetical draw orders must stay distinct across the two sampled futures.
    const transitionKey = `${id}:${positionKey(game)}:${game.bag.join(",")}:${game.deck.join(",")}`;
    const cached = transitions.get(transitionKey);
    if (cached) return cached;
    const result: Node[] = [];
    for (const action of legalActions(game)) {
      if (bounded) check();
      const next = applyAction(game, action);
      next.log = [];
      nodes++;
      const actual = scorePlayer(
        next.players.find((p) => p.id === id)!,
        next.side,
      ).total;
      result.push({
        game: next,
        actions: [action],
        value: orderValue(next, id, actual),
        actual,
      });
    }
    result.sort(compareNodes);
    transitions.set(transitionKey, result);
    return result;
  };
  // A complete legal turn exists before attempting wider/deeper search.
  const greedy = (initial: Game, bounded: boolean): Node => {
    let game = initial;
    const id = game.players[game.active].id,
      actions: Action[] = [];
    for (let i = 0; i < 64; i++) {
      const candidates = expand(game, id, bounded);
      const chosen = candidates[0];
      if (!chosen) throw new Error("AI 未找到合法动作。");
      const action = chosen.actions[0];
      actions.push(action);
      game = chosen.game;
      if (action.type === "endTurn") return { ...chosen, game, actions };
    }
    throw new Error("AI 回合超过动作上限。");
  };
  const fallback = greedy(root, false);
  let best = fallback,
    completedDepth = 0;
  let bestValue = evaluate(fallback.game).value;
  const completeTurns = (initial: Game, width: number): Node[] => {
    let frontier: Node[] = [
      { game: initial, actions: [], value: 0, actual: 0 },
    ];
    const complete = new Map<string, Node>();
    const id = initial.players[initial.active].id;
    try {
      for (let step = 0; frontier.length && step < 64; step++) {
        check();
        const next = new Map<string, Node>();
        for (const node of frontier) {
          const expanded = expand(node.game, id, true);
          // Retain the best action of every operation type, plus the global beam.
          const candidates = expanded.slice(0, width);
          for (const type of new Set(expanded.map((n) => n.actions[0].type))) {
            const first = expanded.find((n) => n.actions[0].type === type)!;
            if (!candidates.includes(first)) candidates.push(first);
          }
          for (const child of candidates) {
            const combined = {
              ...child,
              actions: [...node.actions, ...child.actions],
            };
            const key = positionKey(child.game);
            if (child.actions[0].type === "endTurn") {
              combined.value = evaluate(child.game, id).value;
              const old = complete.get(key);
              if (!old || compareNodes(combined, old) < 0)
                complete.set(key, combined);
            } else {
              const old = next.get(key);
              if (!old || compareNodes(combined, old) < 0)
                next.set(key, combined);
            }
          }
        }
        frontier = [...next.values()].sort(compareNodes).slice(0, width);
      }
    } catch (error) {
      if (!(error instanceof BudgetExceeded)) throw error;
      timedOut = true;
      // A deadline must not discard complete legal turns already discovered.
      if (!complete.size) throw error;
    }
    return [...complete.values()].sort(compareNodes).slice(0, width);
  };
  try {
    const turns = completeTurns(root, config.beamWidth);
    if (turns.length) {
      if (
        !turns.some((n) => compareSequences(n.actions, fallback.actions) === 0)
      )
        turns.push({ ...fallback, value: bestValue });
      turns.sort(compareNodes);
      best = turns[0];
      bestValue = best.value;
      completedDepth = 1;
      // Each extra depth is another complete turn of the root player.
      for (let depth = 2; depth <= config.depth; depth++) {
        const values: { node: Node; value: number }[] = [];
        for (const candidate of turns) {
          let total = 0;
          for (let sample = 0; sample < 2; sample++) {
            check();
            let simulated = sampleGame(
              view,
              config.seed + view.round * 1009 + rootId + sample * 7919,
            );
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
                const options = completeTurns(
                  simulated,
                  Math.min(2, config.beamWidth),
                );
                simulated = options[0]?.game ?? greedy(simulated, true).game;
              }
            }
            total += evaluate(simulated).value;
          }
          values.push({ node: candidate, value: total / 2 });
        }
        values.sort(
          (a, b) => b.value - a.value || compareNodes(a.node, b.node),
        );
        best = values[0].node;
        bestValue = values[0].value;
        completedDepth = depth;
      }
    }
  } catch (error) {
    if (!(error instanceof BudgetExceeded)) throw error;
    timedOut = true;
  }
  // Only executed decision points are logged, not the search tree. No hidden arrays enter the log.
  const decisions: AIDecisionLog[] = [];
  let current = root;
  for (const action of best.actions) {
    const before = evaluate(current),
      candidates = expand(current, rootId, false);
    const selectedNode = candidates.find(
      (n) => compareActions(n.actions[0], action) === 0,
    )!;
    if (!selectedNode) throw new Error("AI 计划与局面不一致。");
    const makeLog = (candidate: Node): CandidateLog => {
      const evaluation = evaluate(candidate.game);
      return {
        action: candidate.actions[0],
        label: actionLabel(current, candidate.actions[0]),
        value: evaluation.value,
        gain: evaluation.value - before.value,
        evaluation,
      };
    };
    const selected = makeLog(selectedNode);
    const legalCounts: AIDecisionLog["legalCounts"] = {},
      evaluatedCounts: AIDecisionLog["evaluatedCounts"] = {};
    const topByOperation: AIDecisionLog["topByOperation"] = {};
    for (const type of new Set(candidates.map((n) => n.actions[0].type))) {
      const group = candidates.filter((n) => n.actions[0].type === type);
      legalCounts[type] = group.length;
      const logs: CandidateLog[] = [];
      for (const candidate of group.slice(0, Math.max(config.topK, 1))) {
        // One representative per operation is always evaluated; further alternatives share the time budget.
        if (logs.length && now() >= traceDeadline) break;
        logs.push(
          compareActions(candidate.actions[0], action) === 0
            ? selected
            : makeLog(candidate),
        );
      }
      if (
        type === action.type &&
        !logs.some((l) => compareActions(l.action, action) === 0)
      )
        logs.push(selected);
      evaluatedCounts[type] = logs.length;
      topByOperation[type] = logs
        .sort((a, b) => b.value - a.value || compareActions(a.action, b.action))
        .slice(0, config.topK);
    }
    decisions.push({
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
      evaluatedCounts,
      targetDepth: config.depth,
      completedDepth,
      searchValue: bestValue,
      elapsedMs: 0,
      nodes,
      timedOut,
      fallback: completedDepth === 0,
    });
    current = selectedNode.game;
  }
  const elapsedMs = Math.round(now() - start);
  decisions.forEach((d) => {
    d.elapsedMs = elapsedMs;
    d.nodes = nodes;
    d.timedOut ||= elapsedMs >= config.timeMs;
  });
  return {
    actions: best.actions,
    decisions,
    completedDepth,
    elapsedMs,
    nodes,
    timedOut: timedOut || elapsedMs >= config.timeMs,
  };
}
