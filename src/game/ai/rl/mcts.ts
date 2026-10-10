import {
  applyAction,
  ranking,
  seededRandom,
  type Action,
  type Game,
} from "../../engine";
import { legalActions, positionKey } from "../actions";
import { observe, sampleGame } from "../observation";
import type { AIObservation } from "../types";
import { actionId, assertSupported, SCORE_SCALE } from "./encoding";
import type { Evaluator, Pair, Values } from "./network";

export interface SearchConfig {
  simulations: number;
  seed: number;
  cpuct: number;
  cpuctLog: number;
  cpuctBase: number;
  fpuReduction: number;
  noiseWeight: number;
  noiseConcentration: number;
  temperature: number;
  rankWeight: number;
  relativeWeight: number;
  absoluteWeight: number;
}
export const defaults: SearchConfig = {
  simulations: 128,
  seed: 1729,
  cpuct: 1.25,
  cpuctLog: 0.5,
  cpuctBase: 19652,
  fpuReduction: 0.2,
  noiseWeight: 0,
  noiseConcentration: 10,
  temperature: 0,
  rankWeight: 1,
  relativeWeight: 0.25,
  absoluteWeight: 0.05,
};
export function searchConfig(config: Partial<SearchConfig> = {}): SearchConfig {
  const c = { ...defaults, ...config };
  if (
    Object.values(c).some((x) => !Number.isFinite(x)) ||
    !Number.isInteger(c.simulations) ||
    c.simulations < 1 ||
    c.simulations > 100000 ||
    !Number.isInteger(c.seed) ||
    c.cpuct <= 0 ||
    c.cpuctBase <= 0 ||
    c.cpuctLog < 0 ||
    c.fpuReduction < 0 ||
    c.noiseWeight < 0 ||
    c.noiseWeight > 1 ||
    c.noiseConcentration <= 0 ||
    c.temperature < 0 ||
    [c.rankWeight, c.relativeWeight, c.absoluteWeight].some((x) => x < 0) ||
    c.rankWeight + c.relativeWeight + c.absoluteWeight === 0
  )
    throw new Error("Invalid MCTS configuration");
  return c;
}
export interface EdgeStats {
  action: Action;
  id: number;
  prior: number;
  visits: number;
  q: number;
  selectionWeight: number;
  target: number;
}
export interface SearchResult {
  action: Action;
  actionId: number;
  policy: EdgeStats[];
  value: Values;
  simulations: number;
  nodes: number;
  chanceOutcomes: number;
}
interface Edge {
  action: Action;
  id: number;
  prior: number;
  visits: number;
  sum: number;
  outcomes: Map<string, Node>;
}
interface Node {
  seat: number;
  visits: number;
  value: Values;
  edges: Edge[];
}

export function terminalValues(game: Game, seat = game.active): Values {
  if (game.phase !== "ended")
    throw new Error("Value targets require a finished game");
  const entries = ranking(game);
  const a = entries.find((e) => e.player.id === game.players[seat].id)!;
  const b = entries.find((e) => e.player.id === game.players[1 - seat].id)!;
  const rank = Math.sign(b.rank - a.rank);
  const relative = (a.score.total - b.score.total) / SCORE_SCALE;
  return {
    rank: [rank, -rank],
    relative: [relative, -relative],
    absolute: [a.score.total / SCORE_SCALE, b.score.total / SCORE_SCALE],
  };
}
const utility = (v: Values, c: SearchConfig): Pair =>
  [0, 1].map(
    (i) =>
      c.rankWeight * v.rank[i] +
      c.relativeWeight * v.relative[i] +
      c.absoluteWeight * v.absolute[i],
  ) as Pair;

function softmax(logits: number[]): number[] {
  if (!logits.length || logits.some((x) => !Number.isFinite(x)))
    throw new Error("Invalid legal policy logits");
  const max = Math.max(...logits),
    exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}
// Marsaglia-Tsang gamma sampler; symmetric Dirichlet is a deliberate simplification.
function gamma(shape: number, random: () => number): number {
  const uniform = () => Math.max(Number.MIN_VALUE, random());
  if (shape < 1)
    return gamma(shape + 1, random) * Math.pow(uniform(), 1 / shape);
  const d = shape - 1 / 3,
    c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x =
      Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * random());
    const v = (1 + c * x) ** 3;
    if (v <= 0) continue;
    const u = uniform();
    if (
      u < 1 - 0.0331 * x ** 4 ||
      Math.log(u) < (x * x) / 2 + d * (1 - v + Math.log(v))
    )
      return d * v;
  }
}
export const exploration = (n: number, c: SearchConfig) =>
  (c.cpuct + c.cpuctLog * Math.log((n + c.cpuctBase) / c.cpuctBase)) *
  Math.sqrt(n + 0.01);

/** Invert PUCT to remove excess exploratory visits, following KataGo's
 * getReducedPlaySelectionWeight/getPlaySelectionValues (equal-weight variant).
 * The stable most-visited move keeps its weight; lower-Q moves lose excess visits.
 */
export function selectionWeights(
  stats: Pick<EdgeStats, "visits" | "prior" | "q">[],
  c: SearchConfig,
): number[] {
  const n = stats.reduce((s, e) => s + e.visits, 0),
    scale = exploration(n, c);
  let best = stats.findIndex((e) => e.visits > 0);
  if (best < 0)
    throw new Error("Policy targets require at least one search visit");
  for (let i = 1; i < stats.length; i++)
    if (
      stats[i].visits > 0 &&
      Math.max(0, stats[i].visits - 1) + 2 * stats[i].prior >
        Math.max(0, stats[best].visits - 1) + 2 * stats[best].prior
    )
      best = i;
  const reference = stats[best];
  const threshold =
    reference.q + (scale * reference.prior) / (1 + reference.visits);
  return stats.map((edge, i) => {
    if (i === best || !edge.visits || threshold <= edge.q) return edge.visits;
    return Math.min(
      edge.visits,
      Math.ceil(Math.max(0, (scale * edge.prior) / (threshold - edge.q) - 1)),
    );
  });
}

export function search(
  view: AIObservation,
  evaluator: Evaluator,
  config: Partial<SearchConfig> = {},
): SearchResult {
  assertSupported(view);
  if (view.phase !== "playing")
    throw new Error("Cannot search a finished game");
  const c = searchConfig(config),
    random = seededRandom(c.seed);
  let nodes = 0,
    chanceOutcomes = 0;
  const makeNode = (game: Game): Node => {
    const publicView = observe(game),
      actions = legalActions(game);
    if (!actions.length)
      throw new Error("Nonterminal state has no legal actions");
    const prediction = evaluator.predict(publicView);
    if (
      (["rank", "relative", "absolute"] as const).some(
        (head) =>
          prediction.value[head].length !== 2 ||
          prediction.value[head].some((x) => !Number.isFinite(x)),
      )
    )
      throw new Error("Invalid NN values");
    const ids = actions.map((a) => actionId(publicView, a));
    const priors = softmax(ids.map((id) => prediction.logits[id]));
    nodes++;
    return {
      seat: game.active,
      visits: 0,
      value: prediction.value,
      edges: actions.map((action, i) => ({
        action,
        id: ids[i],
        prior: priors[i],
        visits: 0,
        sum: 0,
        outcomes: new Map(),
      })),
    };
  };
  const root = makeNode(sampleGame(view, c.seed));
  if (c.noiseWeight) {
    const noise = root.edges.map(() =>
      gamma(c.noiseConcentration / root.edges.length, random),
    );
    const total = noise.reduce((a, b) => a + b, 0);
    root.edges.forEach((edge, i) => {
      edge.prior =
        (1 - c.noiseWeight) * edge.prior +
        c.noiseWeight * (total > 0 ? noise[i] / total : 1 / noise.length);
    });
  }
  const visit = (node: Node, game: Game): Pair => {
    const visitedMass = node.edges.reduce(
      (s, e) => s + (e.visits ? e.prior : 0),
      0,
    );
    const fpu =
      utility(node.value, c)[0] - c.fpuReduction * Math.sqrt(visitedMass);
    const scale = exploration(node.visits, c);
    let selected = node.edges[0],
      best = -Infinity;
    for (const edge of node.edges) {
      const score =
        (edge.visits ? edge.sum / edge.visits : fpu) +
        (scale * edge.prior) / (1 + edge.visits);
      if (score > best) {
        best = score;
        selected = edge;
      }
    }
    const next = applyAction(game, selected.action);
    next.log = [];
    let result: Pair;
    if (next.phase === "ended") {
      result = utility(terminalValues(next, 0), c); // Physical seats, never negate absolute score.
    } else {
      // Different random refills share the parent edge, but never the child state.
      const key =
        selected.action.type === "endTurn"
          ? positionKey(next)
          : "deterministic";
      let child = selected.outcomes.get(key);
      if (!child) {
        child = makeNode(next);
        selected.outcomes.set(key, child);
        if (selected.action.type === "endTurn") chanceOutcomes++;
        const v = utility(child.value, c);
        result = next.active === 0 ? v : [v[1], v[0]];
      } else result = visit(child, next);
    }
    selected.visits++;
    selected.sum += result[node.seat];
    node.visits++;
    return result;
  };
  for (let i = 0; i < c.simulations; i++)
    visit(root, sampleGame(view, Math.floor(random() * 2 ** 32)));
  const stats = root.edges.map((e) => ({
    action: e.action,
    id: e.id,
    prior: e.prior,
    visits: e.visits,
    q: e.visits ? e.sum / e.visits : utility(root.value, c)[0],
  }));
  const weights = selectionWeights(stats, c),
    total = weights.reduce((a, b) => a + b, 0);
  const policy = stats.map((e, i) => ({
    ...e,
    selectionWeight: weights[i],
    target: weights[i] / total,
  }));
  let index = 0;
  if (c.temperature === 0) {
    for (let i = 1; i < policy.length; i++)
      if (
        weights[i] > weights[index] ||
        (weights[i] === weights[index] && policy[i].q > policy[index].q)
      )
        index = i;
  } else {
    const max = Math.max(...weights);
    const probs = weights.map((w) =>
      w ? Math.exp(Math.log(w / max) / c.temperature) : 0,
    );
    let draw = random() * probs.reduce((a, b) => a + b, 0);
    index = probs.findIndex((p) => (draw -= p) < 0);
    if (index < 0) index = probs.length - 1;
  }
  return {
    action: policy[index].action,
    actionId: policy[index].id,
    policy,
    value: root.value,
    simulations: c.simulations,
    nodes,
    chanceOutcomes,
  };
}
