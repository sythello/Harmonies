import assert from "node:assert/strict";
import {
  applyAction,
  clone,
  ranking,
  scorePlayer,
  type Action,
  type Game,
} from "../../src/game/engine";
import { legalActions } from "../../src/game/ai/actions";
import { observe, sampleGame } from "../../src/game/ai/observation";
import {
  actionId,
  encode,
  SCORE_SCALE,
  VERSION,
} from "../../src/game/ai/rl/encoding";
import {
  exploration,
  searchConfig,
  SearchTree,
  terminalValues,
} from "../../src/game/ai/rl/mcts";
import type { Evaluator } from "../../src/game/ai/rl/network";
import { newGame, type Sample } from "../../src/game/ai/rl/trajectory";
import type { AIObservation } from "../../src/game/ai/types";
import type { AuditCase, SourceGame } from "./mcts-audit";

// Only within-turn state: hidden refills and the other player's board cannot
// change this player's final score. Action IDs make pending/card order immaterial.
function key(game: Game) {
  const p = game.players[game.active];
  return JSON.stringify([
    p.board.map((c) => [c.tokens, c.animal]),
    [...p.cards].sort((a, b) => a.id - b.id),
    game.supply,
    game.market,
    game.turn.tookTokens,
    game.turn.tookCard,
    [...game.turn.pending].sort(),
  ]);
}
interface Edge {
  id: number;
  action: Action;
  visits: number;
  sum: number;
  child?: Node;
  terminal?: number;
}
interface Node {
  game: Game;
  score: number;
  visits: number;
  edges: Edge[];
  solved: boolean;
  best: number;
}
export interface ExactTurn {
  score: number;
  root: {
    id: number;
    score: number;
    prefixVisits: number;
    targetVisits: number;
  }[];
  optimalIds: number[];
  plan: Action[];
  // The 512 actual paths allow independent reconstruction of all prefix counts.
  prefixPaths: number[][];
  boostedPath: { id: number; prefixVisits: number; addedVisits: number }[];
  states: number;
  transitions: number;
  transpositions: number;
  simulations: number;
  seconds: number;
}

/** PUCT with uniform priors and current own score. The first 512 simulations
 * use ordinary PUCT (including repeated terminal visits). Afterwards solved
 * branches are excluded until every legal edge has an exact terminal maximum.
 * Transpositions share computation, never probabilistically prune actions.
 */
export function solveTurn(view: AIObservation, timeoutMs = 60_000): ExactTurn {
  assert.equal(view.finalRound, view.round);
  const started = performance.now(),
    c = searchConfig({ rankWeight: 0, relativeWeight: 0, absoluteWeight: 1 });
  const memo = new Map<string, Node>();
  let transitions = 0,
    transpositions = 0,
    simulations = 0;
  function node(game: Game): Node {
    const k = key(game),
      old = memo.get(k);
    if (old) {
      transpositions++;
      return old;
    }
    const publicView = observe(game);
    const n: Node = {
      game,
      score: scorePlayer(game.players[game.active], game.side).total,
      visits: 0,
      solved: false,
      best: -Infinity,
      edges: legalActions(game).map((action) => ({
        action,
        id: actionId(publicView, action),
        visits: 0,
        sum: 0,
      })),
    };
    assert.ok(n.edges.length);
    memo.set(k, n);
    return n;
  }
  const root = node(sampleGame(view, 0));
  const edgeScore = (e: Edge) =>
    e.terminal ?? (e.child?.solved ? e.child.best : undefined);
  function refresh(n: Node) {
    n.solved = n.edges.every((e) => edgeScore(e) !== undefined);
    if (n.solved) n.best = Math.max(...n.edges.map((e) => edgeScore(e)!));
  }
  function visit(n: Node, prove: boolean, path: number[]): number {
    if (performance.now() - started > timeoutMs)
      throw new Error(
        "Exact turn search timed out; no partial label is emitted",
      );
    refresh(n);
    if (prove && n.solved) return n.best;
    const mass = n.edges.filter((e) => e.visits).length / n.edges.length;
    const fpu = n.score / SCORE_SCALE - c.fpuReduction * Math.sqrt(mass);
    const scale = exploration(n.visits, c) / n.edges.length;
    let selected: Edge | undefined,
      best = -Infinity;
    for (const e of n.edges) {
      if (prove && edgeScore(e) !== undefined) continue;
      const q = (e.visits ? e.sum / e.visits : fpu) + scale / (1 + e.visits);
      if (q > best) {
        selected = e;
        best = q;
      }
    }
    const e = selected!;
    path.push(e.id);
    let value: number;
    if (e.terminal !== undefined) value = e.terminal;
    else if (e.child) value = visit(e.child, prove, path);
    else {
      transitions++;
      const next = applyAction(n.game, e.action);
      next.log = [];
      if (e.action.type === "endTurn") {
        e.terminal = scorePlayer(next.players[view.active], next.side).total;
        value = e.terminal;
      } else {
        e.child = node(next);
        value = e.child.solved ? e.child.best : e.child.score;
      }
    }
    e.visits++;
    e.sum += value / SCORE_SCALE;
    n.visits++;
    refresh(n);
    return value;
  }
  const prefixPaths: number[][] = [];
  for (; simulations < 512; simulations++) {
    const path: number[] = [];
    visit(root, false, path);
    prefixPaths.push(path);
  }
  while (!root.solved) {
    visit(root, true, []);
    simulations++;
  }
  const ids: number[] = [];
  let current = root;
  for (;;) {
    const e = current.edges.find((e) => edgeScore(e) === current.best)!;
    ids.push(e.id);
    if (e.terminal !== undefined) break;
    current = e.child!;
  }
  // Replay IDs against the actual order, not the transposition's representative.
  let replay = sampleGame(view, 0);
  const plan = ids.map((id) => {
    const a = legalActions(replay).find(
      (a) => actionId(observe(replay), a) === id,
    )!;
    assert.ok(a);
    replay = applyAction(replay, a);
    replay.log = [];
    return a;
  });
  assert.equal(
    scorePlayer(replay.players[view.active], view.side).total,
    root.best,
  );
  const rows = root.edges.map((e) => {
    const prefixVisits = prefixPaths.filter((p) => p[0] === e.id).length;
    return {
      id: e.id,
      score: edgeScore(e)!,
      prefixVisits,
      targetVisits: prefixVisits + (e.id === ids[0] ? 512 : 0),
    };
  });
  assert.equal(
    rows.reduce((s, e) => s + e.targetVisits, 0),
    1024,
  );
  return {
    score: root.best,
    root: rows,
    optimalIds: rows.filter((e) => e.score === root.best).map((e) => e.id),
    plan,
    prefixPaths,
    boostedPath: ids.map((id, i) => ({
      id,
      addedVisits: 512,
      prefixVisits: prefixPaths.filter(
        (p) => p.length > i && ids.slice(0, i + 1).every((v, j) => p[j] === v),
      ).length,
    })),
    states: memo.size,
    transitions,
    transpositions,
    simulations,
    seconds: (performance.now() - started) / 1000,
  };
}

export interface FitSample extends Sample {
  id: string;
  input: AIObservation;
  exact: ExactTurn;
}
export interface FitGame {
  version: typeof VERSION;
  seed: number;
  first: number;
  gameIndex: number;
  split: "train" | "test";
  phase: "ended";
  scores: number[];
  ranks: number[];
  sourceActionIndex: number;
  finalActions: Action[][];
  actions: Action[];
  samples: FitSample[];
}

export function optimizeGame(
  source: SourceGame,
  start: AuditCase,
  split: FitGame["split"],
): FitGame {
  let game = newGame(source.seed, source.first);
  for (const a of source.actions.slice(0, start.sourceActionIndex)) {
    game = applyAction(game, a);
    game.log = [];
  }
  assert.deepEqual(observe(game), start.original);
  assert.equal(game.active, 0);
  game.finalRound = game.round;
  game.finalReason = "NN 拟合实验最后一回合";
  const samples: FitSample[] = [],
    finalActions: Action[][] = [];
  for (let seat = 0; seat < 2; seat++) {
    assert.equal(game.active, seat);
    assert.equal(game.phase, "playing");
    const input = observe(game),
      exact = solveTurn(input);
    samples.push({
      ...encode(input),
      id: `game-${String(start.gameIndex).padStart(3, "0")}-seat-${seat + 1}`,
      seat,
      input,
      exact,
      legal: exact.root.map((e) => e.id),
      policy: exact.root.map((e) => e.targetVisits / 1024),
      visits: exact.root.map((e) => e.targetVisits),
      value: { rank: [0, 0], relative: [0, 0], absolute: [0, 0] },
      actionId: actionId(input, exact.plan[0]),
      reusedVisits: 0,
      rootVisits: 1024,
    });
    finalActions.push(exact.plan);
    for (const a of exact.plan) {
      assert.ok(
        legalActions(game).some((b) => JSON.stringify(a) === JSON.stringify(b)),
      );
      game = applyAction(game, a);
      game.log = [];
    }
    assert.equal(scorePlayer(game.players[seat], game.side).total, exact.score);
  }
  assert.equal(game.phase, "ended");
  for (const s of samples) s.value = terminalValues(game, s.seat);
  const ranks = ranking(game);
  return {
    version: VERSION,
    gameIndex: start.gameIndex,
    seed: source.seed,
    first: source.first,
    split,
    phase: "ended",
    scores: game.players.map((p) => scorePlayer(p, game.side).total),
    ranks: game.players.map(
      (p) => ranks.find((r) => r.player.id === p.id)!.rank,
    ),
    sourceActionIndex: start.sourceActionIndex,
    finalActions,
    // This is a sparse two-decision supervised dataset, not a full RL rollout.
    actions: finalActions.map((p) => p[0]),
    samples,
  };
}

export function evaluateSample(
  sample: FitSample,
  evaluator: Evaluator,
  seed: number,
  simulations = 128,
) {
  const start = performance.now(),
    prediction = evaluator.predict(sample.input);
  const argmax = sample.legal.reduce((a, b) =>
    prediction.logits[b] > prediction.logits[a] ? b : a,
  );
  function play(search: boolean) {
    let game = sampleGame(sample.input, seed),
      firstId = -1,
      steps = 0,
      inherited = 0;
    const actions: Action[] = [],
      tree = new SearchTree(evaluator);
    for (;;) {
      assert.ok(steps < 64, "Final turn did not terminate");
      const view = observe(game);
      let action: Action;
      if (search) {
        const r = tree.search(view, { simulations, seed: seed + steps * 1009 });
        action = r.action;
        inherited += r.reusedVisits;
      } else {
        const logits = evaluator.predict(view).logits;
        action = legalActions(game).reduce((a, b) =>
          logits[actionId(view, b)] > logits[actionId(view, a)] ? b : a,
        );
      }
      if (!steps) firstId = actionId(view, action);
      steps++;
      actions.push(action);
      game = applyAction(game, action);
      game.log = [];
      if (action.type === "endTurn") break;
    }
    const score = scorePlayer(game.players[sample.seat], game.side).total;
    assert.ok(
      score <= sample.exact.score,
      "Serving exceeded the purported exact optimum",
    );
    return {
      firstId,
      firstHit: sample.exact.optimalIds.includes(firstId),
      hit: score === sample.exact.score,
      score,
      regret: sample.exact.score - score,
      steps,
      inherited,
      actions,
    };
  }
  return {
    id: sample.id,
    seat: sample.seat,
    argmax,
    policyHit: sample.exact.optimalIds.includes(argmax),
    optimalScore: sample.exact.score,
    value: clone(prediction.value),
    greedy: play(false),
    serving: play(true),
    seconds: (performance.now() - start) / 1000,
  };
}
