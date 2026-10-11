import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, clone, type Game } from "../src/game/engine";
import { observe, sampleGame } from "../src/game/ai/observation";
import { legalActions } from "../src/game/ai/actions";
import {
  SearchTree,
  search,
  type SearchResult,
  type RootStats,
} from "../src/game/ai/rl/mcts";
import { uniformEvaluator, type Evaluator } from "../src/game/ai/rl/network";
import {
  collect,
  mctsAgent,
  newGame,
  playGame,
} from "../src/game/ai/rl/trajectory";
import {
  collectCases,
  currentScoreEvaluator,
} from "../scripts/audit/mcts-audit";

const resultStats = (r: SearchResult) => ({
  action: r.action,
  actionId: r.actionId,
  value: r.value,
  policy: r.policy,
  rootVisits: r.rootVisits,
});
const rawStats = (r: SearchResult): RootStats => ({
  visits: r.rootVisits,
  value: r.value,
  policy: r.policy.map(({ action, id, prior, visits, q }) => ({
    action,
    id,
    prior,
    visits,
    q,
  })),
});
const absolute = { rankWeight: 0, relativeWeight: 0, absoluteWeight: 1 };
function beforeEnd(): Game {
  let game = newGame(3);
  game = applyAction(game, { type: "takeTokens", group: 0 });
  for (let cell = 0; cell < 3; cell++)
    game = applyAction(game, { type: "place", pending: 0, cell });
  game.turn.tookCard = true;
  return game;
}

test("same-root continuation exactly matches uninterrupted simulations with noise, temperature and random refills", () => {
  for (const game of [beforeEnd(), newGame(8)]) {
    const view = observe(game);
    const config = { seed: 125, noiseWeight: 0.25, temperature: 1 };
    const tree = new SearchTree(currentScoreEvaluator);
    const first = tree.search(view, { ...config, simulations: 17 });
    const saved = clone(first);
    const second = tree.search(view, { ...config, simulations: 31 });
    const reference = search(view, currentScoreEvaluator, {
      ...config,
      simulations: 48,
    });
    assert.equal(second.reused, true);
    assert.equal(second.reusedVisits, 17);
    assert.equal(second.rootVisits, 48);
    assert.deepEqual(resultStats(second), resultStats(reference));
    assert.equal(first.nodes + second.nodes, reference.nodes);
    assert.equal(
      first.chanceOutcomes + second.chanceOutcomes,
      reference.chanceOutcomes,
    );
    if (game.turn.tookTokens) assert.ok(reference.chanceOutcomes > 1);
    assert.deepEqual(first, saved);
  }
});

test("with deterministic transitions and no noise, inherited K visits equal a fresh K-visit tree", () => {
  // Real final-turn fixture: no stochastic transition can precede game end.
  const view = collectCases(1, 71000).cases[1].input;
  const config = { ...absolute, seed: 91, noiseWeight: 0, temperature: 0 };
  const tree = new SearchTree(currentScoreEvaluator);
  const first = tree.search(view, { ...config, simulations: 256 });
  const edge = first.policy.find((e) => e.id === first.actionId)!;
  assert.notEqual(edge.action.type, "endTurn");
  const next = observe(applyAction(sampleGame(view, 0), edge.action));
  // The first traversal creates/evaluates the child; it does not visit its edges.
  const k = edge.visits - 1;
  assert.ok(k > 0);
  const fresh = search(next, currentScoreEvaluator, {
    ...config,
    simulations: k,
  });
  let before: RootStats | undefined;
  const continued = tree.search(
    next,
    { ...config, simulations: 37 },
    {
      onRoot: (r) => {
        before = r;
      },
    },
  );
  assert.equal(continued.reused, true);
  assert.equal(continued.reusedVisits, k);
  assert.deepEqual(before, rawStats(fresh));
  assert.deepEqual(
    resultStats(continued),
    resultStats(
      search(next, currentScoreEvaluator, { ...config, simulations: k + 37 }),
    ),
  );
  assert.equal(
    continued.policy.reduce((n, e) => n + e.visits, 0),
    k + 37,
  );
});

test("promotion retains the child's NN value without reevaluation and adds root noise once", () => {
  const view = collectCases(1, 71000).cases[1].input;
  const calls = new Map<string, number>();
  const evaluator: Evaluator = {
    predict(v) {
      const key = JSON.stringify(v);
      calls.set(key, (calls.get(key) ?? 0) + 1);
      return currentScoreEvaluator.predict(v);
    },
  };
  const tree = new SearchTree(evaluator);
  const first = tree.search(view, {
    ...absolute,
    simulations: 256,
    seed: 90,
    noiseWeight: 0.25,
  });
  const next = observe(applyAction(sampleGame(view, 90), first.action));
  const key = JSON.stringify(next),
    count = calls.get(key);
  assert.equal(count, 1);
  let promoted: RootStats | undefined;
  const config = {
    ...absolute,
    simulations: 8,
    seed: 101,
    noiseWeight: 0.25,
    temperature: 1,
  };
  const second = tree.search(next, config, {
    onRoot: (r) => {
      promoted = r;
    },
  });
  assert.equal(second.reused, true);
  assert.equal(calls.get(key), count);
  assert.deepEqual(promoted!.value, currentScoreEvaluator.predict(next).value);
  // A concrete counterexample to unconditional fresh-K equivalence: historical
  // child visits used unnoised priors; a fresh root uses noisy priors from visit 1.
  assert.ok(second.reusedVisits > 0);
  const noisyFresh = search(next, currentScoreEvaluator, {
    ...config,
    simulations: second.reusedVisits,
  });
  assert.notDeepEqual(
    promoted!.policy.map((e) => e.visits),
    noisyFresh.policy.map((e) => e.visits),
  );
  let repeated: RootStats | undefined;
  const third = tree.search(next, config, {
    onRoot: (r) => {
      repeated = r;
    },
  });
  assert.equal(third.reusedVisits, second.rootVisits);
  assert.deepEqual(
    repeated!.policy.map((e) => e.prior),
    promoted!.policy.map((e) => e.prior),
  );
  assert.deepEqual(repeated, rawStats(second));
});

test("changed pending indices and public hidden-resource distributions cannot reuse a root", () => {
  let game = newGame(7);
  const group = game.supply.findIndex((g) => new Set(g).size > 1);
  game = applyAction(game, { type: "takeTokens", group });
  const config = { simulations: 32, seed: 71 };
  for (const change of [
    (v: ReturnType<typeof observe>) => {
      const pending = v.turn.pending;
      const i = pending.findIndex((c) => c !== pending[0]);
      [pending[0], pending[i]] = [pending[i], pending[0]];
    },
    (v: ReturnType<typeof observe>) => {
      v.bagCounts[0]--;
      v.bagCounts[1]++;
    },
  ]) {
    const tree = new SearchTree(uniformEvaluator),
      view = observe(game);
    tree.search(view, config);
    change(view);
    const result = tree.search(view, config);
    assert.equal(result.reused, false);
    assert.deepEqual(result, search(view, uniformEvaluator, config));
  }
});

test("reuse follows the actual action even when it differs from the recommendation", () => {
  const game = newGame(6),
    view = observe(game),
    tree = new SearchTree(uniformEvaluator);
  const config = { simulations: 64, seed: 5 };
  const first = tree.search(view, config);
  const alternative = first.policy.find(
    (e) => e.id !== first.actionId && e.visits > 1,
  )!;
  assert.ok(alternative);
  const result = tree.search(
    observe(applyAction(game, alternative.action)),
    config,
  );
  assert.equal(result.reused, true);
  assert.equal(result.reusedVisits, alternative.visits - 1);
  assert.equal(result.rootVisits, result.reusedVisits + 64);
});

test("turn changes, changed public state, skipped actions and undo discard incompatible statistics", () => {
  const config = { simulations: 64, seed: 13 };
  function check(initial: Game, change: (g: Game, r: SearchResult) => Game) {
    const tree = new SearchTree(uniformEvaluator);
    const first = tree.search(observe(initial), config);
    const next = change(clone(initial), first);
    const result = tree.search(observe(next), config);
    assert.equal(result.reused, false);
    assert.equal(result.reusedVisits, 0);
    assert.deepEqual(result, search(observe(next), uniformEvaluator, config));
  }
  check(beforeEnd(), (g) => applyAction(g, { type: "endTurn" }));
  check(newGame(2), (g) => {
    g.round++;
    return g;
  });
  check(newGame(2), (g) => {
    g.players[1].board[0].tokens.push(g.bag.pop()!);
    return g;
  });
  check(newGame(2), (g) => {
    g.finalRound = g.round;
    return g;
  });
  check(newGame(2), (g, r) => {
    const next = applyAction(g, r.action);
    return applyAction(
      next,
      legalActions(next).find((a) => a.type !== "endTurn")!,
    );
  });
  const tree = new SearchTree(uniformEvaluator),
    game = newGame(7),
    view = observe(game);
  const first = tree.search(view, config);
  tree.search(observe(applyAction(game, first.action)), config);
  assert.equal(tree.search(view, config).reused, false);
  tree.reset();
  assert.equal(tree.search(view, config).reused, false);
});

test("changed utility/PUCT settings or same-root seed rebuild; output mutation cannot corrupt retained state", () => {
  const view = observe(newGame(8));
  for (const changed of [
    { rankWeight: 0.5 },
    { cpuct: 2 },
    { noiseWeight: 0.25 },
    { seed: 9 },
  ]) {
    const tree = new SearchTree(uniformEvaluator);
    tree.search(view, { simulations: 20, seed: 1 });
    const c = { simulations: 20, seed: 1, ...changed };
    assert.deepEqual(tree.search(view, c), search(view, uniformEvaluator, c));
  }
  const tree = new SearchTree(uniformEvaluator),
    config = { simulations: 20, seed: 1 };
  const first = tree.search(view, config);
  first.value.rank[0] = 999;
  first.policy[0].visits = 999;
  Object.assign(first.action, { type: "place", cell: 99, pending: 99 });
  const next = tree.search(view, config, {
    onRoot: (r) => {
      r.value.rank[0] = 999;
      r.policy[0].visits = 999;
    },
  });
  assert.deepEqual(
    resultStats(next),
    resultStats(search(view, uniformEvaluator, { ...config, simulations: 40 })),
  );
});

test("self-play targets include inherited visits and reset at each turn; agent sessions reset between games", () => {
  const trajectory = collect(uniformEvaluator, {
    seed: 84,
    model: "uniform",
    search: { simulations: 32 },
  });
  assert.equal(trajectory.treeReuse, "within-turn");
  assert.ok(trajectory.samples.some((s) => s.reusedVisits > 0));
  trajectory.samples.forEach((s, i) => {
    assert.equal(s.rootVisits, s.reusedVisits + 32);
    assert.equal(
      s.visits.reduce((a, b) => a + b, 0),
      s.rootVisits,
    );
    assert.ok(Math.abs(s.policy.reduce((a, b) => a + b, 0) - 1) < 1e-10);
    if (!i || trajectory.actions[i - 1].type === "endTurn")
      assert.equal(s.reusedVisits, 0);
  });
  const a = mctsAgent(uniformEvaluator, { simulations: 4 }),
    b = mctsAgent(uniformEvaluator, { simulations: 4 });
  const first = playGame([a, b], 82);
  assert.deepEqual(playGame([a, b], 82), first);
});
