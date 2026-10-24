import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, clone } from "../src/game/engine";
import { observe, sampleGame } from "../src/game/ai/observation";
import { uniformEvaluator } from "../src/game/ai/rl/network";
import { collect, newGame } from "../src/game/ai/rl/trajectory";
import { terminalValues } from "../src/game/ai/rl/mcts";
import { cheapRollout, moments, rolloutSeeds } from "../scripts/audit/value-variance";

test("conditional variance uses n-1 and is distinct from pooled state variance", () => {
  assert.deepEqual(moments([1, 2, 3]), { n: 3, mean: 2, variance: 1, populationVariance: 2 / 3, min: 1, max: 3 });
  assert.equal(moments([0, 0, 0]).variance, 0);
  assert.equal(moments([10, 10, 10]).variance, 0);
  assert.ok(moments([0, 0, 0, 10, 10, 10]).variance > 0);
});

test("cheap continuations preserve public state, restart trees, and label the starting seat", () => {
  const base = collect(uniformEvaluator, { seed: 14, first: 1, model: "uniform", search: { simulations: 2 } });
  let game = newGame(14, 1);
  for (const action of base.actions.slice(0, -8)) game = applyAction(game, action);
  const view = observe(game), saved = clone(view), seeds = rolloutSeeds(2, 90, 3, 10);
  const a = cheapRollout(view, uniformEvaluator, seeds, 2);
  const b = cheapRollout(view, uniformEvaluator, seeds, 2);
  assert.deepEqual(a, b);
  assert.deepEqual(view, saved);
  const terminal = a.actions.reduce(applyAction, sampleGame(view, seeds.environmentSeed));
  assert.equal(terminal.phase, "ended");
  assert.deepEqual(a.value, terminalValues(terminal, view.active));
  assert.equal(a.addedSimulations, 2 * a.actions.length);
  const next = cheapRollout(view, uniformEvaluator, rolloutSeeds(2, 90, 4, 10), 2);
  assert.notEqual(a.environmentSeed, next.environmentSeed);
  assert.notEqual(a.searchSeed, next.searchSeed);
});

test("rollout RNG streams are unique across source games, states, and repeats", () => {
  const env = new Set<number>(), search = new Set<number>();
  for (let game = 0; game < 100; game++) for (let step = 0; step < 512; step++) for (let repeat = 0; repeat < 10; repeat++) {
    const s = rolloutSeeds(game, step, repeat, 10);
    env.add(s.environmentSeed); search.add(s.searchSeed);
  }
  assert.equal(env.size, 512000); assert.equal(search.size, 512000);
  assert.ok([...env].every((s) => !search.has(s)));
});
