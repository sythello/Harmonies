import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, seededRandom } from "../src/game/engine";
import { observe } from "../src/game/ai/observation";
import { actionId, encode } from "../src/game/ai/rl/encoding";
import { SearchTree, terminalValues } from "../src/game/ai/rl/mcts";
import { uniformEvaluator } from "../src/game/ai/rl/network";
import { collect, newGame } from "../src/game/ai/rl/trajectory";

test("playout caps are drawn per atomic decision; only expensive states get features and terminal labels", () => {
  const options = {
    seed: 56,
    first: 1,
    model: "uniform",
    search: { simulations: 64, seed: 53 },
    playoutCapRandomization: { cheapProbability: 0.75, cheapSimulations: 16 },
  };
  const trajectory = collect(uniformEvaluator, options);
  assert.deepEqual(collect(uniformEvaluator, options), trajectory);
  assert.equal(trajectory.actions.length, trajectory.decisions!.length);
  const random = seededRandom(options.seed ^ 0x6a09e667);
  let game = newGame(options.seed, options.first),
    sampleIndex = 0;
  trajectory.decisions!.forEach((d, step) => {
    const cheap = random() < 0.75;
    assert.equal(d.step, step);
    assert.equal(d.mode, cheap ? "cheap" : "expensive");
    assert.equal(d.simulations, cheap ? 16 : 64);
    assert.equal(d.rootVisits - d.reusedVisits, cheap ? 16 : 64);
    assert.equal(d.noiseWeight, cheap ? 0 : 0.25);
    assert.equal(d.actionId, actionId(observe(game), trajectory.actions[step]));
    if (!cheap) {
      const sample = trajectory.samples[sampleIndex++];
      assert.equal(sample.step, step);
      assert.deepEqual(
        { numeric: sample.numeric, cards: sample.cards },
        encode(observe(game)),
      );
      assert.equal(sample.seat, game.active);
      assert.equal(sample.actionId, d.actionId);
      assert.equal(
        sample.visits.reduce((a, b) => a + b, 0),
        d.rootVisits,
      );
    }
    game = applyAction(game, trajectory.actions[step]);
  });
  assert.equal(game.phase, "ended");
  assert.equal(sampleIndex, trajectory.samples.length);
  assert.ok(sampleIndex > 0 && sampleIndex < trajectory.actions.length);
  for (const s of trajectory.samples)
    assert.deepEqual(s.value, terminalValues(game, s.seat));
});

test("cheap-only games produce no training states; invalid caps fail early", () => {
  const options = { seed: 3, model: "uniform", search: { simulations: 4 } };
  const trajectory = collect(uniformEvaluator, {
    ...options,
    playoutCapRandomization: { cheapProbability: 1, cheapSimulations: 2 },
  });
  assert.equal(trajectory.samples.length, 0);
  assert.ok(
    trajectory.decisions!.every(
      (d) => d.mode === "cheap" && d.noiseWeight === 0,
    ),
  );
  for (const caps of [
    { cheapProbability: NaN, cheapSimulations: 2 },
    { cheapProbability: 1.1, cheapSimulations: 2 },
    { cheapProbability: 0.75, cheapSimulations: 5 },
    { cheapProbability: 0.75, cheapSimulations: 0 },
  ])
    assert.throws(() =>
      collect(uniformEvaluator, { ...options, playoutCapRandomization: caps }),
    );
});

test("switching expensive/cheap preserves child visits but applies only the new root's noise", () => {
  for (const [before, after, budget] of [
    [0.25, 0, 16],
    [0, 0.25, 64],
  ]) {
    const game = newGame(8),
      tree = new SearchTree(uniformEvaluator);
    const first = tree.search(observe(game), {
      simulations: 128,
      noiseWeight: before,
      seed: 7,
    });
    const edge = first.policy.find(
      (e) => e.action.type !== "endTurn" && e.visits > 1,
    )!;
    assert.ok(edge);
    const second = tree.search(observe(applyAction(game, edge.action)), {
      simulations: budget,
      noiseWeight: after,
      seed: 9,
    });
    assert.equal(second.reused, true);
    assert.equal(second.reusedVisits, edge.visits - 1);
    assert.equal(second.rootVisits, second.reusedVisits + budget);
    const priors = second.policy.map((e) => e.prior);
    if (!after)
      assert.ok(priors.every((p) => Math.abs(p - 1 / priors.length) < 1e-12));
    else assert.ok(priors.some((p) => Math.abs(p - 1 / priors.length) > 1e-6));
  }
});
