import assert from "node:assert/strict";
import { applyAction, type Action } from "../../src/game/engine";
import { observe, sampleGame } from "../../src/game/ai/observation";
import type { AIObservation } from "../../src/game/ai/types";
import type { Evaluator, Values } from "../../src/game/ai/rl/network";
import { terminalValues } from "../../src/game/ai/rl/mcts";
import { mctsAgent, terminalStats } from "../../src/game/ai/rl/trajectory";

export const VALUE_HEADS = ["rank", "relative", "absolute"] as const;

export function moments(values: number[]) {
  assert.ok(values.length >= 2 && values.every(Number.isFinite));
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sumSquared = values.reduce((sum, v) => sum + (v - mean) ** 2, 0);
  return {
    n: values.length,
    mean,
    variance: sumSquared / (values.length - 1),
    populationVariance: sumSquared / values.length,
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

export function valueMoments(values: Values[]) {
  return Object.fromEntries(
    VALUE_HEADS.map((head) => [
      head,
      [0, 1].map((seat) => moments(values.map((v) => v[head][seat]))),
    ]),
  );
}

/** Distinct environment/search streams; independent of worker scheduling/resume. */
export function rolloutSeeds(game: number, step: number, repeat: number, repeats: number) {
  assert.ok(game >= 0 && step >= 0 && step < 512 && repeat >= 0 && repeat < repeats);
  const offset = (game * 512 + step) * repeats + repeat;
  assert.ok(Number.isSafeInteger(offset) && offset < 1_000_000_000);
  return { environmentSeed: 1_000_000_000 + offset, searchSeed: 2_000_000_000 + offset };
}

export function cheapRollout(
  view: AIObservation,
  evaluator: Evaluator,
  seeds: { environmentSeed: number; searchSeed: number },
  simulations = 16,
) {
  const startingSeat = view.active;
  let game = sampleGame(view, seeds.environmentSeed);
  assert.deepEqual(observe(game), view);
  let addedSimulations = 0, reusedVisits = 0, decisions = 0;
  // A new agent/tree for every continuation. Each subsequent own-turn child can reuse it.
  const agent = mctsAgent(evaluator, {
    simulations, seed: seeds.searchSeed, noiseWeight: 0, temperature: 1,
  }, (_view, result) => {
    assert.equal(result.simulations, simulations);
    assert.equal(result.rootVisits - result.reusedVisits, simulations);
    if (decisions === 0) assert.equal(result.reusedVisits, 0);
    addedSimulations += result.simulations;
    reusedVisits += result.reusedVisits;
    decisions++;
  });
  const actions: Action[] = [];
  while (game.phase !== "ended") {
    assert.ok(actions.length < 512, "Continuation exceeded 512 actions; never emit truncated labels");
    const action = agent(observe(game), actions.length);
    game = applyAction(game, action);
    game.log = [];
    actions.push(action);
  }
  assert.equal(decisions, actions.length);
  assert.equal(addedSimulations, actions.length * simulations);
  return {
    ...seeds, actions, addedSimulations, reusedVisits,
    value: terminalValues(game, startingSeat),
    ...terminalStats(game),
    finalRound: game.round,
  };
}
