import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, CARDS } from "../src/game/engine";
import { observe, sampleGame } from "../src/game/ai/observation";
import { ACTION_SIZE } from "../src/game/ai/rl/encoding";
import {
  SearchTree,
  checkActionMask,
  policySummary,
  RLDiagnosticError,
  type RootStats,
} from "../src/game/ai/rl/mcts";
import {
  newGame,
  collect,
  evaluateGame,
  mctsAgent,
  playGame,
  terminalStats,
} from "../src/game/ai/rl/trajectory";
import { ranking } from "../src/game/engine";
import { uniformEvaluator } from "../src/game/ai/rl/network";
import {
  currentScoreEvaluator,
  collectCases,
} from "../scripts/audit/mcts-audit";

test("root diagnostics distinguish backed-up search utility from cached NN utility", () => {
  const view = collectCases(1, 71000).cases[1].input;
  const config = {
    simulations: 128,
    seed: 7,
    rankWeight: 0,
    relativeWeight: 0,
    absoluteWeight: 1,
  };
  const tree = new SearchTree(currentScoreEvaluator);
  const first = tree.search(view, config);
  const d = first.diagnostics;
  assert.equal(d.rootWidth, first.policy.length);
  assert.equal(d.networkValue, first.value.absolute[0]);
  const q =
    first.policy.reduce((s, e) => s + e.q * e.visits, 0) / first.rootVisits;
  assert.ok(Math.abs(d.searchValue - q) < 1e-12);
  assert.ok(d.searchValue > d.networkValue);
  assert.equal(d.inheritedSearchValue, null);
  assert.ok(d.meanDepth >= 1 && d.maxDepth >= d.meanDepth);
  assert.ok(d.visitedWidth <= d.rootWidth);
  assert.equal(d.checkedNodes, first.nodes);
  const next = observe(applyAction(sampleGame(view, 0), first.action));
  let before: RootStats | undefined;
  const second = tree.search(next, config, {
    onRoot: (r) => {
      before = r;
    },
  });
  assert.ok(second.reusedVisits > 0);
  const inherited =
    before!.policy.reduce((s, e) => s + e.q * e.visits, 0) / before!.visits;
  assert.ok(
    Math.abs(second.diagnostics.inheritedSearchValue! - inherited) < 1e-12,
  );
});

test("simulation depth counts atomic edges and network policy excludes root noise", () => {
  const tree = new SearchTree(uniformEvaluator);
  const result = tree.search(observe(newGame(3)), {
    simulations: 1,
    noiseWeight: 0.25,
  });
  assert.equal(result.diagnostics.meanDepth, 1);
  assert.equal(result.diagnostics.maxDepth, 1);
  assert.equal(result.diagnostics.inheritedSearchValue, null);
  const n = result.policy.length;
  assert.ok(Math.abs(result.diagnostics.networkPolicy.top1 - 1 / n) < 1e-12);
  assert.ok(
    Math.abs(result.diagnostics.networkPolicy.entropy - Math.log(n)) < 1e-12,
  );
  assert.equal(result.diagnostics.searchPolicy.top1, 1);
  assert.deepEqual(policySummary([0.5, 0.5, 0]), {
    top1: 0.5,
    top5: 1,
    entropy: Math.log(2),
  });
});

test("mask conflicts fail with a countable diagnostic error; whole-game records carry telemetry", () => {
  for (const ids of [[], [1, 1], [-1], [ACTION_SIZE], [0.5]])
    assert.throws(
      () => checkActionMask(ids),
      (e: unknown) =>
        e instanceof RLDiagnosticError && e.kind === "maskConflict",
    );
  const game = collect(uniformEvaluator, {
    seed: 56,
    model: "uniform",
    search: { simulations: 2 },
  });
  assert.ok(game.rounds! > 0);
  assert.equal(
    game.turns!.reduce((a, b) => a + b, 0),
    game.actions.filter((a) => a.type === "endTurn").length,
  );
  assert.ok(
    game.samples.every(
      (s) => s.diagnostics && s.rootVisits - s.reusedVisits === 2,
    ),
  );
  const replayed = game.actions.reduce(
    applyAction,
    newGame(game.seed, game.first),
  );
  const terminal = terminalStats(replayed);
  assert.deepEqual(game.settledAnimals, terminal.settledAnimals);
  assert.deepEqual(game.completedAnimalCards, terminal.completedAnimalCards);
});

test("terminal counts include fully completed animal cards and exclude spirits", () => {
  const game = newGame(11, 1);
  game.players[0].cards = [
    { id: 1, placed: CARDS[1].points.length },
    { id: 2, placed: CARDS[2].points.length - 1 },
    { id: 33, placed: CARDS[33].points.length },
  ];
  for (const [i, animal] of [1, 1, 2, 33].entries())
    game.players[0].board[i].animal = animal;
  const stats = terminalStats(game);
  assert.deepEqual(stats.settledAnimals, [3, 0]);
  assert.deepEqual(stats.completedAnimalCards, [1, 0]);
  assert.equal(
    stats.scores[0],
    ranking(game).find((r) => r.player.id === 1)!.score.total,
  );
});

test("gatekeeper uses serving actions, pairs seats, and attributes diagnostics to model identity", () => {
  const results = [];
  for (const first of [0, 1]) {
    const options = {
      seed: 61,
      first,
      simulations: 8,
      models: ["same", "same"] as [string, string],
    };
    const tested = evaluateGame(uniformEvaluator, uniformEvaluator, options);
    const actions: unknown[] = [];
    const record = (_view: unknown, r: { action: unknown }) =>
      actions.push(r.action);
    const reference = playGame(
      [
        mctsAgent(
          uniformEvaluator,
          { seed: options.seed, simulations: 8 },
          record,
        ),
        mctsAgent(
          uniformEvaluator,
          { seed: options.seed, simulations: 8 },
          record,
        ),
      ],
      options.seed,
      first,
    );
    assert.deepEqual(tested.actions, actions);
    const terminal = terminalStats(reference);
    assert.deepEqual(tested.settledAnimals, terminal.settledAnimals);
    assert.deepEqual(
      tested.completedAnimalCards,
      terminal.completedAnimalCards,
    );
    assert.equal(tested.search.noiseWeight, 0);
    assert.equal(tested.search.temperature, 0);
    assert.equal(
      tested.candidateScore,
      ranking(reference).find((r) => r.player.id === 0)!.score.total,
    );
    assert.equal(
      tested.incumbentScore,
      ranking(reference).find((r) => r.player.id === 1)!.score.total,
    );
    assert.ok(
      tested.decisions.every((d) => d.playerId === (d.seat + first) % 2),
    );
    assert.ok(
      tested.decisions.every((d) => d.rootVisits - d.reusedVisits === 8),
    );
    assert.ok(tested.decisions.some((d) => d.reusedVisits > 0));
    assert.equal("samples" in tested, false);
    results.push(tested.result);
  }
  assert.equal(results[0] + results[1], 1); // Same model is exactly 50% on a seed pair.
});
