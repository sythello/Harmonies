import test from "node:test";
import assert from "node:assert/strict";
import panther from "./fixtures/ai-panther-2026-10-09.json" with { type: "json" };
import {
  applyAction,
  clone,
  createGame,
  scorePlayer,
  matches,
  canPlace,
  CARDS,
  type Game,
  type Action,
} from "../src/game/engine";
import { createEvaluator } from "../src/game/ai/evaluate";
import { applyProject } from "../src/game/ai/projects";
import {
  freeCardSlots,
  legalActions,
  compareActions,
} from "../src/game/ai/actions";
import { observe, sampleGame } from "../src/game/ai/observation";
import { searchTurn } from "../src/game/ai/search";
import type { AIObservation } from "../src/game/ai/types";
const root = () =>
  sampleGame(panther.position as unknown as AIObservation, panther.sampleSeed);
function endPosition(slot: number | null) {
  let game = root();
  if (slot !== null) game = applyAction(game, { type: "takeCard", slot });
  for (const action of panther.actions.slice(1))
    game = applyAction(game, action as Action);
  return game;
}

test("Panther regression: drafting any market card preserves the no-draft evaluation at the same endpoint", () => {
  const legacy = createEvaluator("v1"),
    evaluator = createEvaluator("v3.1");
  const base = evaluator.evaluate(observe(endPosition(null), 1), 1);
  assert.equal(base.value, panther.oldValue);
  assert.ok(
    legacy.evaluate(observe(endPosition(1), 1), 1).value < base.value - 10,
  );
  assert.ok(
    legacy.evaluate(observe(endPosition(2), 1), 1).value < base.value - 3,
  );
  const orderBefore = evaluator.order(observe(root(), 1), 1).value;
  for (let slot = 0; slot < 5; slot++) {
    const before = root(),
      drafted = applyAction(before, { type: "takeCard", slot });
    assert.ok(evaluator.order(observe(drafted, 1), 1).value >= orderBefore);
    assert.ok(
      evaluator.evaluate(observe(drafted, 1), 1).value >=
        evaluator.evaluate(observe(before, 1), 1).value,
    );
    const game = endPosition(slot),
      view = observe(game, 1),
      value = evaluator.evaluate(view, 1);
    assert.ok(value.value >= base.value);
    assert.deepEqual(
      value.actualScore,
      scorePlayer(game.players[1], game.side),
    );
    assert.equal(
      value.value,
      value.actualScore.total + value.projects.reduce((n, p) => n + p.W, 0),
    );
    let player = clone(game.players[1]);
    for (const step of value.projects) {
      assert.ok(!value.ignoredCards!.includes(step.project.cardId ?? -1));
      const actual = scorePlayer(player, game.side);
      const probe = clone(player);
      for (const addition of step.project.additions)
        for (const color of addition.colors) {
          assert.ok(canPlace(probe.board[addition.cell], color));
          probe.board[addition.cell].tokens.push(color);
        }
      if (step.project.kind === "animal")
        assert.ok(
          matches(probe.board, step.project.cardId!).some(
            (m) => m.target === step.project.targetCell,
          ),
        );
      player = applyProject(player, step.project);
      assert.equal(scorePlayer(player, game.side).total - actual.total, step.G);
    }
    const calls = evaluator.computations;
    assert.deepEqual(evaluator.evaluate(view, 1), value);
    assert.equal(evaluator.computations, calls);
  }
});

test("adding zero-placement cards is monotone over every subset, on both board sides, with a nature spirit", () => {
  for (const side of ["A", "B"] as const) {
    const game = createGame({
      names: ["AI"],
      first: 0,
      side,
      spirits: true,
      seed: 7,
    });
    game.players[0].cards = [{ id: 33, placed: 1 }];
    game.players[0].spiritChoices = [];
    const ids = [21, 25, 32],
      evaluator = createEvaluator("v3.1"),
      values = new Map<number, number>();
    // Evaluate supersets before subsets too: correctness cannot depend on cache warm-up order.
    for (let mask = 7; mask >= 0; mask--) {
      const g = clone(game);
      g.players[0].cards.push(
        ...ids
          .filter((_, i) => mask & (1 << i))
          .map((id) => ({ id, placed: 0 })),
      );
      const result = evaluator.evaluate(observe(g));
      assert.ok(!result.ignoredCards!.includes(33));
      values.set(mask, result.value);
    }
    for (let mask = 0; mask < 8; mask++)
      for (let bit = 0; bit < 3; bit++)
        assert.ok(values.get(mask | (1 << bit))! >= values.get(mask)!);
  }
});

test("an interrupted subset comparison is never cached as a complete monotone value", () => {
  const view = observe(endPosition(1), 1),
    evaluator = createEvaluator("v3.1");
  let calls = 0;
  assert.throws(
    () =>
      evaluator.evaluate(view, 1, () => {
        if (++calls === 2) throw new Error("budget");
      }),
    /budget/,
  );
  const resumed = evaluator.evaluate(view, 1),
    fresh = createEvaluator("v3.1").evaluate(view, 1);
  assert.deepEqual(resumed, fresh);
  assert.ok(resumed.value >= panther.oldValue);
});

test("equal-valued final-turn plans and logged alternatives prefer the actual player's free card slots", () => {
  let game = root();
  for (const action of panther.actions.slice(1, -1))
    game = applyAction(game, action as Action);
  game.finalRound = game.round;
  const free = freeCardSlots(game.players[1]);
  const plan = searchTurn(
    observe(game),
    { topK: 10 },
    { now: () => 0, maxNodes: 1000 },
  );
  assert.deepEqual(plan.actions, [{ type: "endTurn" }]);
  const summary = plan.decisions[0].planSummary!;
  assert.equal(summary.topPlans[0].freeCardSlots, free);
  assert.ok(summary.topPlans.some((p) => p.freeCardSlots! < free));
  assert.ok(summary.topPlans.every((p) => p.value === summary.value));
  assert.equal(summary.searchStats!.logEvaluations, 0);
  let current = game;
  for (const decision of plan.decisions) {
    for (const candidate of [
      decision.selected,
      ...Object.values(decision.topByOperation).flat(),
    ]) {
      const after = applyAction(current, candidate.action);
      assert.equal(candidate.freeCardSlots, freeCardSlots(after.players[1]));
    }
    assert.ok(
      legalActions(current).some(
        (a) => compareActions(a, decision.selected.action) === 0,
      ),
    );
    current = applyAction(current, decision.selected.action);
  }
});

test("completed cards and completed nature spirits release real card slots", () => {
  const p = root().players[1];
  p.cards = [
    { id: 4, placed: CARDS[4].points.length },
    { id: 5, placed: 1 },
    { id: 32, placed: 0 },
    { id: 33, placed: 0 },
  ];
  assert.equal(freeCardSlots(p), 1);
  p.cards[3].placed = CARDS[33].points.length;
  assert.equal(freeCardSlots(p), 2);
});
