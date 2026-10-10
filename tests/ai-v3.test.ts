import test from "node:test";
import assert from "node:assert/strict";
import fixtures from "./fixtures/ai-audit-2026-09-27.json" with { type: "json" };
import {
  applyAction,
  createGame,
  scorePlayer,
  type Game,
} from "../src/game/engine";
import { createEvaluator } from "../src/game/ai/evaluate";
import {
  compareActions,
  freeCardSlots,
  legalActions,
  positionKey,
} from "../src/game/ai/actions";
import { observe, sampleGame } from "../src/game/ai/observation";
import { searchTurn } from "../src/game/ai/search";
import {
  evaluatePosition as referenceFull,
  orderValue,
} from "./helpers/reference-ai-v1";
import type { AIObservation, Evaluation } from "../src/game/ai/types";
const last = () =>
  sampleGame(
    fixtures.find((f) => f.index === 117)!.position as unknown as AIObservation,
    10811,
  );
const root = () =>
  createGame({
    names: ["A", "B"],
    first: 0,
    side: "A",
    spirits: false,
    seed: 11,
  });

test("cached v1-formula evaluator preserves original full V and pruning scores", () => {
  const evaluator = createEvaluator("v1");
  for (const fixture of fixtures) {
    const g = sampleGame(fixture.position as unknown as AIObservation, 173);
    const states = [
      g,
      ...legalActions(g)
        .slice(0, 5)
        .map((a) => applyAction(g, a)),
    ];
    for (const state of states) {
      const id = g.players[g.active].id,
        view = observe(state, id);
      const actual = scorePlayer(
        state.players.find((p) => p.id === id)!,
        state.side,
      ).total;
      assert.equal(
        evaluator.order(view, id).value,
        orderValue(state, id, actual),
      );
      const expected = referenceFull(view, id),
        value = evaluator.evaluate(view, id);
      assert.equal(value.value, expected.value);
      assert.equal(value.B, expected.B);
      assert.deepEqual(
        value.projects.map((p) => [p.project, p.G, p.m, p.d, p.B, p.F, p.W]),
        expected.projects.map((p) => [p.project, p.G, p.m, p.d, p.B, p.F, p.W]),
      );
    }
  }
});

test("logs only dump visited scores; disabling logs or changing top-k does not change fixed-work search", () => {
  const view = observe(last());
  const records = new Map<string, Evaluation>();
  const result = searchTurn(
    view,
    { topK: 3 },
    {
      now: () => 0,
      maxNodes: 500,
      onEvaluation(kind, game, value) {
        records.set(`${kind}:${positionKey(game)}`, value);
      },
    },
  );
  for (const config of [
    { log: false, topK: 1 },
    { log: true, topK: 1 },
    { log: true, topK: 10 },
  ]) {
    const plan = searchTurn(
      view,
      { topK: config.topK },
      { now: () => 0, maxNodes: 500, log: config.log },
    );
    assert.deepEqual(plan.actions, result.actions);
    assert.equal(plan.nodes, result.nodes);
    assert.equal(plan.evaluations, result.evaluations);
    if (!config.log) assert.deepEqual(plan.decisions, []);
  }
  assert.equal(result.decisions[0].before, undefined);
  assert.equal(result.decisions[0].selected.gain, undefined);
  assert.equal(result.decisions[0].planSummary!.rootValue, undefined);
  for (const decision of result.decisions) {
    const state = sampleGame(decision.position, decision.sampleSeed!);
    for (const c of [
      decision.selected,
      ...Object.values(decision.topByOperation).flat(),
    ]) {
      assert.equal(c.scoreKind, c.action.type === "endTurn" ? "full" : "order");
      assert.deepEqual(
        c.evaluation,
        records.get(
          `${c.scoreKind}:${positionKey(applyAction(state, c.action))}`,
        ),
      );
    }
    for (const [type, candidates] of Object.entries(decision.topByOperation)) {
      const all = legalActions(state).filter((a) => a.type === type);
      const scored = all
        .flatMap((action) => {
          const evaluation = records.get(
            `${action.type === "endTurn" ? "full" : "order"}:${positionKey(applyAction(state, action))}`,
          );
          return evaluation
            ? [
                {
                  action,
                  evaluation,
                  freeSlots: freeCardSlots(
                    applyAction(state, action).players.find(
                      (p) => p.id === decision.playerId,
                    )!,
                  ),
                },
              ]
            : [];
        })
        .sort(
          (a, b) =>
            b.evaluation.value - a.evaluation.value ||
            b.freeSlots - a.freeSlots ||
            b.evaluation.actualScore.total - a.evaluation.actualScore.total ||
            compareActions(a.action, b.action),
        );
      assert.deepEqual(
        candidates!.map((c) => c.action),
        scored.slice(0, 3).map((c) => c.action),
      );
    }
  }
  const stats = result.decisions[0].planSummary!.searchStats!;
  assert.equal(stats.logEvaluations, 0);
  assert.ok(stats.duplicateStates > 0);
  assert.ok(stats.orderEvaluations < stats.generatedActions);
});

test("timeout during widening returns best completed plan seen so far", () => {
  for (const maxNodes of [0, 150, 400, 900]) {
    const scores: number[] = [];
    const plan = searchTurn(
      observe(last()),
      { beamWidth: 1 },
      {
        now: () => 0,
        maxNodes,
        onComplete(_, value) {
          scores.push(value);
        },
      },
    );
    let game = last();
    for (const action of plan.actions) {
      assert.ok(
        legalActions(game).some((a) => compareActions(a, action) === 0),
      );
      game = applyAction(game, action);
    }
    assert.equal(game.phase, "ended");
    assert.equal(plan.decisions[0].planSummary!.value, Math.max(...scores));
    assert.equal(
      scorePlayer(game.players[1], game.side).total,
      Math.max(...scores),
    );
    assert.equal(plan.actions.filter((a) => a.type === "endTurn").length, 1);
  }
});

test("all scored states belong to this own turn or its immediate end; widening reuses node scores", () => {
  const g = root(),
    seen = new Set<string>();
  const plan = searchTurn(
    observe(g),
    { beamWidth: 1, depth: 3 },
    {
      now: () => 0,
      maxNodes: 1200,
      onEvaluation(kind, state) {
        const key = `${kind}:${positionKey(state)}`;
        assert.ok(!seen.has(key));
        seen.add(key);
        assert.equal(state.players[1].turns, 0);
        assert.ok(state.players[0].turns <= 1);
        assert.equal(state.round, 1);
        assert.ok(kind !== "full" || state.players[0].turns === 1);
      },
    },
  );
  assert.equal(plan.decisions[0].targetDepth, 1);
  const widths = plan.decisions[0].planSummary!.searchStats!.widthsCompleted;
  assert.ok(widths.length > 0);
  widths.forEach((w, i) => assert.equal(w, 2 ** i));
});

test("current-turn search reaches the independent final-turn oracle", () => {
  const g = last(),
    memo = new Map<string, number>();
  const solve = (state: Game): number => {
    const key = positionKey(state),
      cached = memo.get(key);
    if (cached !== undefined) return cached;
    const value =
      state.phase === "ended"
        ? scorePlayer(state.players[1], state.side).total
        : Math.max(
            ...legalActions(state).map((a) => solve(applyAction(state, a))),
          );
    memo.set(key, value);
    return value;
  };
  const result = searchTurn(
    observe(g),
    { beamWidth: 1 },
    { now: () => 0, maxNodes: 20000 },
  );
  assert.equal(result.decisions[0].planSummary!.value, solve(g));
  assert.equal(result.decisions[0].planSummary!.value, 89);
});
