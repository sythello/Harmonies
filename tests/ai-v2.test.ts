import test from "node:test";
import assert from "node:assert/strict";
import fixtureData from "./fixtures/ai-audit-2026-09-27.json" with { type: "json" };
import {
  applyAction,
  canPlace,
  clone,
  createGame,
  matches,
  scorePlayer,
  type Action,
  type Game,
  type Player,
} from "../src/game/engine";
import { boardRotations } from "../src/game/ai/canonical";
import {
  createEvaluator,
  estimateHorizon,
  evaluatePosition,
  rounded,
} from "../src/game/ai/evaluate";
import {
  compareActions,
  legalActions,
  positionKey,
} from "../src/game/ai/actions";
import { applyProject } from "../src/game/ai/projects";
import { observe, sampleGame } from "../src/game/ai/observation";
import { searchTurn } from "../src/game/ai/search-v2";
import { exportGameLog } from "../src/game/ai/log";
import type { AIDecisionLog, AIObservation } from "../src/game/ai/types";

const fixtures = fixtureData as unknown as {
  index: number;
  position: AIObservation;
  action: Action;
}[];
const fixture = (index: number) => fixtures.find((f) => f.index === index)!;
const last = () => sampleGame(fixture(117).position, 10811);
const setup = (n = 2, side: "A" | "B" = "A") =>
  createGame({
    names: Array.from({ length: n }, (_, i) => `P${i}`),
    side,
    spirits: false,
    first: 0,
    seed: 3,
  });

function assertProjectAccounting(view: AIObservation) {
  const evaluation = evaluatePosition(view),
    id = view.players[view.active].id;
  let player = clone(view.players.find((p) => p.id === id)!);
  for (const step of evaluation.projects) {
    const before = scorePlayer(player, view.side);
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
    const after = scorePlayer(player, view.side);
    assert.equal(step.G, after.total - before.total);
    assert.equal(step.terrainGain, after.landscape - before.landscape);
    assert.equal(step.animalGain, after.animals - before.animals);
    assert.equal(step.spiritGain, after.spirit - before.spirit);
    assert.equal(
      step.G,
      step.terrainGain! + step.animalGain! + step.spiritGain!,
    );
  }
  return evaluation;
}

test("v2 horizon preserves all budgets over ordinary turn boundaries for 1–4 players and both sides", () => {
  for (const side of ["A", "B"] as const)
    for (let n = 1; n <= 4; n++) {
      let g = setup(n, side);
      for (let i = 0; i < n; i++) {
        const beforeTake = estimateHorizon(observe(g));
        g = applyAction(g, { type: "takeTokens", group: 0 });
        assert.deepEqual(estimateHorizon(observe(g)), beforeTake);
        for (let cell = 0; cell < 3; cell++)
          g = applyAction(g, { type: "place", pending: 0, cell });
        const before = estimateHorizon(observe(g));
        g = applyAction(g, { type: "endTurn" });
        const after = estimateHorizon(observe(g));
        assert.equal(after.L, before.L);
        assert.deepEqual(after.B, before.B);
      }
    }
  for (const f of fixtures.filter((f) => f.action.type === "endTurn")) {
    const before = estimateHorizon(f.position);
    const after = estimateHorizon(
      observe(applyAction(sampleGame(f.position, 10), f.action)),
    );
    assert.equal(after.L, before.L);
    assert.deepEqual(after.B, before.B);
    if (f.index === 98) assert.equal(before.B[1], 3);
  }
});
test("v2 horizon models solo partial refills, multiplayer order and last-round zero budgets", () => {
  for (const n of [1, 2, 3, 4])
    for (const bagSize of [0, 3, 6, 9]) {
      let g = setup(n);
      g.bag = g.bag.slice(0, bagSize);
      g.turn = { tookTokens: true, tookCard: false, pending: [] };
      g.supply[0] = [];
      const before = estimateHorizon(observe(g));
      assert.equal(
        before.L,
        g.round + Math.floor(Math.ceil(bagSize / (n === 1 ? 9 : 3)) / n),
      );
      g = applyAction(g, { type: "endTurn" });
      const after = estimateHorizon(observe(g));
      assert.deepEqual(after.B, before.B);
    }
  const g = setup(4);
  g.active = 2;
  g.finalRound = g.round;
  assert.deepEqual(estimateHorizon(observe(g)).B, [0, 0, 3, 3]);
  g.turn = { tookTokens: true, tookCard: true, pending: [] };
  assert.deepEqual(estimateHorizon(observe(g)).B, [0, 0, 0, 3]);
  g.phase = "ended";
  assert.deepEqual(estimateHorizon(observe(g)).B, [0, 0, 0, 0]);
});
test("canonical evaluator is rotation invariant and maps cached projects to legal real cells", () => {
  const evaluator = createEvaluator();
  for (const f of [fixture(9), fixture(47), fixture(117)]) {
    const p = f.position.players[f.position.active];
    const base = evaluator.evaluate(f.position);
    for (const transform of boardRotations(p.board)) {
      const rotated = clone(f.position);
      rotated.players[rotated.active].board = p.board.map((c, i) => ({
        ...c,
        tokens: [...p.board[transform.inverse[i]].tokens],
        animal: p.board[transform.inverse[i]].animal,
      }));
      const fresh = assertProjectAccounting(rotated);
      const cached = evaluator.evaluate(rotated);
      assert.ok(Math.abs(fresh.value - base.value) < 1e-9);
      assert.deepEqual(cached, fresh);
    }
  }
  const g = sampleGame(fixture(9).position, 1);
  const a = evaluatePosition(
    observe(applyAction(g, { type: "place", pending: 2, cell: 0 })),
  );
  const b = evaluatePosition(
    observe(applyAction(g, { type: "place", pending: 2, cell: 22 })),
  );
  assert.equal(a.value, b.value);
  const gB = setup(1, "B");
  gB.players[0].cards = [{ id: 15, placed: 0 }];
  gB.players[0].board[0].tokens = [4];
  const original = evaluatePosition(observe(gB));
  for (const t of boardRotations(gB.players[0].board)) {
    const rotated = clone(gB),
      p = gB.players[0];
    rotated.players[0].board = p.board.map((c, i) => ({
      ...c,
      tokens: p.board[t.inverse[i]].tokens,
      animal: p.board[t.inverse[i]].animal,
    }));
    assert.equal(
      assertProjectAccounting(observe(rotated)).value,
      original.value,
    );
  }
});
test("v2 top-k is the exact ranking of every legal action and exports the actual scoring components", () => {
  const g = last();
  const result = searchTurn(
    observe(g),
    { depth: 1, topK: 3 },
    { now: () => 0, maxNodes: 1000, planId: "audit:117" },
  );
  let current = g;
  for (const d of result.decisions) {
    const all = legalActions(current).map((action) => ({
      action,
      evaluation: evaluatePosition(
        observe(applyAction(current, action), d.playerId),
        d.playerId,
      ),
    }));
    for (const type of new Set(all.map((a) => a.action.type))) {
      const group = all
        .filter((a) => a.action.type === type)
        .sort(
          (a, b) =>
            rounded(b.evaluation.value) - rounded(a.evaluation.value) ||
            b.evaluation.actualScore.total - a.evaluation.actualScore.total ||
            compareActions(a.action, b.action),
        );
      assert.equal(d.legalCounts[type], group.length);
      assert.equal(d.evaluatedCounts[type], group.length);
      assert.deepEqual(
        d.topByOperation[type]!.map((c) => c.action),
        group.slice(0, 3).map((c) => c.action),
      );
      group
        .slice(0, 3)
        .forEach((c, i) =>
          assert.equal(d.topByOperation[type]![i].value, c.evaluation.value),
        );
    }
    assert.ok(
      Math.abs(d.scoreGain! + d.potentialGain! - d.selected.gain!) < 1e-9,
    );
    current = applyAction(current, d.selected.action);
  }
  assert.equal(scorePlayer(current.players[1], current.side).total, 89);
  assert.equal(result.decisions[0].planSummary!.value, 89);
  assert.ok(result.decisions.every((d) => d.planId === "audit:117"));
  const legacy = clone(result.decisions[0]);
  delete legacy.evaluationVersion;
  const output = exportGameLog({
    game: g,
    history: [],
    aiLog: [legacy, ...result.decisions],
  });
  assert.equal(output.version, 3);
  assert.deepEqual(output.evaluationVersions, [
    "local-projects-v1",
    "local-projects-v2",
  ]);
  assert.deepEqual(output.aiDecisions[0].before, legacy.before);
});
test("topK does not change fixed-budget decisions, caches, or search coverage", () => {
  const g = last();
  const results = [1, 3, 5].map((topK) =>
    searchTurn(observe(g), { depth: 2, topK }, { now: () => 0, maxNodes: 300 }),
  );
  for (const result of results.slice(1)) {
    assert.deepEqual(result.actions, results[0].actions);
    assert.equal(result.evaluations, results[0].evaluations);
    assert.equal(result.nodes, results[0].nodes);
    assert.equal(result.completedDepth, results[0].completedDepth);
  }
});
test("mid-expansion and deep-layer interruption never publish partial sibling sets or incomplete plans", () => {
  const g = last();
  const baseline = searchTurn(
    observe(g),
    { depth: 3 },
    { now: () => 0, maxNodes: 0 },
  );
  for (const maxNodes of [
    baseline.evaluations! + 1,
    baseline.evaluations! + 12,
    baseline.evaluations! + 80,
  ]) {
    const plan = searchTurn(
      observe(g),
      { depth: 3 },
      { now: () => 0, maxNodes },
    );
    let state = g;
    assert.equal(plan.actions.at(-1)?.type, "endTurn");
    for (const [i, action] of plan.actions.entries()) {
      const d = plan.decisions[i];
      assert.deepEqual(d.evaluatedCounts, d.legalCounts);
      state = applyAction(state, action);
    }
    assert.equal(state.phase, "ended");
    for (const p of plan.decisions[0].planSummary!.topPlans)
      assert.equal(p.scenarioValues.length, plan.completedDepth > 1 ? 2 : 1);
  }
});
test("last-turn exact oracle remains 89/85/83/89/86", () => {
  const root = last(),
    memo = new Map<string, number>();
  function solve(g: Game): number {
    const key = positionKey(g),
      cached = memo.get(key);
    if (cached !== undefined) return cached;
    const score =
      g.phase === "ended"
        ? scorePlayer(g.players[1], g.side).total
        : Math.max(...legalActions(g).map((a) => solve(applyAction(g, a))));
    memo.set(key, score);
    return score;
  }
  assert.deepEqual(
    legalActions(root)
      .filter((a) => a.type === "takeTokens")
      .map((a) => solve(applyAction(root, a))),
    [89, 85, 83, 89, 86],
  );
});

test("adjacency optimization preserves every scoring field against the frozen rules implementation", async () => {
  const { scorePlayer: referenceScore } =
    await import("./helpers/reference-score");
  const { seededRandom, CARDS } = await import("../src/game/engine");
  const rng = seededRandom(9817);
  for (const side of ["A", "B"] as const)
    for (let iteration = 0; iteration < 120; iteration++) {
      const g = setup(1, side),
        p = g.players[0];
      p.cards = [1, 7, 15, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42].map(
        (id) => ({
          id,
          placed: Math.floor(rng() * (CARDS[id].points.length + 1)),
        }),
      );
      for (const cell of p.board) {
        for (let layer = 0; layer < 3; layer++) {
          const choices = [1, 2, 3, 4, 5, 6].filter((c) =>
            canPlace(cell, c as any),
          );
          if (!choices.length || rng() < 0.25) break;
          cell.tokens.push(choices[Math.floor(rng() * choices.length)] as any);
        }
        if (cell.tokens.length && rng() < 0.25)
          cell.animal = p.cards[Math.floor(rng() * p.cards.length)].id;
      }
      assert.deepEqual(scorePlayer(p, side), referenceScore(p, side));
    }
});

test("every expanded sibling is ranked by the independently recomputed complete V", () => {
  let expansions = 0;
  const g = last();
  searchTurn(
    observe(g),
    { depth: 1, topK: 1 },
    {
      now: () => 0,
      maxNodes: 150,
      onExpanded(state, id, candidates) {
        expansions++;
        assert.equal(candidates.length, legalActions(state).length);
        for (const candidate of candidates) {
          const expected = evaluatePosition(
            observe(applyAction(state, candidate.action), id),
            id,
          );
          assert.equal(candidate.value, expected.value);
        }
      },
    },
  );
  assert.ok(expansions >= 5);
});
test("timeout between sampled futures preserves the last complete depth and its plan scores", () => {
  const view = fixture(98).position;
  const shallow = searchTurn(
    view,
    { depth: 1, beamWidth: 1, topK: 3 },
    { now: () => 0, maxNodes: 10000 },
  );
  let clock = 0,
    interrupted = false;
  const result = searchTurn(
    view,
    { depth: 3, beamWidth: 1, topK: 3 },
    {
      now: () => clock,
      maxNodes: 10000,
      onScenario(depth, candidate, sample) {
        if (depth === 2 && candidate === 0 && sample === 1) {
          interrupted = true;
          clock = 10000;
        }
      },
    },
  );
  assert.ok(interrupted);
  assert.equal(result.completedDepth, 1);
  assert.equal(result.timedOut, true);
  assert.deepEqual(result.actions, shallow.actions);
  assert.deepEqual(
    result.decisions[0].planSummary!.topPlans,
    shallow.decisions[0].planSummary!.topPlans,
  );
});
