import test from "node:test";
import assert from "node:assert/strict";
import {
  applyAction,
  canPlace,
  CARDS,
  clone,
  createGame,
  dispatch,
  key,
  neighbor,
  scorePlayer,
  undo,
  type Game,
  type Session,
} from "../src/game/engine";
import {
  completionOptions,
  projects,
  applyProject,
} from "../src/game/ai/projects";
import { discount, evaluatePosition, horizon } from "../src/game/ai/evaluate";
import { observe, sampleGame } from "../src/game/ai/observation";
import { legalActions, positionKey } from "../src/game/ai/actions";
import { searchTurn } from "../src/game/ai/search";
import { exportGameLog } from "../src/game/ai/log";

const game = (n = 1, spirits = false, side: "A" | "B" = "A") =>
  createGame({
    names: Array.from({ length: n }, (_, i) => `P${i}`),
    first: 0,
    side,
    spirits,
    seed: 6,
  });
function noMoreTokens(g: Game) {
  g.discardedTokens.push(...g.bag);
  g.bag = [];
  g.turn = { tookCard: true, tookTokens: true, pending: [] };
  return g;
}
function frogHabitat(waters = 1) {
  const g = noMoreTokens(game());
  const p = g.players[0],
    center = p.board[11];
  center.tokens = [4];
  for (let direction = 0; direction < waters; direction++) {
    const coord = neighbor(center, direction);
    p.board.find((c) => key(c) === key(coord))!.tokens = [1];
  }
  return g;
}

test("AI controllers follow player identity after first-player rotation; old setup remains human", () => {
  const g = createGame({
    names: ["A", "B"],
    side: "A",
    spirits: false,
    first: 1,
    controllers: [{ type: "human" }, { type: "ai", ai: { depth: 3, topK: 5 } }],
  });
  assert.equal(g.players[0].name, "B");
  assert.equal(g.players[0].type, "ai");
  assert.equal(g.players[0].ai?.depth, 3);
  assert.equal(g.players[0].ai?.topK, 5);
  assert.ok(game(4).players.every((p) => p.type === "human"));
});
test("completion uses append-only exact heights and expands all three building bases", () => {
  const c = {
    col: 0,
    row: 0,
    tokens: [] as (1 | 2 | 3 | 4 | 5 | 6)[],
    animal: null as number | null,
  };
  assert.deepEqual(completionOptions(c, [6, 7]), [
    [2, 6],
    [3, 6],
    [6, 6],
  ]);
  c.tokens = [3];
  assert.deepEqual(completionOptions(c, [4, 3]), [[4]]);
  c.tokens = [4];
  assert.deepEqual(completionOptions(c, [4, 3]), []);
  c.tokens = [2, 2];
  assert.deepEqual(completionOptions(c, [2]), []);
  c.animal = 1;
  assert.deepEqual(completionOptions(c, [2, 2]), [[]]);
  assert.deepEqual(completionOptions(c, [2, 2, 2]), []);
});
test("discount matches the specified equation and finite resource constraints", () => {
  const terms = discount(
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1],
    6,
  )!;
  assert.equal(terms.m, 1);
  assert.equal(terms.F, 5);
  assert.deepEqual(terms.d, [0, 0, 0, 0, 0, 0]);
  assert.ok(Math.abs(4 * terms.q - 4 * Math.exp(-1 / 6)) < 1e-12);
  assert.equal(
    discount([2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 5, 5, 5, 5, 5], 6),
    null,
  );
  assert.equal(
    discount([0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [5, 5, 5, 5, 5, 5], 1),
    null,
  );
  const scarce = discount(
    [2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [2, 20, 20, 20, 20, 20],
    6,
  )!;
  const plentiful = discount(
    [2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [20, 2, 2, 2, 2, 2],
    6,
  )!;
  assert.ok(plentiful.q > scarce.q);
});
test("current supply is evaluated group by group and never mixed", () => {
  const g = game();
  g.supply = [
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
  ];
  const value = evaluatePosition(observe(g));
  assert.equal(value.H.filter((n) => n > 0).length, 1);
  assert.equal(
    value.H.reduce((a, b) => a + b),
    3,
  );
  assert.ok(value.supplyGroup !== undefined);
});
test("shared support gives two successive marginal tiers without reusing landing spots", () => {
  const g = frogHabitat(2);
  g.players[0].cards = [{ id: 5, placed: 0 }];
  const before = clone(g),
    value = evaluatePosition(observe(g));
  assert.deepEqual(
    value.projects.map((p) => p.G),
    [2, 2],
  );
  assert.equal(value.potential, 4);
  assert.equal(
    new Set(value.projects.map((p) => p.project.targetCell)).size,
    2,
  );
  assert.ok(value.projects.every((p) => p.m === 0 && p.q === 1));
  assert.deepEqual(g, before);
});
test("two cards competing for one target produce only the higher-value settlement", () => {
  const g = frogHabitat();
  g.players[0].cards = [
    { id: 5, placed: 0 },
    { id: 3, placed: 0 },
  ];
  const p = g.players[0],
    water = p.board.find((c) => c.tokens[0] === 1)!;
  const rock = p.board.find(
    (c) =>
      !c.tokens.length &&
      Array.from({ length: 6 }, (_, d) => key(neighbor(water, d))).includes(
        key(c),
      ),
  )!;
  rock.tokens = [2, 2, 2];
  const value = evaluatePosition(observe(g));
  assert.equal(value.projects.length, 1);
  assert.equal(value.projects[0].project.cardId, 3);
  assert.equal(value.potential, 3);
});
test("project deltas include terrain losses and activated nature spirit changes", () => {
  const g = game(),
    p = g.players[0];
  p.board[11].tokens = [2];
  p.board[12].tokens = [2];
  p.cards = [{ id: 39, placed: 1 }];
  const project = [...projects(p)].find(
    (x) =>
      x.kind === "terrain" &&
      x.additions[0].cell === 11 &&
      x.additions[0].colors[0] === 6,
  )!;
  const after = applyProject(p, project);
  assert.ok(scorePlayer(after, "A").total < scorePlayer(p, "A").total);
  for (const candidate of projects(p))
    for (const addition of candidate.additions) {
      const cell = clone(p.board[addition.cell]);
      for (const color of addition.colors) {
        assert.ok(canPlace(cell, color));
        cell.tokens.push(color);
      }
    }
});
test("six-project accounting never double-spends held colors, future colors, or placement budget", () => {
  const g = game();
  g.players[0].cards = [
    { id: 1, placed: 0 },
    { id: 17, placed: 0 },
  ];
  const value = evaluatePosition(observe(g));
  let H = [...value.H],
    A = [...value.A],
    B = value.B;
  assert.ok(value.projects.length <= 6);
  for (const step of value.projects) {
    assert.deepEqual(step.H, H);
    assert.deepEqual(step.A, A);
    assert.equal(step.B, B);
    step.mByColor.forEach((n, c) => {
      const used = Math.min(H[c], n);
      H[c] -= used;
      A[c] -= n - used;
    });
    B -= step.m;
    assert.ok(B >= 0 && H.every((n) => n >= 0) && A.every((n) => n >= 0));
  }
  assert.equal(
    value.value,
    value.actualScore.total +
      value.projects.reduce((total, p) => total + p.W, 0),
  );
  assert.deepEqual(value, evaluatePosition(observe(g)));
});
test("final-round order and terminal evaluation exclude unavailable future actions", () => {
  const g = game(3);
  g.finalRound = 1;
  g.active = 1;
  assert.equal(horizon(observe(g), g.players[0].id).turns, 0);
  assert.equal(evaluatePosition(observe(g), g.players[0].id).potential, 0);
  assert.equal(horizon(observe(g), g.players[2].id).B, 3);
  g.phase = "ended";
  for (const p of g.players)
    assert.equal(
      evaluatePosition(observe(g), p.id).value,
      scorePlayer(p, g.side).total,
    );
});
test("observation excludes hidden order and other players' unchosen spirits", () => {
  const g = game(2, true),
    other = clone(g);
  other.bag.reverse();
  other.deck.reverse();
  other.players[1].spiritChoices = [41, 42];
  assert.deepEqual(observe(g), observe(other));
  const sampled = sampleGame(observe(g), 44);
  assert.deepEqual(
    sampled.players[0].spiritChoices,
    g.players[0].spiritChoices,
  );
  assert.equal(sampled.bag.length, g.bag.length);
  assert.equal(sampled.deck.length, g.deck.length);
  // Different public discard colors imply different resource pools, even with equal bag sizes.
  const a = game(),
    b = clone(a);
  a.discardedTokens.push(...a.bag.splice(a.bag.indexOf(1), 1));
  b.discardedTokens.push(...b.bag.splice(b.bag.indexOf(2), 1));
  assert.notEqual(positionKey(a), positionKey(b));
});
test("timeout produces a full legal turn and bounded top-k logs with all variables", () => {
  let g = game(),
    session: Session = { id: "test", game: g, history: [] };
  const result = searchTurn(
    observe(g),
    { timeMs: 100, topK: 1, depth: 3 },
    { maxNodes: 0, now: () => 0 },
  );
  assert.equal(result.completedDepth, 0);
  assert.ok(result.timedOut);
  assert.equal(result.actions.at(-1)?.type, "endTurn");
  for (const [i, action] of result.actions.entries()) {
    assert.ok(
      legalActions(g).some((a) => JSON.stringify(a) === JSON.stringify(action)),
    );
    g = applyAction(g, action);
    session = dispatch(session, action);
    const decision = result.decisions[i];
    assert.deepEqual(decision.selected.action, action);
    assert.ok(
      Object.values(decision.topByOperation).every((list) => list.length <= 1),
    );
    for (const step of decision.selected.evaluation.projects)
      for (const variable of ["m", "d", "p", "B", "F", "H", "A", "G", "q", "W"])
        assert.ok(variable in step);
    session.aiLog = [
      ...(session.aiLog ?? []),
      { ...decision, gameLogIndex: g.log.length - 1 },
    ];
  }
  assert.equal(g.players[0].turns, 1);
  const output = exportGameLog(session);
  assert.equal(output.aiDecisions.length, result.actions.length);
  assert.equal(output.events.length, result.actions.length);
  assert.equal("bag" in output.currentPosition, false);
  assert.equal("deck" in output.currentPosition, false);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(output)));
});
test("search is deterministic with fixed node budgets and unchanged public information", () => {
  const g = noMoreTokens(game());
  g.turn.tookCard = true;
  const other = clone(g);
  other.deck.reverse();
  const config = { depth: 3, topK: 1, timeMs: 100 };
  const a = searchTurn(observe(g), config, { now: () => 0, maxNodes: 100 });
  const b = searchTurn(observe(other), config, { now: () => 0, maxNodes: 100 });
  assert.deepEqual(a, b);
  assert.equal(a.completedDepth, 1);
  assert.deepEqual(a.actions, [{ type: "endTurn" }]);
});
test("human-only log export and undo remain supported", () => {
  const initial: Session = { game: game(), history: [] };
  const next = dispatch(initial, { type: "takeTokens", group: 0 });
  assert.equal(exportGameLog(next).aiDecisions.length, 0);
  assert.equal(next.events?.length, 1);
  assert.equal(undo(next).events?.length, 0);
});

test("old saved depths use current-turn search without future own turns", () => {
  let g = createGame({
    names: ["AI"],
    side: "A",
    spirits: false,
    first: 0,
    seed: 3,
  });
  g = applyAction(g, { type: "takeCard", slot: 0 });
  g = applyAction(g, { type: "takeTokens", group: 0 });
  for (let cell = 0; cell < 3; cell++)
    g = applyAction(g, { type: "place", pending: 0, cell });
  const shallow = searchTurn(
    observe(g),
    { depth: 1, beamWidth: 1, topK: 1 },
    { now: () => 0, maxNodes: 5000 },
  );
  const deep = searchTurn(
    observe(g),
    { depth: 3, beamWidth: 1, topK: 1 },
    { now: () => 0, maxNodes: 5000 },
  );
  assert.equal(shallow.completedDepth, 1);
  assert.equal(deep.completedDepth, 1);
  assert.deepEqual(deep.actions, shallow.actions);
  assert.equal(deep.nodes, shallow.nodes);
  assert.notEqual(applyAction(g, deep.actions[0]).phase, "ended");
});

test("three AI players choose their own spirits and complete a round with conserved resources", () => {
  let g = game(3, true, "B");
  for (let player = 0; player < 3; player++) {
    const result = searchTurn(
      observe(g),
      { timeMs: 100, topK: 1 },
      { maxNodes: 0 },
    );
    assert.equal(result.actions[0].type, "chooseSpirit");
    for (const action of result.actions) g = applyAction(g, action);
  }
  assert.equal(g.round, 2);
  assert.equal(g.active, 0);
  assert.ok(
    g.players.every((p) => p.turns === 1 && p.spiritChoices.length === 0),
  );
  assert.equal(
    g.bag.length +
      g.supply.flat().length +
      g.players.flatMap((p) => p.board.flatMap((c) => c.tokens)).length,
    120,
  );
});
