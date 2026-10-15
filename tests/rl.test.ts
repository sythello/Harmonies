import test from "node:test";
import assert from "node:assert/strict";
import {
  applyAction,
  clone,
  CARDS,
  ranking,
  seededRandom,
} from "../src/game/engine";
import { legalActions } from "../src/game/ai/actions";
import { observe } from "../src/game/ai/observation";
import {
  actionId,
  ACTION_SIZE,
  CARD_INPUTS,
  encode,
  FEATURE_SIZE,
  VERSION,
  schema,
} from "../src/game/ai/rl/encoding";
import {
  search,
  searchConfig,
  selectionWeights,
  terminalValues,
} from "../src/game/ai/rl/mcts";
import {
  uniformEvaluator,
  Network,
  type Dense,
  validateWeights,
  type Evaluator,
  type Weights,
} from "../src/game/ai/rl/network";
import { collect, newGame } from "../src/game/ai/rl/trajectory";
import storedSchema from "../rl/schema.json" with { type: "json" };

test("sparse NN inference preserves dense summation and predictions exactly", () => {
  const random = seededRandom(19);
  const dense = (input: number, output: number): Dense => ({
    weight: Array.from({ length: output }, () =>
      Array.from({ length: input }, () => random() - 0.5),
    ),
    bias: Array.from({ length: output }, () => random() - 0.5),
  });
  const weights: Weights = {
    version: VERSION,
    embedding: Array.from({ length: 33 }, (_, i) => [
      i ? random() : 0,
      i ? random() : 0,
    ]),
    trunk: [dense(FEATURE_SIZE + CARD_INPUTS * 2, 16), dense(16, 8)],
    heads: {
      policy: dense(8, ACTION_SIZE),
      rank: dense(8, 2),
      relative: dense(8, 2),
      absolute: dense(8, 2),
    },
  };
  const network = new Network(weights);
  const legacy = (x: number[], layer: Dense) =>
    layer.weight.map((row, i) => {
      let y = layer.bias[i];
      for (let j = 0; j < x.length; j++) y += row[j] * x[j];
      return y;
    });
  let game = newGame(19);
  for (let i = 0; i < 24; i++) {
    const view = observe(game),
      f = encode(view);
    let x = [...f.numeric, ...f.cards.flatMap((id) => weights.embedding[id])];
    for (const layer of weights.trunk)
      x = legacy(x, layer).map((v) => Math.max(0, v));
    const expected = {
      logits: legacy(x, weights.heads.policy),
      value: {
        rank: legacy(x, weights.heads.rank).map(Math.tanh),
        relative: legacy(x, weights.heads.relative),
        absolute: legacy(x, weights.heads.absolute),
      },
    };
    assert.deepEqual(network.predict(view), expected);
    const legal = legalActions(game);
    game = applyAction(game, legal[Math.floor(random() * legal.length)]);
  }
});

test("RL schema includes full stacks, occupation, all cards, market, pending and seat", () => {
  assert.deepEqual(schema, storedSchema);
  const game = newGame(1),
    features = encode(observe(game));
  assert.equal(features.numeric.length, FEATURE_SIZE);
  assert.equal(features.cards.length, CARD_INPUTS);
  assert.equal(FEATURE_SIZE, 1614);
  assert.equal(CARD_INPUTS, 5);
  assert.equal(FEATURE_SIZE + CARD_INPUTS * 8, 1654);
  assert.ok(features.numeric.every(Number.isFinite));
  game.players[0].board[0].tokens = [2];
  const one = encode(observe(game));
  game.players[0].board[0].tokens = [2, 2];
  assert.notDeepEqual(encode(observe(game)), one);
  const two = encode(observe(game));
  game.players[0].board[0].animal = 1;
  assert.notDeepEqual(encode(observe(game)), two);
  game.players[0].cards = [
    { id: 1, placed: CARDS[1].points.length },
    { id: 2, placed: 0 },
  ];
  assert.deepEqual(
    encode(observe(game)).cards,
    game.market.map((id) => id ?? 0),
  );
  const cards = encode(observe(game));
  game.players[0].cards[1].placed++;
  assert.notDeepEqual(encode(observe(game)), cards);
  const before = encode(observe(game));
  game.turn.pending = [2];
  assert.notDeepEqual(encode(observe(game)), before);
  const view = observe(game);
  view.players.reverse();
  view.active = 1;
  // Same active player and opponent features; only seat one-hot differs.
  assert.equal(
    encode(view).numeric.filter(
      (v, i) => v !== encode(observe(game)).numeric[i],
    ).length,
    2,
  );
});

test("owned-card rows stay tied to IDs through acquisition, completion and reordering", () => {
  const game = newGame(1);
  game.players[0].cards = [
    { id: 8, placed: 1 },
    { id: 20, placed: 0 },
    { id: 32, placed: 0 },
  ];
  game.players[1].cards = [
    { id: 2, placed: 0 },
    { id: 4, placed: 1 },
  ];
  const rows = (seat: number) => {
    // Each player block: 23 x 29 board one-hot, 32 x 3 owned-card fields, 3 scalars.
    const start = seat * 766 + 667;
    return encode(observe(game)).numeric.slice(start, start + 96);
  };
  const row = (seat: number, id: number) =>
    rows(seat).slice((id - 1) * 3, id * 3);
  assert.deepEqual(row(0, 1), [0, 0, 0]);
  assert.deepEqual(row(0, 8), [1, 1, CARDS[8].points.length]);
  assert.deepEqual(row(0, 32), [1, 0, CARDS[32].points.length]);
  assert.deepEqual(row(0, 2), [0, 0, 0]);
  assert.deepEqual(row(1, 2), [1, 0, CARDS[2].points.length]);
  assert.deepEqual(row(1, 4), [1, 1, CARDS[4].points.length]);
  const before = rows(0),
    opponent = rows(1);
  game.players[0].cards.push({ id: 5, placed: 0 });
  assert.deepEqual(row(0, 5), [1, 0, CARDS[5].points.length]);
  for (let id = 1; id <= 32; id++)
    if (id !== 5)
      assert.deepEqual(row(0, id), before.slice((id - 1) * 3, id * 3));
  assert.deepEqual(rows(1), opponent);
  game.players[0].cards.find((c) => c.id === 20)!.placed =
    CARDS[20].points.length;
  assert.deepEqual(row(0, 20), [
    1,
    CARDS[20].points.length,
    CARDS[20].points.length,
  ]);
  const encoded = encode(observe(game));
  game.players.forEach((p) => p.cards.reverse());
  assert.deepEqual(encode(observe(game)), encoded);
  assert.deepEqual(
    encoded.cards,
    game.market.map((id) => id ?? 0),
  );
  game.market[0] = null;
  assert.equal(encode(observe(game)).cards[0], 0);
});

test("old owned-card embedding checkpoints are rejected before inference", () => {
  assert.throws(
    () => validateWeights({ version: "harmonies-rl-v1" } as unknown as Weights),
    /Incompatible RL model/,
  );
});

test("RL rejects unsupported variants and invalid budgets", () => {
  const view = observe(newGame(1));
  assert.throws(() => search({ ...view, side: "B" }, uniformEvaluator));
  assert.throws(() => search({ ...view, spirits: true }, uniformEvaluator));
  assert.throws(() =>
    search({ ...view, players: [view.players[0]] }, uniformEvaluator),
  );
  for (const simulations of [0, -1, NaN, 1.5])
    assert.throws(() => searchConfig({ simulations }));
  assert.throws(() => searchConfig({ temperature: -1 }));
});

test("action IDs are legal, unique, stable across equivalent pending permutations", () => {
  const game = newGame(9);
  game.turn = { tookTokens: true, tookCard: true, pending: [2, 2, 3] };
  const ids = legalActions(game).map((a) => actionId(observe(game), a));
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.every((i) => i >= 0 && i < ACTION_SIZE));
  game.turn.pending = [3, 2, 2];
  assert.deepEqual(
    legalActions(game)
      .map((a) => actionId(observe(game), a))
      .sort((a, b) => a - b),
    ids.sort((a, b) => a - b),
  );
});

test("MCTS legal mask excludes arbitrarily large illegal logits and is seed deterministic", () => {
  const game = newGame(7),
    view = observe(game);
  const legal = new Set(legalActions(game).map((a) => actionId(view, a)));
  const evaluator: Evaluator = {
    predict: (v) => ({
      ...uniformEvaluator.predict(v),
      logits: Array.from({ length: ACTION_SIZE }, (_, i) =>
        legal.has(i) ? 0 : 1e30,
      ),
    }),
  };
  const config = {
    simulations: 12,
    noiseWeight: 0.25,
    temperature: 1,
    seed: 3,
  };
  const result = search(view, evaluator, config);
  assert.deepEqual(result, search(view, evaluator, config));
  assert.equal(
    result.policy.reduce((n, e) => n + e.visits, 0),
    12,
  );
  assert.ok(
    Math.abs(result.policy.reduce((n, e) => n + e.target, 0) - 1) < 1e-10,
  );
  assert.deepEqual(new Set(result.policy.map((e) => e.id)), legal);
  assert.ok(legal.has(result.actionId));
  assert.doesNotThrow(() => applyAction(game, result.action));
  assert.deepEqual(view, observe(game));
});

test("real deck/bag order is invisible, and refills have independent child states", () => {
  let game = newGame(3);
  game = applyAction(game, { type: "takeTokens", group: 0 });
  for (let cell = 0; cell < 3; cell++)
    game = applyAction(game, { type: "place", pending: 0, cell });
  game.turn.tookCard = true;
  const other = clone(game);
  other.bag.reverse();
  other.deck.reverse();
  assert.deepEqual(observe(game), observe(other));
  const a = search(observe(game), uniformEvaluator, { simulations: 20 });
  assert.deepEqual(
    a,
    search(observe(other), uniformEvaluator, { simulations: 20 }),
  );
  assert.ok(a.chanceOutcomes > 1);
});

test("backup uses acting-player identity across consecutive actions and turn changes", () => {
  const evaluator: Evaluator = {
    predict: (v) => ({
      ...uniformEvaluator.predict(v),
      value: { rank: [0.75, -0.5], relative: [0, 0], absolute: [0.2, 0.9] },
    }),
  };
  const game = newGame(4);
  game.turn.tookCard = true;
  const config = { simulations: 1, relativeWeight: 0, absoluteWeight: 0 };
  const same = search(observe(game), evaluator, config);
  assert.equal(same.policy.find((e) => e.visits)!.q, 0.75);
  game.turn.tookTokens = true;
  const switched = search(observe(game), evaluator, config);
  assert.equal(switched.policy.find((e) => e.visits)!.q, -0.5);
  const absolute = search(observe(game), evaluator, {
    ...config,
    rankWeight: 0,
    absoluteWeight: 1,
  });
  assert.equal(absolute.policy.find((e) => e.visits)!.q, 0.9);
});

test("terminal labels respect animal-count tiebreak, draws, and absolute scores", () => {
  const game = newGame(1);
  game.phase = "ended";
  assert.deepEqual(terminalValues(game).rank, [0, -0]);
  game.players[0].board[0].tokens = [1];
  game.players[0].board[0].animal = 1;
  game.players[1].board[0].tokens = [1];
  assert.deepEqual(terminalValues(game, 0).rank, [1, -1]);
  assert.deepEqual(terminalValues(game, 1).rank, [-1, 1]);
  assert.deepEqual(terminalValues(game).relative, [0, -0]);
  game.phase = "playing";
  game.active = 1;
  game.finalRound = 1;
  game.turn = { tookTokens: true, tookCard: true, pending: [] };
  const result = search(observe(game), uniformEvaluator, {
    simulations: 1,
    relativeWeight: 0,
    absoluteWeight: 0,
  });
  assert.equal(result.policy[0].q, -1);
});

test("policy target removes excess exploratory visits based on Q", () => {
  const stats = [
    { visits: 80, prior: 0.5, q: 0.8 },
    { visits: 20, prior: 0.5, q: -0.8 },
  ];
  const weights = selectionWeights(stats, searchConfig());
  assert.equal(weights[0], 80);
  assert.ok(weights[1] < 20);
  assert.ok(weights[1] >= 0);
  assert.equal(
    selectionWeights(
      [
        { visits: 1, prior: 0.01, q: 0 },
        { visits: 0, prior: 0.99, q: 1 },
      ],
      searchConfig(),
    )[0],
    1,
  );
});

test("complete self-play replays exactly and supplies only terminal value labels", () => {
  const trajectory = collect(uniformEvaluator, {
    seed: 11,
    first: 1,
    model: "uniform",
    search: { simulations: 2 },
  });
  let game = newGame(trajectory.seed, trajectory.first);
  trajectory.actions.forEach((action, i) => {
    const row = trajectory.samples[i],
      view = observe(game);
    assert.deepEqual({ numeric: row.numeric, cards: row.cards }, encode(view));
    assert.deepEqual(
      row.legal,
      legalActions(game).map((a) => actionId(view, a)),
    );
    assert.equal(row.actionId, actionId(view, action));
    game = applyAction(game, action);
  });
  assert.equal(game.phase, "ended");
  assert.deepEqual(
    trajectory.scores,
    game.players.map(
      (p) => ranking(game).find((r) => r.player.id === p.id)!.score.total,
    ),
  );
  for (const row of trajectory.samples)
    assert.deepEqual(row.value, terminalValues(game, row.seat));
  assert.throws(
    () =>
      collect(uniformEvaluator, {
        seed: 1,
        model: "uniform",
        maxActions: 1,
        search: { simulations: 1 },
      }),
    /maxActions/,
  );
});
