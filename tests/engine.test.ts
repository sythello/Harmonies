import test from "node:test";
import assert from "node:assert/strict";
import {
  CARDS,
  activeCards,
  applyAction,
  canPlace,
  clone,
  createGame,
  dispatch,
  emptyBoard,
  isComplete,
  key,
  matches,
  neighbor,
  patternCoordinates,
  ranking,
  riverLength,
  scorePlayer,
  seededRandom,
  soloSuns,
  spiritScore,
  stackMatches,
  undo,
  type Cell,
  type Color,
  type Game,
  type Player,
  type Session,
} from "../src/game/engine";

const setup = (count = 2, spirits = false, side: "A" | "B" = "A") =>
  createGame({
    names: Array.from({ length: count }, (_, i) => `玩家${i + 1}`),
    side,
    spirits,
    first: 0,
    seed: 2024,
  });
const cell = (
  tokens: Color[],
  col = 0,
  row = 0,
  animal: number | null = null,
): Cell => ({ tokens, col, row, animal });
const player = (board: Cell[]): Player => ({
  id: 0,
  name: "旅人",
  type: "human",
  board,
  cards: [],
  spiritChoices: [],
  turns: 0,
});
function ready(game: Game) {
  let g = game;
  if (g.players[g.active].spiritChoices.length)
    g = applyAction(g, {
      type: "chooseSpirit",
      card: g.players[g.active].spiritChoices[0],
    });
  g = applyAction(g, {
    type: "takeTokens",
    group: g.supply.findIndex((s) => s.length === 3),
  });
  while (g.turn.pending.length) {
    const c = g.players[g.active].board.findIndex((c) =>
      canPlace(c, g.turn.pending[0]),
    );
    g = applyAction(g, { type: "place", cell: c, pending: 0 });
  }
  return g;
}
test("setup uses the exact finite bag, 32 animals, 10 spirits and the two official board geometries", () => {
  for (let n = 1; n <= 4; n++)
    for (const side of ["A", "B"] as const) {
      const g = setup(n, true, side);
      const tokens = [...g.bag, ...g.supply.flat()];
      assert.deepEqual(
        [1, 2, 3, 4, 5, 6].map((c) => tokens.filter((x) => x === c).length),
        [23, 23, 21, 19, 19, 15],
      );
      assert.equal(g.market.length, n === 1 ? 3 : 5);
      assert.equal(g.supply.length, n === 1 ? 3 : 5);
      assert.equal(g.deck.length + g.market.length, 32);
      assert.equal(g.players[0].board.length, side === "A" ? 23 : 28);
      assert.equal(
        new Set(g.players.flatMap((p) => p.spiritChoices)).size,
        n * 2,
      );
    }
  assert.equal(Object.keys(CARDS).length, 42);
});
test("starting-player ordering and all setup inputs are validated", () => {
  const g = createGame({
    names: ["一", "二", "三"],
    side: "B",
    spirits: false,
    first: 1,
    seed: 1,
  });
  assert.deepEqual(
    g.players.map((p) => p.name),
    ["二", "三", "一"],
  );
  assert.throws(() =>
    createGame({ names: [], side: "A", spirits: false, first: 0 }),
  );
});
test("all legal and illegal stacking combinations, including blocked animal cells", () => {
  for (const color of [1, 2, 3, 4, 5, 6] as Color[])
    assert.ok(canPlace(cell([]), color));
  const legal: [[Color[], Color], ...[Color[], Color][]] = [
    [[2], 2],
    [[2, 2], 2],
    [[3], 3],
    [[3], 4],
    [[3, 3], 4],
    [[2], 6],
    [[3], 6],
    [[6], 6],
  ];
  for (const [stack, c] of legal) assert.ok(canPlace(cell(stack), c));
  for (const [stack, c] of [
    [[2, 2, 2], 2],
    [[3, 3], 3],
    [[4], 4],
    [[3, 4], 6],
    [[2, 2], 6],
    [[6, 6], 6],
    [[1], 1],
    [[5], 6],
    [[3], 2],
    [[6], 4],
  ] as [Color[], Color][])
    assert.equal(canPlace(cell(stack), c), false);
  for (const color of [1, 2, 3, 4, 5, 6] as Color[])
    assert.equal(canPlace(cell([2], 0, 0, 21), color), false);
});
test("building patterns require completed two-level buildings with any permitted base", () => {
  for (const base of [2, 3, 6] as Color[])
    assert.ok(stackMatches(cell([base, 6]), [6, 7]));
  assert.equal(stackMatches(cell([6]), [6, 7]), false);
});
test("all 42 habitat definitions match in all six rotations with exact target and height", () => {
  for (const card of Object.values(CARDS))
    for (let rotation = 0; rotation < 6; rotation++) {
      const points = patternCoordinates(card, { col: 3, row: 3 }, rotation);
      const board = points.map((p, i) => ({
        ...p,
        tokens:
          card.pattern[i].colors[1] === 7
            ? ([2, 6] as Color[])
            : ([...card.pattern[i].colors].reverse() as Color[]),
        animal: null,
      }));
      const expected = card.pattern.findIndex((p) => p.allowCube);
      assert.ok(
        matches(board, card.id).some((m) => m.target === expected),
        `card ${card.id} rotation ${rotation}`,
      );
      board[expected].tokens = [];
      assert.equal(matches(board, card.id).length, 0);
    }
});
test("a cube only blocks its own location; supporting habitats can be shared", () => {
  const origin = { col: 2, row: 2 },
    n = neighbor(origin, 3);
  const b = [cell([4], origin.col, origin.row, 18), cell([1], n.col, n.row)];
  assert.equal(matches(b, 5).length, 1);
  b[1].animal = 6;
  assert.equal(matches(b, 5).length, 0);
});
test("asymmetric patterns cannot be mirrored", () => {
  const card = clone(CARDS[2]);
  card.id = 999;
  card.pattern[0].colors = [4];
  CARDS[999] = card;
  const p = patternCoordinates(card, { col: 3, row: 3 }, 0);
  const b = p.map((v, i) => ({
    ...v,
    tokens: card.pattern[i].colors as Color[],
    animal: null,
  }));
  const reflected = b.map((c) => ({ ...c, row: -c.row - (c.col % 2) }));
  assert.ok(matches(b, 999).length);
  assert.equal(matches(reflected, 999).length, 0);
  delete CARDS[999];
});
test("mandatory tokens and optional actions interleave; market only refills at turn end", () => {
  let g = setup();
  g.market[0] = 5;
  g.supply[0] = [4, 1, 2];
  g = applyAction(g, { type: "takeTokens", group: 0 });
  g = applyAction(g, { type: "place", pending: 0, cell: 11 });
  g = applyAction(g, { type: "takeCard", slot: 0 });
  assert.equal(g.market[0], null);
  assert.throws(() => applyAction(g, { type: "takeCard", slot: 1 }));
  assert.throws(() => applyAction(g, { type: "endTurn" }));
  const target = g.players[0].board.findIndex(
    (c) => key(c) === key(neighbor(g.players[0].board[11], 3)),
  );
  g = applyAction(g, { type: "place", pending: 0, cell: target });
  g = applyAction(g, { type: "settle", card: 5, cell: target });
  assert.equal(g.players[0].cards[0].placed, 1);
  assert.equal(scorePlayer(g.players[0], "A").animals, 2);
  assert.throws(() =>
    applyAction(g, { type: "settle", card: 5, cell: target }),
  );
  g = applyAction(g, { type: "place", pending: 0, cell: 0 });
  g = applyAction(g, { type: "endTurn" });
  assert.ok(g.market[0]);
  assert.equal(g.active, 1);
});
test("completed cards free one of four active slots; spirits count until settled", () => {
  let g = setup(2, true);
  assert.throws(() => applyAction(g, { type: "takeTokens", group: 0 }));
  g = applyAction(g, {
    type: "chooseSpirit",
    card: g.players[0].spiritChoices[0],
  });
  g.players[0].cards.push(
    { id: 1, placed: 0 },
    { id: 2, placed: 0 },
    { id: 3, placed: 0 },
  );
  assert.equal(activeCards(g.players[0]).length, 4);
  assert.throws(() => applyAction(g, { type: "takeCard", slot: 0 }));
  g.players[0].cards[1].placed = 3;
  assert.equal(activeCards(g.players[0]).length, 3);
  assert.ok(applyAction(g, { type: "takeCard", slot: 0 }));
});
test("undo restores exact supply, cards, cubes and first-turn spirit choice, but never crosses turns", () => {
  const initial = setup(2, true);
  let s: Session = { game: initial, history: [] };
  s = dispatch(s, {
    type: "chooseSpirit",
    card: initial.players[0].spiritChoices[0],
  });
  const before = s.game;
  s = dispatch(s, { type: "takeTokens", group: 0 });
  s = dispatch(s, { type: "place", pending: 0, cell: 0 });
  assert.equal(undo(s).game.turn.pending.length, 3);
  assert.deepEqual(undo(s, true).game, initial);
  assert.deepEqual(undo(undo(s)).game, before);
  s = { game: ready(initial), history: [initial] };
  s = dispatch(s, { type: "endTurn" });
  assert.equal(s.history.length, 0);
  assert.deepEqual(undo(s), s);
});
test("official score-sheet example: trees 4, mountains 10, fields 10, buildings 10, river 19 = 53", () => {
  const columns: Color[][][] = [
    [[5], [5], [1], [1], [4]],
    [[1], [1], [5], [1]],
    [[1], [2, 6], [5], [5], [1]],
    [[5], [6, 6], [2], [3, 4]],
    [[2, 2, 2], [2], [2], [3, 6], [2]],
  ];
  const board = emptyBoard("A");
  board.forEach((c) => (c.tokens = columns[c.col][c.row]));
  const p = player(board);
  const s = scorePlayer(p, "A");
  assert.deepEqual(
    [s.trees, s.mountains, s.fields, s.buildings, s.water, s.landscape],
    [4, 10, 10, 10, 19, 53],
  );
  p.cards = [
    { id: 25, placed: 1 },
    { id: 9, placed: 2 },
    { id: 17, placed: 1 },
    { id: 30, placed: 2 },
    { id: 7, placed: 3 },
    { id: 3, placed: 4 },
  ];
  assert.equal(scorePlayer(p, "A").animals, 63);
  assert.equal(scorePlayer(p, "A").total, 116);
});
test("river takes graph diameter by shortest paths, not longest winding path", () => {
  const center = { col: 3, row: 3 };
  const ring = Array.from({ length: 6 }, (_, i) => ({
    ...neighbor(center, i),
    tokens: [1] as Color[],
    animal: null,
  }));
  assert.equal(riverLength(ring), 4);
  assert.equal(scorePlayer(player(ring), "A").water, 8);
  assert.equal(scorePlayer(player([cell([1])]), "A").water, 0);
});
test("islands include empty spaces, with a minimum of one island", () => {
  const b = emptyBoard("B");
  assert.equal(scorePlayer(player(b), "B").water, 5);
  b.filter((c) => c.col === 3).forEach((c) => (c.tokens = [1]));
  assert.equal(scorePlayer(player(b), "B").water, 10);
});
test("fields form connected groups and single red tokens do not score as buildings", () => {
  const p = player([
    cell([5]),
    cell([5], 0, 1),
    cell([5], 0, 2),
    cell([5], 5, 5),
    cell([6], 1, 1),
  ]);
  assert.equal(scorePlayer(p, "A").fields, 5);
  assert.equal(scorePlayer(p, "A").buildings, 0);
});
test("all ten spirit scoring functions, including the corrected stork value of six", () => {
  const b = [
    cell([5], 0, 0),
    cell([5], 0, 1),
    cell([5], 0, 2),
    cell([5], 5, 0),
    cell([4], 8, 0),
    cell([3, 4], 8, 1),
    cell([3, 3, 4], 8, 2),
    cell([6, 6], 11, 0),
    cell([3, 6], 11, 1),
    cell([2, 6], 15, 0),
    cell([6], 17, 0),
    cell([2], 20, 0),
    cell([2, 2], 20, 1),
    cell([2, 2, 2], 20, 2),
    cell([1], 24, 0),
    cell([1], 24, 1),
    cell([1], 28, 0),
  ];
  assert.deepEqual(
    Array.from({ length: 10 }, (_, i) => spiritScore(b, i + 33)),
    [12, 10, 8, 7, 8, 6, 8, 7, 7, 6],
  );
  const p = player(b);
  p.cards = [{ id: 33, placed: 0 }];
  assert.equal(scorePlayer(p, "A").spirit, 0);
  p.cards[0].placed = 1;
  assert.equal(scorePlayer(p, "A").spirit, 12);
});
test("solo discards six unused tokens permanently and optional card refresh is end-turn-only", () => {
  let g = ready(setup(1));
  const old = g.market[1],
    next = g.deck[0],
    bag = g.bag.length;
  g = applyAction(g, { type: "endTurn", discardSlot: 1 });
  assert.equal(g.discardedTokens.length, 6);
  assert.equal(g.bag.length, bag - 9);
  assert.equal(g.market[1], next);
  assert.deepEqual(g.discardedCards, [old]);
  assert.equal(g.round, 2);
  let drafted = setup(1);
  drafted = applyAction(drafted, { type: "takeCard", slot: 0 });
  drafted = ready(drafted);
  assert.throws(() =>
    applyAction(drafted, { type: "endTurn", discardSlot: 1 }),
  );
  assert.throws(() =>
    applyAction(ready(setup()), { type: "endTurn", discardSlot: 1 }),
  );
});
test("solo final three tokens form a final supply group; no tokens return to the bag", () => {
  let g = ready(setup(1));
  g.bag = [2, 3, 4];
  g = applyAction(g, { type: "endTurn" });
  assert.deepEqual(g.supply, [[2, 3, 4], [], []]);
  assert.equal(g.phase, "playing");
  g = ready(g);
  g = applyAction(g, { type: "endTurn" });
  assert.equal(g.phase, "ended");
});
test("board completion finishes the current round, not an extra round; all get equal turns", () => {
  let g = setup(3);
  g.players[0].board.slice(0, 21).forEach((c) => (c.tokens = [1]));
  g.turn = { tookTokens: true, tookCard: false, pending: [] };
  g = applyAction(g, { type: "endTurn" });
  assert.equal(g.finalRound, 1);
  assert.equal(g.active, 1);
  for (let i = 0; i < 2; i++) g = applyAction(ready(g), { type: "endTurn" });
  assert.equal(g.phase, "ended");
  assert.deepEqual(
    g.players.map((p) => p.turns),
    [1, 1, 1],
  );
  assert.throws(() => applyAction(g, { type: "takeTokens", group: 0 }));
});
test("bag becomes empty after a refill, then triggers at the following refill", () => {
  let g = ready(setup(2));
  g.bag = [1, 2, 3];
  g = applyAction(g, { type: "endTurn" });
  assert.equal(g.finalRound, null);
  g = applyAction(ready(g), { type: "endTurn" });
  assert.equal(g.phase, "ended");
});
test("score tie breaks by animal cubes only; further ties share rank", () => {
  const g = setup(3);
  g.players[0].board = [cell([1], 0, 0, 1)];
  g.players[1].board = [cell([1], 0, 0, 33)];
  g.players[2].board = [cell([1], 0, 0, 2)];
  assert.deepEqual(
    ranking(g).map((r) => [r.player.id, r.rank]),
    [
      [0, 1],
      [2, 1],
      [1, 3],
    ],
  );
});
test("solo suns use board and chosen spirit category even if spirit never settles", () => {
  const p = player([]);
  assert.deepEqual(soloSuns(p, "A"), {
    base: 0,
    sideBonus: 1,
    spiritBonus: 2,
    total: 3,
  });
  p.cards = [{ id: 33, placed: 0 }];
  assert.equal(soloSuns(p, "B").total, 1);
  p.cards = [{ id: 35, placed: 0 }];
  assert.equal(soloSuns(p, "B").total, 0);
});
test("seeded complete games preserve finite components, legal stacks, card ownership and equal turns", () => {
  for (let n = 1; n <= 4; n++)
    for (const side of ["A", "B"] as const)
      for (const spirits of [false, true])
        for (let seed = 1; seed <= 4; seed++) {
          let g = createGame({
              names: Array.from({ length: n }, (_, i) => `P${i}`),
              side,
              spirits,
              first: seed % n,
              seed,
            }),
            actions = 0;
          const rng = seededRandom(seed + 10);
          while (g.phase === "playing" && actions++ < 1000) {
            let p = g.players[g.active];
            if (p.spiritChoices.length)
              g = applyAction(g, {
                type: "chooseSpirit",
                card: p.spiritChoices[0],
              });
            p = g.players[g.active];
            if (activeCards(p).length < 4) {
              const slot = g.market.findIndex((c) => c !== null);
              if (slot >= 0) g = applyAction(g, { type: "takeCard", slot });
            }
            const choices = g.supply
              .map((v, i) => (v.length === 3 ? i : -1))
              .filter((i) => i >= 0);
            assert.ok(choices.length);
            g = applyAction(g, {
              type: "takeTokens",
              group: choices[Math.floor(rng() * choices.length)],
            });
            while (g.turn.pending.length) {
              const pending = Math.floor(rng() * g.turn.pending.length),
                color = g.turn.pending[pending];
              const possible = g.players[g.active].board
                .map((c, i) => (canPlace(c, color) ? i : -1))
                .filter((i) => i >= 0);
              assert.ok(possible.length);
              g = applyAction(g, {
                type: "place",
                pending,
                cell: possible[Math.floor(rng() * possible.length)],
              });
            }
            for (const card of [...activeCards(g.players[g.active])]) {
              let targets = matches(g.players[g.active].board, card.id);
              while (
                targets.length &&
                !isComplete(
                  g.players[g.active].cards.find((c) => c.id === card.id)!,
                )
              ) {
                g = applyAction(g, {
                  type: "settle",
                  card: card.id,
                  cell: targets[0].target,
                });
                targets = matches(g.players[g.active].board, card.id);
              }
            }
            g = applyAction(g, { type: "endTurn" });
            assert.equal(
              g.bag.length +
                g.supply.flat().length +
                g.discardedTokens.length +
                g.players.flatMap((p) => p.board.flatMap((c) => c.tokens))
                  .length +
                g.turn.pending.length,
              120,
            );
            assert.equal(
              new Set([
                ...g.deck,
                ...g.market.filter((x): x is number => x !== null),
                ...g.discardedCards,
                ...g.players.flatMap((p) =>
                  p.cards.filter((c) => !CARDS[c.id].spirit).map((c) => c.id),
                ),
              ]).size,
              32,
            );
            for (const p of g.players) {
              assert.ok(activeCards(p).length <= 4);
              for (const c of p.board) {
                const built = cell([]);
                for (const color of c.tokens) {
                  assert.ok(canPlace(built, color));
                  built.tokens.push(color);
                }
              }
            }
          }
          assert.equal(g.phase, "ended");
          assert.equal(new Set(g.players.map((p) => p.turns)).size, 1);
          assert.ok(actions < 1000);
        }
});
