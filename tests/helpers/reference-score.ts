// Frozen pre-v2 scoring implementation, used only for semantic equivalence tests.
import {
  CARDS,
  key,
  neighbors,
  top,
  type Cell,
  type Player,
  type Side,
  type Score,
} from "../../src/game/engine";
export function groups(
  board: Cell[],
  predicate: (c: Cell) => boolean,
): Cell[][] {
  const left = new Map(board.filter(predicate).map((c) => [key(c), c])),
    result: Cell[][] = [];
  while (left.size) {
    const first = left.values().next().value!;
    const group: Cell[] = [];
    const queue = [first];
    left.delete(key(first));
    for (let i = 0; i < queue.length; i++) {
      const c = queue[i];
      group.push(c);
      for (const n of neighbors(c)) {
        const next = left.get(key(n));
        if (next) {
          left.delete(key(n));
          queue.push(next);
        }
      }
    }
    result.push(group);
  }
  return result;
}
export function riverLength(board: Cell[]): number {
  const water = new Map(
    board.filter((c) => top(c) === 1).map((c) => [key(c), c]),
  );
  let longest = 0;
  for (const start of water.values()) {
    const distances = new Map([[key(start), 1]]),
      queue = [start];
    for (let i = 0; i < queue.length; i++) {
      const cell = queue[i],
        distance = distances.get(key(cell))!;
      longest = Math.max(longest, distance);
      for (const n of neighbors(cell)) {
        const k = key(n);
        if (water.has(k) && !distances.has(k)) {
          distances.set(k, distance + 1);
          queue.push(water.get(k)!);
        }
      }
    }
  }
  return longest;
}
export function spiritScore(board: Cell[], id: number): number {
  const trees = board.filter((c) => top(c) === 4),
    mountains = board.filter((c) => top(c) === 2);
  const fieldGroups = () => groups(board, (c) => top(c) === 5),
    buildings = () =>
      groups(board, (c) => top(c) === 6 && c.tokens.length === 2);
  switch (id) {
    case 33:
      return fieldGroups().reduce((s, g) => s + (g.length >= 3 ? 10 : 2), 0);
    case 34:
      return fieldGroups().length * 5;
    case 35:
      return trees.filter((c) => c.tokens.length >= 2).length * 4;
    case 36:
      return trees.reduce((s, c) => s + (c.tokens.length === 3 ? 1 : 3), 0);
    case 37:
      return buildings().length * 4;
    case 38:
      return buildings().filter((g) => g.length >= 2).length * 6;
    case 39:
      return mountains.filter((c) => c.tokens.length >= 2).length * 4;
    case 40:
      return mountains.reduce((s, c) => s + (c.tokens.length === 3 ? 1 : 3), 0);
    case 41:
      return (
        groups(board, (c) => top(c) === 1).filter((g) => g.length >= 2).length *
        7
      );
    case 42:
      return board.filter((c) => top(c) === 1).length * 2;
    default:
      return 0;
  }
}
export function scorePlayer(player: Player, side: Side): Score {
  const board = player.board,
    lookup = new Map(board.map((c) => [key(c), c]));
  const height = [0, 1, 3, 7];
  const trees = board
    .filter((c) => top(c) === 4)
    .reduce((s, c) => s + height[c.tokens.length], 0);
  const mountains = board
    .filter(
      (c) =>
        top(c) === 2 &&
        neighbors(c).some(
          (n) =>
            top(lookup.get(key(n)) ?? { ...n, tokens: [], animal: null }) === 2,
        ),
    )
    .reduce((s, c) => s + height[c.tokens.length], 0);
  const fields =
    groups(board, (c) => top(c) === 5).filter((g) => g.length >= 2).length * 5;
  const buildings =
    board.filter(
      (c) =>
        top(c) === 6 &&
        c.tokens.length === 2 &&
        new Set(
          neighbors(c)
            .map((n) => lookup.get(key(n)))
            .filter((n): n is Cell => !!n && n.tokens.length > 0)
            .map(top),
        ).size >= 3,
    ).length * 5;
  const river = riverLength(board),
    islands = Math.max(1, groups(board, (c) => top(c) !== 1).length);
  const water =
    side === "B"
      ? islands * 5
      : river > 6
        ? 15 + (river - 6) * 4
        : [0, 0, 2, 5, 8, 11, 15][river];
  const cardScores = player.cards.map((c) => ({
    id: c.id,
    placed: c.placed,
    score:
      c.placed === 0
        ? 0
        : CARDS[c.id].spirit
          ? spiritScore(board, c.id)
          : CARDS[c.id].points[c.placed - 1],
  }));
  const animals = cardScores
      .filter((c) => !CARDS[c.id].spirit)
      .reduce((s, c) => s + c.score, 0),
    spirit = cardScores
      .filter((c) => CARDS[c.id].spirit)
      .reduce((s, c) => s + c.score, 0);
  const landscape = trees + mountains + fields + buildings + water,
    animalTotal = animals + spirit;
  return {
    trees,
    mountains,
    fields,
    buildings,
    water,
    landscape,
    animals,
    spirit,
    animalTotal,
    total: landscape + animalTotal,
    cubes: board.filter((c) => c.animal !== null && !CARDS[c.animal].spirit)
      .length,
    river,
    islands,
    cardScores,
  };
}
