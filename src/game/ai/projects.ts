import {
  CARDS,
  canPlace,
  isComplete,
  key,
  patternCoordinates,
  type Cell,
  type Color,
  type Player,
} from "../engine";
import type { Project } from "./types";

export function completionOptions(cell: Cell, expected: number[]): Color[][] {
  const stacks: Color[][] =
    expected[0] === 6 && expected[1] === 7
      ? [
          [2, 6],
          [3, 6],
          [6, 6],
        ]
      : [[...expected].reverse() as Color[]];
  return stacks.flatMap((stack) => {
    if (
      cell.tokens.length > stack.length ||
      cell.tokens.some((c, i) => c !== stack[i])
    )
      return [];
    const colors = stack.slice(cell.tokens.length);
    const probe = { ...cell, tokens: [...cell.tokens] };
    for (const color of colors) {
      if (!canPlace(probe, color)) return [];
      probe.tokens.push(color);
    }
    return [colors];
  });
}
const templates = new Map<
  string,
  { origin: number; rotation: number; cells: number[]; target: number }[]
>();
function placements(board: Cell[], cardId: number) {
  const cacheKey = `${cardId}:${board.map(key).join(";")}`;
  let result = templates.get(cacheKey);
  if (result) return result;
  result = [];
  const lookup = new Map(board.map((c, i) => [key(c), i]));
  const card = CARDS[cardId];
  for (let origin = 0; origin < board.length; origin++)
    for (let rotation = 0; rotation < 6; rotation++) {
      const cells = patternCoordinates(card, board[origin], rotation).map(
        (c) => lookup.get(key(c)) ?? -1,
      );
      if (cells.every((i) => i >= 0))
        result.push({
          origin,
          rotation,
          cells,
          target: cells[card.pattern.findIndex((p) => p.allowCube)],
        });
    }
  templates.set(cacheKey, result);
  return result;
}
export function* projects(
  player: Player,
  includeTerrain = true,
): Generator<Project> {
  for (const owned of player.cards
    .filter((c) => !isComplete(c))
    .sort((a, b) => a.id - b.id)) {
    const card = CARDS[owned.id];
    for (const layout of placements(player.board, card.id)) {
      if (player.board[layout.target].animal !== null) continue;
      const options = layout.cells.map((index, i) =>
        completionOptions(player.board[index], card.pattern[i].colors),
      );
      if (options.some((o) => !o.length)) continue;
      function* combinations(
        i: number,
        additions: Project["additions"],
      ): Generator<Project> {
        if (i === options.length) {
          yield {
            kind: "animal",
            cardId: card.id,
            targetCell: layout.target,
            additions,
            key: [
              0,
              card.id,
              layout.origin,
              layout.rotation,
              ...additions.flatMap((a) => [
                a.cell,
                a.colors.length,
                ...a.colors,
              ]),
            ],
          };
          return;
        }
        for (const colors of options[i])
          yield* combinations(
            i + 1,
            colors.length
              ? [...additions, { cell: layout.cells[i], colors }]
              : additions,
          );
      }
      yield* combinations(0, []);
    }
  }
  if (!includeTerrain) return;
  for (let cell = 0; cell < player.board.length; cell++)
    for (let c = 1; c <= 6; c++) {
      const color = c as Color;
      if (canPlace(player.board[cell], color))
        yield {
          kind: "terrain",
          additions: [{ cell, colors: [color] }],
          key: [1, 0, cell, 0, color],
        };
    }
}
export function applyProject(player: Player, project: Project): Player {
  const next: Player = {
    ...player,
    board: player.board.map((c) => ({ ...c })),
    cards: player.cards.map((c) => ({ ...c })),
  };
  for (const addition of project.additions)
    next.board[addition.cell].tokens = [
      ...next.board[addition.cell].tokens,
      ...addition.colors,
    ];
  if (project.kind === "animal") {
    next.board[project.targetCell!].animal = project.cardId!;
    next.cards.find((c) => c.id === project.cardId)!.placed++;
  }
  return next;
}
export function compareKeys(a: number[], b: number[]): number {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const difference = (a[i] ?? -1) - (b[i] ?? -1);
    if (difference) return difference;
  }
  return 0;
}
