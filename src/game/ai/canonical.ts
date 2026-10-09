import { key, type Cell, type Player } from "../engine";
import { compareKeys } from "./projects";

export interface BoardRotation {
  rotation: number;
  /** Source cell -> destination cell. */
  forward: number[];
  inverse: number[];
}
const geometry = new Map<string, BoardRotation[]>();
export function boardRotations(board: Cell[]): BoardRotation[] {
  const shape = board.map(key).join(";");
  const cached = geometry.get(shape);
  if (cached) return cached;
  const original = board.map((c) => [c.col, c.row - Math.floor(c.col / 2)]);
  const lookup = new Map(original.map((c, i) => [c.join(","), i]));
  const minQ = Math.min(...original.map((c) => c[0]));
  const minR = Math.min(...original.map((c) => c[1]));
  let coords = original;
  const result: BoardRotation[] = [];
  for (let rotation = 0; rotation < 6; rotation++) {
    const dq = minQ - Math.min(...coords.map((c) => c[0]));
    const dr = minR - Math.min(...coords.map((c) => c[1]));
    const forward = coords.map(
      ([q, r]) => lookup.get(`${q + dq},${r + dr}`) ?? -1,
    );
    if (forward.every((i) => i >= 0)) {
      const inverse: number[] = [];
      forward.forEach((dest, source) => {
        inverse[dest] = source;
      });
      result.push({ rotation, forward, inverse });
    }
    coords = coords.map(([q, r]) => [-r, q + r]);
  }
  geometry.set(shape, result);
  return result;
}
export function canonicalize(player: Player) {
  let best: { transform: BoardRotation; code: number[] } | undefined;
  for (const transform of boardRotations(player.board)) {
    const code = transform.inverse.flatMap((source) => {
      const c = player.board[source];
      return [
        c.tokens.length,
        c.tokens[0] ?? 0,
        c.tokens[1] ?? 0,
        c.tokens[2] ?? 0,
        c.animal ?? 0,
      ];
    });
    if (!best || compareKeys(code, best.code) < 0) best = { transform, code };
  }
  const { transform, code } = best!;
  return {
    ...transform,
    key: code.join(","),
    player: {
      ...player,
      board: player.board.map((coord, dest) => ({
        ...coord,
        tokens: player.board[transform.inverse[dest]].tokens,
        animal: player.board[transform.inverse[dest]].animal,
      })),
    },
  };
}
