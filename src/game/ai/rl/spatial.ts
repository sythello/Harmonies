import { emptyBoard } from "../../engine";
import { CELLS, CELL_TYPES, PLAYER_FEATURE_SIZE } from "./encoding";

/** Axial q=col, r=row-floor(col/2); +q and +r are adjacent directions. */
export const KERNEL_OFFSETS: Record<string, [number, number][]> = {
  triangle3_down: [
    [0, 0],
    [1, 0],
    [0, 1],
  ],
  triangle3_up: [
    [0, 0],
    [-1, 0],
    [0, -1],
  ],
  triangle6_down: [
    [0, 0],
    [1, 0],
    [2, 0],
    [0, 1],
    [1, 1],
    [0, 2],
  ],
  triangle6_up: [
    [0, 0],
    [-1, 0],
    [-2, 0],
    [0, -1],
    [-1, -1],
    [0, -2],
  ],
  hexagon7: [
    [0, 0],
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
  ],
};
const board = emptyBoard("A").map((c) => [
  c.col,
  c.row - Math.floor(c.col / 2),
]);
export const spatialSchema = {
  version: "harmonies-spatial-v1",
  cells: CELLS,
  cellTypes: CELL_TYPES,
  channels: 16,
  playerOffsets: [0, PLAYER_FEATURE_SIZE],
  // Index CELLS is a padding cell: 29 zeros followed by a boundary one.
  paddingIndex: CELLS,
  kernels: Object.entries(KERNEL_OFFSETS).map(([name, offsets]) => ({
    name,
    offsets,
    indices: board.map(([q, r]) =>
      offsets.map(([dq, dr]) => {
        const index = board.findIndex(([x, y]) => x === q + dq && y === r + dr);
        return index < 0 ? CELLS : index;
      }),
    ),
  })),
};
export const SPATIAL_SIZE =
  2 * CELLS * spatialSchema.kernels.length * spatialSchema.channels;
