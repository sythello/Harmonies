import test from "node:test";
import assert from "node:assert/strict";
import { emptyBoard, neighbor, type Coord } from "../src/game/engine";
import { spatialSchema, SPATIAL_SIZE } from "../src/game/ai/rl/spatial";
import stored from "../rl/spatial.json" with { type: "json" };
import { seededRandom } from "../src/game/engine";
import { spatialFeatures, type Dense } from "../src/game/ai/rl/network";
import { FEATURE_SIZE } from "../src/game/ai/rl/encoding";

test("sparse spatial filters equal dense convolution including boundary and fractional inputs", () => {
  const rng = seededRandom(94);
  const numeric = Array.from({ length: FEATURE_SIZE }, () =>
    rng() < 0.9 ? 0 : rng() - 0.5,
  );
  const filters: Record<string, Dense> = {};
  const expected: number[] = [];
  for (const kernel of spatialSchema.kernels) {
    const layer = (filters[kernel.name] = {
      weight: Array.from({ length: 16 }, () =>
        Array.from({ length: kernel.offsets.length * 30 }, () => rng() - 0.5),
      ),
      bias: Array.from({ length: 16 }, () => rng() - 0.5),
    });
    for (const start of spatialSchema.playerOffsets)
      for (const patch of kernel.indices) {
        const x = patch.flatMap((cell) =>
          cell === 23
            ? [...Array(29).fill(0), 1]
            : [...numeric.slice(start + cell * 29, start + (cell + 1) * 29), 0],
        );
        layer.weight.forEach((row, channel) => {
          let y = layer.bias[channel];
          x.forEach((value, i) => {
            y += row[i] * value;
          });
          expected.push(Math.max(0, y));
        });
      }
  }
  assert.deepEqual(spatialFeatures(numeric, filters), expected);
});

test("spatial schema matches Python and every patch follows true hex adjacency", () => {
  assert.deepEqual(spatialSchema, stored);
  assert.equal(SPATIAL_SIZE, 3680);
  const board = emptyBoard("A");
  const step = (p: Coord, d: number, n = 1): Coord =>
    n ? step(neighbor(p, d), d, n - 1) : p;
  for (const [i, anchor] of board.entries()) {
    const shapes = [
      [anchor, step(anchor, 0), step(anchor, 5)],
      [anchor, step(anchor, 3), step(anchor, 2)],
      [
        anchor,
        step(anchor, 0),
        step(anchor, 0, 2),
        step(anchor, 5),
        step(step(anchor, 0), 5),
        step(anchor, 5, 2),
      ],
      [
        anchor,
        step(anchor, 3),
        step(anchor, 3, 2),
        step(anchor, 2),
        step(step(anchor, 3), 2),
        step(anchor, 2, 2),
      ],
      [anchor, ...Array.from({ length: 6 }, (_, d) => step(anchor, d))],
    ];
    shapes.forEach((shape, k) => {
      const expected = shape.map((c) => {
        const index = board.findIndex(
          (p) => c.col === p.col && c.row === p.row,
        );
        return index < 0 ? 23 : index;
      });
      assert.deepEqual(spatialSchema.kernels[k].indices[i], expected);
    });
  }
  for (const kernel of spatialSchema.kernels) {
    assert.ok(
      kernel.indices.some((patch) => patch.includes(23)),
      "edge padding must be present",
    );
    assert.ok(
      kernel.indices.some((patch) => !patch.includes(23)),
      "interior patches must be present",
    );
  }
});

test("triangle orientations are reversed and have 3/6 distinct cells", () => {
  for (const k of [0, 2]) {
    const forward = spatialSchema.kernels[k].offsets;
    const reversed = spatialSchema.kernels[k + 1].offsets;
    assert.deepEqual(
      reversed,
      forward.map(([q, r]) => [q === 0 ? 0 : -q, r === 0 ? 0 : -r]),
    );
    assert.equal(new Set(forward.map((p) => p.join())).size, k === 0 ? 3 : 6);
  }
});
