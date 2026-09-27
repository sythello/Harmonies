import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import {
  applyAction,
  createGame,
  scorePlayer,
  seededRandom,
  type Game,
  type Side,
} from "../src/game/engine";
import { legalActions } from "../src/game/ai/actions";
import { observe } from "../src/game/ai/observation";
import { searchTurn } from "../src/game/ai/search";

const budget = Number(process.env.AI_BENCH_MS ?? 3500);
const cases: [number, Side, boolean, number][] = [
  [1, "A", false, 11],
  [1, "B", true, 13],
  [2, "A", true, 17],
  [4, "B", false, 19],
];
const output = [];
function check(game: Game) {
  assert.equal(
    game.bag.length +
      game.supply.flat().length +
      game.turn.pending.length +
      game.discardedTokens.length +
      game.players.flatMap((p) => p.board.flatMap((c) => c.tokens)).length,
    120,
  );
  assert.equal(
    new Set([
      ...game.deck,
      ...game.market.filter((id): id is number => id !== null),
      ...game.discardedCards,
      ...game.players.flatMap((p) =>
        p.cards.filter((c) => c.id <= 32).map((c) => c.id),
      ),
    ]).size,
    32,
  );
}
for (const [n, side, spirits, seed] of cases) {
  for (const mode of ["random", "greedy", "ai"] as const) {
    let game = createGame({
      names: Array.from({ length: n }, (_, i) => `P${i}`),
      first: 0,
      side,
      spirits,
      seed,
    });
    const random = seededRandom(seed + 100);
    const times: number[] = [],
      depths: number[] = [];
    let turns = 0;
    while (game.phase !== "ended" && turns < 120) {
      if (mode === "ai") {
        const result = searchTurn(observe(game), {
          depth: 2,
          timeMs: budget,
          topK: 1,
        });
        times.push(result.elapsedMs);
        depths.push(result.completedDepth);
        for (const action of result.actions) game = applyAction(game, action);
      } else {
        for (let step = 0; step < 64; step++) {
          const actions = legalActions(game);
          assert.ok(actions.length);
          const playerId = game.players[game.active].id;
          const action =
            mode === "random"
              ? actions[Math.floor(random() * actions.length)]
              : actions
                  .map((action, index) => {
                    const next = applyAction(game, action);
                    return {
                      action,
                      index,
                      score: scorePlayer(
                        next.players.find((p) => p.id === playerId)!,
                        side,
                      ).total,
                    };
                  })
                  .sort((a, b) => b.score - a.score || a.index - b.index)[0]
                  .action;
          game = applyAction(game, action);
          if (action.type === "endTurn") break;
        }
      }
      turns++;
      check(game);
      if (mode === "ai")
        console.log(
          `${n}p ${side} spirits=${spirits} turn=${turns} depth=${depths.at(-1)} ${times.at(-1)}ms`,
        );
    }
    assert.equal(game.phase, "ended");
    assert.equal(new Set(game.players.map((p) => p.turns)).size, 1);
    times.sort((a, b) => a - b);
    const row = {
      n,
      side,
      spirits,
      seed,
      mode,
      turns,
      scores: game.players.map((p) => scorePlayer(p, side).total),
      budget,
      medianMs: times[Math.floor(times.length * 0.5)] ?? 0,
      p95Ms:
        times[Math.min(times.length - 1, Math.floor(times.length * 0.95))] ?? 0,
      maxMs: times.at(-1) ?? 0,
      depthCounts: [0, 1, 2, 3].map(
        (d) => depths.filter((x) => x === d).length,
      ),
    };
    output.push(row);
    console.log(JSON.stringify(row));
    await writeFile(
      "/tmp/harmonies-ai-benchmark.json",
      JSON.stringify(output, null, 2),
    );
  }
}
