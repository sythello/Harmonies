import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, scorePlayer, type Game } from "../src/game/engine";
import { legalActions } from "../src/game/ai/actions";
import { observe } from "../src/game/ai/observation";
import { actionId } from "../src/game/ai/rl/encoding";
import { newGame } from "../src/game/ai/rl/trajectory";
import { solveTurn, optimizeGame } from "../scripts/audit/nn-fit";
import { collectCases } from "../scripts/audit/mcts-audit";

test("exhaustive PUCT agrees with unmerged enumeration for every first action and preserves the 512+512 target", () => {
  const game = newGame(17);
  game.active = 1;
  game.finalRound = game.round;
  game.market.fill(null);
  game.supply = [[1, 1, 5], [5, 5, 1], [], [], []];
  for (const cell of game.players[1].board) cell.tokens = [4];
  for (let i = 0; i < 4; i++) game.players[1].board[i].tokens = [];
  function brute(g: Game): number {
    return Math.max(
      ...legalActions(g).map((a) => {
        const n = applyAction(g, a);
        return a.type === "endTurn"
          ? scorePlayer(n.players[1], "A").total
          : brute(n);
      }),
    );
  }
  const view = observe(game),
    exact = solveTurn(view);
  for (const a of legalActions(game))
    assert.equal(
      exact.root.find((e) => e.id === actionId(view, a))!.score,
      brute(applyAction(game, a)),
    );
  assert.equal(exact.score, brute(game));
  assert.ok(exact.transpositions > 0);
  assert.equal(
    exact.root.reduce((s, e) => s + e.prefixVisits, 0),
    512,
  );
  assert.equal(
    exact.root.reduce((s, e) => s + e.targetVisits, 0),
    1024,
  );
  assert.ok(exact.optimalIds.includes(actionId(view, exact.plan[0])));
  exact.boostedPath.forEach((e, i) => {
    assert.equal(e.addedVisits, 512);
    assert.equal(
      e.prefixVisits,
      exact.prefixPaths.filter(
        (p) =>
          p.length > i &&
          exact.boostedPath.slice(0, i + 1).every((x, j) => x.id === p[j]),
      ).length,
    );
  });
  assert.throws(() => solveTurn(view, 0), /timed out/);
});

test("small fully solved trees still retain exactly 512 prefix simulations", () => {
  const game = newGame(18);
  game.active = 1;
  game.finalRound = game.round;
  game.market.fill(null);
  game.turn = { tookTokens: true, tookCard: true, pending: [] };
  const exact = solveTurn(observe(game));
  assert.equal(exact.states, 1);
  assert.equal(exact.simulations, 512);
  assert.deepEqual(
    exact.prefixPaths,
    Array.from({ length: 512 }, () => [884]),
  );
  assert.equal(exact.root[0].targetVisits, 1024);
});

test("two optimized samples follow the real game and share terminal values in opposite perspectives", () => {
  const raw = collectCases(1, 102000),
    row = optimizeGame(raw.games[0], raw.cases[0], "train");
  let game = newGame(raw.games[0].seed, raw.games[0].first);
  for (const a of raw.games[0].actions.slice(0, row.sourceActionIndex)) {
    game = applyAction(game, a);
    game.log = [];
  }
  game.finalRound = game.round;
  game.finalReason = "NN 拟合实验最后一回合";
  row.samples.forEach((s, i) => {
    assert.deepEqual(observe(game), s.input);
    for (const a of row.finalActions[i]) {
      game = applyAction(game, a);
      game.log = [];
    }
    assert.equal(s.exact.score, row.scores[i]);
  });
  assert.equal(game.phase, "ended");
  assert.notDeepEqual(
    row.samples[1].input.players[0].board,
    raw.cases[1].input.players[0].board,
  );
  for (const head of ["rank", "relative", "absolute"] as const)
    assert.deepEqual(
      row.samples[0].value[head],
      [...row.samples[1].value[head]].reverse(),
    );
});
