import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, clone, scorePlayer } from "../src/game/engine";
import { legalActions } from "../src/game/ai/actions";
import { observe } from "../src/game/ai/observation";
import { newGame } from "../src/game/ai/rl/trajectory";
import { search } from "../src/game/ai/rl/mcts";
import {
  collectCases,
  currentScoreEvaluator,
  endpointKey,
  enumerateTurn,
  verifyPlans,
} from "../scripts/audit/mcts-audit";

test("random-game fixtures are the actual last turn starts of both players", () => {
  const { cases, games } = collectCases(3, 71000);
  assert.equal(cases.length, 6);
  games.forEach((source, i) => {
    let game = newGame(source.seed, source.first);
    const positions = cases.filter((c) => c.gameIndex === i + 1);
    source.actions.forEach((action, index) => {
      for (const c of positions.filter((c) => c.sourceActionIndex === index)) {
        assert.deepEqual(observe(game), c.original);
        assert.deepEqual(game.turn, {
          tookTokens: false,
          tookCard: false,
          pending: [],
        });
        const normalized = clone(c.original);
        normalized.finalRound = normalized.round;
        normalized.finalReason = "MCTS 最后一回合验证";
        assert.deepEqual(c.input, normalized);
      }
      game = applyAction(game, action);
      game.log = [];
    });
    assert.equal(game.phase, "ended");
    for (const c of positions)
      assert.equal(
        c.original.players[c.original.active].turns + 1,
        game.players[c.original.active].turns,
      );
    assert.deepEqual(
      game.players.map((p) => scorePlayer(p, "A").total),
      source.scores,
    );
  });
});

test("turn observer does not change MCTS and only captures root-player complete turns", () => {
  const game = newGame(10);
  game.finalRound = game.round;
  game.market.fill(null);
  game.turn = { tookTokens: true, tookCard: false, pending: [] };
  game.players[0].board[0].tokens = [4];
  game.players[1].board[0].tokens = [2, 2];
  const config = {
    simulations: 64,
    seed: 87,
    rankWeight: 0,
    relativeWeight: 0,
    absoluteWeight: 1,
  };
  const ordinary = search(observe(game), currentScoreEvaluator, config);
  let ends = 0;
  const observed = search(observe(game), currentScoreEvaluator, config, {
    onTurnEnd(actions, next) {
      ends++;
      assert.deepEqual(actions, [{ type: "endTurn" }]);
      assert.equal(next.active, 1);
      assert.equal(scorePlayer(next.players[0], "A").total, 1);
    },
  });
  assert.deepEqual(observed, ordinary);
  assert.equal(ends, 64);
  assert.ok(Math.abs(observed.policy[0].q - 0.01) < 1e-12);
});

test("memoized enumeration agrees with independent unmerged traversal and marks timeout", () => {
  let game = newGame(12);
  game.finalRound = game.round;
  game.active = 1;
  game.market.fill(null);
  game.supply = [[1, 1, 1], [], [], [], []];
  for (const cell of game.players[1].board) cell.tokens = [4];
  game.players[1].board[0].tokens = [];
  game.players[1].board[1].tokens = [];
  game.players[1].board[2].tokens = [];
  game.players[1].board[3].tokens = [];
  const expected = new Map<string, number>();
  function brute(state: typeof game) {
    for (const action of legalActions(state)) {
      const next = applyAction(state, action);
      if (action.type === "endTurn")
        expected.set(
          endpointKey(next.players[1]),
          scorePlayer(next.players[1], "A").total,
        );
      else brute(next);
    }
  }
  brute(game);
  const exact = enumerateTurn(observe(game), 10000);
  assert.equal(exact.complete, true);
  assert.ok(exact.duplicates > 0);
  assert.equal(exact.uniquePlans, expected.size);
  const top = [...expected]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 3);
  assert.deepEqual(
    exact.plans.map((p) => [p.key, p.score.total]),
    top,
  );
  verifyPlans(observe(game), exact.plans);
  let clock = 0;
  const timed = enumerateTurn(observe(game), 3, () => clock++);
  assert.equal(timed.complete, false);
});
