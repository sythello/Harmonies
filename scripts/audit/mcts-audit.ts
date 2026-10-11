import assert from "node:assert/strict";
import {
  applyAction,
  clone,
  scorePlayer,
  seededRandom,
  type Action,
  type Game,
  type Player,
  type Score,
} from "../../src/game/engine";
import { actionLabel, legalActions } from "../../src/game/ai/actions";
import { observe, sampleGame } from "../../src/game/ai/observation";
import type { AIObservation } from "../../src/game/ai/types";
import { ACTION_SIZE, SCORE_SCALE } from "../../src/game/ai/rl/encoding";
import type { Evaluator, Pair } from "../../src/game/ai/rl/network";
import { newGame } from "../../src/game/ai/rl/trajectory";
import {
  search,
  searchConfig,
  type EdgeStats,
  type SearchConfig,
} from "../../src/game/ai/rl/mcts";

export interface AuditCase {
  id: string;
  gameIndex: number;
  seed: number;
  first: number;
  sourceActionIndex: number;
  original: AIObservation;
  input: AIObservation;
}
export interface SourceGame {
  seed: number;
  first: number;
  actions: Action[];
  scores: number[];
  finalRound: number;
}
/** Uniform over legal atomic actions, with independently seeded games and policies. */
export function collectCases(count: number, seed: number) {
  const cases: AuditCase[] = [],
    games: SourceGame[] = [];
  for (let i = 0; i < count; i++) {
    const gameSeed = seed + i,
      first = i % 2;
    let game = newGame(gameSeed, first);
    const random = seededRandom(gameSeed ^ 0x6a09e667);
    const actions: Action[] = [];
    const starts: { view: AIObservation; actionIndex: number }[] = [];
    while (game.phase !== "ended") {
      assert.ok(actions.length < 512, "Random game did not terminate");
      if (!actions.length || actions.at(-1)!.type === "endTurn")
        starts[game.active] = {
          view: observe(game),
          actionIndex: actions.length,
        };
      const legal = legalActions(game);
      assert.ok(legal.length);
      const action = legal[Math.floor(random() * legal.length)];
      actions.push(action);
      game = applyAction(game, action);
      game.log = [];
    }
    assert.equal(starts.length, 2);
    starts.forEach(({ view, actionIndex }, seat) => {
      assert.equal(view.round, game.finalRound);
      assert.equal(view.players[seat].turns + 1, game.players[seat].turns);
      const input = clone(view);
      input.finalRound = input.round;
      input.finalReason = "MCTS 最后一回合验证";
      cases.push({
        id: `game-${String(i + 1).padStart(3, "0")}-seat-${seat + 1}`,
        gameIndex: i + 1,
        seed: gameSeed,
        first,
        sourceActionIndex: actionIndex,
        original: view,
        input,
      });
    });
    games.push({
      seed: gameSeed,
      first,
      actions,
      scores: game.players.map((p) => scorePlayer(p, game.side).total),
      finalRound: game.finalRound!,
    });
  }
  return { cases, games };
}

/** Uniform priors; use absolute current score / 100, preserving production PUCT scale. */
export const currentScoreEvaluator: Evaluator = {
  predict(view) {
    const absolute = [view.active, 1 - view.active].map(
      (seat) => scorePlayer(view.players[seat], view.side).total / SCORE_SCALE,
    ) as Pair;
    return {
      logits: Array(ACTION_SIZE).fill(0),
      value: { rank: [0, 0], relative: [0, 0], absolute },
    };
  },
};

export function endpointKey(player: Player): string {
  return JSON.stringify([
    player.board.map((c) => [c.tokens, c.animal]),
    [...player.cards].sort((a, b) => a.id - b.id),
  ]);
}
/** All mutable within-turn rule state; ignore only order of cards/pending tokens. */
function turnKey(game: Game): string {
  return JSON.stringify([
    endpointKey(game.players[game.active]),
    game.supply,
    game.market,
    game.turn.tookTokens,
    game.turn.tookCard,
    [...game.turn.pending].sort(),
  ]);
}
export interface AuditPlan {
  key: string;
  actions: Action[];
  labels: string[];
  score: Score;
  gain: number;
  player: Player;
  visits: number;
}
export class PlanCollector {
  private seen = new Set<string>();
  top: AuditPlan[] = [];
  completions = 0;
  constructor(readonly root: Game) {}
  get unique() {
    return this.seen.size;
  }
  add(actions: readonly Action[], endpoint: Game) {
    this.completions++;
    const player = endpoint.players[this.root.active],
      key = endpointKey(player);
    const existing = this.top.find((p) => p.key === key);
    if (existing) {
      existing.visits++;
      return;
    }
    if (this.seen.has(key)) return;
    this.seen.add(key);
    const score = scorePlayer(player, this.root.side);
    if (this.top.length === 3 && score.total < this.top[2].score.total) return;
    let game = this.root;
    const labels = actions.map((a) => {
      const label = actionLabel(game, a);
      game = applyAction(game, a);
      game.log = [];
      return label;
    });
    assert.equal(endpointKey(game.players[this.root.active]), key);
    this.top.push({
      key,
      actions: [...actions],
      labels,
      score,
      gain:
        score.total -
        scorePlayer(this.root.players[this.root.active], this.root.side).total,
      player: clone(player),
      visits: 1,
    });
    this.top.sort(
      (a, b) => b.score.total - a.score.total || a.key.localeCompare(b.key),
    );
    this.top.length = Math.min(3, this.top.length);
  }
}
export interface ExactResult {
  complete: boolean;
  elapsedMs: number;
  states: number;
  transitions: number;
  duplicates: number;
  uniquePlans: number;
  completions: number;
  plans: AuditPlan[];
}
export interface MctsResult {
  simulations: number;
  elapsedMs: number;
  nodes: number;
  chanceOutcomes: number;
  completions: number;
  uniquePlans: number;
  plans: AuditPlan[];
  selectedAction: Action;
  selectedLabel: string;
  rootPolicy: EdgeStats[];
  config: SearchConfig;
}
export function auditMcts(
  view: AIObservation,
  simulations: number,
  seed: number,
): MctsResult {
  const root = sampleGame(view, 0),
    collector = new PlanCollector(root);
  const config = searchConfig({
    simulations,
    seed,
    rankWeight: 0,
    relativeWeight: 0,
    absoluteWeight: 1,
    noiseWeight: 0,
    temperature: 0,
  });
  const start = performance.now();
  const result = search(view, currentScoreEvaluator, config, {
    onTurnEnd: (actions, game) => collector.add(actions, game),
  });
  return {
    simulations,
    elapsedMs: performance.now() - start,
    nodes: result.nodes,
    chanceOutcomes: result.chanceOutcomes,
    completions: collector.completions,
    uniquePlans: collector.unique,
    plans: collector.top,
    selectedAction: result.action,
    selectedLabel: actionLabel(root, result.action),
    rootPolicy: result.policy,
    config,
  };
}
export interface CaseResult {
  position: AuditCase;
  mcts: MctsResult[];
  exact: ExactResult;
}
/** Enumerate every legal action, merging equivalent states without heuristic pruning. */
export function enumerateTurn(
  view: AIObservation,
  timeMs: number,
  now = () => performance.now(),
): ExactResult {
  const root = sampleGame(view, 0),
    collector = new PlanCollector(root);
  const seen = new Set<string>(),
    start = now(),
    deadline = start + timeMs;
  let transitions = 0,
    duplicates = 0,
    complete = true;
  const path: Action[] = [];
  function visit(game: Game) {
    if (now() >= deadline) {
      complete = false;
      return;
    }
    const key = turnKey(game);
    if (seen.has(key)) {
      duplicates++;
      return;
    }
    seen.add(key);
    for (const action of legalActions(game)) {
      if (now() >= deadline) {
        complete = false;
        return;
      }
      const next = applyAction(game, action);
      next.log = [];
      transitions++;
      path.push(action);
      if (action.type === "endTurn") collector.add(path, next);
      else visit(next);
      path.pop();
      if (!complete) return;
    }
  }
  visit(root);
  return {
    complete,
    elapsedMs: now() - start,
    states: seen.size,
    transitions,
    duplicates,
    uniquePlans: collector.unique,
    completions: collector.completions,
    plans: collector.top,
  };
}

export function verifyPlans(view: AIObservation, plans: AuditPlan[]) {
  for (const plan of plans) {
    let game = sampleGame(view, 0);
    assert.equal(plan.actions.at(-1)?.type, "endTurn");
    for (let i = 0; i < plan.actions.length; i++) {
      const action = plan.actions[i];
      assert.ok(
        legalActions(game).some(
          (a) => JSON.stringify(a) === JSON.stringify(action),
        ),
      );
      assert.equal(game.active, view.active);
      assert.equal(game.round, view.round);
      assert.equal(plan.labels[i], actionLabel(game, action));
      game = applyAction(game, action);
      game.log = [];
    }
    assert.ok(game.phase === "ended" || game.active !== view.active);
    assert.deepEqual(
      scorePlayer(game.players[view.active], game.side),
      plan.score,
    );
    assert.equal(endpointKey(game.players[view.active]), plan.key);
  }
}
