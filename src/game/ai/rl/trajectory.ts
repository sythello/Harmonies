import {
  applyAction,
  createGame,
  ranking,
  seededRandom,
  type Action,
  type Game,
} from "../../engine";
import { legalActions } from "../actions";
import { observe, sampleGame } from "../observation";
import type { AIObservation } from "../types";
import { encode, VERSION, type Features } from "./encoding";
import {
  search,
  searchConfig,
  terminalValues,
  type SearchConfig,
} from "./mcts";
import type { Evaluator, Values } from "./network";

export interface Sample extends Features {
  seat: number;
  legal: number[];
  policy: number[];
  visits: number[];
  value: Values;
  actionId: number;
}
export interface Trajectory {
  version: typeof VERSION;
  model: string;
  seed: number;
  first: number;
  search: SearchConfig;
  actions: Action[];
  samples: Sample[];
  scores: number[];
  ranks: number[];
  phase: "ended";
}
export type Agent = (view: AIObservation, step: number) => Action;
export const newGame = (seed: number, first = 0) =>
  createGame({
    names: ["P0", "P1"],
    first,
    seed,
    side: "A",
    spirits: false,
  });
export function playGame(
  agents: [Agent, Agent],
  seed: number,
  first = 0,
  maxActions = 512,
): Game {
  let game = newGame(seed, first),
    step = 0;
  while (game.phase !== "ended") {
    if (step >= maxActions)
      throw new Error("Game exceeded maxActions; no terminal labels emitted");
    const view = observe(game);
    game = applyAction(
      game,
      agents[game.players[game.active].id](view, step++),
    );
    game.log = [];
  }
  return game;
}
export function collect(
  evaluator: Evaluator,
  options: {
    seed: number;
    first?: number;
    model: string;
    search?: Partial<SearchConfig>;
    maxActions?: number;
  },
): Trajectory {
  const config = searchConfig({
    noiseWeight: 0.25,
    temperature: 1,
    ...options.search,
  });
  const samples: Sample[] = [],
    actions: Action[] = [];
  const agent: Agent = (view, step) => {
    const result = search(view, evaluator, {
      ...config,
      seed: config.seed + step * 1009,
    });
    samples.push({
      ...encode(view),
      seat: view.active,
      legal: result.policy.map((e) => e.id),
      policy: result.policy.map((e) => e.target),
      visits: result.policy.map((e) => e.visits),
      value: result.value,
      actionId: result.actionId,
    });
    actions.push(result.action);
    return result.action;
  };
  const first = options.first ?? 0;
  const game = playGame(
    [agent, agent],
    options.seed,
    first,
    options.maxActions,
  );
  // Overwrite every value with the actual terminal outcome, never a search bootstrap.
  for (const sample of samples)
    sample.value = terminalValues(game, sample.seat);
  const ranks = ranking(game);
  return {
    version: VERSION,
    model: options.model,
    seed: options.seed,
    first,
    search: config,
    actions,
    samples,
    phase: "ended",
    scores: game.players.map(
      (p) => ranks.find((r) => r.player.id === p.id)!.score.total,
    ),
    ranks: game.players.map(
      (p) => ranks.find((r) => r.player.id === p.id)!.rank,
    ),
  };
}
export function randomAgent(seed: number): Agent {
  const random = seededRandom(seed);
  return (view) => {
    const actions = legalActions(sampleGame(view, 0));
    return actions[Math.floor(random() * actions.length)];
  };
}

export function evaluate(candidate: Agent, opponent: Agent, seeds: number[]) {
  const games = seeds.flatMap((seed) =>
    [0, 1].map((first) => {
      const game = playGame([candidate, opponent], seed, first),
        entries = ranking(game);
      const a = entries.find((e) => e.player.id === 0)!,
        b = entries.find((e) => e.player.id === 1)!;
      return {
        seed,
        first,
        candidate: a.score.total,
        opponent: b.score.total,
        result: a.rank === b.rank ? 0.5 : a.rank < b.rank ? 1 : 0,
      };
    }),
  );
  const mean = (f: (g: (typeof games)[number]) => number) =>
    games.reduce((s, g) => s + f(g), 0) / games.length;
  return {
    games,
    wins: games.filter((g) => g.result === 1).length,
    draws: games.filter((g) => g.result === 0.5).length,
    losses: games.filter((g) => g.result === 0).length,
    winRate: mean((g) => g.result),
    meanScore: mean((g) => g.candidate),
    meanMargin: mean((g) => g.candidate - g.opponent),
  };
}
