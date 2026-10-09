// Frozen evaluator and ordering formula from f36c7cc; imports use equivalent current rules.
import {
  emptyCount,
  scorePlayer,
  CARDS,
  spiritScore,
  type Game,
} from "../../src/game/engine";
import { MAX_PROJECTS } from "../../src/game/ai/config";
import { counts, sum, observe } from "../../src/game/ai/observation";
import {
  applyProject,
  compareKeys,
  projects,
} from "../../src/game/ai/projects";
import type {
  AIObservation,
  Counts,
  Evaluation,
  ProjectEstimate,
} from "../../src/game/ai/types";

export function horizon(view: AIObservation, playerId: number) {
  const index = view.players.findIndex((p) => p.id === playerId);
  const bagRounds =
    1 +
    Math.floor(
      view.bagCount / (view.players.length === 1 ? 9 : 3 * view.players.length),
    );
  const boardRounds = Math.max(
    1,
    Math.min(
      ...view.players.map((p) => Math.ceil(Math.max(0, emptyCount(p) - 2) / 2)),
    ),
  );
  const R =
    view.phase === "ended"
      ? 0
      : view.finalRound ||
          !view.bagCount ||
          emptyCount(view.players[view.active]) <= 2
        ? 1
        : Math.min(bagRounds, boardRounds);
  const turns = Math.max(0, R - Number(index < view.active));
  const current = index === view.active;
  const B =
    turns === 0
      ? 0
      : current
        ? (view.turn.tookTokens ? view.turn.pending.length : 3) + 3 * (R - 1)
        : 3 * turns;
  return { R, B, turns, current };
}
export function discount(mByColor: Counts, H: Counts, A: Counts, B: number) {
  const m = sum(mByColor),
    F = Math.max(0, B - sum(H));
  const d = mByColor.map((n, i) => Math.max(0, n - H[i]));
  const total = sum(A),
    p = A.map((n) => (total ? n / total : 0));
  if (m > B || sum(d) > F || d.some((n, i) => n > A[i])) return null;
  const q = Math.exp(
    -m / Math.max(1, B) - 0.5 * sum(d.map((n, i) => (n * n) / (1 + F * p[i]))),
  );
  return { m, F, d, p, q };
}
const rounded = (value: number) => Math.round(value * 1e6);
export function evaluatePosition(
  view: AIObservation,
  playerId = view.players[view.active].id,
): Evaluation {
  const h = horizon(view, playerId);
  const player = view.players.find((p) => p.id === playerId)!;
  const actualScore = scorePlayer(player, view.side);
  const pool = view.bagCounts.map((n, i) => n + counts(view.supply.flat())[i]);
  const hands: { H: Counts; A: Counts; group?: number }[] = [];
  if (h.turns && h.current && !view.turn.tookTokens) {
    view.supply.forEach((group, i) => {
      if (group.length !== 3) return;
      const H = counts(group);
      hands.push({ H, A: pool.map((n, c) => n - H[c]), group: i });
    });
  } else
    hands.push({
      H: h.current ? counts(view.turn.pending) : [0, 0, 0, 0, 0, 0],
      A: pool,
    });
  if (!hands.length) hands.push({ H: [0, 0, 0, 0, 0, 0], A: pool });
  let best: Evaluation | undefined;
  for (const hand of hands) {
    const initial = discount([0, 0, 0, 0, 0, 0], hand.H, hand.A, h.B)!;
    const result: Evaluation = {
      actualScore,
      value: actualScore.total,
      potential: 0,
      R: h.R,
      B: h.B,
      F: initial.F,
      H: [...hand.H],
      A: [...hand.A],
      p: initial.p,
      supplyGroup: hand.group,
      projects: [],
    };
    let virtual = player,
      score = actualScore.total,
      B = h.B;
    const H = [...hand.H],
      A = [...hand.A];
    for (let iteration = 0; h.turns && iteration < MAX_PROJECTS; iteration++) {
      let chosen: ProjectEstimate | undefined;
      let nextPlayer = virtual;
      for (const project of projects(virtual)) {
        const mByColor = counts(project.additions.flatMap((a) => a.colors));
        const terms = discount(mByColor, H, A, B);
        if (!terms) continue;
        const candidate = applyProject(virtual, project);
        const G = scorePlayer(candidate, view.side).total - score;
        if (G <= 0) continue;
        const W = G * terms.q;
        if (
          !chosen ||
          rounded(W) > rounded(chosen.W) ||
          (rounded(W) === rounded(chosen.W) &&
            (terms.m < chosen.m ||
              (terms.m === chosen.m &&
                compareKeys(project.key, chosen.project.key) < 0)))
        ) {
          chosen = {
            project,
            G,
            mByColor,
            ...terms,
            W,
            B,
            H: [...H],
            A: [...A],
          };
          nextPlayer = candidate;
        }
      }
      if (!chosen) break;
      result.projects.push(chosen);
      result.potential += chosen.W;
      B -= chosen.m;
      chosen.mByColor.forEach((n, c) => {
        const held = Math.min(H[c], n);
        H[c] -= held;
        A[c] -= n - held;
      });
      virtual = nextPlayer;
      score += chosen.G;
    }
    result.value += result.potential;
    if (!best || rounded(result.value) > rounded(best.value)) best = result;
  }
  return best!;
}

export function orderValue(
  game: Game,
  playerId: number,
  actual: number,
): number {
  const player = game.players.find((p) => p.id === playerId)!;
  if (game.phase === "ended") return actual;
  const view = observe(game, playerId),
    h = horizon(view, playerId);
  if (!h.turns) return actual;
  const H =
    game.players[game.active].id === playerId
      ? counts(game.turn.pending)
      : [0, 0, 0, 0, 0, 0];
  const supplyCounts = counts(game.supply.flat());
  const A = view.bagCounts.map((n, c) => n + supplyCounts[c]);
  const best = new Map<number, number>();
  for (const project of projects(player, false)) {
    const card = player.cards.find((c) => c.id === project.cardId)!;
    const data = CARDS[card.id];
    const G = data.spirit
      ? spiritScore(player.board, card.id)
      : data.points[card.placed] - (data.points[card.placed - 1] ?? 0);
    const terms = discount(
      counts(project.additions.flatMap((a) => a.colors)),
      H,
      A,
      h.B,
    );
    if (terms) best.set(card.id, Math.max(best.get(card.id) ?? 0, G * terms.q));
  }
  return actual + [...best.values()].reduce((a, b) => a + b, 0);
}
