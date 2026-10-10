import {
  emptyCount,
  CARDS,
  spiritScore,
  scorePlayer,
  type Player,
  type Score,
  type Side,
} from "../engine";
import { MAX_PROJECTS } from "./config";
import { counts, sum } from "./observation";
import { applyProject, compareKeys, projects } from "./projects";
import { canonicalize } from "./canonical";
import type {
  AIObservation,
  Counts,
  Evaluation,
  Project,
  ProjectEstimate,
} from "./types";

export function estimateHorizon(view: AIObservation) {
  const { round: r, active: a, players } = view;
  if (view.phase === "ended")
    return { L: r, R: 0, T: players.map(() => 0), B: players.map(() => 0) };
  const n = players.length,
    u = view.turn.tookTokens ? view.turn.pending.length : 3;
  const boardEnds = players.map((p, i) => {
    const e = emptyCount(p);
    return i === a
      ? r + Math.ceil(Math.max(0, 3 * (e - 2) - 2 * u) / 6)
      : r + Number(i < a) + Math.max(1, Math.ceil((e - 2) / 2)) - 1;
  });
  const bagEnd =
    r + Math.floor((a + Math.ceil(view.bagCount / (n === 1 ? 9 : 3))) / n);
  const L = view.finalRound ?? Math.min(bagEnd, ...boardEnds);
  const T = players.map((_, i) => Math.max(0, L - r + 1 - Number(i < a)));
  const B = T.map((t, i) => (t === 0 ? 0 : i === a ? u + 3 * (t - 1) : 3 * t));
  return { L, R: L - r + 1, T, B };
}
export function horizon(view: AIObservation, playerId: number) {
  const i = view.players.findIndex((p) => p.id === playerId),
    h = estimateHorizon(view);
  return {
    L: h.L,
    R: h.R,
    B: h.B[i],
    turns: h.T[i],
    current: i === view.active,
  };
}
/** Frozen v1 horizon, used only by the v1-formula search comparison. */
export function legacyHorizon(view: AIObservation, playerId: number) {
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
export const rounded = (value: number) => Math.round(value * 1e6);

/** Caches belong to a single planning request. Cached project coordinates are canonical. */
export function createEvaluator(version: "v1" | "v2" | "v3.1" = "v2") {
  const cache = new Map<string, Evaluation>();
  const monotoneCache = new Map<string, Evaluation>();
  const orderCache = new Map<string, Evaluation>();
  const scores = new Map<string, Score>();
  const menus = new Map<
    string,
    { project: Project; mByColor: Counts; after?: Score }[]
  >();
  const playerKey = (p: Player, side: Side) =>
    `${side}|${p.board.map((c) => (c.tokens[0] ?? 0) + 7 * (c.tokens[1] ?? 0) + 49 * (c.tokens[2] ?? 0) + 343 * (c.animal ?? 0)).join(",")}|${p.cards.map((c) => `${c.id}:${c.placed}`).join(",")}`;
  let computations = 0;
  let orderComputations = 0;
  const score = (p: Player, side: Side) => {
    const key = playerKey(p, side);
    let result = scores.get(key);
    if (!result) {
      result = scorePlayer(p, side);
      if (scores.size >= 8192) scores.delete(scores.keys().next().value!);
      scores.set(key, result);
    }
    return result;
  };
  const menu = (p: Player, side: Side, terrain = true) => {
    const key = `${terrain}:${playerKey(p, side)}`;
    let result = menus.get(key);
    if (!result) {
      // Exact equivalence only: identical edits + settlement have identical G/m
      // and successors. Retain the smallest specified tie-break key.
      const unique = new Map<string, Project>();
      for (const project of projects(p, terrain)) {
        const edits = [...project.additions]
          .sort((a, b) => a.cell - b.cell)
          .map((a) => `${a.cell}:${a.colors.join("")}`)
          .join(";");
        const effect = `${project.kind}:${project.cardId ?? 0}:${project.targetCell ?? -1}:${edits}`;
        const old = unique.get(effect);
        if (!old || compareKeys(project.key, old.key) < 0)
          unique.set(effect, project);
      }
      result = [...unique.values()].map((project) => ({
        project,
        mByColor: counts(project.additions.flatMap((a) => a.colors)),
      }));
      if (menus.size >= 512) menus.delete(menus.keys().next().value!);
      menus.set(key, result);
    }
    return result;
  };
  // This is exactly v1's pruning score, with its winning project terms captured
  // during ranking. Independent cards do not share a virtual resource budget.
  function order(
    view: AIObservation,
    playerId: number,
    beforeCompute?: () => void,
  ): Evaluation {
    const h = legacyHorizon(view, playerId);
    const player = view.players.find((p) => p.id === playerId)!;
    const H = h.current ? counts(view.turn.pending) : [0, 0, 0, 0, 0, 0];
    const supply = counts(view.supply.flat());
    const A = view.bagCounts.map((n, c) => n + supply[c]);
    const key = JSON.stringify([playerKey(player, view.side), h, H, A]);
    const cached = orderCache.get(key);
    if (cached) return cached;
    beforeCompute?.();
    orderComputations++;
    const actualScore = score(player, view.side);
    const total = sum(A),
      F = Math.max(0, h.B - sum(H));
    const result: Evaluation = {
      actualScore,
      value: actualScore.total,
      potential: 0,
      R: h.R,
      T: h.turns,
      B: h.B,
      F,
      H,
      A,
      p: A.map((n) => (total ? n / total : 0)),
      projects: [],
    };
    if (h.turns) {
      const best = new Map<number, ProjectEstimate>();
      for (const { project, mByColor } of menu(player, view.side, false)) {
        const card = player.cards.find((c) => c.id === project.cardId)!;
        const data = CARDS[card.id];
        const G = data.spirit
          ? spiritScore(player.board, card.id)
          : data.points[card.placed] - (data.points[card.placed - 1] ?? 0);
        const terms = discount(mByColor, H, A, h.B);
        if (!terms) continue;
        const W = G * terms.q,
          old = best.get(card.id);
        if (!old || W > old.W)
          best.set(card.id, {
            project,
            G,
            mByColor,
            ...terms,
            W,
            B: h.B,
            H,
            A,
          });
      }
      result.projects = [...best.values()];
      result.potential = result.projects.reduce((s, p) => s + p.W, 0);
      result.value += result.potential;
    }
    orderCache.set(key, result);
    return result;
  }
  function evaluateGreedy(
    view: AIObservation,
    playerId = view.players[view.active].id,
    beforeCompute?: () => void,
  ): Evaluation {
    const h =
      version === "v2"
        ? horizon(view, playerId)
        : legacyHorizon(view, playerId);
    const original = view.players.find((p) => p.id === playerId)!;
    const canonical =
      version === "v2"
        ? canonicalize(original)
        : {
            player: original,
            key: playerKey(original, view.side),
            rotation: 0,
            inverse: original.board.map((_, i) => i),
          };
    const player = canonical.player;
    // These are all semantic inputs to V; other players affect only the horizon.
    const key = JSON.stringify([
      playerId,
      view.side,
      canonical.key,
      player.cards,
      h,
      view.turn,
      view.supply,
      view.bagCounts,
    ]);
    let best = cache.get(key);
    if (!best) {
      beforeCompute?.();
      computations++;
      const actualScore = score(player, view.side);
      const supply = counts(view.supply.flat());
      const pool = view.bagCounts.map((n, i) => n + supply[i]);
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
      for (const hand of hands) {
        const initial = discount([0, 0, 0, 0, 0, 0], hand.H, hand.A, h.B)!;
        const result: Evaluation = {
          actualScore,
          value: actualScore.total,
          potential: 0,
          L: "L" in h ? (h.L as number) : undefined,
          R: h.R,
          T: h.turns,
          B: h.B,
          F: initial.F,
          H: [...hand.H],
          A: [...hand.A],
          p: initial.p,
          supplyGroup: hand.group,
          projects: [],
        };
        let virtual = player,
          previousScore = actualScore,
          B = h.B;
        const H = [...hand.H],
          A = [...hand.A];
        for (
          let iteration = 0;
          h.turns && iteration < MAX_PROJECTS;
          iteration++
        ) {
          let chosen: ProjectEstimate | undefined,
            nextPlayer = virtual,
            nextScore = previousScore;
          for (const entry of menu(virtual, view.side)) {
            const { project, mByColor } = entry;
            const terms = discount(mByColor, H, A, B);
            if (!terms) continue;
            const after = (entry.after ??= score(
              applyProject(virtual, project),
              view.side,
            ));
            const G = after.total - previousScore.total;
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
                terrainGain: after.landscape - previousScore.landscape,
                animalGain: after.animals - previousScore.animals,
                spiritGain: after.spirit - previousScore.spirit,
              };
              nextScore = after;
            }
          }
          if (!chosen) break;
          nextPlayer = applyProject(virtual, chosen.project);
          result.projects.push(chosen);
          result.potential += chosen.W;
          B -= chosen.m;
          chosen.mByColor.forEach((n, c) => {
            const held = Math.min(H[c], n);
            H[c] -= held;
            A[c] -= n - held;
          });
          virtual = nextPlayer;
          previousScore = nextScore;
        }
        result.value += result.potential;
        if (!best || rounded(result.value) > rounded(best.value)) best = result;
      }
      cache.set(key, best!);
    }
    if (version !== "v2") return best!;
    return {
      ...best!,
      canonicalRotation: canonical.rotation,
      projects: best!.projects.map((p) => ({
        ...p,
        project: {
          ...p.project,
          targetCell:
            p.project.targetCell === undefined
              ? undefined
              : canonical.inverse[p.project.targetCell],
          additions: p.project.additions.map((a) => ({
            ...a,
            cell: canonical.inverse[a.cell],
          })),
        },
      })),
    };
  }
  function evaluate(
    view: AIObservation,
    playerId = view.players[view.active].id,
    beforeCompute?: () => void,
  ): Evaluation {
    if (version !== "v3.1")
      return evaluateGreedy(view, playerId, beforeCompute);
    const player = view.players.find((p) => p.id === playerId)!;
    const key = JSON.stringify([
      playerId,
      playerKey(player, view.side),
      legacyHorizon(view, playerId),
      view.turn,
      view.supply,
      view.bagCounts,
    ]);
    const cached = monotoneCache.get(key);
    if (cached) return cached;
    const raw = evaluateGreedy(view, playerId, beforeCompute);
    let best = raw,
      ignoredCards: number[] = [];
    // These cards have neither points nor cubes on the board. Removing them
    // only disables future projects; keep all started cards and nature spirits.
    const optional = player.cards.filter(
      (c) => c.placed === 0 && !CARDS[c.id].spirit,
    );
    const fullMask = (1 << optional.length) - 1;
    for (let mask = fullMask - 1; mask >= 0 && raw.T; mask--) {
      const ignored = optional
        .filter((_, i) => !(mask & (1 << i)))
        .map((c) => c.id);
      const variant = {
        ...view,
        players: view.players.map((p) =>
          p.id === playerId
            ? { ...p, cards: p.cards.filter((c) => !ignored.includes(c.id)) }
            : p,
        ),
      };
      const candidate = evaluateGreedy(variant, playerId, beforeCompute);
      // Exact max: rounding here could violate monotonicity by a small amount.
      if (candidate.value > best.value) {
        best = candidate;
        ignoredCards = ignored;
      }
    }
    const result: Evaluation = {
      ...best,
      actualScore: raw.actualScore,
      greedyValue: raw.value,
      ignoredCards,
    };
    // Never cache a partial maximum: interruption must preserve the previous
    // fully evaluated search plan, not publish a non-monotone estimate.
    monotoneCache.set(key, result);
    return result;
  }
  return {
    evaluate,
    order,
    get orderComputations() {
      return orderComputations;
    },
    get computations() {
      return computations;
    },
  };
}
export function evaluatePosition(
  view: AIObservation,
  playerId = view.players[view.active].id,
): Evaluation {
  return createEvaluator().evaluate(view, playerId);
}
