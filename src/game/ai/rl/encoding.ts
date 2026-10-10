import { CARDS, emptyBoard, scorePlayer, type Action } from "../../engine";
import type { AIObservation } from "../types";

// Changing any ordering or scaling here requires a new schema version.
export const VERSION = "harmonies-rl-v1";
export const CELLS = 23;
export const CARD_SLOTS = 32; // Include completed cards, which still score.
export const CARD_INPUTS = 2 * CARD_SLOTS + 5;
export const SCORE_SCALE = 100;
const STACKS = [
  [],
  [1],
  [2],
  [2, 2],
  [2, 2, 2],
  [3],
  [3, 3],
  [4],
  [3, 4],
  [3, 3, 4],
  [5],
  [6],
  [2, 6],
  [3, 6],
  [6, 6],
];
const CELL_TYPES = 1 + 2 * (STACKS.length - 1);
export const FEATURE_SIZE =
  2 * CELLS * CELL_TYPES +
  2 * (CARD_SLOTS + 3) +
  5 * 6 +
  6 +
  2 +
  1 +
  2 +
  2 +
  6 +
  1 +
  32;
const PLACE = 10,
  SETTLE = PLACE + 6 * CELLS;
export const ACTION_SIZE = SETTLE + 32 * CELLS + 1;
export const schema = {
  version: VERSION,
  featureSize: FEATURE_SIZE,
  cardInputs: CARD_INPUTS,
  cardVocabulary: 33,
  actionSize: ACTION_SIZE,
  scoreScale: SCORE_SCALE,
};
export interface Features {
  numeric: number[];
  cards: number[];
}

export function assertSupported(view: AIObservation) {
  if (
    view.side !== "A" ||
    view.players.length !== 2 ||
    view.spirits ||
    view.needsSpiritChoice.length ||
    view.players.some(
      (p) =>
        p.spiritChoices.length || p.cards.some((c) => c.id < 1 || c.id > 32),
    )
  )
    throw new Error("RL v1 only supports side A, two players, and no spirits");
  const shape = emptyBoard("A");
  if (
    view.players.some(
      (p) =>
        p.board.length !== CELLS ||
        p.board.some(
          (c, i) => c.col !== shape[i].col || c.row !== shape[i].row,
        ),
    )
  )
    throw new Error("RL v1 requires the engine's canonical A-side cell order");
}

/** Always active player first; the seat one-hot retains first/second-player order. */
export function encode(view: AIObservation): Features {
  assertSupported(view);
  const numeric: number[] = [],
    cards: number[] = [];
  const players = [view.players[view.active], view.players[1 - view.active]];
  for (const player of players) {
    for (const cell of player.board) {
      const stack = STACKS.findIndex((s) => s.join() === cell.tokens.join());
      if (stack < 0 || (!stack && cell.animal !== null))
        throw new Error("Invalid terrain stack");
      const category =
        stack === 0 ? 0 : 1 + (stack - 1) * 2 + Number(cell.animal !== null);
      for (let i = 0; i < CELL_TYPES; i++) numeric.push(Number(i === category));
    }
    const owned = [...player.cards].sort((a, b) => a.id - b.id);
    for (let i = 0; i < CARD_SLOTS; i++) {
      cards.push(owned[i]?.id ?? 0);
      numeric.push(
        owned[i] ? owned[i].placed / CARDS[owned[i].id].points.length : 0,
      );
    }
    const active = owned.filter(
      (c) => c.placed < CARDS[c.id].points.length,
    ).length;
    numeric.push(
      (4 - active) / 4,
      player.turns / 20,
      scorePlayer(player, "A").total / SCORE_SCALE,
    );
  }
  cards.push(...view.market.map((id) => id ?? 0));
  for (const group of view.supply)
    for (let c = 1; c <= 6; c++)
      numeric.push(group.filter((x) => x === c).length / 3);
  for (let c = 1; c <= 6; c++)
    numeric.push(view.turn.pending.filter((x) => x === c).length / 3);
  numeric.push(
    Number(view.turn.tookTokens),
    Number(view.turn.tookCard),
    view.round / 20,
    Number(view.active === 0),
    Number(view.active === 1),
    Number(view.finalRound !== null),
    (view.finalRound ?? view.round) / 20,
  );
  numeric.push(...view.bagCounts.map((n) => n / 23), view.deckCount / 32);
  for (let c = 1; c <= 32; c++)
    numeric.push(Number(view.unseenAnimals.includes(c)));
  if (numeric.length !== FEATURE_SIZE || cards.length !== CARD_INPUTS)
    throw new Error("Feature schema mismatch");
  return { numeric, cards };
}

/** Placement IDs use color, not the shifting index in pending. */
export function actionId(view: AIObservation, action: Action): number {
  switch (action.type) {
    case "takeTokens":
      return action.group;
    case "takeCard":
      return 5 + action.slot;
    case "place":
      return (
        PLACE + (view.turn.pending[action.pending] - 1) * CELLS + action.cell
      );
    case "settle":
      return SETTLE + (action.card - 1) * CELLS + action.cell;
    case "endTurn":
      if (action.discardSlot !== undefined)
        throw new Error("Solo action not supported");
      return ACTION_SIZE - 1;
    default:
      throw new Error("Spirit action not supported");
  }
}
