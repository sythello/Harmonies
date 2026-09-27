import { CARDS, clone, seededRandom, type Color, type Game } from "../engine";
import { INITIAL_COUNTS } from "./config";
import type { AIObservation, Counts } from "./types";

export const counts = (colors: readonly number[]): Counts => {
  const result = [0, 0, 0, 0, 0, 0];
  for (const color of colors) result[color - 1]++;
  return result;
};
export const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

/** This is the only boundary that receives real hidden arrays. Never copy their contents. */
export function observe(
  game: Game,
  viewerId = game.players[game.active].id,
): AIObservation {
  const used = counts([
    ...game.players.flatMap((p) => p.board.flatMap((c) => c.tokens)),
    ...game.supply.flat(),
    ...game.turn.pending,
    ...game.discardedTokens,
  ]);
  const knownCards = new Set([
    ...game.players.flatMap((p) => p.cards.map((c) => c.id)),
    ...game.market,
    ...game.discardedCards,
  ]);
  return {
    version: game.version,
    players: game.players.map((p) => ({
      ...clone(p),
      spiritChoices: p.id === viewerId ? [...p.spiritChoices] : [],
    })),
    active: game.active,
    round: game.round,
    side: game.side,
    spirits: game.spirits,
    supply: clone(game.supply),
    market: [...game.market],
    discardedTokens: [...game.discardedTokens],
    discardedCards: [...game.discardedCards],
    turn: clone(game.turn),
    finalRound: game.finalRound,
    finalReason: game.finalReason,
    phase: game.phase,
    bagCount: game.bag.length,
    deckCount: game.deck.length,
    bagCounts: INITIAL_COUNTS.map((n, i) => Math.max(0, n - used[i])),
    unseenAnimals: Array.from({ length: 32 }, (_, i) => i + 1).filter(
      (id) => !knownCards.has(id),
    ),
    needsSpiritChoice: game.players
      .filter((p) => p.spiritChoices.length)
      .map((p) => p.id),
  };
}
export function sampleGame(view: AIObservation, seed: number): Game {
  const rng = seededRandom(seed);
  const shuffled = <T>(values: T[]) => {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  };
  const knownSpirits = new Set(
    view.players.flatMap((p) => [
      ...p.cards.map((c) => c.id),
      ...p.spiritChoices,
    ]),
  );
  const spirits = shuffled(
    Object.values(CARDS)
      .filter((c) => c.spirit && !knownSpirits.has(c.id))
      .map((c) => c.id),
  );
  return {
    version: view.version,
    players: view.players.map((p) => ({
      ...clone(p),
      spiritChoices:
        view.needsSpiritChoice.includes(p.id) && !p.spiritChoices.length
          ? spirits.splice(0, 2)
          : [...p.spiritChoices],
    })),
    active: view.active,
    round: view.round,
    side: view.side,
    spirits: view.spirits,
    bag: shuffled(
      view.bagCounts.flatMap((n, i) => Array<Color>(n).fill((i + 1) as Color)),
    ),
    deck: shuffled([...view.unseenAnimals]).slice(0, view.deckCount),
    supply: clone(view.supply),
    market: [...view.market],
    discardedTokens: [...view.discardedTokens],
    discardedCards: [...view.discardedCards],
    turn: clone(view.turn),
    finalRound: view.finalRound,
    finalReason: view.finalReason,
    phase: view.phase,
    log: [],
  };
}
