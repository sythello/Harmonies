import {
  activeCards,
  canEndTurn,
  canPlace,
  CARDS,
  COLORS,
  matches,
  type Action,
  type Game,
  type Player,
} from "../engine";
import { compareKeys } from "./projects";

export const freeCardSlots = (player: Player) => 4 - activeCards(player).length;

export function legalActions(game: Game): Action[] {
  if (game.phase === "ended") return [];
  const player = game.players[game.active];
  if (player.spiritChoices.length)
    return player.spiritChoices.map((card) => ({ type: "chooseSpirit", card }));
  const actions: Action[] = [];
  if (!game.turn.tookTokens)
    game.supply.forEach((group, i) => {
      if (group.length === 3) actions.push({ type: "takeTokens", group: i });
    });
  if (!game.turn.tookCard && activeCards(player).length < 4)
    game.market.forEach((id, slot) => {
      if (id) actions.push({ type: "takeCard", slot });
    });
  const seenColors = new Set<number>();
  game.turn.pending.forEach((color, pending) => {
    if (seenColors.has(color)) return;
    seenColors.add(color);
    player.board.forEach((cell, index) => {
      if (canPlace(cell, color))
        actions.push({ type: "place", pending, cell: index });
    });
  });
  for (const card of activeCards(player))
    for (const match of matches(player.board, card.id))
      actions.push({ type: "settle", card: card.id, cell: match.target });
  if (canEndTurn(game)) {
    actions.push({ type: "endTurn" });
    if (game.players.length === 1 && !game.turn.tookCard && game.deck.length)
      game.market.forEach((id, discardSlot) => {
        if (id) actions.push({ type: "endTurn", discardSlot });
      });
  }
  return actions.sort(compareActions);
}
export const actionKey = (action: Action): number[] => {
  switch (action.type) {
    case "chooseSpirit":
      return [0, action.card];
    case "takeCard":
      return [1, action.slot];
    case "takeTokens":
      return [2, action.group];
    case "place":
      return [3, action.pending, action.cell];
    case "settle":
      return [4, action.card, action.cell];
    case "endTurn":
      return [5, action.discardSlot ?? -1];
  }
};
export const compareActions = (a: Action, b: Action) =>
  compareKeys(actionKey(a), actionKey(b));
export function actionLabel(game: Game, action: Action): string {
  const player = game.players[game.active];
  const cellName = (index: number) =>
    `${player.board[index].col + 1}-${player.board[index].row + 1}`;
  switch (action.type) {
    case "chooseSpirit":
      return `选择${CARDS[action.card].name}`;
    case "takeCard":
      return `拿取${CARDS[game.market[action.slot]!].name}`;
    case "takeTokens":
      return `拿取第 ${action.group + 1} 组：${game.supply[action.group].map((c) => COLORS[c].name).join("、")}`;
    case "place":
      return `在 ${cellName(action.cell)} 放置${COLORS[game.turn.pending[action.pending]].name}`;
    case "settle":
      return `${CARDS[action.card].name}入住 ${cellName(action.cell)}`;
    case "endTurn":
      return action.discardSlot === undefined
        ? "结束回合"
        : `结束回合并替换${CARDS[game.market[action.discardSlot]!].name}`;
  }
}
export function positionKey(game: Game): string {
  const discarded = [0, 0, 0, 0, 0, 0];
  for (const color of game.discardedTokens) discarded[color - 1]++;
  return JSON.stringify([
    game.side,
    discarded,
    [...game.discardedCards].sort((a, b) => a - b),
    game.active,
    game.round,
    game.phase,
    game.finalRound,
    game.bag.length,
    game.deck.length,
    game.supply,
    game.market,
    game.turn,
    game.players.map((p) => [
      p.id,
      p.spiritChoices,
      p.turns,
      p.cards,
      p.board.map((c) => [c.tokens, c.animal]),
    ]),
  ]);
}
